
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from torch.nn.utils import spectral_norm

class FrequencyDecomposer(nn.Module):

    def __init__(self, configs):
        super(FrequencyDecomposer, self).__init__()
        self.decomp_levels = max(3, getattr(configs, 'down_sampling_layers', 0) + 1)
        self.down_sampling_window = max(1, int(getattr(configs, 'down_sampling_window', 1)))
        self.low_gain_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.decomp_levels)
        ])
        self.high_gain_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.decomp_levels)
        ])

    def _depthwise_conv(self, x_bct, kernel_1d, dilation):

        B, C, T = x_bct.shape
        device = x_bct.device
        dtype = x_bct.dtype
        K = len(kernel_1d)
        weight = torch.tensor(kernel_1d, device=device, dtype=dtype).view(1, 1, K)
        weight = weight.repeat(C, 1, 1)
        total_pad = dilation * (K - 1)
        pad_left = int(math.ceil(total_pad / 2))
        pad_right = total_pad - pad_left
        x_padded = F.pad(x_bct, (pad_left, pad_right))
        return F.conv1d(x_padded, weight, bias=None, stride=1, padding=0, dilation=dilation, groups=C)

    def forward(self, x_btc):

        h = [0.5, 0.5]
        g = [0.5, -0.5]

        x_bct = x_btc.permute(0, 2, 1)  # [B, C, T]
        approx = x_bct
        details = []

        for i in range(self.decomp_levels):
            dilation = self.down_sampling_window ** i
            low = self._depthwise_conv(approx, h, dilation)
            high = self._depthwise_conv(approx, g, dilation)

            low_gain = 1.0 + 0.5 * torch.tanh(self.low_gain_params[i])
            high_gain = 1.0 + 0.5 * torch.tanh(self.high_gain_params[i])
            low = low * low_gain
            high = high * high_gain
            details.append(high)  
            approx = low          

        level = approx.permute(0, 2, 1)  # [B, T, C]
        details_ts = [d.permute(0, 2, 1) for d in details]  

        L = len(details_ts)
        if L == 1:
            trend = torch.zeros_like(level)
            season = torch.zeros_like(level)
            explained = level
        elif L == 2:
            trend = details_ts[1]
            season = torch.zeros_like(level)
            explained = level + trend
        else:
            trend = details_ts[-1]                    
            mid_details = details_ts[1:-1]
            if len(mid_details) > 0:
                season = torch.stack(mid_details, dim=0).sum(dim=0)  
            else:
                season = torch.zeros_like(level)
            explained = level + trend + season

        residual = x_btc - explained 

        return level, season, trend, residual


class StablePredictor(nn.Module):

    def __init__(self, configs):
        super(StablePredictor, self).__init__()
        
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.down_sampling_layers = configs.down_sampling_layers
        self.down_sampling_window = configs.down_sampling_window
        self.dropout = getattr(configs, 'dropout', 0.0)

        self.mlp_expansion = int(getattr(configs, 'stable_mlp_expansion', 1))

        self.scale_lengths = [
            configs.seq_len // (self.down_sampling_window ** i)
            for i in range(self.down_sampling_layers + 1)
        ]

        if getattr(configs, 'enable_channel_fuse', False) and (configs.enc_in == configs.c_out):

            self.channel_fuse = nn.Conv1d(configs.c_out, configs.c_out, kernel_size=1, bias=True)
        else:
            self.channel_fuse = nn.Identity()
        self.season_down_mix = nn.ModuleList()
        for i in range(self.down_sampling_layers):
            t_in = self.scale_lengths[i]
            t_out = self.scale_lengths[i + 1]
            self.season_down_mix.append(
                nn.Sequential(
                    nn.Linear(t_in, t_out),
                    nn.GELU(),
                    nn.Linear(t_out, t_out),
                )
            )

        self.trend_up_mix = nn.ModuleList()
        for i in range(self.down_sampling_layers):
            t_in = self.scale_lengths[i + 1]
            t_out = self.scale_lengths[i]
            self.trend_up_mix.append(
                nn.Sequential(
                    nn.Linear(t_in, t_out),
                    nn.GELU(),
                    nn.Linear(t_out, t_out),
                )
            )

        class PreNormTimeMLP(nn.Module):
            def __init__(self, t_in: int, pred_len: int, hidden: int, dropout: float):
                super().__init__()
                self.t_in = t_in
                self.pred_len = pred_len
                self.norm = nn.LayerNorm(t_in)
                self.ff = nn.Sequential(
                    nn.Linear(t_in, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, pred_len),
                )

            def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
                # x_flat: [B*C, T_in]
                x_norm = self.norm(x_flat)
                out = self.ff(x_norm)
                if self.pred_len == self.t_in:
                    out = out + x_flat
                return out

        self.scale_predictors = nn.ModuleList()
        for i in range(self.down_sampling_layers + 1):
            t_in = self.scale_lengths[i]
            hidden = max(self.pred_len, int(self.mlp_expansion * t_in))
            self.scale_predictors.append(
                PreNormTimeMLP(t_in=t_in, pred_len=self.pred_len, hidden=hidden, dropout=self.dropout)
            )
        
        self.num_scales = self.down_sampling_layers + 1
        self.scale_weight_logits = nn.Parameter(torch.zeros(self.num_scales, self.c_out))
        
    def forward(self, level_list, season_list, trend_list):

        num_scales = len(level_list)


        mixed_season = [s.clone() for s in season_list]
        mixed_trend = [t.clone() for t in trend_list]

        def apply_time_mlp(x_bct, mlp: nn.Module):
            B, C, T_in = x_bct.shape
            x_flat = x_bct.reshape(B * C, T_in)
            y_flat = mlp(x_flat)
            T_out = y_flat.shape[-1]
            return y_flat.view(B, C, T_out)

        for i in range(num_scales - 1):
            res = apply_time_mlp(mixed_season[i], self.season_down_mix[i])
            mixed_season[i + 1] = mixed_season[i + 1] + res


        for i in reversed(range(num_scales - 1)):
            res = apply_time_mlp(mixed_trend[i + 1], self.trend_up_mix[i])
            mixed_trend[i] = mixed_trend[i] + res

        y_list = []  
        for i in range(num_scales):
            level_c = level_list[i]     
            season_c = mixed_season[i] 
            trend_c = mixed_trend[i]    
            stable_component = level_c + season_c + trend_c  

            stable_component = self.channel_fuse(stable_component)

            pred_current = self.scale_predictors[i](
                stable_component.reshape(-1, stable_component.shape[-1])
            ).view(stable_component.shape[0], stable_component.shape[1], self.pred_len)
            y_i = pred_current.transpose(1, 2)
            y_list.append(y_i)

        weights = F.softmax(self.scale_weight_logits, dim=0)  
        y_stack = torch.stack(y_list, dim=2)  
        y_stable = (y_stack * weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)  
        return y_stable


class UncertaintyPredictor(nn.Module):

    def __init__(self, configs):
        super(UncertaintyPredictor, self).__init__()
        
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.trend_detach_in_uncertainty = getattr(configs, 'trend_detach_in_uncertainty', True)
        
        depth_groups = configs.enc_in
        self.depthwise_conv1 = nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=3, padding=1, dilation=1, groups=depth_groups)
        self.depthwise_conv2 = nn.Conv1d(configs.enc_in, configs.enc_in, kernel_size=3, padding=2, dilation=2, groups=depth_groups)

        pw_groups = configs.enc_in if (configs.c_out % configs.enc_in == 0) else 1
        self.pointwise_proj = nn.Conv1d(configs.enc_in, configs.c_out, kernel_size=1, groups=pw_groups)
        self.act = nn.ReLU()
        
        self.mu_head = nn.Conv1d(configs.c_out * 2, configs.c_out, kernel_size=1, groups=configs.c_out)
        self.log_sigma_head = nn.Conv1d(configs.c_out * 2, configs.c_out, kernel_size=1, groups=configs.c_out)


    def forward(self, residual_list, trend_list):

        residual = residual_list[0]  
        trend = trend_list[0].permute(0, 2, 1) 
        if self.trend_detach_in_uncertainty:
            trend = trend.detach()

        residual_conv = residual.transpose(1, 2)  
        trend_conv = trend.transpose(1, 2)       

        residual_dw = self.act(self.depthwise_conv1(residual_conv))
        residual_dw = self.act(self.depthwise_conv2(residual_dw))
        residual_encoded = self.pointwise_proj(residual_dw) 

        trend_dw = self.act(self.depthwise_conv1(trend_conv))
        trend_dw = self.act(self.depthwise_conv2(trend_dw))
        trend_encoded = self.pointwise_proj(trend_dw)       

        combined_seq = torch.cat([residual_encoded, trend_encoded], dim=1)  
        mu_seq = self.mu_head(combined_seq)          
        log_sigma_raw = self.log_sigma_head(combined_seq) 
        sigma_seq = F.softplus(log_sigma_raw)         

        mu_seq_up = F.interpolate(mu_seq, size=self.pred_len, mode='linear', align_corners=False)     
        sigma_seq_up = F.interpolate(sigma_seq, size=self.pred_len, mode='linear', align_corners=False)  
        y_residual = mu_seq_up.transpose(1, 2) 
        risk_pred = sigma_seq_up.transpose(1, 2) 

        return y_residual, risk_pred


class RiskAwareFusion(nn.Module):

    def __init__(self, configs):
        super(RiskAwareFusion, self).__init__()
        
        self.c_out = configs.c_out
        
        self.alpha_raw = nn.Parameter(torch.zeros(self.c_out))
        self.tau_raw = nn.Parameter(torch.zeros(self.c_out))
        self.beta = nn.Parameter(torch.zeros(self.c_out))

        self.risk_scale_raw = nn.Parameter(torch.zeros(self.c_out))
        self.risk_bias = nn.Parameter(torch.zeros(self.c_out))
        self.risk_clip_k = 3.0  
        
    def forward(self, y_stable, y_residual, risk_pred):

        B, pred_len, c_out = risk_pred.shape

        risk_tc = risk_pred.permute(0, 2, 1)  
        risk_tc = F.pad(risk_tc, (1, 1), mode='replicate')
        risk_smooth_tc = F.avg_pool1d(risk_tc, kernel_size=3, stride=1)  
        risk_smooth = risk_smooth_tc.permute(0, 2, 1) 

        q_low = torch.quantile(risk_smooth, q=0.10, dim=1, keepdim=True)   
        q_high = torch.quantile(risk_smooth, q=0.90, dim=1, keepdim=True)  
        denom = torch.clamp(q_high - q_low, min=1e-6)
        risk_cal = torch.clamp((risk_smooth - q_low) / denom, 0.0, 1.0)    

        alpha = F.softplus(self.alpha_raw).view(1, 1, -1)               
        tau = F.softplus(self.tau_raw).view(1, 1, -1) + 1e-3            
        beta = self.beta.view(1, 1, -1)                                
        gate_logits = (alpha * risk_cal + beta) / tau                  
        gate_values = torch.sigmoid(gate_logits)
        
        y_final = y_stable + gate_values * y_residual  
        
        return y_final, gate_values


class RATE(nn.Module):

    
    def __init__(self, configs):
        super(RATE, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        self.decomposition = FrequencyDecomposer(configs)
        
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        
        self.variable_interact_linear = spectral_norm(
            nn.Linear(self.enc_in, self.enc_in, bias=False)
        )
        offdiag = torch.ones(self.enc_in, self.enc_in) - torch.eye(self.enc_in)
        self.register_buffer('variable_interact_offdiag_mask', offdiag)

        self.s_max_interact = float(getattr(configs, 's_max_interact', 0))
        init_ratio = min(0.1, max(0.001, 0.05 / max(self.s_max_interact, 1e-6)))
        init_logit = math.log(init_ratio / (1.0 - init_ratio))
        self.interact_gate_raw = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        

        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.film_proj = nn.Linear(configs.d_model, 2 * configs.enc_in)
        self.beta_gate_raw = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        
        self.stable_predictor = StablePredictor(configs)
        
        self.uncertainty_predictor = UncertaintyPredictor(configs)
        
        self.risk_aware_fusion = RiskAwareFusion(configs)

        k = 3
        stride = max(1, int(self.configs.down_sampling_window))
        padding = k // 2
        self.downsample_conv = nn.Conv1d(
            in_channels=self.enc_in,
            out_channels=self.enc_in,
            kernel_size=k,
            stride=stride,
            padding=padding,
            groups=self.enc_in,
            bias=False,
        )
        with torch.no_grad():
            lp = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32).view(1, 1, k)
            weight = lp.repeat(self.enc_in, 1, 1)
            self.downsample_conv.weight.copy_(weight)
        for p in self.downsample_conv.parameters():
            p.requires_grad = False

    def __multi_scale_process_inputs(self, x_enc):
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc

        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for _ in range(self.configs.down_sampling_layers):
            x_enc_sampling = self.downsample_conv(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def forward(self, x_enc, x_mark_enc=None):

        B, L, C = x_enc.shape
        

        x_list = self.__multi_scale_process_inputs(x_enc)

        enhanced_x_list = []
        time_emb_base = None
        if x_mark_enc is not None:
            time_emb_base = self.enc_embedding(None, x_mark_enc)
            base_len = time_emb_base.shape[1]

        for i, x in enumerate(x_list):
            x_norm = self.normalize_layers[i](x, 'norm')
            if time_emb_base is not None:
                cur_len = x_norm.shape[1]
                if cur_len != base_len:
                    time_emb_scaled = F.interpolate(
                        time_emb_base.transpose(1, 2),  
                        size=cur_len,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  
                else:
                    time_emb_scaled = time_emb_base

                film_params = self.film_proj(time_emb_scaled) 
                gamma, beta_raw = torch.chunk(film_params, 2, dim=-1)  
                beta = torch.sigmoid(self.beta_gate_raw) * beta_raw
                x_norm = (1.0 + torch.tanh(gamma)) * x_norm + beta

            if self.s_max_interact <= 0.0:
                enhanced_x_list.append(x_norm)
            else:
                offdiag_mask = self.variable_interact_offdiag_mask.to(self.variable_interact_linear.weight.device)
                masked_w = self.variable_interact_linear.weight * offdiag_mask  # [C, C]

                masked_w = masked_w.to(x_norm.device)
                interact_part = F.linear(x_norm, masked_w, bias=None)  # [B, T, C]
                lambda_interact = torch.sigmoid(self.interact_gate_raw) * self.s_max_interact
                x_enhanced = x_norm + lambda_interact * interact_part
                enhanced_x_list.append(x_enhanced)

        level_list = []
        season_list = []
        trend_list = []
        residual_list = []
        
        for x in enhanced_x_list:
        
            level, season, trend, residual = self.decomposition(x)
            
            level_list.append(level.permute(0, 2, 1))     
            season_list.append(season.permute(0, 2, 1))   
            trend_list.append(trend.permute(0, 2, 1))     
            residual_list.append(residual)              
        
        decomposed_level = level_list[0].permute(0, 2, 1)
        decomposed_season = season_list[0].permute(0, 2, 1)
        decomposed_trend = trend_list[0].permute(0, 2, 1)
        decomposed_residual = residual_list[0]
        
        y_stable = self.stable_predictor(level_list, season_list, trend_list)
        
        y_residual, risk_pred = self.uncertainty_predictor(residual_list, trend_list)
        
        y_final, gate_values = self.risk_aware_fusion(y_stable, y_residual, risk_pred)
        
        y_final = self.normalize_layers[0](y_final, 'denorm')
        y_stable = self.normalize_layers[0](y_stable, 'denorm')
        y_residual = self.normalize_layers[0](y_residual, 'denorm')

        decomposed_level = self.normalize_layers[0](decomposed_level, 'denorm')
        decomposed_season = self.normalize_layers[0](decomposed_season, 'denorm')
        decomposed_trend = self.normalize_layers[0](decomposed_trend, 'denorm')
        decomposed_residual = self.normalize_layers[0](decomposed_residual, 'denorm')
        
        return y_final, y_stable, y_residual, risk_pred, gate_values, decomposed_level, decomposed_season, decomposed_trend, decomposed_residual