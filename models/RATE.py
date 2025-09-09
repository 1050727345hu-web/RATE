import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

class FrequencyDecomposer(nn.Module):

    def __init__(self, configs):
        super(FrequencyDecomposer, self).__init__()
        self.decomp_levels = max(3, getattr(configs, 'down_sampling_layers', 0) + 1)

    def _depthwise_conv(self, x_bct, kernel_1d, dilation):

        B, C, T = x_bct.shape
        device = x_bct.device
        dtype = x_bct.dtype
        K = len(kernel_1d)
        weight = torch.tensor(kernel_1d, device=device, dtype=dtype).view(1, 1, K)
        weight = weight.repeat(C, 1, 1)  # [C,1,K]

        total_pad = dilation * (K - 1)
        pad_left = int(math.ceil(total_pad / 2))
        pad_right = total_pad - pad_left
        x_padded = F.pad(x_bct, (pad_left, pad_right))
        return F.conv1d(x_padded, weight, bias=None, stride=1, padding=0, dilation=dilation, groups=C)

    def forward(self, x_btc):

        h = [0.5, 0.5]
        g = [0.5, -0.5]

        x_bct = x_btc.permute(0, 2, 1)
        approx = x_bct
        details = []

        for i in range(self.decomp_levels):
            dilation = 1 << i
            low = self._depthwise_conv(approx, h, dilation)
            high = self._depthwise_conv(approx, g, dilation)
            details.append(high)
            approx = low
        level = approx.permute(0, 2, 1)
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
        
        # 为每个尺度创建预测器
        self.scale_predictors = nn.ModuleList([
            nn.Linear(
                configs.seq_len // (configs.down_sampling_window ** i), 
                configs.pred_len
            )
            for i in range(configs.down_sampling_layers + 1)
        ])
        
    def forward(self, level_list, season_list, trend_list):

        pred_coarse = None
        
        for i in reversed(range(len(level_list))):
            # 当前尺度的稳定分量
            level = level_list[i].permute(0, 2, 1)
            season = season_list[i].permute(0, 2, 1)
            trend = trend_list[i].permute(0, 2, 1)
            
            # 组合稳定分量
            stable_component = level + season + trend
            
            # 如果有来自粗尺度的预测，则融入当前尺度
            if pred_coarse is not None:
                # 将粗尺度预测上采样到当前尺度的序列长度
                current_seq_len = stable_component.shape[1]
                pred_upsampled = F.interpolate(
                    pred_coarse.transpose(1, 2),
                    size=current_seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                
                # 残差融入：用粗尺度的长期预测指导细尺度输入
                stable_component = stable_component + pred_upsampled

            pred_current = self.scale_predictors[i](stable_component.transpose(1, 2)).transpose(1, 2)
            

            pred_coarse = pred_current

        y_stable = pred_coarse
        
        return y_stable


class UncertaintyPredictor(nn.Module):

    def __init__(self, configs):
        super(UncertaintyPredictor, self).__init__()
        
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        

        self.tcn_encoder = nn.Sequential(
            nn.Conv1d(configs.enc_in, configs.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        

        self.residual_projector = nn.Linear(configs.d_model * 2, configs.pred_len * configs.c_out)
        self.risk_projector = nn.Linear(configs.d_model * 2, configs.pred_len * configs.c_out)

    def forward(self, residual_list, trend_list):

        prev_risk_feature = None
        
        residual_pred_list = []
        risk_pred_list = []

        for i in reversed(range(len(residual_list))):
            residual = residual_list[i]
            trend = trend_list[i].permute(0, 2, 1)

            residual_conv = residual.transpose(1, 2)
            residual_encoded = self.tcn_encoder(residual_conv)
            residual_pooled = self.global_pool(residual_encoded).squeeze(-1)

            trend_conv = trend.transpose(1, 2)
            trend_encoded = self.tcn_encoder(trend_conv)
            trend_pooled = self.global_pool(trend_encoded).squeeze(-1)

            combined_feature = torch.cat([residual_pooled, trend_pooled], dim=-1)

            if prev_risk_feature is not None:
                combined_feature = combined_feature + prev_risk_feature

            residual_pred = self.residual_projector(combined_feature).view(-1, self.pred_len, self.c_out)

            risk_pred = self.risk_projector(combined_feature).view(-1, self.pred_len, self.c_out)
            risk_pred = F.softplus(risk_pred)
            
            residual_pred_list.append(residual_pred)
            risk_pred_list.append(risk_pred)

            prev_risk_feature = combined_feature
        
        # 聚合多尺度预测
        y_residual = torch.stack(residual_pred_list, dim=-1).mean(dim=-1)
        risk_pred = torch.stack(risk_pred_list, dim=-1).mean(dim=-1)
        
        return y_residual, risk_pred


class RiskAwareFusion(nn.Module):

    def __init__(self, configs):
        super(RiskAwareFusion, self).__init__()
        
        self.c_out = configs.c_out

        self.risk_to_gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.risk_bias = nn.Parameter(torch.zeros(self.c_out))
        
    def forward(self, y_stable, y_residual, risk_pred):

        B, pred_len, c_out = risk_pred.shape

        adjusted_risk = risk_pred + self.risk_bias.view(1, 1, -1)
        

        adjusted_risk_flat = adjusted_risk.view(-1, 1)
        gate_values_flat = 1.0 - self.risk_to_gate(adjusted_risk_flat)
        gate_values = gate_values_flat.view(B, pred_len, c_out)

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
        

        self.variable_interact_matrix = nn.Parameter(torch.randn(self.enc_in, self.enc_in) * 0.01)
        

        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.film_proj = nn.Linear(configs.d_model, 2 * configs.enc_in)
        

        self.stable_predictor = StablePredictor(configs)
        

        self.uncertainty_predictor = UncertaintyPredictor(configs)
        

        self.risk_aware_fusion = RiskAwareFusion(configs)

    def __multi_scale_process_inputs(self, x_enc):

        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        

        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
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
            # 归一化
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
                gamma, beta = torch.chunk(film_params, 2, dim=-1)
                x_norm = (1.0 + torch.tanh(gamma)) * x_norm + beta

            x_enhanced = x_norm + torch.matmul(x_norm, self.variable_interact_matrix)
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