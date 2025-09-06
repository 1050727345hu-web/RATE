"""
RiskMixer: Risk-Aware Time Series Forecasting with Dynamic Fusion
基于风险感知的时间序列预测模型，支持动态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize

class FrequencyDecomposer(nn.Module):
    """
    基于互不重叠频带的极简四分法（à trous/Haar 深度可分离卷积）
    - 使用固定的 Haar 滤波器核在多尺度上进行平滑(近似低通)与细节(带通)提取
    - 频带划分：
        level  : 最低频(最终近似)
        trend  : 次低频细节(最粗尺度的细节)
        season : 中频细节(中间尺度细节之和)
        residual: x - (level + trend + season) 作为不可解释/非平稳部分
    - 设计目标：互不重叠频带(按尺度带通划分) + 极简、可微、无额外依赖
    """
    def __init__(self, configs):
        super(FrequencyDecomposer, self).__init__()
        # 至少 3 个频带以形成 level/trend/season 三段；更多层数 -> 更细频带
        self.decomp_levels = max(3, getattr(configs, 'down_sampling_layers', 0) + 1)

    def _depthwise_conv(self, x_bct, kernel_1d, dilation):
        """
        x_bct: [B, C, T]
        kernel_1d: list/tuple of length K
        depthwise conv with groups=C, same-length padding
        """
        B, C, T = x_bct.shape
        device = x_bct.device
        dtype = x_bct.dtype
        K = len(kernel_1d)
        weight = torch.tensor(kernel_1d, device=device, dtype=dtype).view(1, 1, K)
        weight = weight.repeat(C, 1, 1)  # [C,1,K]
        # 为确保输出长度与输入一致：对偶长度核(K=2)与任意 dilation 时采用非对称预填充
        # 令总填充量 = dilation * (K - 1)
        total_pad = dilation * (K - 1)
        pad_left = int(math.ceil(total_pad / 2))
        pad_right = total_pad - pad_left
        x_padded = F.pad(x_bct, (pad_left, pad_right))
        return F.conv1d(x_padded, weight, bias=None, stride=1, padding=0, dilation=dilation, groups=C)

    def forward(self, x_btc):
        """
        Args:
            x_btc: [B, T, C]
        Returns:
            level, season, trend, residual: 全部为 [B, T, C]
        """
        # Haar 核（简化能量缩放，保持极简）：
        # 低通: [0.5, 0.5]  高通: [0.5, -0.5]
        h = [0.5, 0.5]
        g = [0.5, -0.5]

        x_bct = x_btc.permute(0, 2, 1)  # [B, C, T]
        approx = x_bct
        details = []

        for i in range(self.decomp_levels):
            dilation = 1 << i  # 2^i
            low = self._depthwise_conv(approx, h, dilation)
            high = self._depthwise_conv(approx, g, dilation)
            details.append(high)  # 该尺度的带通信号
            approx = low          # 送入下一尺度

        level = approx.permute(0, 2, 1)  # [B, T, C]
        details_ts = [d.permute(0, 2, 1) for d in details]  # 每个为 [B, T, C]

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
            trend = details_ts[-1]                     # 最粗尺度细节 -> 次低频带
            mid_details = details_ts[1:-1]
            if len(mid_details) > 0:
                season = torch.stack(mid_details, dim=0).sum(dim=0)  # 中频带合成
            else:
                season = torch.zeros_like(level)
            explained = level + trend + season

        residual = x_btc - explained  # 不可解释/非平稳部分

        return level, season, trend, residual


class StablePredictor(nn.Module):
    """
    稳定分量预测器 (级联精炼设计)
    输入: Level (长期基准) + Season + Trend (可预测的周期性和趋势)
    输出: 未来稳定预测 y_stable (级联精炼，从粗尺度到细尺度)
    注意: Level是最稳定的基准，Season+Trend是可预测但有一定波动的模式
    """
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
        """
        级联精炼流程：从最粗尺度开始，逐步精炼到最细尺度
        Args:
            level_list: 多尺度基础值序列 [B, C, T_i]
            season_list: 多尺度季节序列 [B, C, T_i]
            trend_list: 多尺度趋势序列 [B, C, T_i]
        Returns:
            y_stable: 稳定分量预测 [B, pred_len, c_out]
        """
        # 倒序处理：从最粗尺度到最细尺度
        pred_coarse = None
        
        for i in reversed(range(len(level_list))):
            # 当前尺度的稳定分量
            level = level_list[i].permute(0, 2, 1)    # [B, C, T_i] -> [B, T_i, C]
            season = season_list[i].permute(0, 2, 1)  # [B, C, T_i] -> [B, T_i, C]
            trend = trend_list[i].permute(0, 2, 1)    # [B, C, T_i] -> [B, T_i, C]
            
            # 组合稳定分量
            stable_component = level + season + trend  # [B, T_i, C]
            
            # 如果有来自粗尺度的预测，则融入当前尺度
            if pred_coarse is not None:
                # 将粗尺度预测上采样到当前尺度的序列长度
                current_seq_len = stable_component.shape[1]
                pred_upsampled = F.interpolate(
                    pred_coarse.transpose(1, 2),  # [B, pred_len, C] -> [B, C, pred_len]
                    size=current_seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # [B, C, T_i] -> [B, T_i, C]
                
                # 残差融入：用粗尺度的长期预测指导细尺度输入
                stable_component = stable_component + pred_upsampled
            
            # 当前尺度预测：[B, T_i, C] -> [B, C, T_i] -> [B, C, pred_len] -> [B, pred_len, C]
            pred_current = self.scale_predictors[i](stable_component.transpose(1, 2)).transpose(1, 2)
            
            # 更新粗尺度预测
            pred_coarse = pred_current
        
        # 最终预测来自最细尺度（原始尺度）
        y_stable = pred_coarse  # [B, pred_len, c_out]
        
        return y_stable


class UncertaintyPredictor(nn.Module):
    """
    不确定性/风险预测器 (极简化设计 + 趋势感知 + 层级预测)
    输入: Residual + Trend (残差波动 + 趋势不确定性)
    输出: y_residual (残差预测) + risk_pred (通道级风险强度)
    注意: 同时关注高频残差和低频趋势的不确定性，实现层级式风险预测
    """
    def __init__(self, configs):
        super(UncertaintyPredictor, self).__init__()
        
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        
        # 轻量级TCN编码器 - 使用1D卷积捕捉局部波动特征
        self.tcn_encoder = nn.Sequential(
            nn.Conv1d(configs.enc_in, configs.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(configs.d_model, configs.d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 全局平均池化 - 将序列信息压缩为特征向量
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分离式预测头
        self.residual_projector = nn.Linear(configs.d_model * 2, configs.pred_len * configs.c_out)  # 融合残差+趋势特征
        self.risk_projector = nn.Linear(configs.d_model * 2, configs.pred_len * configs.c_out)  # 通道级风险预测

    def forward(self, residual_list, trend_list):
        """
        层级式不确定性预测：从粗尺度到细尺度，融合残差+趋势不确定性
        Args:
            residual_list: 多尺度残差列表 [B, T_i, C]
            trend_list: 多尺度趋势列表 [B, C, T_i]
        Returns:
            y_residual: 残差预测 [B, pred_len, c_out]
            risk_pred: 通道级风险强度预测 [B, pred_len, c_out]
        """
        # 层级式预测：从最粗尺度开始，逐步精炼
        prev_risk_feature = None
        
        residual_pred_list = []
        risk_pred_list = []
        
        # 倒序处理：从最粗尺度到最细尺度
        for i in reversed(range(len(residual_list))):
            residual = residual_list[i]  # [B, T_i, C]
            trend = trend_list[i].permute(0, 2, 1)  # [B, C, T_i] -> [B, T_i, C]
            
            # 残差特征编码
            residual_conv = residual.transpose(1, 2)  # [B, C, T_i]
            residual_encoded = self.tcn_encoder(residual_conv)  # [B, d_model, T_i]
            residual_pooled = self.global_pool(residual_encoded).squeeze(-1)  # [B, d_model]
            
            # 趋势特征编码
            trend_conv = trend.transpose(1, 2)  # [B, C, T_i]
            trend_encoded = self.tcn_encoder(trend_conv)  # [B, d_model, T_i]
            trend_pooled = self.global_pool(trend_encoded).squeeze(-1)  # [B, d_model]
            
            # 融合残差+趋势特征
            combined_feature = torch.cat([residual_pooled, trend_pooled], dim=-1)  # [B, d_model * 2]
            
            # 层级融合：如果有来自粗尺度的风险特征，则进行融合
            if prev_risk_feature is not None:
                combined_feature = combined_feature + prev_risk_feature
            
            # 残差预测: [B, d_model * 2] -> [B, pred_len * c_out] -> [B, pred_len, c_out]
            residual_pred = self.residual_projector(combined_feature).view(-1, self.pred_len, self.c_out)
            
            # 通道级风险预测: [B, d_model * 2] -> [B, pred_len * c_out] -> [B, pred_len, c_out]
            risk_pred = self.risk_projector(combined_feature).view(-1, self.pred_len, self.c_out)
            risk_pred = F.softplus(risk_pred)  # 保证非负
            
            residual_pred_list.append(residual_pred)
            risk_pred_list.append(risk_pred)
            
            # 更新粗尺度风险特征供下一层使用
            prev_risk_feature = combined_feature
        
        # 聚合多尺度预测
        y_residual = torch.stack(residual_pred_list, dim=-1).mean(dim=-1)  # [B, pred_len, c_out]
        risk_pred = torch.stack(risk_pred_list, dim=-1).mean(dim=-1)  # [B, pred_len, c_out]
        
        return y_residual, risk_pred


class RiskAwareFusion(nn.Module):
    """
    风险感知动态融合器 (通道级门控)
    实现: y_final(t,c) = y_stable(t,c) + gate(t,c) * y_residual(t,c)
    其中 gate(t,c) 由每个通道的 risk_pred(t,c) 独立控制
    """
    def __init__(self, configs):
        super(RiskAwareFusion, self).__init__()
        
        self.c_out = configs.c_out
        
        # 风险到门控的映射 (处理单个风险值)
        self.risk_to_gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出[0,1]范围的门控值
        )
        
        # 可学习的风险阈值参数 (每个通道独立)
        self.risk_bias = nn.Parameter(torch.zeros(self.c_out))
        
    def forward(self, y_stable, y_residual, risk_pred):
        """
        Args:
            y_stable: 稳定分量预测 [B, pred_len, c_out]
            y_residual: 残差分量预测 [B, pred_len, c_out]
            risk_pred: 通道级风险强度 [B, pred_len, c_out]
        Returns:
            y_final: 最终融合预测 [B, pred_len, c_out]
            gate_values: 通道级门控值 [B, pred_len, c_out] (用于分析)
        """
        B, pred_len, c_out = risk_pred.shape
        
        # 调整风险强度 (每个通道独立的偏置)
        adjusted_risk = risk_pred + self.risk_bias.view(1, 1, -1)  # [B, pred_len, c_out]
        
        # 计算门控值: 风险低时gate接近1，风险高时gate接近0
        # 对每个风险值独立处理
        adjusted_risk_flat = adjusted_risk.view(-1, 1)  # [B*pred_len*c_out, 1]
        gate_values_flat = 1.0 - self.risk_to_gate(adjusted_risk_flat)  # [B*pred_len*c_out, 1]
        gate_values = gate_values_flat.view(B, pred_len, c_out)  # [B, pred_len, c_out]
        
        # 动态融合: y_final = y_stable + gate * y_residual
        y_final = y_stable + gate_values * y_residual  # [B, pred_len, c_out]
        
        return y_final, gate_values


class RiskMixer(nn.Module):
    """
    RiskMixer主模型 - 基于级联分解的风险感知时间序列预测
    
    核心设计：
    1. 级联分解：Level(基准) + Season+Trend(可预测) + Residual(动态波动)
    2. 并行预测：StablePredictor处理确定性，UncertaintyPredictor处理不确定性
    3. 风险融合：根据预测风险动态调整稳定预测和残差预测的权重
    """
    
    def __init__(self, configs):
        super(RiskMixer, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        
        # 1. 序列分解器（四分法）
        self.decomposition = FrequencyDecomposer(configs)
        
        # 2. 数据归一化层
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.enc_in, affine=True, non_norm=False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        
        # 3. 全局变量交互矩阵
        self.variable_interact_matrix = nn.Parameter(torch.randn(self.enc_in, self.enc_in) * 0.01)
        
        # 4. 嵌入层
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        # FiLM: 时间嵌入到通道维度的缩放与偏置 [d_model] -> [2*C]
        self.film_proj = nn.Linear(configs.d_model, 2 * configs.enc_in)
        
        # 5. 稳定分量预测器 (Level + Season + Trend)
        self.stable_predictor = StablePredictor(configs)
        
        # 6. 不确定性预测器 (Residual -> y_residual + risk_pred)
        self.uncertainty_predictor = UncertaintyPredictor(configs)
        
        # 7. 风险感知融合器
        self.risk_aware_fusion = RiskAwareFusion(configs)

    def __multi_scale_process_inputs(self, x_enc):
        """多尺度处理输入序列"""
        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        
        # B,T,C -> B,C,T
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
        """
        前向传播：并行预测 + 风险动态融合 (趋势感知 + 通道级门控 + 层级式不确定性)
        Args:
            x_enc: 输入序列 [B, seq_len, enc_in]
            x_mark_enc: 时间特征 [B, seq_len, time_features] (可选)
        Returns:
            y_final: 最终融合预测 [B, pred_len, c_out]
            y_stable: 稳定分量预测 [B, pred_len, c_out]
            y_residual: 残差分量预测 [B, pred_len, c_out]
            risk_pred: 通道级风险强度 [B, pred_len, c_out]
            gate_values: 通道级门控值 [B, pred_len, c_out]
        """
        B, L, C = x_enc.shape
        
        # 1. 多尺度处理
        x_list = self.__multi_scale_process_inputs(x_enc)
        
        # 2. 归一化 + FiLM 时间调制 + 全局变量交互
        enhanced_x_list = []
        # 计算原始长度的时间嵌入，后续各尺度做对齐
        time_emb_base = None
        if x_mark_enc is not None:
            # 仅取时间嵌入 [B, L, d_model]
            time_emb_base = self.enc_embedding(None, x_mark_enc)
            base_len = time_emb_base.shape[1]

        for i, x in enumerate(x_list):
            # 归一化
            x_norm = self.normalize_layers[i](x, 'norm')
            # FiLM: 利用时间嵌入对每个时间步、每个通道进行条件缩放与偏置
            if time_emb_base is not None:
                cur_len = x_norm.shape[1]
                if cur_len != base_len:
                    # 线性插值到当前尺度长度
                    time_emb_scaled = F.interpolate(
                        time_emb_base.transpose(1, 2),  # [B, d_model, L]
                        size=cur_len,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # [B, cur_len, d_model]
                else:
                    time_emb_scaled = time_emb_base

                film_params = self.film_proj(time_emb_scaled)  # [B, cur_len, 2*C]
                gamma, beta = torch.chunk(film_params, 2, dim=-1)  # [B, cur_len, C] each
                x_norm = (1.0 + torch.tanh(gamma)) * x_norm + beta

            # 全局变量交互: x_enhanced = x_norm + x_norm @ W_interact
            # x_norm: [B, T, C], variable_interact_matrix: [C, C]
            x_enhanced = x_norm + torch.matmul(x_norm, self.variable_interact_matrix)
            enhanced_x_list.append(x_enhanced)
        
        # 3. 级联分解：Level + Season + Trend + Residual
        level_list = []
        season_list = []
        trend_list = []
        residual_list = []
        
        for x in enhanced_x_list:
            # 使用 FrequencyDecomposer 进行四分法分解
            level, season, trend, residual = self.decomposition(x)
            
            level_list.append(level.permute(0, 2, 1))     # [B, C, T]
            season_list.append(season.permute(0, 2, 1))   # [B, C, T]
            trend_list.append(trend.permute(0, 2, 1))     # [B, C, T]
            residual_list.append(residual)                # [B, T, C]
        
        # 为可视化提取原始尺度的分量
        decomposed_level = level_list[0].permute(0, 2, 1)
        decomposed_season = season_list[0].permute(0, 2, 1)
        decomposed_trend = trend_list[0].permute(0, 2, 1)
        decomposed_residual = residual_list[0]
        
        # 4. 并行一步到位预测
        # 4.1 稳定分量预测 (Level基准 + Season季节性 + Trend趋势性 -> y_stable)
        y_stable = self.stable_predictor(level_list, season_list, trend_list)
        
        # 4.2 不确定性预测 (Residual + Trend -> y_residual + 通道级risk_pred)
        y_residual, risk_pred = self.uncertainty_predictor(residual_list, trend_list)
        
        # 5. 风险感知动态融合
        y_final, gate_values = self.risk_aware_fusion(y_stable, y_residual, risk_pred)
        
        # 6. 反归一化
        y_final = self.normalize_layers[0](y_final, 'denorm')
        y_stable = self.normalize_layers[0](y_stable, 'denorm')
        y_residual = self.normalize_layers[0](y_residual, 'denorm')

        # 为可视化反归一化分解的分量
        decomposed_level = self.normalize_layers[0](decomposed_level, 'denorm')
        decomposed_season = self.normalize_layers[0](decomposed_season, 'denorm')
        decomposed_trend = self.normalize_layers[0](decomposed_trend, 'denorm')
        decomposed_residual = self.normalize_layers[0](decomposed_residual, 'denorm')
        
        return y_final, y_stable, y_residual, risk_pred, gate_values, decomposed_level, decomposed_season, decomposed_trend, decomposed_residual