"""
多尺度时序分割模块 (Multi-Scale Time Series Decomposition Module)
该模块整合了：
1. FFT周期识别与Top-k选择
2. 多尺度片段分割
3. 片段内/跨片段编码增强
4. 多尺度协同预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional
from layers.Embed import PatchEmbedding
from layers.Autoformer_EncDec import series_decomp_multi


class MultiScaleFFTAnalyzer:
    """多尺度FFT分析器，用于识别时间序列中的主要周期模式"""
    
    @staticmethod
    def get_top_k_periods(x: torch.Tensor, top_k: int = 5) -> List[int]:
        """
        使用FFT识别时间序列的Top-k周期
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, n_vars]
            top_k: 返回的周期数量
            
        Returns:
            periods: Top-k周期列表
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            
        batch_size, seq_len, n_vars = x.shape
        all_periods = []
        
        for i in range(batch_size):
            # 对每个batch进行FFT
            ft = np.fft.rfft(x[i], axis=0)
            freqs = np.fft.rfftfreq(seq_len, 1)
            
            # 计算平均幅度谱
            mags = abs(ft).mean(axis=-1)
            
            # 寻找峰值
            inflection = np.diff(np.sign(np.diff(mags)))
            peaks = (inflection < 0).nonzero()[0] + 1
            
            # 选择Top-k峰值对应的周期
            batch_periods = []
            for _ in range(min(top_k, len(peaks))):
                if len(peaks) > 0:
                    max_index = np.argmax(mags[peaks])
                    max_peak = peaks[max_index]
                    signal_freq = freqs[max_peak]
                    if signal_freq > 0:  # 避免除零
                        period = int(1 / signal_freq)
                        batch_periods.append(period)
                    peaks = np.delete(peaks, max_index)
            
            all_periods.extend(batch_periods)
        
        # 统计最频繁的周期
        period_counter = Counter(all_periods)
        top_periods = [p[0] for p in period_counter.most_common(top_k)]
        
        # 确保返回top_k个周期，如果不足则补充默认值
        while len(top_periods) < top_k:
            default_period = seq_len // (len(top_periods) + 2)
            if default_period not in top_periods and default_period > 1:
                top_periods.append(default_period)
            else:
                top_periods.append(max(2, seq_len // (len(top_periods) + 3)))
                
        return top_periods[:top_k]


class MultiScalePatchBlock(nn.Module):
    """多尺度片段块，处理单个尺度的时序片段"""
    
    def __init__(self, configs, patch_len: int):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        
        # 计算片段数量
        self.patch_num = int((self.seq_len - patch_len) / self.stride + 2)
        
        # 片段嵌入层
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, self.stride, configs.dropout
        )
        
        # 多尺度序列分解
        # 确保conv_kernel是列表格式
        conv_kernel = configs.conv_kernel
        if isinstance(conv_kernel, str):
            conv_kernel = eval(conv_kernel)
        
        decomp_kernel = []
        for k in conv_kernel:
            decomp_kernel.append(k + 1 if k % 2 == 0 else k)
        self.decomp_multi = series_decomp_multi(decomp_kernel)
        
        # 跨片段线性融合层（季节性和趋势性）
        self.linear_seasonal = nn.Linear(self.patch_num, self.patch_num)
        self.linear_trend = nn.Linear(self.patch_num, self.patch_num)
        
        # 初始化权重
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / self.patch_num)
        nn.init.constant_(self.linear_trend.weight, 1.0 / self.patch_num)
        
        # 片段内特征增强MLP
        self.intra_patch_mlp = nn.Sequential(
            nn.Linear(patch_len, configs.d_model),
            nn.BatchNorm1d(configs.enc_in),
            nn.ReLU(inplace=True),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(inplace=True)
        )
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(self.patch_num * configs.d_model, configs.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, self.pred_len)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, n_vars, seq_len]
            
        Returns:
            output: [batch_size, pred_len, n_vars]
        """
        batch_size = x.shape[0]
        n_vars = x.shape[1]
        
        # 片段嵌入
        enc_out, _ = self.patch_embedding(x)
        # enc_out: [batch_size * n_vars, patch_num, d_model]
        
        # 多尺度分解与跨片段融合
        enc_out = self._cross_patch_fusion(enc_out)
        # enc_out: [batch_size * n_vars, patch_num, d_model]
        
        # 重塑回批次维度
        enc_out = enc_out.reshape(batch_size, n_vars, self.patch_num, self.d_model)
        enc_out = enc_out.reshape(batch_size, n_vars, -1)  # [B, n_vars, patch_num*d_model]
        
        # 输出投影
        output = self.output_projection(enc_out)
        output = output.permute(0, 2, 1)  # [B, pred_len, n_vars]
        
        return output
    
    def _cross_patch_fusion(self, x):
        """跨片段特征融合"""
        # 多尺度分解
        seasonal, trend = self.decomp_multi(x)
        
        # 转置以便进行跨片段处理
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        
        # 跨片段线性融合
        seasonal_out = self.linear_seasonal(seasonal)
        trend_out = self.linear_trend(trend)
        
        # 合并季节性和趋势性成分
        x = seasonal_out + trend_out
        
        return x.permute(0, 2, 1)


class AdaptiveMultiScaleFusion(nn.Module):
    """自适应多尺度融合模块"""
    
    def __init__(self, n_scales: int, d_model: int, pred_len: int):
        super().__init__()
        self.n_scales = n_scales
        self.d_model = d_model
        self.pred_len = pred_len
        
        # 可学习的尺度权重
        self.scale_weights = nn.Parameter(torch.ones(n_scales))
        
        # 自适应门控机制
        self.gate_network = nn.Sequential(
            nn.Linear(pred_len * n_scales, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_scales),
            nn.Softmax(dim=-1)
        )
        
        # 跨尺度注意力机制
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=pred_len,
            num_heads=4,
            dropout=0.1
        )
        
    def forward(self, multi_scale_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multi_scale_outputs: List of [batch_size, pred_len, n_vars]
            
        Returns:
            fused_output: [batch_size, pred_len, n_vars]
        """
        # 堆叠多尺度输出
        stacked = torch.stack(multi_scale_outputs, dim=-1)  # [B, pred_len, n_vars, n_scales]
        batch_size, pred_len, n_vars, n_scales = stacked.shape
        
        # 计算自适应权重
        flat_features = stacked.permute(0, 2, 1, 3).reshape(batch_size * n_vars, -1)
        adaptive_weights = self.gate_network(flat_features)  # [B*n_vars, n_scales]
        adaptive_weights = adaptive_weights.view(batch_size, n_vars, n_scales)
        
        # 结合静态和自适应权重
        combined_weights = F.softmax(self.scale_weights, dim=0) * adaptive_weights
        combined_weights = combined_weights.unsqueeze(1)  # [B, 1, n_vars, n_scales]
        
        # 加权融合
        fused_output = torch.sum(stacked * combined_weights, dim=-1)
        
        return fused_output


class MultiScaleTimeSeriesDecomposer(nn.Module):
    """
    完整的多尺度时序分割模块
    整合FFT分析、多尺度分割、编码增强和协同预测
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # 确保configs的参数被正确解析
        if hasattr(configs, 'conv_kernel') and isinstance(configs.conv_kernel, str):
            configs.conv_kernel = eval(configs.conv_kernel)
        if hasattr(configs, 'patch_list') and isinstance(configs.patch_list, str):
            configs.patch_list = eval(configs.patch_list)
        
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k
        
        # FFT分析器
        self.fft_analyzer = MultiScaleFFTAnalyzer()
        
        # 初始化时获取默认周期
        self.default_periods = self._get_default_periods()
        
        # 多尺度片段块（延迟初始化）
        self.multi_scale_blocks = None
        
        # 自适应融合模块（延迟初始化）
        self.fusion_module = None
        
        # 是否已初始化
        self.initialized = False
        
    def _get_default_periods(self):
        """获取默认周期列表"""
        periods = []
        for i in range(self.top_k):
            period = self.seq_len // (2 ** (i + 1))
            if period >= 2:
                periods.append(period)
        
        # 确保有足够的周期
        while len(periods) < self.top_k:
            periods.append(max(2, self.seq_len // (len(periods) + 2)))
            
        return periods[:self.top_k]
    
    def initialize_with_data(self, x: torch.Tensor):
        """使用数据初始化多尺度块"""
        if not self.initialized:
            # 使用FFT分析获取最优周期
            periods = self.fft_analyzer.get_top_k_periods(x, self.top_k)
            
            # 创建多尺度片段块
            self.multi_scale_blocks = nn.ModuleList([
                MultiScalePatchBlock(self.configs, patch_len=period)
                for period in periods
            ])
            
            # 创建融合模块
            self.fusion_module = AdaptiveMultiScaleFusion(
                n_scales=self.top_k,
                d_model=self.configs.d_model,
                pred_len=self.pred_len
            )
            
            # 将模块移到正确的设备
            device = x.device
            self.multi_scale_blocks = self.multi_scale_blocks.to(device)
            self.fusion_module = self.fusion_module.to(device)
            
            self.initialized = True
            self.periods = periods
            
            #print(f"多尺度时序分割模块初始化完成，识别的周期: {periods}")
    
    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_enc: [batch_size, seq_len, n_vars]
            
        Returns:
            output: [batch_size, pred_len, n_vars]
        """
        # 首次前向传播时初始化
        if not self.initialized:
            self.initialize_with_data(x_enc)
        
        # 转置输入 [B, seq_len, n_vars] -> [B, n_vars, seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        
        # 多尺度处理
        multi_scale_outputs = []
        for scale_block in self.multi_scale_blocks:
            scale_output = scale_block(x_enc)
            multi_scale_outputs.append(scale_output)
        
        # 自适应融合
        fused_output = self.fusion_module(multi_scale_outputs)
        
        return fused_output
    
    def get_scale_info(self) -> dict:
        """获取当前的尺度信息"""
        if self.initialized:
            return {
                'periods': self.periods,
                'n_scales': len(self.periods),
                'fusion_weights': self.fusion_module.scale_weights.detach().cpu().numpy()
            }
        else:
            return {
                'periods': self.default_periods,
                'n_scales': len(self.default_periods),
                'fusion_weights': None
            } 