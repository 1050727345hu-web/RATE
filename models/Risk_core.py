import torch
import torch.nn as nn
import torch.nn.functional as F
from .RATE import RATE


class RiskCore(nn.Module):

    
    def __init__(self, configs):
        super(RiskCore, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.risk_mixer = RATE(configs)

        initial_alpha = getattr(configs, 'alpha', 1.0)
        initial_beta = getattr(configs, 'beta', 0.1)
        initial_gamma = getattr(configs, 'gamma', 0.05)
        initial_delta = getattr(configs, 'delta', 0.02)

        self.loss_weights = nn.Parameter(torch.tensor([
            initial_alpha, initial_beta, initial_gamma, initial_delta
        ], dtype=torch.float32))

    
    def forward(self, x_enc, x_mark_enc=None, y_true=None):

        y_final, y_stable, y_residual, risk_pred, gate, decomposed_level, decomposed_season, decomposed_trend, decomposed_residual = self.risk_mixer(x_enc, x_mark_enc)

        if y_true is not None:

            L_pred = F.mse_loss(y_final, y_true)
            

            with torch.no_grad():
                true_risk = torch.abs(y_true - y_stable)
            L_risk = F.mse_loss(risk_pred, true_risk)
            

            with torch.no_grad():
                y_fusion_expected = y_stable + gate * y_residual
            L_fusion = F.mse_loss(y_final, y_fusion_expected)
            

            L_gate = torch.tensor(0.0, device=y_final.device)
            if self.pred_len > 1:
                gate_diff = torch.abs(gate[:, 1:, :] - gate[:, :-1, :])
                L_gate = gate_diff.mean()


            L_smoothness = torch.tensor(0.0, device=y_final.device)
            if self.pred_len > 1:
                with torch.no_grad():
                    stable_smoothness = torch.abs(y_stable[:, 1:, :] - y_stable[:, :-1, :]).mean()
                    residual_smoothness = torch.abs(y_residual[:, 1:, :] - y_residual[:, :-1, :]).mean()
                L_smoothness = F.relu(stable_smoothness - residual_smoothness)


            weights = F.softplus(self.loss_weights)

            alpha_learnable = torch.tensor(1.0, device=y_final.device, dtype=weights.dtype)
            beta_learnable = weights[1]
            gamma_learnable = weights[2]
            delta_learnable = weights[3]
            

            L_total = (alpha_learnable * L_pred + 
                       beta_learnable * L_risk + 
                       gamma_learnable * L_fusion + 
                       delta_learnable * L_gate + 
                       0.01 * L_smoothness)
            
            return {
                'total_loss': L_total,
                'pred_loss': L_pred,
                'risk_loss': L_risk,
                'fusion_loss': L_fusion,
                'gate_loss': L_gate,
                'y_final': y_final,
                'gate': gate
            }

        else:
            return y_final, y_stable, y_residual, risk_pred, gate, decomposed_level, decomposed_season, decomposed_trend, decomposed_residual
    
    def compute_loss(self, x_enc, y_true, x_mark_enc=None):

        loss_dict = self.forward(x_enc, x_mark_enc, y_true=y_true)
        return loss_dict
    
    def get_risk_analysis(self, x_enc, x_mark_enc=None):

        with torch.no_grad():
            y_final, y_stable, y_residual, risk_pred, gate_values, _, _, _, _ = self.forward(x_enc, x_mark_enc)
            
            # 计算风险统计
            risk_stats = {
                'mean_risk': risk_pred.mean().item(),
                'max_risk': risk_pred.max().item(),
                'min_risk': risk_pred.min().item(),
                'mean_gate': gate_values.mean().item(),
                'gate_std': gate_values.std().item(),
                'stable_contribution': (1 - gate_values).mean().item(),
                'residual_contribution': gate_values.mean().item()
            }
            
            # 分析稳定性和波动性
            stability_analysis = {
                'stable_variance': y_stable.var(dim=1).mean().item(),
                'residual_variance': y_residual.var(dim=1).mean().item(),
                'final_variance': y_final.var(dim=1).mean().item(),
                'risk_trend': (risk_pred[:, -1, :] - risk_pred[:, 0, :]).mean().item()
            }
            
            risk_analysis = {
                'risk_pred': risk_pred.cpu(),
                'gate_values': gate_values.cpu(),
                'stable_prediction': y_stable.cpu(),
                'residual_prediction': y_residual.cpu(),
                'final_prediction': y_final.cpu(),
                'risk_statistics': risk_stats,
                'stability_analysis': stability_analysis
            }
        
        return risk_analysis
    
    def get_prediction_decomposition(self, x_enc, x_mark_enc=None):

        with torch.no_grad():
            y_final, y_stable, y_residual, risk_pred, gate_values, _, _, _, _ = self.forward(x_enc, x_mark_enc)
            
            # 计算各分量的贡献
            stable_contribution = y_stable  # 稳定分量的直接贡献
            residual_contribution = gate_values * y_residual  # 残差分量的加权贡献
            
            decomposition = {
                'final_prediction': y_final.cpu(),
                'stable_component': stable_contribution.cpu(),
                'residual_component': residual_contribution.cpu(),
                'gate_weights': gate_values.cpu(),
                'risk_scores': risk_pred.cpu(),
                # 验证：stable + weighted_residual = final
                'reconstruction_error': torch.abs(y_final - (y_stable + gate_values * y_residual)).mean().item()
            }
        
        return decomposition
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        with torch.no_grad():
            current_weights = F.softplus(self.loss_weights)
        
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'RiskMixer with Parallel Prediction & Risk-Aware Fusion',
            'design_principle': 'Decompose-Parallel_Predict-Risk_Fusion',
            'core_components': ['StablePredictor', 'UncertaintyPredictor', 'RiskAwareFusion'],
            'loss_components': ['L_pred', 'L_risk', 'L_fusion', 'L_gate', 'L_smoothness'],
            'loss_weights': {
                'alpha_learnable': current_weights[0].item(),
                'beta_learnable': current_weights[1].item(),
                'gamma_learnable': current_weights[2].item(),
                'delta_learnable': current_weights[3].item(),
                'optimization_type': 'End-to-End Learnable Parameters'
            },
            'key_features': [
                'One-shot parallel prediction (no autoregression)',
                'Risk-aware dynamic fusion',
                'Four-component decomposition (Level+Season+Trend+Residual)',
                'Gate-controlled uncertainty integration',
                'Adaptive loss weight optimization'
            ]
        }
        
        return model_info 