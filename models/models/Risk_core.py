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
        
        self.lambda_g = 0.6               
        self.lambda_tv = 0.01             
        self.lambda_res = 0.015             
        self.lambda_hf = 0.0005             
        self.lambda_lf = 0.0005              
        self.lambda_comp = float(getattr(configs, 'lambda_comp', 0.002))


    def forward(self, x_enc, x_mark_enc=None, y_true=None):


        y_final, y_stable, y_residual, risk_pred, gate, decomposed_level, decomposed_season, decomposed_trend, decomposed_residual = self.risk_mixer(x_enc, x_mark_enc)
        
        if y_true is not None:
            L_pred = F.mse_loss(y_final, y_true)

            target_residual = y_true - y_stable
            L_res = F.l1_loss(y_residual, target_residual)

            with torch.no_grad():
                dy_true = torch.abs(y_true[:, 1:, :] - y_true[:, :-1, :])
                dy_tc = dy_true.permute(0, 2, 1)  
                dy_tc = F.pad(dy_tc, (1, 1), mode='replicate')
                dy_smooth = F.avg_pool1d(dy_tc, kernel_size=3, stride=1).permute(0, 2, 1)  
                q_low = torch.quantile(dy_smooth, q=0.10, dim=1, keepdim=True)
                q_high = torch.quantile(dy_smooth, q=0.90, dim=1, keepdim=True)
                denom = torch.clamp(q_high - q_low, min=1e-6)
                v = torch.clamp((dy_smooth - q_low) / denom, 0.0, 1.0)  
                v = torch.cat([v, v[:, -1:, :]], dim=1)  
                gate_teacher = v

            L_gate = F.l1_loss(gate, gate_teacher)

            if self.pred_len > 1:
                tv = torch.abs(gate[:, 1:, :] - gate[:, :-1, :]).mean()
            else:
                tv = torch.tensor(0.0, device=y_final.device)

            if self.pred_len > 1:
                y_stable_diff = y_stable[:, 1:, :] - y_stable[:, :-1, :]
                L_hf = (y_stable_diff * y_stable_diff).mean()
            else:
                L_hf = torch.tensor(0.0, device=y_final.device)
            y_res_tc = y_residual.permute(0, 2, 1) 
            y_res_tc = F.pad(y_res_tc, (1, 1), mode='replicate')
            y_res_low = F.avg_pool1d(y_res_tc, kernel_size=3, stride=1) 
            y_res_low = y_res_low.permute(0, 2, 1)
            L_lf = (y_res_low * y_res_low).mean()
            def cosine_sq(a, b, eps: float = 1e-6):
                # a,b: [B, T, C]
                a2 = torch.clamp((a * a).sum(dim=1, keepdim=False), min=eps)  
                b2 = torch.clamp((b * b).sum(dim=1, keepdim=False), min=eps)  
                ab = (a * b).sum(dim=1, keepdim=False)                        
                cos = ab / torch.sqrt(a2 * b2)
                return (cos * cos).mean()

            L_comp = cosine_sq(y_stable, y_residual)

            L_total = (
                L_pred
                + self.lambda_g * L_gate
                + self.lambda_tv * tv
                + self.lambda_res * L_res
                + self.lambda_hf * L_hf
                + self.lambda_lf * L_lf
                + self.lambda_comp * L_comp
            )

            return {
                'total_loss': L_total,
                'pred_loss': L_pred,
                'gate_loss': L_gate,
                'tv_loss': tv,
                'residual_loss': L_res,
                'hf_penalty': L_hf,
                'lf_penalty': L_lf,
                'comp_loss': L_comp,
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

            risk_stats = {
                'mean_risk': risk_pred.mean().item(),
                'max_risk': risk_pred.max().item(),
                'min_risk': risk_pred.min().item(),
                'mean_gate': gate_values.mean().item(),
                'gate_std': gate_values.std().item(),
                'stable_contribution': (1 - gate_values).mean().item(),
                'residual_contribution': gate_values.mean().item()
            }
            
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

            stable_contribution = y_stable  
            residual_contribution = gate_values * y_residual  
            
            decomposition = {
                'final_prediction': y_final.cpu(),
                'stable_component': stable_contribution.cpu(),
                'residual_component': residual_contribution.cpu(),
                'gate_weights': gate_values.cpu(),
                'risk_scores': risk_pred.cpu(),
                'reconstruction_error': torch.abs(y_final - (y_stable + gate_values * y_residual)).mean().item()
            }
        
        return decomposition
