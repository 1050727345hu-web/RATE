import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

def plot_risk_storyboard(x_enc, y_true, y_final, y_stable, risk_pred, gate_values, 
                        file_path, sample_idx=0, feature_idx=0, title_suffix=""):
    """
    Create a comprehensive risk-aware forecasting storyboard visualization.
    
    Args:
        x_enc (torch.Tensor): Historical input sequence. Shape: [B, seq_len, C].
        y_true (torch.Tensor): Ground truth future values. Shape: [B, pred_len, C].
        y_final (torch.Tensor): Final model predictions. Shape: [B, pred_len, C].
        y_stable (torch.Tensor): Stable component predictions. Shape: [B, pred_len, C].
        risk_pred (torch.Tensor): Risk intensity predictions. Shape: [B, pred_len, 1].
        gate_values (torch.Tensor): Gate values (0-1). Shape: [B, pred_len, 1].
        file_path (str): Path to save the plot.
        sample_idx (int): Index of the sample in the batch to plot.
        feature_idx (int): Index of the feature/channel to plot.
        title_suffix (str): Additional suffix for the plot title.
    """
    # Use professional publication style
    plt.style.use('default')
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.1)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # Extract data for the specified sample and feature
    hist_data = x_enc[sample_idx, :, feature_idx].cpu().numpy()
    true_data = y_true[sample_idx, :, feature_idx].cpu().numpy()
    final_pred = y_final[sample_idx, :, feature_idx].cpu().numpy()
    stable_pred = y_stable[sample_idx, :, feature_idx].cpu().numpy()
    risk_intensity = risk_pred[sample_idx, :, 0].cpu().numpy()  # Risk is single-dimensional
    gate_vals = gate_values[sample_idx, :, 0].cpu().numpy()    # Gate is single-dimensional
    
    seq_len = len(hist_data)
    pred_len = len(true_data)
    total_len = seq_len + pred_len
    
    # Create time axis
    hist_time = np.arange(seq_len)
    pred_time = np.arange(seq_len, total_len)
    all_time = np.arange(total_len)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    # ============ Main Prediction Panel ============
    ax_main = fig.add_subplot(gs[0, 0])
    
    # Risk Background
    colors = ['#E8F4FD', '#FFF3E0', '#FFEBEE', '#FFCDD2']
    risk_cmap = LinearSegmentedColormap.from_list("risk", colors, N=256)
    risk_norm = plt.Normalize(vmin=0, vmax=np.max(risk_intensity) * 1.1)
    
    for i, (t, risk) in enumerate(zip(pred_time, risk_intensity)):
        color = risk_cmap(risk_norm(risk))
        ax_main.axvspan(t-0.5, t+0.5, alpha=0.5, color=color, zorder=0)
    
    # Time series lines
    ax_main.plot(hist_time, hist_data, 'k-', linewidth=2, label='Historical')
    ax_main.axvline(x=seq_len-0.5, color='gray', linestyle='--', alpha=0.7)
    ax_main.plot(pred_time, stable_pred, 'b-', linewidth=2, label='Stable', alpha=0.8)
    ax_main.plot(pred_time, final_pred, 'r-', linewidth=2.5, label='Final Prediction')
    ax_main.plot(pred_time, true_data, 'gold', linewidth=2, label='Ground Truth')
    
    ax_main.set_ylabel('Value')
    ax_main.set_title(f'Risk-Aware Forecasting{title_suffix}', fontsize=12)
    ax_main.legend(loc='upper left', frameon=False)
    ax_main.grid(True, alpha=0.3)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # ============ Risk Intensity Panel ============
    ax_risk = fig.add_subplot(gs[1, 0])
    
    ax_risk.fill_between(pred_time, 0, risk_intensity, alpha=0.6, color='orange')
    ax_risk.plot(pred_time, risk_intensity, 'darkorange', linewidth=2)
    
    ax_risk.set_ylabel('Risk Level')
    ax_risk.set_title('Risk Intensity')
    ax_risk.grid(True, alpha=0.3)
    ax_risk.spines['top'].set_visible(False)
    ax_risk.spines['right'].set_visible(False)
    ax_risk.set_xlim(pred_time[0]-0.5, pred_time[-1]+0.5)
    
    # ============ Gate Values Panel ============
    ax_gate = fig.add_subplot(gs[2, 0])
    
    gate_colors = ['lightcoral' if g < 0.5 else 'lightgreen' for g in gate_vals]
    ax_gate.bar(pred_time, gate_vals, alpha=0.6, color=gate_colors)
    ax_gate.plot(pred_time, gate_vals, 'navy', linewidth=2)
    ax_gate.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax_gate.set_ylabel('Gate Value')
    ax_gate.set_xlabel('Time Step')
    ax_gate.set_title('Gate Values')
    ax_gate.grid(True, alpha=0.3)
    ax_gate.spines['top'].set_visible(False)
    ax_gate.spines['right'].set_visible(False)
    ax_gate.set_xlim(pred_time[0]-0.5, pred_time[-1]+0.5)
    ax_gate.set_ylim(-0.05, 1.05)
    

    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_batch_risk_storyboards(x_enc, y_true, y_final, y_stable, risk_pred, gate_values,
                               folder_path, max_samples=5, features_to_plot=None):
    """
    Create risk storyboard visualizations for multiple samples and features in a batch.
    
    Args:
        x_enc, y_true, y_final, y_stable, risk_pred, gate_values: Model outputs
        folder_path (str): Directory to save the plots
        max_samples (int): Maximum number of samples to visualize
        features_to_plot (list): List of feature indices to plot. If None, plot first feature only.
    """
    batch_size = x_enc.shape[0]
    num_features = x_enc.shape[2]
    
    if features_to_plot is None:
        features_to_plot = [0]  # Default to first feature only
    
    # Limit samples to visualize
    samples_to_plot = min(max_samples, batch_size)
    
    for sample_idx in range(samples_to_plot):
        for feature_idx in features_to_plot:
            if feature_idx < num_features:
                file_name = f'risk_storyboard_sample_{sample_idx}_feature_{feature_idx}.png'
                file_path = os.path.join(folder_path, file_name)
                title_suffix = f' (Sample {sample_idx}, Feature {feature_idx})'
                
                plot_risk_storyboard(
                    x_enc, y_true, y_final, y_stable, risk_pred, gate_values,
                    file_path, sample_idx, feature_idx, title_suffix
                )
    
    print(f"Saved {samples_to_plot * len(features_to_plot)} risk storyboard plots to {folder_path}") 