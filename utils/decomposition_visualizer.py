import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import numpy as np

def plot_decomposition(original, level, season, trend, residual, file_path, sample_idx=0, feature_idx=0):
    """
    Plots the time series decomposition components.

    Args:
        original (torch.Tensor): The original input sequence (x_enc). Shape: [B, T, C].
        level (torch.Tensor): The level component. Shape: [B, T, C].
        season (torch.Tensor): The seasonal component. Shape: [B, T, C].
        trend (torch.Tensor): The trend component. Shape: [B, T, C].
        residual (torch.Tensor): The residual component. Shape: [B, T, C].
        file_path (str): Path to save the plot.
        sample_idx (int): The index of the sample in the batch to plot.
        feature_idx (int): The index of the feature/channel to plot.
    """
    # Use a clean style suitable for publications
    plt.style.use('default')
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.2)
    # Use system-available serif fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Convert tensors to numpy for plotting
    original_np = original[sample_idx, :, feature_idx].cpu().numpy()
    level_np = level[sample_idx, :, feature_idx].cpu().numpy()
    season_np = season[sample_idx, :, feature_idx].cpu().numpy()
    trend_np = trend[sample_idx, :, feature_idx].cpu().numpy()
    residual_np = residual[sample_idx, :, feature_idx].cpu().numpy()

    seq_len = original_np.shape[0]
    x_axis = np.arange(seq_len)

    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'Time Series Decomposition', 
                 fontsize=14, fontweight='bold', y=0.98)

    # Define colors from a professional sci palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

    # Configure all subplots with consistent styling
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(False)

    # 1. Original Series
    axes[0].plot(x_axis, original_np, label='Original Signal', color=colors[0], linewidth=1.5)
    axes[0].legend(loc='upper right', frameon=False, fontsize=10)
    axes[0].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[0].set_title('(a) Original Input Signal', fontsize=12, fontweight='bold', pad=10)

    # 2. Level Component
    axes[1].plot(x_axis, level_np, label='Level Component', color=colors[1], linewidth=1.5)
    axes[1].legend(loc='upper right', frameon=False, fontsize=10)
    axes[1].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[1].set_title('(b) Level Component (Long-term Baseline)', fontsize=12, fontweight='bold', pad=10)

    # 3. Seasonal Component
    axes[2].plot(x_axis, season_np, label='Seasonal Component', color=colors[2], linewidth=1.5)
    axes[2].legend(loc='upper right', frameon=False, fontsize=10)
    axes[2].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[2].set_title('(c) Seasonal Component (Periodical Patterns)', fontsize=12, fontweight='bold', pad=10)

    # 4. Trend Component
    axes[3].plot(x_axis, trend_np, label='Trend Component', color=colors[3], linewidth=1.5)
    axes[3].legend(loc='upper right', frameon=False, fontsize=10)
    axes[3].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[3].set_title('(d) Trend Component (Long-term Direction)', fontsize=12, fontweight='bold', pad=10)

    # 5. Residual Component
    axes[4].plot(x_axis, residual_np, label='Residual Component', color=colors[4], linewidth=1.5, alpha=0.9)
    axes[4].axhline(0, color='#666666', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[4].legend(loc='upper right', frameon=False, fontsize=10)
    axes[4].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[4].set_xlabel('Time Step', fontsize=11, fontweight='bold')
    axes[4].set_title('(e) Residual Component (Uncertainty & Noise)', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig) 