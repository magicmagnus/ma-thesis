import os
import random
import numpy as np
import torch
import json
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd
from argparse import Namespace


def setup_gridspec_figure(nrows, ncols, fs, title, fs_title, y_adj, title_height_ratio, sp_width=1.5, sp_height=1.5, height_correction=0):
    """Create figure with gridspec layout and title rows."""
    # Basic layout parameters
    data_height_ratio = 1.0
    height_ratios = [title_height_ratio if i % 2 == 0 else data_height_ratio 
                    for i in range(2 * nrows)]
    wspace = 0.05
    hspace = 0.05

    # Calculate figure size
    subplot_width = sp_width
    subplot_height = sp_height
    fig_width = subplot_width * ncols + (ncols - 1) * wspace * subplot_width
    fig_height = (subplot_height * sum(height_ratios) + (2 * nrows - 1) * hspace * subplot_height) + height_correction

    # Create figure and gridspec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(2 * nrows, ncols, figure=fig, 
                          wspace=wspace, hspace=hspace, 
                          height_ratios=height_ratios)
    
    # Create title axes
    title_axes = []
    for i in range(0, 2 * nrows, 2):
        ax = fig.add_subplot(gs[i, :])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)  # Hide axis lines
        title_axes.append(ax)
    
    fig.suptitle(title, fontsize=fs_title, fontweight="bold", y=y_adj)
    
    return fig, gs, title_axes

def plot_heatmaps(args, fpr_grid, tpr_grid, title_suffix, xticks, yticks, levels=0.01, contour_label='FPR = 0.01'):
    xlabel = 'RID Thresholds (higher = more likely WM)'
    ylabel = 'GS Thresholds (higher = more likely WM)'

    title_FPR = f'FPR Grid for {title_suffix}'
    title_TPR = f'TPR Grid for {title_suffix}' 
    contour_label = f'FPR={levels}'

    # Create 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Use imshow for heatmaps ---
    # Define extent for imshow: [left, right, bottom, top]
    # Coordinates correspond to the *edges* of the pixels/cells
    

    # FPR plot using imshow
    im1 = ax1.imshow(fpr_grid, cmap="viridis", vmin=0, vmax=1, aspect='auto', origin='lower')
    fig.colorbar(im1, ax=ax1, label='False Positive Rate (↓)')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title_FPR, fontsize=20)
    # Set ticks manually for imshow
    ax1.set_xticks(np.arange(len(xticks)))
    ax1.set_yticks(np.arange(len(yticks)))
    ax1.set_xticklabels(xticks, rotation=90)
    ax1.set_yticklabels(yticks)
    # ax1.invert_yaxis() # origin='lower' handles the y-axis direction

    # TPR plot using imshow
    im2 = ax2.imshow(tpr_grid, cmap="viridis", vmin=0, vmax=1, aspect='auto', origin='lower')
    fig.colorbar(im2, ax=ax2, label='True Positive Rate (↑)')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(title_TPR, fontsize=20)
    # Set ticks manually for imshow
    ax2.set_xticks(np.arange(len(xticks)))
    ax2.set_yticks(np.arange(len(yticks)))
    ax2.set_xticklabels(xticks, rotation=90)
    ax2.set_yticklabels(yticks)
    # ax2.invert_yaxis() # origin='lower' handles the y-axis direction

    # --- Contour plot ---
    # Create coordinates for contour centers (matching imshow pixel centers)
    x_centers = np.arange(fpr_grid.shape[1]) - 0.5
    y_centers = np.arange(fpr_grid.shape[0]) - 0.5
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers)

    # Generate FPR contour on FPR plot
    fpr_CS = ax1.contour(X_centers, Y_centers, fpr_grid, levels=[levels], colors='red', linewidths=1.5, linestyles='-')
    ax1.clabel(fpr_CS, fmt={levels: contour_label}, inline=True, fontsize=10)

    # Display the same FPR contour on TPR plot
    fpr_contour_on_tpr = ax2.contour(X_centers, Y_centers, fpr_grid, levels=[levels], colors='red', linewidths=1.5, linestyles='-')
    ax2.clabel(fpr_contour_on_tpr, fmt={levels: contour_label}, inline=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, f'grids_{title_suffix}_heatmaps.pdf'))
    plt.show()


# for RID and TR
def plot_wm_pattern_fft(num_channels, pattern, title, save_path):
    
    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 1.2
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 10
        y_adj = 1.05
        title_height_ratio = 0.15
    """Plot watermark pattern FFT."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=2, ncols=num_channels, 
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )
    
    # Add row titles
    title_axes[0].text(0.5, 0.25, 'Watermark pattern (real)', 
                      fontsize=fs, fontweight="bold", ha="center", va="center")
    title_axes[1].text(0.5, 0.25, 'Watermark pattern (imag)', 
                      fontsize=fs, fontweight="bold", ha="center", va="center")
    
    # Plot data
    for col in range(num_channels):
        ax1 = fig.add_subplot(gs[1, col])
        ax3 = fig.add_subplot(gs[3, col])
        
        ax1.imshow(pattern[0, col].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
        ax3.imshow(pattern[0, col].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
        
        for ax in [ax1, ax3]:
            ax.axvline(32, color='r', linestyle='--', linewidth=1)
            ax.axhline(32, color='r', linestyle='--', linewidth=1)
            ax.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# for PRC and GS
def plot_wm_pattern_spatial_domain(num_channels, pattern, title, save_path):
    
    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 1.2
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 10
        y_adj = 1.05
        title_height_ratio = 0.15

    """Plot watermark pattern in spatial domain."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=1, ncols=num_channels, 
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )

    # Add row title
    title_axes[0].text(0.5, 0.25, 'Watermark pattern (spatial domain)', 
                      fontsize=fs, fontweight="bold", ha="center", va="center")
    
    # Plot data
    for col in range(num_channels):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(pattern[0, col].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
        ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# for RID and TR
def plot_wm_latents_fft(num_channels, init_latents_fft, init_latents_watermarked_fft,
                        init_latents_watermarked, diff, title, save_path):
    
    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 1.0
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 10
        y_adj = 0.95
        title_height_ratio = 0.15

    """Plot watermarked latents FFT and spatial domain."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=5, ncols=num_channels,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )
    
    # Add row titles
    row_titles = [
        'FFT of original latents (real)',
        'FFT after watermarking (real)',
        'FFT after watermarking (imag)',
        'Latents after watermarking (spatial domain)',
        'Abs. Difference (spatial domain)'
    ]
    for ax, title_text in zip(title_axes, row_titles):
        ax.text(0.5, 0.25, title_text, fontsize=fs, fontweight="bold",
                ha="center", va="center")
    
    # Plot data
    for col in range(num_channels):
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(5)]
        plots = [
            (init_latents_fft[0, col].real.cpu().numpy(), 'GnBu', -50, 50),
            (init_latents_watermarked_fft[0, col].real.cpu().numpy(), 'GnBu', -50, 50),
            (init_latents_watermarked_fft[0, col].imag.cpu().numpy(), 'GnBu', -50, 50),
            (init_latents_watermarked[0, col].cpu().numpy(), 'OrRd', -4, 4),
            (np.abs(diff[0, col].cpu().numpy()), 'gray', -4, 4)
        ]
        
        for ax, (data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# for RID and TR
def visualize_reversed_latents_fft(num_channels, 
                               reversed_latents_wm_fft,
                               reversed_latents_nowm_fft,
                               diff_wm_wm_fft,
                               diff_wm_nowm_fft,
                               diff_wm_true_fft,
                               diff_nowm_true_fft,
                               ch_mean_abs_diff_wm_wm_fft,
                               ch_mean_abs_diff_wm_nowm_fft,
                               mean_abs_diff_wm_true_fft,
                               mean_abs_diff_nowm_true_fft,
                               title, save_path):

    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 0.95
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 9
        y_adj = 0.91
        title_height_ratio = 0.15

    """Visualize reversed latents in frequency domain."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=12, ncols=num_channels,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )

    # Add row titles
    row_titles = [
        'FFT of Reversed WM latents (real)',
        'FFT of Reversed WM latents (imag)',
        '|WM latents - masked WM| (real)',
        '|WM latents - masked WM| (imag)',
        f'|WM latents - true WM latents| (real) (per-pixel mean={mean_abs_diff_wm_true_fft:.2f})',
        '|WM latents - true WM latents (imag)',
        'FFT of Reversed NoWM latents (real)',
        'FFT of Reversed NoWM latents (imag)',
        '|NoWM latents - masked WM| (real)',
        '|NoWM latents - masked WM| (imag)',
        f'|NoWM latents - true NoWM latents| (real) (per-pixel mean={mean_abs_diff_nowm_true_fft:.2f})',
        '|NoWM latents - true NoWM latents| (imag)'
    ]

    for ax, title_text in zip(title_axes, row_titles):
        ax.text(0.5, 0.25, title_text, fontsize=fs, fontweight="bold",
                ha="center", va="center")
        
    # Plot data
    for col in range(num_channels):
        # first all for WM
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(6)]
        plots = [
            (0 , reversed_latents_wm_fft[0, col].real.cpu().numpy(), 'GnBu', -50, 50),
            (1 , reversed_latents_wm_fft[0, col].imag.cpu().numpy(), 'GnBu', -50, 50),
            (2 , np.abs(diff_wm_wm_fft[0, col].real.cpu().numpy()), 'gray', 0, 150),
            (3 , np.abs(diff_wm_wm_fft[0, col].imag.cpu().numpy()), 'gray', 0, 150),
            (4 , np.abs(diff_wm_true_fft[0, col].real.cpu().numpy()), 'gray', 0, 150),
            (5 , np.abs(diff_wm_true_fft[0, col].imag.cpu().numpy()), 'gray', 0, 150)
        ]
        
        for ax, (idx, data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            if idx == 2:
                # add text to top rght corner, the abs_diff_wm_fft[col]
                ax.text(0.02, 0.98, f"{ch_mean_abs_diff_wm_wm_fft[col]:.2f}", 
                        fontsize=fs, ha="left", va="top",
                        transform=ax.transAxes, color='white')
                
        # then for NoWM
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(6, 12)]
        plots = [
            (0 , reversed_latents_nowm_fft[0, col].real.cpu().numpy(), 'GnBu', -50, 50),
            (1 , reversed_latents_nowm_fft[0, col].imag.cpu().numpy(), 'GnBu', -50, 50),
            (2 , np.abs(diff_wm_nowm_fft[0, col].real.cpu().numpy()), 'gray', 0, 150),
            (3 , np.abs(diff_wm_nowm_fft[0, col].imag.cpu().numpy()), 'gray', 0, 150),
            (4 , np.abs(diff_nowm_true_fft[0, col].real.cpu().numpy()), 'gray', 0, 150),
            (5 , np.abs(diff_nowm_true_fft[0, col].imag.cpu().numpy()), 'gray', 0, 150)
        ]

        for ax, (idx, data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            if idx == 2:
                # add text to top rght corner, the abs_diff_nowm_fft[col]
                ax.text(0.02, 0.98, f"{ch_mean_abs_diff_wm_nowm_fft[col]:.2f}", 
                        fontsize=fs, ha="left", va="top",
                        transform=ax.transAxes, color='white')
                
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# for PRC and GS
def visualize_reversed_latents_spatial_domain(num_channels, 
                               reversed_latents_wm,
                               reversed_latents_nowm,
                               diff_wm_wm,
                               diff_nowm_wm,
                               diff_wm_true,
                               diff_nowm_true,
                               metric_wm,
                               metric_nowm,
                               mean_abs_diff_wm_wm,
                               mean_abs_diff_nowm_wm,
                               mean_abs_diff_wm_true,
                               mean_abs_diff_nowm_true,
                               title, save_path):

    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 0.95
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 8
        y_adj = 0.91
        title_height_ratio = 0.15

    """Visualize reversed latents in spatial domain."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=6, ncols=num_channels,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )

    # Add row titles
    row_titles = [
        'Reversed WM latents (spatial domain)',
        f'|WM latents - WM pattern| (per-pixel mean={mean_abs_diff_wm_wm:.2f}) with metric {metric_wm:.2f}',
        f'|WM latents - true WM latents| (mean per-pixel={mean_abs_diff_wm_true:.2f})',
        'Reversed NoWM latents (spatial domain)',
        f'|NoWM latents - WM pattern| (per-pixel mean={mean_abs_diff_nowm_wm:.2f}) with metric {metric_nowm:.2f}',
        f'|NoWM latents - true NoWM latents| (per-pixel mean={mean_abs_diff_nowm_true:.2f})'
    ]

    for ax, title_text in zip(title_axes, row_titles):
        ax.text(0.5, 0.25, title_text, fontsize=fs, fontweight="bold",
                ha="center", va="center")
        
    # Plot data
    for col in range(num_channels):
        # first all for WM
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(3)]
        plots = [
            (0 , reversed_latents_wm[0, col].cpu().numpy(), 'OrRd', -4, 4),
            (1 , np.abs(diff_wm_wm[0, col].cpu().numpy()), 'gray', 0, 4),
            (2 , np.abs(diff_wm_true[0, col].cpu().numpy()), 'gray', 0, 4)

        ]

        for ax, (idx, data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
                  
                
        # then for NoWM
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(3, 6)]
        plots = [
            (0 , reversed_latents_nowm[0, col].cpu().numpy(), 'OrRd', -4, 4),
            (1 , np.abs(diff_nowm_wm[0, col].cpu().numpy()), 'gray', 0, 4),
            (2 , np.abs(diff_nowm_true[0, col].cpu().numpy()), 'gray', 0, 4)
        ]

        for ax, (idx, data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
                
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

################################################################################################
################################################################################################
################################################################################################

# Plotting functions for the Results section

################################################################################################
################################################################################################
################################################################################################


# dict for saving names of axes/values
ATTACK_NAME_MAPPING = {
    'gaussian_std': {
        'name': 'Noise\n(Gaussian std)',
        'x_axis': 'Standard deviation',
        'order': 'high-to-low',
        'cast_to_int': False
    },
    'brightness_factor': {
        'name': 'Brightness\n(factor)',
        'x_axis': 'Factor',
        'order': 'high-to-low',
        'cast_to_int': False
    },
    'crop_scale': {
        'name': 'Crop&Scale\n(ratio)',
        'x_axis': 'Crop ratio and scale ratio',
        'order': 'low-to-high',
        'cast_to_int': False
    },
    'crop': {
        'name': 'Crop\n(ratio)',
        'x_axis': 'Crop ratio',
        'order': 'low-to-high',
        'cast_to_int': False
    },
    'jpeg_ratio': {
        'name': 'JPEG\n(quality factor)',
        'x_axis': 'Quality factor', 
        'order': 'low-to-high',
        'cast_to_int': True
    },
    'r_degree': {
        'name': 'Rotation\n(degrees)',
        'x_axis': 'Angle',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'gaussian_blur_r': {
        'name': 'Blur\n(radius)',
        'x_axis': 'Radius',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_surr_resnet18': {
        'name': 'Adv. Surr.\nRN18 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_surr_resnet50': {
        'name': 'Adv. Surr.\nRN50 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_klvae8': {	
        'name': 'Adv. Embed\nKLVAE8 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_klvae16': {
        'name': 'Adv. Embed KLVAE16\n(eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_resnet18': {
        'name': 'Adv. Embed RN18\n(eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_resnet50': {
        'name': 'Adv. Embed RN50\n(eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_resnet101': {
        'name': 'Adv. Embed RN101\n(eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_sdxlvae': {
        'name': 'Adv. Embed SDXLVAE\n(eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'no_attack': {
        'name': 'No Attack',
        'x_axis': 'Attack strength\n(eps)',
        'order': 'low-to-high'
    }
}

MODEL_NAME_MAPPING = {
    'sd': {
        'name': 'Stable Diffusion v2-1-base',
        'marker': 'o',
        'line': '-',
        'color': '#1447e6'
    },
    'flux': {
        'name': 'FLUX.1 [dev]',
        'marker': 'o',
        'line': '-',
        'color': '#a800b7'
    },
    'flux_s': {
        'name': 'FLUX.1 [schnell]',
        'marker': 'o',
        'line': '-',
        'color': '#ff04d5'
    },

}

METHODS_NAME_MAPPING = {
    'prc': 'PRC',
    'gs': 'Gaussian Shading',
    'tr': 'Tree-Ring',
    'rid': 'Ring ID',
    'grids': 'GRIDS',
}

def order_attack_strengths(order, attack_strengths, attack_results, ci_lower, ci_upper, cast_to_int=False):
    """Orders attack strengths based on difficulty"""
    # Convert series to numpy for easier manipulation
    strengths = attack_strengths.values
    results = attack_results.values
    ci_lower = ci_lower.values
    ci_upper = ci_upper.values

    # print dtypes of strengths and results
   
    # elements in strengths are strings, convert to float
    strengths = strengths.astype(float) 

    if cast_to_int:
        strengths = strengths.astype(int)

    
    if order == 'high-to-low':
        # Keep original order for attacks where lower values = easier
        idx = np.argsort(strengths)   # Sort in ascending order
    else: # 'low-to-high'
        # Reverse the order for attacks where higher values = easier
        idx = np.argsort(-strengths)  # Sort in descending order
        
        
    return strengths[idx], results[idx], ci_lower[idx], ci_upper[idx]



