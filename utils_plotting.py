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

# Add palatino-clone fonts to matplotlib
import matplotlib.font_manager as fm
for font_file in os.listdir('fonts'):
    if font_file.endswith('.otf'):
        fm.fontManager.addfont(os.path.join('fonts', font_file))
# Configure matplotlib
plt.rcParams['font.family'] = 'TeX Gyre Pagella'
plt.rcParams['mathtext.fontset'] = 'stix'


def setup_gridspec_figure(nrows, ncols, fs, fs_title, y_adj, title_height_ratio, title=None, sp_width=1.5, sp_height=1.5, height_correction=0, wspace=0.05, hspace=0.05):
    """Create figure with gridspec layout and title rows."""
    # Basic layout parameters
    data_height_ratio = 1.0
    height_ratios = [title_height_ratio if i % 2 == 0 else data_height_ratio 
                    for i in range(2 * nrows)]
    

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
    
    if title is not None:
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

    # # Also, save just the 4. channel of the watermarked latents in spatial domain as an image with colormap OrRd 
    # # so that we can use it for the paper

    
    # # Save the 4th channel of the watermarked latents
    # wm_latent_4 = init_latents_watermarked[0, 3].cpu().numpy()
    # wm_latent_4_normalized = ((wm_latent_4 - wm_latent_4.min()) / (wm_latent_4.max() - wm_latent_4.min()) * 255).astype(np.uint8)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(wm_latent_4_normalized, interpolation='nearest', cmap='YlGn', vmin=50, vmax=205)
    # plt.axis('off')
    # plt.savefig(save_path.replace('.pdf', '_wm_latent_4.png'), 
    #             dpi=72, 
    #             bbox_inches='tight', 
    #             pad_inches=0)
    
    # # same for channel 0, 
    # wm_latent_0 = init_latents_watermarked[0, 1].cpu().numpy()
    # wm_latent_0_normalized = ((wm_latent_0 - wm_latent_0.min()) / (wm_latent_0.max() - wm_latent_0.min()) * 255).astype(np.uint8)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(wm_latent_0_normalized, interpolation='nearest', cmap='YlGn', vmin=50, vmax=205)
    # plt.axis('off')
    # plt.savefig(save_path.replace('.pdf', '_wm_latent_1.png'), 
    #             dpi=72, 
    #             bbox_inches='tight', 
    #             pad_inches=0)
    
    # # Now the 4th channel of the watermarked latents in fourier domain as an image but with cmap 'GnBu'
    # wm_latent_4_fft = init_latents_watermarked_fft[0, 3].real.cpu().numpy()
    # wm_latent_4_fft_normalized = ((wm_latent_4_fft - wm_latent_4_fft.min()) / (wm_latent_4_fft.max() - wm_latent_4_fft.min()) * 255).astype(np.uint8)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(wm_latent_4_fft_normalized, interpolation='nearest', cmap='GnBu', vmin=50, vmax=205)
    # plt.axis('off')
    # plt.savefig(save_path.replace('.pdf', '_wm_latent_4_fft.png'),
    #             dpi=72, 
    #             bbox_inches='tight', 
    #             pad_inches=0)




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
        'name': 'Noise',
        'x_axis': 'Standard deviation',
        'order': 'high-to-low',
        'cast_to_int': False
    },
   'gaussian_blur_r': {
        'name': 'Blur',
        'x_axis': 'Radius',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'brightness_factor': {
        'name': 'Brightness',
        'x_axis': 'Factor',
        'order': 'high-to-low',
        'cast_to_int': False
    },
     'crop': {
        'name': 'Crop',
        'x_axis': 'Crop ratio',
        'order': 'low-to-high',
        'cast_to_int': False
    },
    'crop_scale': {
        'name': 'Crop&Scale',
        'x_axis': 'Crop ratio and scale ratio',
        'order': 'low-to-high',
        'cast_to_int': False
    },
    'jpeg_ratio': {
        'name': 'JPEG',
        'x_axis': 'Quality factor', 
        'order': 'low-to-high',
        'cast_to_int': True
    },
    'r_degree': {
        'name': 'Rotation',
        'x_axis': 'Angle',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_surr_resnet18': {
        'name': 'Adv. Surr.',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_surr_resnet50': {
        'name': 'Adv.Surr.',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_klvae8': {	
        'name': 'Adv.Embed.',
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
        'name': 'Stable Diffusion v2.1',
        'marker': 'D',
        'line': '-',
        'color': '#092B8C',
        'size': 4,
        'linewidth': 1,
    },
    'flux': {
        'name': 'Flux.1 [dev]',
        'marker': '^',
        'line': '-',
        'color': '#00B4D8',
        'size': 5,
        'linewidth': 1,
    },
    'flux_s': {
        'name': 'Flux.1 [schnell]',
        'marker': 'v',
        'line': '-',
        'color': '#6CD874',
        'size': 5,
        'linewidth': 1,
    },

}

METHODS_NAME_MAPPING = {
    'tr': 'Tree-Ring',
    'rid': 'RingID',
    'gs': 'Gaussian Shading',
    'prc': 'PRC',
    'grids': 'GRIDS',
}

WMCH_NAME_MAPPING = {
    4: {
        'name': '4 WmCh',
        'marker': 'o',
        'line': '-',
        'color': '#A800B7'
    },
    8: {
        'name': '8 WmCh',
        'marker': 'x',
        'line': '-',
        'color': '#C54690'
    },
    12: {
        'name': '12 WmCh',
        'marker': 'o',
        'line': '-',
        'color': '#E28D69'
    },
    16: {
        'name': '16 WmCh',
        'marker': 'x',
        'line': '-',
        'color': '#FFD342'
    },

}

TIF_NAME_MAPPING = {
    4: {
        'name': '4 test inf steps',
        'marker': 'o',
        'line': '-',
        'color': '#A800B7'
    },
    6: {
        'name': '6 test inf steps',
        'marker': 'x',
        'line': '-',
        'color': '#BE359A'
    },
    8: {
        'name': '8 test inf steps',
        'marker': 'o',
        'line': '-',
        'color': '#D46A7D'
    },
    10: {
        'name': '10 test inf steps',
        'marker': 'x',
        'line': '-',
        'color': '#E99E5F'
    },
    12: {
        'name': '12 test inf steps',
        'marker': 'o',
        'line': '-',
        'color': '#FFD342'
    },
    


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


# for Exp2 and Exp3, plot TPR per attack for all attacks, with the lines as either the wm_ch or the test_inf_steps
# both assume the df contains just data for one model
def plot_tpr_per_attack_compare_variable(args, results_df, result_metric='tpr_empirical', compare_variable='wm_ch', compare_variable_mapping=None):

    results_df['set_fpr'].unique()[0] # set_fpr should be the same for all experiments, so we can just take the first value

    attack_names = results_df['attack_name'].unique()
    wm_methods = results_df['wm_method'].unique()
    compare_variables = results_df[compare_variable].unique()
    
    # order the attacks and methods based on the order in name_mapping
    attack_names = np.array(sorted(attack_names, key=lambda x: list(ATTACK_NAME_MAPPING.keys()).index(x)))
    wm_methods = np.array(sorted(wm_methods, key=lambda x: list(METHODS_NAME_MAPPING.keys()).index(x)))
    compare_variables = np.array(sorted(compare_variables))
    print(f'compare_variable: {compare_variable}')
    print(f'compare_variables: {compare_variables}')

    # for each attack (rows), plot all 4 WM methods in 4 sublpots (cols), all 2 models as lines
    ncols = wm_methods.shape[0] + 1 # per method, plus one for title
    nrows = attack_names.shape[0] # for each attack
    fs = 10
    fs_xticks = 8
    fs_yticks = 8
    fs_title = 14
    y_adj = 1.01
    title_height_ratio = 0.15
    height_correction = 0
    

    fig, gs, title_axes = setup_gridspec_figure(
        nrows=nrows, ncols=ncols ,
        fs=fs, title=args.title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio,
        sp_width=2, sp_height=1.75, height_correction=height_correction,
    )

    # # set the titles for each row, as the attack names
    # for i, ax in enumerate(title_axes):
    #     ax.text(0.5, 0.4, ATTACK_NAME_MAPPING[attack_names[i]]['name'], fontsize=fs_title, fontweight="bold", ha="center", va="center")
                      
    handles, labels = [], []

    if result_metric == 'tpr_empirical':
        ylabel = "TPR@FPR=0.01"
        ylim = [-0.1, 1.1]
        yticks = np.arange(0, 1.1, 0.25)
    elif result_metric == 'tpr_empirical_mean':
        ylabel = "Mean TPR@FPR=0.01"
        ylim = [-0.1, 1.1]
        yticks = np.round(np.arange(0, 1.1, 0.25), 2)
    elif result_metric == 'wm_diff':
        ylabel = "Mean Pixel Difference"
        ylim = [0.5, 1.3]
        yticks = np.round(np.arange(0.6, 1.2, 0.25), 2)
    elif result_metric == 'clip_score_wm':
        ylabel = "CLIP Score (WM)"
        ylim = [0.29, 0.39]
        yticks = np.round(np.arange(0.3, 0.38, 0.02), 2)
    elif result_metric == 'clip_score_nowm':
        ylabel = "CLIP Score (No WM)"
        ylim = [0.29, 0.39]
        yticks = np.round(np.arange(0.3, 0.38, 0.1), 2)
    elif result_metric == 'fid_wm_coco':
        ylabel = "FID (WM vs. COCO)"
        ylim = [-0.1, 300]
        yticks = int(np.arange(0, 301, 50))
    elif result_metric == 'fid_nowm_coco':
        ylabel = "FID (No WM vs. COCO)"
        ylim = [-0.1, 300]
        yticks = int(np.arange(0, 301, 50))
    elif result_metric == 'fid_wm_nowm':
        ylabel = "FID (WM vs. No WM)"
        ylim = [-0.1, 300]
        yticks = int(np.arange(0, 301, 50))
    elif result_metric == 'acc':
        ylabel = "Accuracy"
        ylim = [-0.1, 1.1]
        yticks = np.round(np.arange(0, 1.1, 0.25), 2)


    # loop through all attacks (rows), and then per attack, loop through all WM methods
    for i, attack_name in enumerate(attack_names): # rows
        attack_df = results_df[results_df['attack_name'] == attack_name]
        if attack_name not in ATTACK_NAME_MAPPING:
            continue

        axes = [fig.add_subplot(gs[2*i +1, j]) for j in range(ncols)]
        for j, wm_method in enumerate(np.concatenate((wm_methods, ["title"]))): # columns
            if wm_method == "title": # last column is title of the attack
                axes[j].axis('off')
                axes[j].text(0.1, 0.5, ATTACK_NAME_MAPPING[attack_name]['name'], fontsize=fs, fontweight="bold", ha="left", va="center")
                if i == 0:
                    axes[j].set_title('Attacktype', fontsize=fs)
            else:
                wm_df = attack_df[attack_df['wm_method'] == wm_method]
                
                # Set axis direction based on attack type
                if ATTACK_NAME_MAPPING[attack_name]['order'] == 'low-to-high':
                    axes[j].invert_xaxis()
                    
                if i == 0:
                    axes[j].set_title(METHODS_NAME_MAPPING[wm_method], fontsize=fs)
                
                axes[j].set_yticks(yticks)
                axes[j].set_yticklabels(yticks, fontsize=fs_yticks)
                axes[j].set_ylim(ylim)
                axes[j].grid(True, linestyle='--', alpha=0.5)
                axes[j].spines['top'].set_visible(False)
                axes[j].spines['right'].set_visible(False)

                if j == 0:# Add y-axis label to the first plot in each row
                    axes[j].set_ylabel(ylabel)
                else:# disable y-axis labels for all but the first column
                    plt.setp(axes[j].get_yticklabels(), visible=False)
                    plt.setp(axes[j].get_yticklines(), visible=False)

                for comp_var in compare_variables: # lines
                    compare_variable_df = wm_df[wm_df[compare_variable] == comp_var]
                    # Check if the wm_ch_df is empty
                    if compare_variable_df.empty:
                        print(f"Warning: No data for {attack_name}, {wm_method}, {comp_var}\n")
                        continue

                    if attack_name == 'no_attack':
                        # No need to order the attack strengths for the no attack case
                        strengths = compare_variable_df['attack_strength'].unique()
                        results = compare_variable_df[result_metric].values
                        ci_lower = compare_variable_df['tpr_ci_lower_percentile'].values
                        ci_upper = compare_variable_df['tpr_ci_upper_percentile'].values
                    else:
                        strengths, results, ci_lower, ci_upper = order_attack_strengths(
                            ATTACK_NAME_MAPPING[attack_name]['order'],
                            compare_variable_df['attack_strength'], 
                            compare_variable_df[result_metric],
                            compare_variable_df['tpr_ci_lower_percentile'],
                            compare_variable_df['tpr_ci_upper_percentile'],
                            ATTACK_NAME_MAPPING[attack_name]['cast_to_int'],
                        )
                    
                    label = compare_variable_mapping[comp_var]['name']
                    
                    # Plot using actual strength values
                    line, = axes[j].plot(strengths, results,
                                marker=compare_variable_mapping[comp_var]['marker'],
                                linestyle=compare_variable_mapping[comp_var]['line'],
                                label=label,
                                color=compare_variable_mapping[comp_var]['color'])
                    
                    if 'tpr' in result_metric and ((not np.isnan(ci_lower).any() and not np.isnan(ci_upper).any()) or (len(ci_lower) > 0 and len(ci_upper) > 0)):
                        axes[j].fill_between(strengths, ci_lower, ci_upper, color=compare_variable_mapping[comp_var]['color'], alpha=0.2)
                        if attack_name == 'no_attack':
                            axes[j].plot(strengths, ci_lower, color=compare_variable_mapping[comp_var]['color'], alpha=0.2, marker='x', linestyle='--')
                            axes[j].plot(strengths, ci_upper, color=compare_variable_mapping[comp_var]['color'], alpha=0.2, marker='x', linestyle='--')

                                
                    if label not in labels:
                        handles.append(line)
                        labels.append(label)

                    # Set only the actual strength values as ticks
                    axes[j].set_xticks(strengths)
                    axes[j].set_xticklabels(strengths, fontsize=fs_xticks)
                    #axes[j].set_xlim([strengths[0]-0.1, strengths[-1]+0.1])
            
            

    
    fig.legend(loc='upper center', bbox_to_anchor=(0.2, 0.44, 0.5, 0.5), ncol=len(compare_variables), handles=handles, labels=labels)
    

    plt.savefig(args.output_plot, bbox_inches='tight', dpi=300)
    #plt.show()
    plt.close()
    print(f"Plot saved to {args.output_plot}")
