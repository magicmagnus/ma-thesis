import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import setup_gridspec_figure

# dict for saving names of axes/values
attack_name_mapping = {
    'gaussian_std': {
        'name': 'Noise (Gaussian std)',
        'x_axis': 'Standard deviation',
        'order': 'high-to-low',
        'cast_to_int': False
    },
    'brightness_factor': {
        'name': 'Brightness (factor)',
        'x_axis': 'Factor',
        'order': 'high-to-low',
        'cast_to_int': False
    },
    'crop_scale': {
        'name': 'Crop&Scale (ratio)',
        'x_axis': 'Crop ratio and scale ratio',
        'order': 'low-to-high',
        'cast_to_int': False
    },
    'crop': {
        'name': 'Crop (ratio)',
        'x_axis': 'Crop ratio',
        'order': 'low-to-high',
        'cast_to_int': False
    },
    'jpeg_ratio': {
        'name': 'JPEG (quality factor)',
        'x_axis': 'Quality factor', 
        'order': 'low-to-high',
        'cast_to_int': True
    },
    'r_degree': {
        'name': 'Rotation (degrees)',
        'x_axis': 'Angle',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'gaussian_blur_r': {
        'name': 'Blur (radius)',
        'x_axis': 'Radius',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_surr_resnet18': {
        'name': 'Adv. Surrogate RN18 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_surr_resnet50': {
        'name': 'Adv. Surrogate RN50 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_klvae8': {	
        'name': 'Adv. Embed KLVAE8 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_klvae16': {
        'name': 'Adv. Embed KLVAE16 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_resnet18': {
        'name': 'Adv. Embed RN18 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_resnet50': {
        'name': 'Adv. Embed RN50 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_resnet101': {
        'name': 'Adv. Embed RN101 (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'adv_embed_sdxlvae': {
        'name': 'Adv. Embed SDXLVAE (eps)',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low',
        'cast_to_int': True
    },
    'no_attack': {
        'name': 'No Attack',
        'x_axis': 'Attack strength (eps)',
        'order': 'low-to-high'
    }
}

diff_model_markers = {
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

wm_methods_names = {
    'prc': 'PRC',
    'gs': 'Gaussian Shading',
    'tr': 'Tree-Ring',
    'rid': 'Ring ID',
    'grids': 'GRIDS',
}

def merge_csv_for_dataset_identifier(experiments_dir, dataset_identifiers, prompt_dataset, output_file):
    csv_files = []
    # print(f'Looking in {experiments_dir} for CSV files for dataset_identifier: {dataset_identifiers}')
    # Walk through the directory structure
    for root, dirs, files in os.walk(experiments_dir): 
        # Check if the last part of the path matches the dataset_identifier
        # print(f'1 Checking {root}')
        if os.path.basename(root) in dataset_identifiers:
            new_root = os.path.join(root, 'decode_imgs', 'logs')
            # print(f'1 Found dataset_identifier: {new_root}')
            for root2, dirs, files in os.walk(new_root):
                # print(f'2 Checking {root2}')
                # print(f'2 with files: {files}')
                for file in files:
                    if file.endswith('.csv') and prompt_dataset in root2:
                        file_path = os.path.join(root2, file)
                        csv_files.append(file_path)
                        print(f'Found CSV file: {file_path}')

    if not csv_files:
        print(f'No CSV files found for dataset_identifier: {dataset_identifiers}')
        return
    
    print(f'Merging {len(csv_files)} CSV files for dataset_identifier: {dataset_identifiers} and prompt_dataset: {prompt_dataset}')

    # Combine all CSV files
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    combined_df = pd.concat(dfs).drop_duplicates()

    # Save the merged CSV
    combined_df.to_csv(output_file, index=False)
    print(f'Saved merged CSV to {output_file}')

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

def plot_tpr_per_attack(args,results_df):

    #results_df = pd.read_csv(args.output_csv)

    results_df['set_fpr'].unique()[0] # set_fpr should be the same for all experiments, so we can just take the first value

    attack_names = results_df['attack_name'].unique()
    wm_methods = results_df['wm_method'].unique()
    models = results_df['model_id'].unique()

    # order the attacks and methods based on the order in name_mapping
    attack_names = np.array(sorted(attack_names, key=lambda x: list(attack_name_mapping.keys()).index(x)))
    wm_methods = np.array(sorted(wm_methods, key=lambda x: list(wm_methods_names.keys()).index(x)))

    # for each attack, plot all 4 WM methods in 4 sublpots, all 2 models as lines

    ncols = wm_methods.shape[0]  # per method
    nrows = attack_names.shape[0] # for each attack
    fs = 10
    fs_title = 14
    y_adj = 0.95
    title_height_ratio = 0.65
    title = ( 
        f'Performance of watermarking methods under different attacks\n'
        f'for dataset "{args.prompt_dataset}" for experiments in \n'
        f'{args.dataset_identifier}'
    )

    fig, gs, title_axes = setup_gridspec_figure(
        nrows=nrows, ncols=ncols,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio,
        sp_width=2, sp_height=1.75
    )

    # set the titles for each row, as the attack names
    for i, ax in enumerate(title_axes):
        ax.text(0.5, 0.4, attack_name_mapping[attack_names[i]]['name'], fontsize=fs_title, fontweight="bold", ha="center", va="center")
                      
    handles, labels = [], []

    # loop through all attacks, and then per attack, loop through all WM methods
    for i, attack_name in enumerate(attack_names):
        attack_df = results_df[results_df['attack_name'] == attack_name]
        if attack_name not in attack_name_mapping:
            continue

        axes = [fig.add_subplot(gs[2*i +1, j]) for j in range(ncols)]
        for j, wm_method in enumerate(wm_methods):
            wm_df = attack_df[attack_df['wm_method'] == wm_method]

            if j != 0:
                # disable y-axis labels for all but the first column
                plt.setp(axes[j].get_yticklabels(), visible=False)
            else:
                # Add y-axis label to the first plot in each row
                axes[j].set_ylabel("TPR@FPR=0.01")

            for model in models:
                model_df = wm_df[wm_df['model_id'] == model]

                if attack_name == 'no_attack':
                    # No need to order the attack strengths for the no attack case
                    strengths = model_df['attack_strength'].unique()
                    results = model_df['tpr_empirical'].values
                    ci_lower = model_df['tpr_ci_lower_percentile'].values
                    ci_upper = model_df['tpr_ci_upper_percentile'].values
                else:
                    strengths, results, ci_lower, ci_upper = order_attack_strengths(
                        attack_name_mapping[attack_name]['order'],
                        model_df['attack_strength'], 
                        model_df['tpr_empirical'],
                        model_df['tpr_ci_lower_percentile'],
                        model_df['tpr_ci_upper_percentile'],
                        attack_name_mapping[attack_name]['cast_to_int'],
                    )
                
                label = diff_model_markers[model]['name']
                
                # Plot using actual strength values
                line, = axes[j].plot(strengths, results,
                            marker=diff_model_markers[model]['marker'],
                            linestyle=diff_model_markers[model]['line'],
                            label=label,
                            color=diff_model_markers[model]['color'])
                # pllot the CI as a shaded region
                #only if there are no NaN values in the CI or the lists are not empty
                if (not np.isnan(ci_lower).any() and not np.isnan(ci_upper).any()) or (len(ci_lower) > 0 and len(ci_upper) > 0):
                    print(f'\t\tplotting CI for {attack_name}, {wm_method}, {model}')
                    print(f'\t\tci_lower: {ci_lower}')
                    axes[j].fill_between(strengths, ci_lower, ci_upper, color=diff_model_markers[model]['color'], alpha=0.2)
                    if attack_name == 'no_attack':
                        axes[j].plot(strengths, ci_lower, color=diff_model_markers[model]['color'], alpha=0.2, marker='x', linestyle='--')
                        axes[j].plot(strengths, ci_upper, color=diff_model_markers[model]['color'], alpha=0.2, marker='x', linestyle='--')

                            
                if label not in labels:
                    handles.append(line)
                    labels.append(label)

            # Set only the actual strength values as ticks
            axes[j].set_xticks(strengths)
            axes[j].set_xticklabels(strengths)
            
            # Set axis direction based on attack type
            if attack_name_mapping[attack_name]['order'] == 'low-to-high':
                axes[j].invert_xaxis()
                
            axes[j].grid(True)
            axes[j].set_title(wm_methods_names[wm_method])
            #axes[j].set_xlabel(attack_name_mapping[attack_name]['x_axis'])
            axes[j].set_ylim([-0.1, 1.1])

    
    fig.legend(loc='lower center', ncol=len(models), handles=handles, labels=labels)
    

    plt.savefig(args.output_plot)
    plt.show()
    print(f"\nPlot saved to {args.output_plot}")



def plot_tpr_per_metric(args, results_df, metric_name, metric_column, title_suffix, xlabel, xlim):
    """
    Generic plotting function that can use any metric for the x-axis
    
    Parameters:
    - args: The command line arguments
    - metric_name: String identifier for the metric (used in filenames)
    - metric_column: Name of the column to use for x-axis values
    - title_suffix: Text to add to the plot title
    - xlabel: Label for the x-axis
    """
    # results_df = pd.read_csv(args.output_csv)
    
    attack_names = results_df['attack_name'].unique()
    wm_methods = results_df['wm_method'].unique()
    models = results_df['model_id'].unique()

    attack_names = np.array(sorted(attack_names, key=lambda x: list(attack_name_mapping.keys()).index(x)))
    wm_methods = np.array(sorted(wm_methods, key=lambda x: list(wm_methods_names.keys()).index(x)))

    # Setup figure with same layout
    ncols = wm_methods.shape[0]  # per method
    nrows = attack_names.shape[0]  # for each attack
    fs = 10
    fs_title = 14
    y_adj = 0.95
    title_height_ratio = 0.8
    title = (
        f'Watermarking performance vs {title_suffix}\n'
        f'for dataset "{args.prompt_dataset}" for experiments in \n'
        f'{args.dataset_identifier}'
    )

    fig, gs, title_axes = setup_gridspec_figure(
        nrows=nrows, ncols=ncols,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio,
        sp_width=2, sp_height=1.75
    )

    # Set row titles (attack names)
    for i, ax in enumerate(title_axes):
        ax.text(0.5, 0.25, attack_name_mapping[attack_names[i]]['name'], 
                fontsize=fs_title, fontweight="bold", ha="center", va="center")
                      
    handles, labels = [], []

    # Loop through attacks and watermarking methods
    for i, attack_name in enumerate(attack_names):
        attack_df = results_df[results_df['attack_name'] == attack_name]
        if attack_name not in attack_name_mapping:
            continue

        axes = [fig.add_subplot(gs[2*i +1, j]) for j in range(ncols)]
        for j, wm_method in enumerate(wm_methods):
            wm_df = attack_df[attack_df['wm_method'] == wm_method]

            if j != 0:
                # Disable y-axis labels for all but the first column
                plt.setp(axes[j].get_yticklabels(), visible=False)
            else:
                # Add y-axis label to the first plot in each row
                axes[j].set_ylabel("TPR@FPR=0.01")


            for model in models:
                model_df = wm_df[wm_df['model_id'] == model]
                
                # Check if the metric column exists
                if metric_column not in model_df.columns:
                    print(f"Warning: {metric_column} not found for {attack_name}, {wm_method}, {model}")
                    continue

                # Sort by the metric column
                df_sorted = model_df.sort_values(by=metric_column)
                x_values = df_sorted[metric_column].values
                tpr_values = df_sorted['tpr_empirical'].values
                attack_strengths = df_sorted['attack_strength'].values

                label = diff_model_markers[model]['name']
                
                line, = axes[j].plot(x_values, tpr_values,
                            marker=diff_model_markers[model]['marker'],
                            linestyle=diff_model_markers[model]['line'],
                            label=label,
                            color=diff_model_markers[model]['color'])
                
                # Add attack strength as text near each point for reference
                for k, (x, y, strength) in enumerate(zip(x_values, tpr_values, attack_strengths)):
                    if k % 2 == 0:  # Only label every other point to avoid clutter
                        axes[j].annotate(f"{strength}", (x, y), 
                                         textcoords="offset points", 
                                         xytext=(0, 5), 
                                         ha='center',
                                         fontsize=7)
                            
                if label not in labels:
                    handles.append(line)
                    labels.append(label)

            axes[j].grid(True)
            axes[j].set_title(wm_methods_names[wm_method])
            axes[j].set_xlabel(xlabel)
            axes[j].set_ylim([-0.1, 1.1])
            axes[j].set_xlim(xlim)
            
            # For quality metrics (like CLIP similarity score), higher is better, 
            # so have higher values to the left
            if "score" in metric_column.lower() or "similarity" in metric_column.lower():
                #print(f"enter score for {metric_column}")
                if axes[j].get_xlim()[0] < axes[j].get_xlim()[1]:  # If lower values are on left
                    #print(f"enter score for {metric_column} invert")
                    axes[j].invert_xaxis()  # Invert so higher values are on left
            # For distance metrics (like FID), lower is better, so have lower values to the left
            if "fid" in metric_column.lower() or "distance" in metric_column.lower():
                #print(f"enter fid for {metric_column}")
                if axes[j].get_xlim()[0] > axes[j].get_xlim()[1]:  # If higher values are on left
                    #print(f"enter fid for {metric_column} invert")
                    axes[j].invert_xaxis()  # Invert so lower values are on left

    fig.legend(loc='lower center', ncol=len(models), handles=handles, labels=labels)

    output_plot = args.output_plot.replace('.pdf', f'_{metric_name}.pdf')
    plt.savefig(output_plot)
    plt.show()
    print(f"\n{title_suffix} plot saved to {output_plot}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot results of watermarking methods under different attacks')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Directory to save the merged CSV file')
    args = parser.parse_args()
    

    # specify which experimental setup we want to plot
    args.num_imgs = 200
    args.exp_name = 'exp1'
    args.prompt_dataset = 'coco'

    # for now, we merge results over wmch_16 and wmch_4
    # later we might want to include only one of them
    # to compare external
    args.dataset_identifier = [f'num_{args.num_imgs}_fpr_0.01_cfg_3.0_wmch_16', f'num_{args.num_imgs}_fpr_0.01_cfg_3.0_wmch_4'] 
    

    # create the output directories and ffilenames
    args.input_dir = os.path.join('experiments', args.exp_name)
    args.output_dir = os.path.join(args.output_dir, args.exp_name, '_results', args.prompt_dataset,  args.dataset_identifier[0])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_plot = os.path.join(args.output_dir, args.dataset_identifier[0] + '_plot.pdf')
    args.output_csv = os.path.join(args.output_dir, args.dataset_identifier[0] + '_merged.csv')

    # merge all .csv files in the args.input_dir
    # matching the dataset_identifiers and args.prompt_dataset
    # into the args.output_csv
    merge_csv_for_dataset_identifier(args.input_dir, args.dataset_identifier, args.prompt_dataset, args.output_csv)
    results_df = pd.read_csv(args.output_csv)

    # 1. plot TPR vs attack strength
    plot_tpr_per_attack(args, results_df)

    # 2. plot TPR vs CLIP 
    xmin = results_df['clip_score_wm'].min()
    xmax = results_df['clip_score_wm'].max()
    plot_tpr_per_metric(
        args, 
        results_df, 
        metric_name="clip_score", 
        metric_column="clip_score_wm",
        title_suffix="CLIP similarity score",
        xlabel="CLIP score (↑)",
        xlim=[xmin, xmax]
    )
    
    # 3. plot TPR vs diff 
    xmin = results_df['wm_diff'].min()
    xmax = results_df['wm_diff'].max()
    plot_tpr_per_metric(
        args, 
        results_df, 
        metric_name="wm_diff", 
        metric_column="wm_diff",
        title_suffix="Abs. Mean Difference (originial - recovered)",
        xlabel="Diff (↓)",
        xlim=[xmin, xmax]
    )
    
    # 4. plot TPR vs FID (WM vs COCO)
    xmin = results_df['fid_wm_coco'].min()
    xmax = results_df['fid_wm_coco'].max()
    plot_tpr_per_metric(
        args, 
        results_df, 
        metric_name="fid_coco", 
        metric_column="fid_wm_coco",
        title_suffix="FID (WM vs COCO)",
        xlabel="FID (↓)",
        xlim=[xmin, xmax]
    )
    
    # 5. plot TPR vs FID (WM vs NOWM)
    xmin = results_df['fid_wm_nowm'].min()
    xmax = results_df['fid_wm_nowm'].max()
    plot_tpr_per_metric(
        args, 
        results_df, 
        metric_name="fid_wm_nowm", 
        metric_column="fid_wm_nowm",
        title_suffix="FID (WM vs NOWM)",
        xlabel="FID (↓)",
        xlim=[xmin, xmax]
    )
