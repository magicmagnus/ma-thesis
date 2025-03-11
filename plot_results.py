import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import setup_gridspec_figure

# dict for saving names of axes/values
attack_name_mapping = {
    'crop_scale': {
        'name': 'Crop&Scale',
        'x_axis': 'Crop ratio and scale ratio',
        'order': 'low-to-high'
    },
    'jpeg_ratio': {
        'name': 'JPEG',
        'x_axis': 'Quality factor', 
        'order': 'low-to-high'
    },
    'gaussian_std': {
        'name': 'Noise',
        'x_axis': 'Standard deviation',
        'order': 'high-to-low'
    },
    'r_degree': {
        'name': 'Rotate',
        'x_axis': 'Angle',
        'order': 'high-to-low'
    },
    'gaussian_blur_r': {
        'name': 'Blur',
        'x_axis': 'Radius',
        'order': 'high-to-low'
    },
    'brightness_factor': {
        'name': 'Brightness',
        'x_axis': 'Factor',
        'order': 'high-to-low'
    },
    'adv_surr_resnet18': {
        'name': 'Adv. Surrogate RN18',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low'
    },
    'adv_surr_resnet50': {
        'name': 'Adv. Surrogate RN50',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low'
    },
    'adv_embed_klvae8': {	
        'name': 'Adv. Embed KLVAE8',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low'
    },
    'adv_embed_klvae16': {
        'name': 'Adv. Embed KLVAE16',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low'
    },
    'adv_embed_resnet18': {
        'name': 'Adv. Embed RN18',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low'
    },
    'adv_embed_resnet50': {
        'name': 'Adv. Embed RN50',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low'
    },
    'adv_embed_resnet101': {
        'name': 'Adv. Embed RN101',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low'
    },
    'adv_embed_sdxlvae': {
        'name': 'Adv. Embed SDXLVAE',
        'x_axis': 'eps = x/255',
        'order': 'high-to-low'
    },
    'no_attack': {
        'name': 'No Attack',
        'x_axis': 'Attack strength',
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
}

wm_methods_names = {
    'prc': 'PRC',
    'gs': 'Gaussian Shading',
    'rid': 'Ring ID',
    'tr': 'Tree-Ring'
}

def merge_csv_for_dataset_identifier(experiments_dir, dataset_identifiers, output_file):
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
                    if file.endswith('.csv'):
                        file_path = os.path.join(root2, file)
                        csv_files.append(file_path)
                        print(f'Found CSV file: {file_path}')

    if not csv_files:
        print(f'No CSV files found for dataset_identifier: {dataset_identifiers}')
        return
    
    print(f'Merging {len(csv_files)} CSV files for dataset_identifier: {dataset_identifiers}')

    # Combine all CSV files
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    combined_df = pd.concat(dfs).drop_duplicates()

    # Save the merged CSV
    combined_df.to_csv(output_file, index=False)
    print(f'Saved merged CSV to {output_file}')

def order_attack_strengths(order, attack_strengths, attack_results):
    """Orders attack strengths based on difficulty"""
    # Convert series to numpy for easier manipulation
    strengths = attack_strengths.values
    results = attack_results.values

    # print dtypes of strengths and results
    #print(f'strengths: {strengths.dtype}, results: {results.dtype}')
    #print(f'strengths: {strengths}, results: {results}')
    # elements in strengths are strings, convert to float
    strengths = strengths.astype(float)
    
    if order == 'high-to-low':
        # Keep original order for attacks where lower values = easier
        idx = np.argsort(strengths)   # Sort in ascending order
    else: # 'low-to-high'
        # Reverse the order for attacks where higher values = easier
        idx = np.argsort(-strengths)  # Sort in descending order
        
        
    return strengths[idx], results[idx]

def plot_per_attack(args):


    
    
    

    results_df = pd.read_csv(args.output_csv)

    set_fpr = 0.01 # results_df['set_fpr'].unique()[0]

    attack_names = results_df['attack_name'].unique()
    wm_methods = results_df['wm_method'].unique()
    models = results_df['model_id'].unique()

    # for each attack, plot all 4 WM methods in 4 sublpots, all 2 models as lines

    ncols = 4 # per method
    nrows = attack_names.shape[0] # for each attack
    fs = 10
    fs_title = 14
    y_adj = 0.95
    title_height_ratio = 0.8
    title = f'Performance of watermarking methods under different attacks\n for experiments in {args.dataset_identifier}'

    fig, gs, title_axes = setup_gridspec_figure(
        nrows=nrows, ncols=ncols,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio,
        sp_width=2, sp_height=1.75
    )

    # set the titles for each row, as the attack names
    for i, ax in enumerate(title_axes):
        ax.text(0.5, 0.25, attack_name_mapping[attack_names[i]]['name'], fontsize=fs_title, fontweight="bold", ha="center", va="center")
                      
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

            for model in models:
                model_df = wm_df[wm_df['model_id'] == model]

                if attack_name == 'no_attack':
                    # No need to order the attack strengths for the no attack case
                    strengths = model_df['attack_strength'].unique()
                    results = model_df['tpr_empirical'].values
                else:
                    strengths, results = order_attack_strengths(
                        attack_name_mapping[attack_name]['order'],
                        model_df['attack_strength'], 
                        model_df['tpr_empirical']
                    )

                label = diff_model_markers[model]['name']
                
                # Plot using actual strength values
                line, = axes[j].plot(strengths, results,
                            marker=diff_model_markers[model]['marker'],
                            linestyle=diff_model_markers[model]['line'],
                            label=label,
                            color=diff_model_markers[model]['color'])
                            
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
            axes[j].set_xlabel(attack_name_mapping[attack_name]['x_axis'])
            axes[j].set_ylim([-0.1, 1.1])

    
    fig.legend(loc='lower center', ncol=len(models), handles=handles, labels=labels)
    

    plt.savefig(args.output_plot)
    plt.show()

        



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot results of watermarking methods under different attacks')
    parser.add_argument('--input_dir', type=str, default='experiments', help='Directory containing the CSV files')
    parser.add_argument('--dataset_identifier', type=list, default=['num_5_fpr_0.01_cfg_3.0_wmch_16', 'num_5_fpr_0.01_cfg_3.0_wmch_4'] )
    parser.add_argument('--output_dir', type=str, default='experiments', help='Directory to save the merged CSV file')
    

    args = parser.parse_args()

    # if we want to compare sd and flux, we merge wmch_16 and wmch_4
    args.dataset_identifier = ['num_10_fpr_0.01_cfg_3.0_wmch_16', 'num_10_fpr_0.01_cfg_3.0_wmch_4'] 
    # if, for any reason later, we want to compare only one of them, we can change the dataset_identifier

    # extra 
    args.output_dir = os.path.join(args.output_dir, '_results')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # where the plot will be saved
    args.output_plot = os.path.join(args.output_dir, args.dataset_identifier[0] + '_plot.png')

    # where the merged csv will be saved
    args.output_csv = os.path.join(args.output_dir, args.dataset_identifier[0] + '_merged.csv')

    # merge all csv matching the dataset_identifier in the input_dir into the output_csv
    merge_csv_for_dataset_identifier(args.input_dir, args.dataset_identifier, args.output_csv)

    # plot the results
    plot_per_attack(args)   