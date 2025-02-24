import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


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

def merge_csv_files(input_dir, output_file):
    csv_files = []
    # walk through all files in the input_dir
    for root, dirs, files in os.walk(input_dir): 
        for file in files:
            if file.endswith('.csv'):
                # save not only the file name, but the full path
                file_path = os.path.join(root, file)
                csv_files.append(file_path)
                print(f'Found csv file: {file_path}')

    print(f'Merging {len(csv_files)} csv files')

    # combine all csv files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(input_dir, csv_file))
        dfs.append(df)
    # combine all dataframes, delete duplicates (if we have a csv files thats already a merge of multiple csv files, we might have duplicates)
    combined_df = pd.concat(dfs).drop_duplicates()

    # save to output file
    combined_df.to_csv(output_file, index=False)

def order_attack_strengths(order, attack_strengths, attack_results):
    """Orders attack strengths based on difficulty"""
    # Convert series to numpy for easier manipulation
    strengths = attack_strengths.values
    results = attack_results.values

    # print dtypes of strengths and results
    print(f'strengths: {strengths.dtype}, results: {results.dtype}')
    print(f'strengths: {strengths}, results: {results}')
    # elements in strengths are strings, convert to float
    strengths = strengths.astype(float)
    
    if order == 'high-to-low':
        # Keep original order for attacks where lower values = easier
        idx = np.argsort(strengths)   # Sort in ascending order
    else: # 'low-to-high'
        # Reverse the order for attacks where higher values = easier
        idx = np.argsort(-strengths)  # Sort in descending order
        
        
    return strengths[idx], results[idx]

def main(args):

    merge_csv_files(args.input_dir, args.output_file)

    results_df = pd.read_csv(args.output_file)

    set_fpr = 0.01 # results_df['set_fpr'].unique()[0]

    attack_names = results_df['attack_name'].unique()
    #wm_methods = results_df['wm_method'].unique()
    models = results_df['model_id'].unique()

    cols = 4
    rows = 2

    fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), sharey=True)
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D for easier indexing
    fig.supylabel(f'TPR@FPR={set_fpr}')
    
    # Collect handles and labels for the legend
    handles, labels = [], []

    # until now, one such plot represents one (of 4) WM methods, plots all 2 models, all 10 attacks
    for i, attack_name in enumerate(attack_names):
        attack_df = results_df[results_df['attack_name'] == attack_name]
        if attack_name not in attack_name_mapping:
            continue
            
        print(f'\n\nPlotting {attack_name}')
        for model in models:
            print(f'\nPlotting {model}')
            wm_df = attack_df[attack_df['model_id'] == model]

            if attack_name == 'no_attack':
                # No need to order the attack strengths for the no attack case
                strengths = wm_df['attack_strength'].unique()
                results = wm_df['tpr_empirical'].values
            else:
                strengths, results = order_attack_strengths(
                    attack_name_mapping[attack_name]['order'],
                    wm_df['attack_strength'], 
                    wm_df['tpr_empirical']
                )

            label = diff_model_markers[model]['name']
            
            # Plot using actual strength values
            line, = axs[i].plot(strengths, results,
                        marker=diff_model_markers[model]['marker'],
                        linestyle=diff_model_markers[model]['line'],
                        label=label,
                        color=diff_model_markers[model]['color'])
                        
            if label not in labels:
                handles.append(line)
                labels.append(label)

        # Set only the actual strength values as ticks
        axs[i].set_xticks(strengths)
        axs[i].set_xticklabels(strengths)
        
        # Set axis direction based on attack type
        if attack_name_mapping[attack_name]['order'] == 'low-to-high':
            axs[i].invert_xaxis()
            
        axs[i].grid(True)
        axs[i].set_title(attack_name_mapping[attack_name]['name'])
        axs[i].set_xlabel(attack_name_mapping[attack_name]['x_axis'])
        axs[i].set_ylim([-0.1, 1.1])
    
    method_name = results_df['wm_method'].unique()[0] 
    fig.suptitle(f'Performance of watermarking method {method_name} under different attacks\n for experiments in {args.input_dir}', fontsize=16)
    fig.legend(loc='lower center', ncol=len(models), handles=handles, labels=labels)
    plt.tight_layout() 
    fig.subplots_adjust(bottom=0.1)


    plt.savefig(args.output_plot)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot results of watermarking methods under different attacks')
    parser.add_argument('--input_dir', type=str, help='Directory containing csv files with results')
    parser.add_argument('--output_file', type=str, help='Output file for merged csv files')
    

    args = parser.parse_args()

    args.output_plot = args.output_file.replace('.csv', '.png')
    main(args)