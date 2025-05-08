import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import setup_gridspec_figure


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



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot results of watermarking methods under different attacks')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Directory to save the merged CSV file')
    args = parser.parse_args()
    

    # specify which experimental setup we want to plot
    args.num_imgs = 100
    args.exp_name = 'exp2'
    args.prompt_dataset = 'coco'

    # depending on the experiment, we need merge over all number of channels, as they should also be 
    # specfied in their respeictive csv files
    args.dataset_identifier = [f'num_{args.num_imgs}_fpr_0.01_cfg_3.0_wmch_16', 
                               f'num_{args.num_imgs}_fpr_0.01_cfg_3.0_wmch_12',
                               f'num_{args.num_imgs}_fpr_0.01_cfg_3.0_wmch_8',
                               f'num_{args.num_imgs}_fpr_0.01_cfg_3.0_wmch_4'] 

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
    # results_df = pd.read_csv(args.output_csv)

    # # 1. plot TPR vs attack strength
    # plot_tpr_per_attack(args, results_df)

    # # 2. plot TPR vs CLIP 
    # xmin = results_df['clip_score_wm'].min()
    # xmax = results_df['clip_score_wm'].max()
    # plot_tpr_per_metric(
    #     args, 
    #     results_df, 
    #     metric_name="clip_score", 
    #     metric_column="clip_score_wm",
    #     title_suffix="CLIP similarity score",
    #     xlabel="CLIP score (↑)",
    #     xlim=[xmin, xmax]
    # )
    
    # # 3. plot TPR vs diff 
    # xmin = results_df['wm_diff'].min()
    # xmax = results_df['wm_diff'].max()
    # plot_tpr_per_metric(
    #     args, 
    #     results_df, 
    #     metric_name="wm_diff", 
    #     metric_column="wm_diff",
    #     title_suffix="Abs. Mean Difference (originial - recovered)",
    #     xlabel="Diff (↓)",
    #     xlim=[xmin, xmax]
    # )
    
    # # 4. plot TPR vs FID (WM vs COCO)
    # xmin = results_df['fid_wm_coco'].min()
    # xmax = results_df['fid_wm_coco'].max()
    # plot_tpr_per_metric(
    #     args, 
    #     results_df, 
    #     metric_name="fid_coco", 
    #     metric_column="fid_wm_coco",
    #     title_suffix="FID (WM vs COCO)",
    #     xlabel="FID (↓)",
    #     xlim=[xmin, xmax]
    # )
    
    # # 5. plot TPR vs FID (WM vs NOWM)
    # xmin = results_df['fid_wm_nowm'].min()
    # xmax = results_df['fid_wm_nowm'].max()
    # plot_tpr_per_metric(
    #     args, 
    #     results_df, 
    #     metric_name="fid_wm_nowm", 
    #     metric_column="fid_wm_nowm",
    #     title_suffix="FID (WM vs NOWM)",
    #     xlabel="FID (↓)",
    #     xlim=[xmin, xmax]
    # )
