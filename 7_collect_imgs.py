import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse


def collect_imgs_over_experiment(experiments_dir, dataset_identifiers, prompt_dataset, output_dir, args):

    for i, method in enumerate(args.methods):
        method_dir = os.path.join(experiments_dir, method)
        if not os.path.exists(method_dir):
            print(f"Method directory {method_dir} does not exist. Skipping.")
            continue

        for j, model in enumerate(args.models):
            model_dir = os.path.join(method_dir, model, args.prompt_dataset)
            if not os.path.exists(model_dir):
                print(f"Model directory {model_dir} does not exist. Skipping.")
                continue


            for k, dataset in enumerate(dataset_identifiers):
                dataset_dir = os.path.join(model_dir, dataset)
                if not os.path.exists(dataset_dir):
                    print(f"Dataset directory {dataset_dir} does not exist. Skipping.")
                    continue

                #print(f"Processing {method} on {dataset} with model {model}...")
                print(f"\ncurretly processing {dataset_dir}")

                wm_path = os.path.join(dataset_dir, 'encoded_imgs', 'data', 'wm')
                wm_img_path = os.path.join(wm_path, f"{args.img_id}.png")
                
                if os.path.exists(wm_img_path):
                    wm_output_path = os.path.join(output_dir, f"{model}-{method}.png")
                    os.makedirs(os.path.dirname(wm_output_path), exist_ok=True)
                    plt.imsave(wm_output_path, plt.imread(wm_img_path))
                    print(f"Saved WM image to {wm_output_path}")
                else:
                    print(f"WM image {wm_img_path} does not exist. Skipping.")

                
                nowm_path = os.path.join(dataset_dir, 'encoded_imgs', 'data', 'nowm')
                nowm_img_path = os.path.join(nowm_path, f"{args.img_id}.png")

                if os.path.exists(nowm_img_path):
                    nowm_output_path = os.path.join(output_dir, f"{model}-nowm.png")
                    os.makedirs(os.path.dirname(nowm_output_path), exist_ok=True)
                    plt.imsave(nowm_output_path, plt.imread(nowm_img_path))
                    print(f"Saved NOWM image to {nowm_output_path}")
                else:
                    print(f"NOWM image {nowm_img_path} does not exist. Skipping.")
                
                # load the prompts file 
                prompts = os.path.join(dataset_dir, 'encoded_imgs', 'data', 'prompts.txt')
                with open(prompts, 'r') as f:
                    prompts_data = f.readlines()
                prompts_data = [line.strip() for line in prompts_data if line.strip()]
                prompt = prompts_data[args.img_id].split(': ', 1)[-1]  # Get the prompt text after the first space
                print(f"Processing prompt: {prompt}")
                
                # save a text file with the prompt 
                prompt_output_path = os.path.join(output_dir, f"prompt.txt")
                os.makedirs(os.path.dirname(prompt_output_path), exist_ok=True)
                with open(prompt_output_path, 'w') as f:
                    f.write(prompt)


                
if __name__ == '__main__':
    args = argparse.Namespace()


    # specify which experimental setup we want to plot
    args.num_imgs = 2000
    args.exp_name = 'exp1_mjprompts'
    args.prompt_dataset = 'mjprompts'

    # specify which image we want to collect
    args.img_id = 1614


    args.methods = ['grids', 'gs', 'prc', 'rid', 'tr']
    args.models = ['flux', 'flux_s', 'sd', ]

    args.dataset_identifier = [f'num_{args.num_imgs}_fpr_0.01_cfg_3.0_wmch_16_infsteps_50', # flux
                                   f'num_{args.num_imgs}_fpr_0.01_cfg_0_wmch_16_infsteps_4', # flux_s
                                   f'num_{args.num_imgs}_fpr_0.01_cfg_3.0_wmch_4_infsteps_50']  # sd

    # create the output directories and ffilenames
    args.input_dir = os.path.join('experiments', args.exp_name)
    args.output_dir = os.path.join('x_imgs_comp', args.exp_name, args.prompt_dataset, str(args.img_id))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    print(f"Collecting images for experiment {args.exp_name} with prompt dataset {args.prompt_dataset} and image ID {args.img_id}...")
    collect_imgs_over_experiment(args.input_dir, args.dataset_identifier, args.prompt_dataset, args.output_dir, args)