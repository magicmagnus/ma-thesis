import os
if "is/sg2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import torch
import pickle
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils import seed_everything, image_distortion, print2file

sys.path.append(os.path.join(os.path.dirname(__file__), 'waves', 'adversarial'))
from waves.adversarial.train import train_surrogate_classifier


def main(args):

    # Set random seed
    seed_everything(42)

    if "is/sg2" in os.getcwd():
        HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    else:
        HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting Training...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')

    # load dataset
    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_{args.model_tag}_{args.dataset_tag}'
    input_path = f'./results/{exp_id}'


    args_train = argparse.Namespace()

    args_train.image_size = 512
    args_train.num_classes = 2
    args_train.train_ratio = 0.8
    args_train.seed = 42
    if args.adv_surr_method == "nowm_wm":
        args_train.train_data_path_class0 = os.path.join(input_path, "nowm")
        args_train.train_data_path_class1 = os.path.join(input_path, "wm")
    elif args.adv_surr_method == "real_wm":
        args_train.train_data_path_class0 = os.path.join('coco', 'val2017')
        args_train.train_data_path_class1 = os.path.join(input_path, "wm")
    args_train.train_size = None
    args_train.surrogate_model = "ResNet18"
    args_train.model_save_path = os.path.join(input_path)
    args_train.model_save_name = os.path.join(f'adv_cls_{args.method}_{args.adv_surr_method}')
    args_train.learning_rate = 1e-3
    args_train.num_epochs = 20
    args_train.batch_size = 128
    args_train.do_eval = True
    args_train.normalize = True
    args_train.device = device
    args_train.log_file = args.log_file
    args_train.log_dir = args.log_dir

    train_surrogate_classifier(args_train)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configurations from the JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Update args namespace with configurations
    for key, value in config.items():
        setattr(args, key, value)

    # create a custom folder based on the current time in the name
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = f'./experiments/{date}_att_train_surr_{args.run_name}'
    os.makedirs(args.log_dir)

    args.model_tag = "SD" if args.model_id == 'stabilityai/stable-diffusion-2-1-base' else "Flux"
    args.dataset_tag = "coco" if args.dataset_id == 'coco' else "SDprompts"	

    logfile_name = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_{args.run_name}_{args.model_tag}_{args.dataset_tag}'
    
    # create a log file
    args.log_file = open(f'{args.log_dir}/{logfile_name}.txt', 'w', buffering=1)  # Use line buffering
    
    main(args)