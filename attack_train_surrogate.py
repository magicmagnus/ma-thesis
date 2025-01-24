import os
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

    #HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'
    HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting Attack...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}'
    # load dataset
    if args.dataset_id == 'coco':
        exp_id = exp_id + '_coco'


    args_train = argparse.Namespace()

    args_train.image_size = 512
    args_train.num_classes = 2
    args_train.train_ratio = 0.8
    args_train.seed = 42
    if args.adv_surr_method == "nowm_wm":
        args_train.train_data_path_class0 = os.path.join('results', exp_id, "nowm")
        args_train.train_data_path_class1 = os.path.join('results', exp_id, "wm")
    elif args.adv_surr_method == "real_wm":
        args_train.train_data_path_class0 = os.path.join('coco', 'val2017')
        args_train.train_data_path_class1 = os.path.join('results', exp_id, "wm")
    args_train.train_size = None
    args_train.surrogate_model = "ResNet18"
    args_train.model_save_path = os.path.join('results', exp_id)
    args_train.model_save_name = os.path.join(f'adv_cls_{args.method}_{args.adv_surr_method}')
    args_train.learning_rate = 1e-3
    args_train.num_epochs = 10
    args_train.batch_size = 128
    args_train.do_eval = True
    args_train.normalize = True
    args_train.device = device

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

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_{args.run_name}'
    if args.dataset_id == 'coco':
        exp_id += '_coco'
    # create a log file
    args.log_file = open(f'{args.log_dir}/{exp_id}.txt', 'w', buffering=1)  # Use line buffering
    
    main(args)