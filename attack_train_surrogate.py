import os
if "is/sg2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
import matplotlib.pyplot as plt

from utils import seed_everything, image_distortion, print2file, get_dirs

sys.path.append(os.path.join(os.path.dirname(__file__), 'waves', 'adversarial'))
from waves.adversarial.train import train_surrogate_classifier


def main(args):

    # Set random seed
    seed_everything(43)

    if "is/sg2" in os.getcwd():
        HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    else:
        HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    log_dir, args.data_dir = get_dirs(args, "attack_imgs")# , extra=args.run_name)
    args.log_dir = os.path.join(log_dir, f'{args.date}_{args.run_name}')
    os.makedirs(args.log_dir)
    
    logfile_name = f'{args.run_name}'
    args.log_file = open(os.path.join(args.log_dir, f'{logfile_name}.txt'), 'w', buffering=1)  # Use line buffering

    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting Training...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')

    args_train = argparse.Namespace()

    args_train.image_size = 512
    args_train.num_classes = 2
    args_train.train_ratio = 0.8
    args_train.seed = 42
    if args.adv_surr_method == "nowm_wm":
        args_train.train_data_path_class0 = os.path.join(args.data_dir, "nowm")
        args_train.train_data_path_class1 = os.path.join(args.data_dir, "wm")
    elif args.adv_surr_method == "real_wm":
        args_train.train_data_path_class0 = os.path.join('coco', 'val2017')
        args_train.train_data_path_class1 = os.path.join(args.data_dir, "wm")
    args_train.train_size = None
    args_train.surrogate_model = args.adv_surr_model
    args_train.model_save_path = args.data_dir
    args_train.model_save_name = f'adv_cls_{args.method}_{args.adv_surr_method}_{args.adv_surr_model}'
    args_train.learning_rate = 1e-3
    args_train.num_epochs = 40
    args_train.batch_size = 64
    args_train.do_eval = True
    args_train.normalize = True
    args_train.device = device
    args_train.log_file = args.log_file
    args_train.log_dir = args.log_dir

    train_accs, val_accs, losses, lrs = train_surrogate_classifier(args_train)

    # Save the training results as plots
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.plot(train_accs, label='Train Accuracy', color='tab:blue')
    ax1.plot(val_accs, label='Validation Accuracy', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss (log scale)', color='tab:red')
    ax2.plot(losses, label='Loss', color='tab:red')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')

    
    plt.title('Training Results')
    plt.grid()
    plt.savefig(os.path.join(args.log_dir, 'training_results.png'))
    plt.close()

    # plt the lrs 
    plt.plot(lrs)
    plt.title('Learning Rate')
    plt.grid()
    plt.savefig(os.path.join(args.log_dir, 'learning_rate.png'))
    plt.close()




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
    args.date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    main(args)