import os
if 'is/sg2' in os.getcwd():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

from datasets import load_dataset

from utils import seed_everything, image_distortion, print2file, get_dirs

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
import treeringwatermark.open_clip as open_clip
from treeringwatermark.optim_utils import measure_similarity
from treeringwatermark.pytorch_fid.fid_score import calculate_fid_given_paths

sys.path.append(os.path.join(os.path.dirname(__file__), 'waves', 'adversarial'))
from waves.adversarial.embedding import adv_emb_attack_custom, adv_emb_attack
from waves.adversarial.surrogate import adv_surrogate_model_attack


def main(args):

    if 'is/sg2' in os.getcwd():
        HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    else:
        HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # check if the latent_channels_wm is correct for the model
    if args.model_id == 'sd' and args.latent_channels_wm != 4:
        print('Warning: For the sd model, the latent_channels_wm should be 4\nSetting it to 4')
        args.latent_channels_wm = 4
    
    log_dir, args.data_dir = get_dirs(args, 'attack_imgs')# , extra=args.run_name)
    args.log_dir = os.path.join(log_dir, f'{args.date}_{args.run_name}')
    os.makedirs(args.log_dir)
    
    logfile_name = f'{args.run_name}'
    args.log_file = open(os.path.join(args.log_dir, f'{logfile_name}.txt'), 'w', buffering=1)  # Use line buffering

    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting Attack...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')
   
    
    # load raw, then attack and save the images
    print2file(args.log_file, f'\nLoading raw images from {args.data_dir}')

    # set seed for internal WM viz and prompt loading
    seed_everything(0) # should be 0 cause it gets set to 0 later in the loop

    # load the prompts
    if args.dataset_id == 'coco':
        with open('coco/captions_val2017.json') as f:
            all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
    elif args.dataset_id == 'sdprompts':
        all_prompts = [sample['Prompt'] for sample in load_dataset('Gustavosta/Stable-Diffusion-Prompts')['test']]
    elif args.dataset_id == 'mjprompts':
        all_prompts = [sample['caption'] for sample in load_dataset('bghira/mj-v52-redux')['Collection_10']]
    else:
        print2file(args.log_file, 'Invalid dataset_id')
        return
    # sample the prompts
    prompts = random.sample(all_prompts, args.num_images)
    print2file(args.log_file,  '\nPrompts:')
    for i, prompt in enumerate(prompts):
        print2file(args.log_file, f'{i}: {prompt}')

    
    distortions = ['r_degree', 'jpeg_ratio', 'crop_scale', 'crop_ratio', 'gaussian_blur_r', 'gaussian_std', 'brightness_factor', ]
    adversarial_embeds = ['adv_embed_resnet18', 'adv_embed_clip', 'adv_embed_klvae8', 'adv_embed_sdxlvae', 'adv_embed_klvae16']
    adversarial_surr = ['adv_surr_resnet18', 'adv_surr_resnet50']
    attack_vals = ['no_attack']
    attack_name = 'no_attack'
    attack_type = 'no_attack'

    # determine attack type
    for arg in vars(args):
        if getattr(args, arg) is not None and arg in distortions:
            print2file(args.log_file, f'\nlooping over {arg}: {getattr(args, arg)}')
            attack_vals = getattr(args, arg)
            attack_name = arg
            attack_type = 'distortion'
            break   
        elif getattr(args, arg) is not None and arg in adversarial_embeds:
            print2file(args.log_file, f'\nlooping over {arg}: {getattr(args, arg)}')
            attack_vals = getattr(args, arg)
            attack_name = arg
            attack_type = 'adversarial_embed'
            break
        elif getattr(args, arg) is not None and arg in adversarial_surr:
            print2file(args.log_file, f'\nlooping over {arg}: {getattr(args, arg)}')
            attack_vals = getattr(args, arg)
            attack_name = arg
            attack_type = 'adversarial_surr'
            break
        else:
            continue
            

    # start the attacks
    print2file(args.log_file, '\n\nStarting to attack...\n')
    for strength in range(len(attack_vals)):
        print2file(args.log_file, f'\nAttacktype "{attack_type}" with Attack "{attack_name}": {attack_vals[strength]}' if attack_name != 'no_attack' else '\nNo attack')
        
        # get dirs of the attacked images, per attack type
        if attack_type == 'distortion' or attack_type == 'adversarial_embed':
            path_attacked_wm = os.path.join(args.data_dir, 'wm', attack_name, str(attack_vals[strength]))
            path_attacked_nowm = os.path.join(args.data_dir, 'nowm', attack_name, str(attack_vals[strength]))
        elif attack_type == 'adversarial_surr':
            path_attacked_wm = os.path.join(args.data_dir, 'wm', args.run_name, str(attack_vals[strength]))
            path_attacked_nowm = os.path.join(args.data_dir, 'nowm', args.run_name, str(attack_vals[strength]))
        elif attack_type == 'no_attack':
            # nothing to attack
            print2file(args.log_file, 'No attack\n\n Skipping...')
            return
        else:
            RuntimeError('Invalid attack type')
        
        # clear the attacked image dirs of possible previous attacks
        print2file(args.log_file, f'\nOverwriting attacked images in {path_attacked_wm} \nand {path_attacked_nowm}')
        os.makedirs(path_attacked_wm, exist_ok=True)
        os.makedirs(path_attacked_nowm, exist_ok=True)
        for f in os.listdir(path_attacked_wm):
            os.remove(os.path.join(path_attacked_wm, f))
        for f in os.listdir(path_attacked_nowm):
            os.remove(os.path.join(path_attacked_nowm, f))

        seed_everything(strength)

        print2file(args.log_file, f'\nAttacking images...')
        
        # per img-dir attack
        if attack_type == 'adversarial_surr':
            # attack the wm images to be classified as nowm (label 0), and the nowm images to be classified as wm (label 1)
            print2file(args.log_file, f'\nAttacking with adversarial surrogate model...')

            # class 0 and class 1 paths depend on the adv_surr_method
            if args.adv_surr_method == 'nowm_wm':
                path_class0 = os.path.join(args.data_dir, 'nowm')
                path_class1 = os.path.join(args.data_dir, 'wm')
            elif args.adv_surr_method == 'real_wm':
                path_class0 = os.path.join('coco', 'val2017')
                path_class1 = os.path.join(args.data_dir, 'wm')
            elif args.adv_surr_method == 'wm1_wm2':
                # TODO 
                pass

            print2file(args.log_file, f'\nLoading images from {path_class0} \nand {path_class1}')

            batch_size = 64 if args.num_images > 64 else args.num_images
    
            # attack class 0 to be classified as class 1
            adv_surrogate_model_attack(
                data_path=path_class0,
                model_path=args.adv_surr_model_path,
                strength=attack_vals[strength],
                output_path=path_attacked_nowm, # attacked nowm images, so save attack here
                target_label=1,
                batch_size=batch_size,
                warmup=True,
                device=device,
                args=args,
            )
            # attack class 1 to be classified as class 0
            adv_surrogate_model_attack(
                data_path=path_class1,
                model_path=args.adv_surr_model_path,
                strength=attack_vals[strength],
                output_path=path_attacked_wm, # attacked wm images, so save attack here
                target_label=0,
                batch_size=batch_size,
                warmup=True,
                device=device,
                args=args,
            )
        # also per img-dir attack
        elif attack_type == 'adversarial_embed':
            batch_size = 8 if args.num_images > 8 else args.num_images
            adv_emb_attack(wm_img_path=os.path.join(args.data_dir, 'wm'),
                                                encoder=attack_name,
                                                strength=attack_vals[strength],
                                                output_path=path_attacked_wm,
                                                device=device,
                                                batch_size=batch_size)
            adv_emb_attack(wm_img_path=os.path.join(args.data_dir, 'nowm'),
                                                encoder=attack_name,
                                                    strength=attack_vals[strength],
                                                    output_path=path_attacked_nowm,
                                                    device=device,
                                                    batch_size=batch_size)
        # per image attack, loop over all images
        elif attack_type == 'distortion':
            # only loop per-image for distortion attacks
            for i in tqdm(range(args.num_images)):

                seed_everything(i)
                # raw images
                img_wm = Image.open(os.path.join(args.data_dir, 'wm', f'{i}.png'))
                img_nowm = Image.open(os.path.join(args.data_dir, 'nowm', f'{i}.png'))

                img_wm_attacked, img_nowm_attacked = image_distortion(img_wm, img_nowm, i, args, strength, i==0)
                
                img_wm_attacked.save(os.path.join(path_attacked_wm, f'{i}.png'))
                img_nowm_attacked.save(os.path.join(path_attacked_nowm, f'{i}.png'))  
        else:
            RuntimeError('Invalid attack type')

        print2file(args.log_file, '\nFinished attacking images')


        print2file(args.log_file, '\n\n' + '#'*100 + '\n')
    




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
    args.date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    main(args)