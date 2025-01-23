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

from datasets import load_dataset

from utils import seed_everything, image_distortion, print2file

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
import treeringwatermark.open_clip as open_clip
from treeringwatermark.optim_utils import measure_similarity
from treeringwatermark.pytorch_fid.fid_score import calculate_fid_given_paths

sys.path.append(os.path.join(os.path.dirname(__file__), 'waves', 'adversarial'))
from waves.adversarial.embedding import adv_emb_attack_custom 

CALC_FID = True
CALC_CLIP = True

def main(args):

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

    print2file(args.log_file, '\nLoading imgs from', f'results/{exp_id}')


    # load the prompts
    seed_everything(42)
    if args.dataset_id == 'coco':
        with open('coco/captions_val2017.json') as f:
            all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
    else:
        all_prompts = [sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']]
    # sample the prompts
    prompts = random.sample(all_prompts, args.num_images)
    print2file(args.log_file,  '\nPrompts:')
    for i, prompt in enumerate(prompts):
        print2file(args.log_file, f'{i}: {prompt}')


    # load the reference CLIP model
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
        args.reference_model, 
        pretrained=args.reference_model_pretrain, 
        device=device,
        cache_dir=HF_CACHE_DIR)
    ref_tokenizer = open_clip.get_tokenizer(args.reference_model)
    
    
    distortions = ['r_degree', 'jpeg_ratio', 'crop_scale', 'crop_ratio', 'gaussian_blur_r', 'gaussian_std', 'brightness_factor', ]
    adversarial_embeds = ['adv_embed_resnet18', 'adv_embed_clip', 'adv_embed_klvae8', 'adv_embed_sdxlvae', 'adv_embed_klvae16']
    attack_vals = [None]
    attack_name = None
    attack_type = None

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
        else:
            attack_type = None

    # start the attacks
    print2file(args.log_file, '\n\nStarting to attack...\n')
    for strength in range(len(attack_vals)):
       
        # saved attacked images in the results/attack_name/attack_val folder
        path_attack_wm = f'results/{exp_id}/wm/{attack_name}/{attack_vals[strength]}'
        path_attack_nowm = f'results/{exp_id}/nowm/{attack_name}/{attack_vals[strength]}'
        os.makedirs(path_attack_wm, exist_ok=True)
        os.makedirs(path_attack_nowm, exist_ok=True)

        clip_scores_wm = []
        clip_scores_nowm = []

        print2file(args.log_file, f'\nAttack {attack_name}: {attack_vals[strength]}' if attack_name is not None else '\n\nNo attack')
        for i in tqdm(range(args.num_images)):

            seed_everything(i)

            img_wm = Image.open(f'results/{exp_id}/wm/{i}.png')
            img_nowm = Image.open(f'results/{exp_id}/nowm/{i}.png')

            if attack_type == 'distortion' or attack_type is None:
                img_wm_attacked, img_nowm_attacked = image_distortion(img_wm, img_nowm, i, args, strength, i==0)
            elif attack_type == 'adversarial_embed':
                img_wm_attacked = adv_emb_attack_custom(img_wm, attack_name, attack_vals[strength], device)
                img_nowm_attacked = adv_emb_attack_custom(img_nowm, attack_name, attack_vals[strength], device)
            img_wm_attacked.save(os.path.join(path_attack_wm, f'{i}.png'))
            img_nowm_attacked.save(os.path.join(path_attack_nowm, f'{i}.png'))

            # clip scores
            sims = measure_similarity([img_nowm_attacked, img_wm_attacked], prompts[i], ref_model, ref_clip_preprocess, ref_tokenizer, device)
            clip_scores_nowm.append(sims[0].item())
            clip_scores_wm.append(sims[1].item())
        print2file(args.log_file, '\nFinished attacking images')

        fid_score_wm = None
        fid_score_nowm = None
        clip_score_wm = None
        clip_score_nowm = None
        

        if CALC_FID:
            # measure the FID between original and attacked images, both with and without watermark
            fid_score_wm = calculate_fid_given_paths([path_attack_wm, '/is/sg2/mkaut/ma-thesis/coco/val2017'], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048,
                                                    max_samples=args.num_images)
            fid_score_nowm = calculate_fid_given_paths([path_attack_nowm, '/is/sg2/mkaut/ma-thesis/coco/val2017'], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048,
                                                    max_samples=args.num_images)
            print2file(args.log_file, '''
            ______ _____ _____  
            |  ____|_   _|  __ \ 
            | |__    | | | |  | |
            |  __|   | | | |  | |
            | |     _| |_| |__| |
            |_|    |_____|_____/ 
                        ''')
            print2file(args.log_file, f'\nFID score with watermark for attack {attack_name}={attack_vals[strength]} for {args.num_images} samples: \n\n\t{fid_score_wm}')
            print2file(args.log_file, f'\nFID score without watermark for attack {attack_name}={attack_vals[strength]} for {args.num_images} samples: \n\n\t{fid_score_nowm}')

        if CALC_CLIP:
            # calculate CLIP score between the generated images with and without watermark to the prompt with the reference model
            clip_score_wm = np.mean(clip_scores_wm)
            clip_score_nowm = np.mean(clip_scores_nowm)
            print2file(args.log_file, '''
              _____ _      _____ _____  
             / ____| |    |_   _|  __ \ 
            | |    | |      | | | |__) |
            | |    | |      | | |  ___/ 
            | |____| |____ _| |_| |     
             \_____|______|_____|_|     
                                    ''')
            print2file(args.log_file, f'\nCLIP score with watermark: \n\n\t{clip_score_wm}')
            print2file(args.log_file, f'\nCLIP score without watermark: \n\n\t{clip_score_nowm}')

        results = []
        results.append({
            'attack': attack_name,
            'strength': attack_vals[strength],
            'fid_score_wm': fid_score_wm if args.dataset_id == 'coco' else None,
            'fid_score_nowm': fid_score_nowm if args.dataset_id == 'coco' else None,
            'clip_score_wm': clip_score_wm,
            'clip_score_nowm': clip_score_nowm

        })
        
        # save results
        with open(f'{args.log_dir}/results_{attack_name}_{attack_vals[strength]}.pkl', 'wb') as f:
            pickle.dump(results, f)

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
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.log_dir = f'./experiments/{date}_attack_{args.run_name}'
    os.makedirs(args.log_dir)

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_{args.run_name}'
    if args.dataset_id == 'coco':
        exp_id += '_coco'
    # create a log file
    args.log_file = open(f'{args.log_dir}/{exp_id}.txt', 'w', buffering=1)  # Use line buffering
    
    main(args)