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

from datasets import load_dataset

from utils import seed_everything, image_distortion, print2file

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
import treeringwatermark.open_clip as open_clip
from treeringwatermark.optim_utils import measure_similarity
from treeringwatermark.pytorch_fid.fid_score import calculate_fid_given_paths

sys.path.append(os.path.join(os.path.dirname(__file__), 'waves', 'adversarial'))
from waves.adversarial.embedding import adv_emb_attack_custom 
from waves.adversarial.surrogate import adv_surrogate_model_attack

CALC_FID = True
CALC_CLIP = True

def main(args):

    if "is/sg2" in os.getcwd():
        HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    else:
        HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

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


    if args.overwrite_attacked_imgs: 
        # load raw, then attack and save the images
        print2file(args.log_file, f'\nLoading raw images from results/{exp_id}')
        print2file(args.log_file, f'\nOverwriting attacked images in results/{exp_id}')
    else:
        # load attacked images, only calculate the scores
        print2file(args.log_file, f'\nLoading attacked images from results/{exp_id}')
    


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

    if args.calc_CLIP:
        # load the reference CLIP model
        print2file(args.log_file, f'\nLoading reference CLIP model {args.reference_model}')
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model, 
            pretrained=args.reference_model_pretrain, 
            device=device,
            cache_dir=HF_CACHE_DIR)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)
    
    
    distortions = ['r_degree', 'jpeg_ratio', 'crop_scale', 'crop_ratio', 'gaussian_blur_r', 'gaussian_std', 'brightness_factor', ]
    adversarial_embeds = ['adv_embed_resnet18', 'adv_embed_clip', 'adv_embed_klvae8', 'adv_embed_sdxlvae', 'adv_embed_klvae16']
    adversarial_surr = ['adv_surr_resnet18', 'adv_surr_clip', 'adv_surr_klvae8', 'adv_surr_sdxlvae', 'adv_surr_klvae16']
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
        print2file(args.log_file, f'\nAttacktype "{attack_type}" with Attack "{attack_name}": {attack_vals[strength]}' if attack_name is not None else '\n\nNo attack')
        
        # saved new attacked images in or load pre-attacked images from the results/attack_name/attack_val folder
        path_attack_wm = f'results/{exp_id}/wm/{attack_name}/{attack_vals[strength]}'
        path_attack_nowm = f'results/{exp_id}/nowm/{attack_name}/{attack_vals[strength]}'
        

        clip_scores_wm = []
        clip_scores_nowm = []

        seed_everything(strength)

        # load raw imgs and genrete attacked images
        if args.overwrite_attacked_imgs:
            print2file(args.log_file, f'\nAttacking images...')
            
            if attack_type == 'adversarial_surr':
                # attack the wm images to be classified as nowm (label 0), and the nowm images to be classified as wm (label 1)
                print2file(args.log_file, f'\nAttacking with adversarial surrogate model...')

                # class 0 and class 1 paths depend on the adv_surr_method
                if args.adv_surr_method == "nowm_wm":
                    path_class0 = f'results/{exp_id}/nowm'
                    path_class1 = f'results/{exp_id}/wm'
                elif args.adv_surr_method == "real_wm":
                    path_class0 = f'coco/val2017'
                    path_class1 = f'results/{exp_id}/wm'
                elif args.adv_surr_method == "wm1_wm2":
                    # TODO 
                    pass

                
                if args.adv_surr_model_path is None:
                    model_path = os.path.join('results', exp_id, f'adv_cls_{args.method}_{args.adv_surr_method}.pth')
                else:
                    model_path = args.adv_surr_model_path

                print2file(args.log_file, f'\nLoading images from {path_class0} and {path_class1}')

                # for adv_surr, the path names are slightly different
                path_attack_wm = f'results/{exp_id}/wm/{attack_name}_{args.adv_surr_method}/{attack_vals[strength]}'
                path_attack_nowm = f'results/{exp_id}/nowm/{attack_name}_{args.adv_surr_method}/{attack_vals[strength]}'
                os.makedirs(path_attack_wm, exist_ok=True)
                os.makedirs(path_attack_nowm, exist_ok=True)
                for f in os.listdir(path_attack_wm):
                    os.remove(os.path.join(path_attack_wm, f))
                for f in os.listdir(path_attack_nowm):
                    os.remove(os.path.join(path_attack_nowm, f))
                print2file(args.log_file, f'\nOverwriting attacked images in {path_attack_wm} and {path_attack_nowm}')
                
                # attack class 0 to be classified as class 1
                adv_surrogate_model_attack(
                    data_path=path_class0,
                    model_path=model_path,
                    strength=attack_vals[strength],
                    output_path=path_attack_nowm, # attacked nowm images, so save attack here
                    target_label=1,
                    batch_size=(32 if args.num_images > 32 else args.num_images),
                    warmup=True,
                    device=device,
                )
                # attack class 1 to be classified as class 0
                adv_surrogate_model_attack(
                    data_path=path_class1,
                    model_path=model_path,
                    strength=attack_vals[strength],
                    output_path=path_attack_wm, # attacked wm images, so save attack here
                    target_label=0,
                    batch_size=(32 if args.num_images > 32 else args.num_images),
                    warmup=True,
                    device=device,
                )

            else:
                os.makedirs(path_attack_wm, exist_ok=True)
                os.makedirs(path_attack_nowm, exist_ok=True)
                for f in os.listdir(path_attack_wm):
                    os.remove(os.path.join(path_attack_wm, f))
                for f in os.listdir(path_attack_nowm):
                    os.remove(os.path.join(path_attack_nowm, f))
                print2file(args.log_file, f'\nOverwriting attacked images in {path_attack_wm} and {path_attack_nowm}')
                
                # only loop per-image for non-surrogate attacks
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
                    if args.calc_CLIP:
                        sims = measure_similarity([img_nowm_attacked, img_wm_attacked], prompts[i], ref_model, ref_clip_preprocess, ref_tokenizer, device)
                        clip_scores_nowm.append(sims[0].item())
                        clip_scores_wm.append(sims[1].item())
            print2file(args.log_file, '\nFinished attacking images')
        # load pre-attacked images, calculate the scores
        else:
            # check that the attacked images exist, if not, raise an error
            if not os.path.exists(path_attack_wm) or not os.path.exists(path_attack_nowm):
                raise FileNotFoundError(f'Attacked images not found in {path_attack_wm} or {path_attack_nowm}')
            print2file(args.log_file, f'\nLoading attacked images from {path_attack_wm} and {path_attack_nowm}')

            if args.calc_CLIP:
                # also loop per-image, to calculate the clip scores
                for i in tqdm(range(args.num_images)):
                    img_wm = Image.open(f'{path_attack_wm}/{i}.png')
                    img_nowm = Image.open(f'{path_attack_nowm}/{i}.png')
                    sims = measure_similarity([img_nowm, img_wm], prompts[i], ref_model, ref_clip_preprocess, ref_tokenizer, device)
                    clip_scores_nowm.append(sims[0].item())
                    clip_scores_wm.append(sims[1].item())

        fid_score_wm = None
        fid_score_nowm = None
        clip_score_wm = None
        clip_score_nowm = None
        

        if args.calc_FID:
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

        if args.calc_CLIP:
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