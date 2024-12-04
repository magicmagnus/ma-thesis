import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import argparse
import torch
import pickle
import json

from tqdm import tqdm
import random
import numpy as np
from datasets import load_dataset
import datetime

# Add the source repositories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prc'))
from prc.export import PRCWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussianshading'))
from gaussianshading.export import GSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
from treeringwatermark.export import TRWatermark
import treeringwatermark.open_clip as open_clip
from treeringwatermark.optim_utils import measure_similarity

def print2file(logfile, *args):
    print(*args)
    print(file=logfile, *args)

def main(args):
    
    # depending on the server, the cache dir might be different
    #HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub' 
    HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting encode...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')


    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}'

    # first genrate all the keys per method
    if args.method == 'prc':
        prc_watermark = PRCWatermark(model_id=args.model_id, 
                                     inf_steps=args.inf_steps, 
                                     fpr=args.fpr, 
                                     prc_t=args.prc_t, 
                                     num_images=args.num_images, 
                                     guidance_scale=args.guidance_scale,
                                     hf_cache_dir=HF_CACHE_DIR)
        
    elif args.method == 'gs':
        gs_watermark = GSWatermark(model_id=args.model_id,
                                      inf_steps=args.inf_steps,
                                      fpr=args.fpr,
                                      num_images=args.num_images,
                                      chacha=args.gs_chacha,
                                      ch_factor=args.gs_ch_factor,
                                      hw_factor=args.gs_hw_factor,
                                      user_number=args.gs_user_number,
                                      guidance_scale=args.guidance_scale,
                                      hf_cache_dir=HF_CACHE_DIR)
    elif args.method == 'tr':
        tr_watermark = TRWatermark(model_id=args.model_id,
                                      inf_steps=args.inf_steps,
                                      fpr=args.fpr,
                                      num_images=args.num_images,
                                      guidance_scale=args.guidance_scale,
                                      w_seed=args.w_seed,
                                      w_channel=args.w_channel,
                                      w_pattern=args.w_pattern,
                                      w_mask_shape=args.w_mask_shape,
                                      w_radius=args.w_radius,
                                      w_measurement=args.w_measurement,
                                      w_injection=args.w_injection,
                                      w_pattern_const=args.w_pattern_const,
                                      hf_cache_dir=HF_CACHE_DIR)
        
    else:
        print2file(args.log_file, 'Invalid method')
        return
    
    # load dataset
    if args.dataset_id == 'coco':
        save_folder = f'./results/{exp_id}_coco'
    else:
        save_folder = f'./results/{exp_id}'

    # create save folders 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(f'{save_folder}/wm')
        os.makedirs(f'{save_folder}/nowm')
    print2file(args.log_file, f'\nSaving original images to {save_folder}')

    # load the prompts
    random.seed(42)
    if args.dataset_id == 'coco':
        with open('coco/captions_val2017.json') as f:
            all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
    else:
        all_prompts = [sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']]

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

    def seed_everything(seed, workers=False):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
        return seed
    
    clip_scores_wm = []
    clip_scores_nowm = []

    # generate images
    print2file(args.log_file, '\n\nStarting to generate images...\n')
    for i in tqdm(range(args.num_images)):
    
        wm_img = None
        nowm_img = None

        seed_everything(1)

        current_prompt = prompts[i]
        for nowm in [0, 1]: # 0 = with watermark, 1 = without watermark
            
            if args.method == 'prc':
                orig_image = prc_watermark.generate_img(current_prompt, nowm=nowm, num_images_per_prompt=args.num_images_per_prompt)
            elif args.method == 'gs':
                orig_image = gs_watermark.generate_img(current_prompt, nowm=nowm, num_images_per_prompt=args.num_images_per_prompt)
            elif args.method == 'tr':
                orig_image = tr_watermark.generate_img(current_prompt, nowm=nowm, num_images_per_prompt=args.num_images_per_prompt)
        
            if nowm:
                orig_image.save(f'{save_folder}/nowm/{i}.png')
                nowm_img = orig_image
            else:
                orig_image.save(f'{save_folder}/wm/{i}.png')
                wm_img = orig_image
        
        # calculate CLIP score between the generated images with and without watermark to the prompt with the reference model
        sims = measure_similarity([nowm_img, wm_img], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
        clip_scores_nowm.append(sims[0].item())
        clip_scores_wm.append(sims[1].item())
    
    print2file(args.log_file, '''
            _____ _      _____ _____             ______ _____ _____  
           / ____| |    |_   _|  __ \    ___    |  ____|_   _|  __ \ 
          | |    | |      | | | |__) |  ( _ )   | |__    | | | |  | |
          | |    | |      | | |  ___/   / _ \/\ |  __|   | | | |  | |
          | |____| |____ _| |_| |      | (_>  < | |     _| |_| |__| |
           \_____|______|_____|_|       \___/\/ |_|    |_____|_____/ 
                                                           ''')

    print2file(args.log_file, f'\nCLIP score with watermark: ')
    print2file(args.log_file, f'\n\t{np.mean(clip_scores_wm)}')
    print2file(args.log_file, f'\nCLIP score without watermark: ')
    print2file(args.log_file, f'\n\t{np.mean(clip_scores_nowm)}')

    print2file(args.log_file, '\n' + '#'*100 + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ################### general #######################
    # parser.add_argument('--num_images', type=int, default=10)
    # parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
    # parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
    # parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') # coco 
    # parser.add_argument('--inf_steps', type=int, default=50)
    # parser.add_argument('--fpr', type=float, default=0.00001)
    # parser.add_argument('--guidance_scale', type=float, default=7.5) # in tr, gs = 7.5, prc = 3.0
    # parser.add_argument('--num_images_per_prompt', type=int, default=1)
    # parser.add_argument('--run_name', type=str, default='test')
    

    # ################### per method ####################
    # args = parser.parse_args()
    # if args.method == 'prc':
    #     parser.add_argument('--prc_t', type=int, default=3)
    # if args.method == 'gs':
    #     parser.add_argument('--gs_chacha', type=bool, default=True)
    #     parser.add_argument('--gs_ch_factor', type=int, default=1)
    #     parser.add_argument('--gs_hw_factor', type=int, default=8)
    #     parser.add_argument('--gs_user_number', type=int, default=1000000)
    # if args.method == 'tr':
    #     parser.add_argument('--w_seed', type=int, default=0)
    #     parser.add_argument('--w_channel', type=int, default=0)
    #     parser.add_argument('--w_pattern', type=str, default='rand')
    #     parser.add_argument('--w_mask_shape', type=str, default='circle')
    #     parser.add_argument('--w_radius', type=int, default=10)
    #     parser.add_argument('--w_measurement', type=str, default='l1_complex')
    #     parser.add_argument('--w_injection', type=str, default='complex')
    #     parser.add_argument('--w_pattern_const', type=int, default=0)
    

    # ################### for testing ################### 
    # # clip related
    # parser.add_argument('--reference_model', default='ViT-g-14')
    # parser.add_argument('--reference_model_pretrain', default='laion2b_s12b_b42k')
    
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
    args.log_dir = f'./experiments/encode_{date}_{args.run_name}'
    os.makedirs(args.log_dir)

    

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_{args.run_name}'
    if args.dataset_id == 'coco':
        exp_id += '_coco'
    # create a log file
    args.log_file = open(f'{args.log_dir}/{exp_id}.txt', 'w', buffering=1)  # Use line buffering
    
    main(args)

