import os
import sys
import argparse
import torch
import pickle
import json
from tqdm import tqdm
import random
import numpy as np
from datasets import load_dataset



# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prc'))
from prc.export import PRCWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussianshading'))
from gaussianshading.export import GSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
from treeringwatermark.export import TRWatermark



def main(args):

    print('\n\n\nStarting encode...')
    print('\nArgs:\n')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    # hf_cache_dir = '/home/mkaut/.cache/huggingface/hub'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # codeword_length = 4 * 64 * 64  # the length of a PRC codeword

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}'

    # first genrate all the keys per method
    if args.method == 'prc':
        prc_watermark = PRCWatermark(model_id=args.model_id, 
                                     inf_steps=args.inf_steps, 
                                     fpr=args.fpr, 
                                     prc_t=args.prc_t, 
                                    #  nowm=args.nowm, 
                                     num_images=args.num_images, 
                                     guidance_scale=args.guidance_scale)
        
    elif args.method == 'gs':
        gs_watermark = GSWatermark(model_id=args.model_id,
                                      inf_steps=args.inf_steps,
                                      fpr=args.fpr,
                                    #   nowm=args.nowm,
                                      num_images=args.num_images,
                                      chacha=args.gs_chacha,
                                      ch_factor=args.gs_ch_factor,
                                      hw_factor=args.gs_hw_factor,
                                      user_number=args.gs_user_number,
                                      guidance_scale=args.guidance_scale)
    elif args.method == 'tr':
        tr_watermark = TRWatermark(model_id=args.model_id,
                                      inf_steps=args.inf_steps,
                                      fpr=args.fpr,
                                    #   nowm=args.nowm,
                                      num_images=args.num_images,
                                      guidance_scale=args.guidance_scale,
                                      w_seed=args.w_seed,
                                      w_channel=args.w_channel,
                                      w_pattern=args.w_pattern,
                                      w_mask_shape=args.w_mask_shape,
                                      w_radius=args.w_radius,
                                      w_measurement=args.w_measurement,
                                      w_injection=args.w_injection,
                                      w_pattern_const=args.w_pattern_const)
        
    else:
        print('Invalid method')
        return
    
    # load dataset
    if args.dataset_id == 'coco':
        save_folder = f'./results/{exp_id}_coco'
    else:
        save_folder = f'./results/{exp_id}'


    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(f'{save_folder}/wm')
        os.makedirs(f'{save_folder}/nowm')
    print(f'Saving original images to {save_folder}')

    random.seed(42)
    if args.dataset_id == 'coco':
        with open('coco/captions_val2017.json') as f:
            all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
    else:
        all_prompts = [sample['Prompt'] for sample in load_dataset(args.dataset_id)['test']]

    prompts = random.sample(all_prompts, args.num_images)
    print( 'Prompts:')
    for i, prompt in enumerate(prompts):
        print(f'{i}: {prompt}')

    

    def seed_everything(seed, workers=False):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
        return seed
    
    # generate images
    print('\n\nStarting to generate images...\n')
    for nowm in [0, 1]: # 0 = with watermark, 1 = without watermark
        
        for i in tqdm(range(args.num_images)):
            seed_everything(1)

            current_prompt = prompts[i]
            
                
            if args.method == 'prc':
                orig_image = prc_watermark.generate_img(current_prompt, nowm=nowm, num_images_per_prompt=args.num_images_per_prompt)
            elif args.method == 'gs':
                orig_image = gs_watermark.generate_img(current_prompt, nowm=nowm, num_images_per_prompt=args.num_images_per_prompt)
            elif args.method == 'tr':
                orig_image = tr_watermark.generate_img(current_prompt, nowm=nowm, num_images_per_prompt=args.num_images_per_prompt)
                    
            
            if nowm:
                orig_image.save(f'{save_folder}/nowm/{i}.png')
            else:
                orig_image.save(f'{save_folder}/wm/{i}.png')

    print('Done')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')

    # general
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
    parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') # coco 
    parser.add_argument('--inf_steps', type=int, default=50)
    # parser.add_argument('--nowm', type=int, default=0) # 0 = with watermark, 1 = without watermark
    parser.add_argument('--fpr', type=float, default=0.00001)
    parser.add_argument('--guidance_scale', type=float, default=3.0)
    parser.add_argument('--num_images_per_prompt', type=int, default=1)
    

    # prc related
    parser.add_argument('--prc_t', type=int, default=3)

    # gs related
    parser.add_argument('--gs_chacha', type=bool, default=True)
    parser.add_argument('--gs_ch_factor', type=int, default=1)
    parser.add_argument('--gs_hw_factor', type=int, default=8)
    parser.add_argument('--gs_user_number', type=int, default=1000000)

    # tr related
    parser.add_argument('--w_seed', type=int, default=0)
    parser.add_argument('--w_channel', type=int, default=0)
    parser.add_argument('--w_pattern', type=str, default='rand')
    parser.add_argument('--w_mask_shape', type=str, default='circle')
    parser.add_argument('--w_radius', type=int, default=10)
    parser.add_argument('--w_measurement', type=str, default='l1_complex')
    parser.add_argument('--w_injection', type=str, default='complex')
    parser.add_argument('--w_pattern_const', type=int, default=0)



   
    


    args = parser.parse_args()

   
    
    main(args)