import os
if "is/sg2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import json
import torch
import random
import datetime
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset

from utils import seed_everything, print2file

# Add the source repositories to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prc'))
from prc.export import PRCWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussianshading'))
from gaussianshading.export import GSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'rid'))
from ringid.export import RingIDWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
from treeringwatermark.export import TRWatermark
import treeringwatermark.open_clip as open_clip
from treeringwatermark.optim_utils import measure_similarity
from treeringwatermark.pytorch_fid.fid_score import calculate_fid_given_paths



def main(args):
    
    if "is/sg2" in os.getcwd():
        HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    else:
        HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting Encode...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}'

    # first genrate all the keys per method
    if args.method == 'prc':
        prc_watermark = PRCWatermark(args, hf_cache_dir=HF_CACHE_DIR)
        encoder = prc_watermark
    elif args.method == 'gs':
        gs_watermark = GSWatermark(args, hf_cache_dir=HF_CACHE_DIR)
        encoder = gs_watermark
    elif args.method == 'tr':
        tr_watermark = TRWatermark(args, hf_cache_dir=HF_CACHE_DIR)
        encoder = tr_watermark
    elif args.method == 'rid':
        rid_watermark = RingIDWatermark(args, hf_cache_dir=HF_CACHE_DIR)
        encoder = rid_watermark
    else:
        print2file(args.log_file, 'Invalid method')
        return

    # load dataset
    if args.dataset_id == 'coco':
        exp_id = exp_id + '_coco'
    
    save_folder = f'./results/{exp_id}'

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
    print2file(args.log_file, f'\nLoading reference CLIP model {args.reference_model}')
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
        args.reference_model, 
        pretrained=args.reference_model_pretrain, 
        device=device,
        cache_dir=HF_CACHE_DIR)
    ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    
    # create the save folders
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(f'{save_folder}/wm')
        os.makedirs(f'{save_folder}/nowm')
    print2file(args.log_file, f'\nSaving original images to {save_folder}')
    # also save the config file
    with open(f'{save_folder}/config.json', 'w') as f:
        temp = vars(args).copy()
        temp.pop('log_file')
        json.dump(temp, f, indent=4)
    
    clip_scores_wm = []
    clip_scores_nowm = []

    # generate images
    print2file(args.log_file, '\n\nStarting to generate images...\n')
    for i in tqdm(range(args.num_images)):
    
        seed_everything(i)

        current_prompt = prompts[i]
            
        orig_image_wm = encoder.generate_img(current_prompt, 
                                            nowm=0,  
                                            num_images_per_prompt=args.num_images_per_prompt, 
                                            pattern_index=args.pattern_index if args.method == 'rid' else None)
        orig_image_nowm = encoder.generate_img(current_prompt, 
                                            nowm=1, 
                                            num_images_per_prompt=args.num_images_per_prompt, 
                                            pattern_index=args.pattern_index if args.method == 'rid' else None)
        
        orig_image_wm.save(f'{save_folder}/wm/{i}.png')
        orig_image_nowm.save(f'{save_folder}/nowm/{i}.png')

        # calculate CLIP score between the generated images with and without watermark to the prompt with the reference model
        sims = measure_similarity([orig_image_nowm, orig_image_wm], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
        clip_scores_nowm.append(sims[0].item())
        clip_scores_wm.append(sims[1].item())
    
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



    # calculate FID score between the generated images with and without watermark
    fid_score_wm = calculate_fid_given_paths([f'{save_folder}/wm', '/is/sg2/mkaut/ma-thesis/coco/val2017'], 
                                                batch_size=50, 
                                                device=device, 
                                                dims=2048,
                                                max_samples=args.num_images)
    fid_score_nowm = calculate_fid_given_paths([f'{save_folder}/nowm', '/is/sg2/mkaut/ma-thesis/coco/val2017'],
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
    print2file(args.log_file, f'\nFID score with watermark for {args.num_images} samples: \n\n\t{fid_score_wm}')
    print2file(args.log_file, f'\nFID score without watermarkfor {args.num_images} samples: \n\n\t{fid_score_nowm}')
    

    print2file(args.log_file, '\n' + '#'*100 + '\n')



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
    args.log_dir = f'./experiments/{date}_encode_{args.run_name}'
    os.makedirs(args.log_dir)

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_{args.run_name}'
    if args.dataset_id == 'coco':
        exp_id += '_coco'
    # create a log file
    args.log_file = open(f'{args.log_dir}/{exp_id}.txt', 'w', buffering=1)  # Use line buffering
    
    main(args)

