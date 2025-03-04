import os
if 'is/sg2' in os.getcwd():
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import sys
import json
import torch
import random
import datetime
import argparse
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset

from utils import seed_everything, print2file, get_dirs, create_and_save_decode_confs

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
    
    if 'is/sg2' in os.getcwd():
        HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    else:
        HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

    # paramters that could theoretically be moved to the config file, but are always the same
    NUM_IMAGES_PER_PROMPT = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # check if the latent_channels_wm is correct for the model
    if args.model_id == 'sd' and args.latent_channels_wm != 4:
        print('Warning: For the sd model, the latent_channels_wm should be 4\nSetting it to 4')
        args.latent_channels_wm = 4

    log_dir, args.data_dir = get_dirs(args, 'encoded_imgs')
    args.log_dir = os.path.join(log_dir, args.date)
    os.makedirs(args.log_dir)
    
    logfile_name = f'{args.run_name}'
    args.log_file = open(os.path.join(args.log_dir, f'{logfile_name}.txt'), 'w', buffering=1)  # Use line buffering

    create_and_save_decode_confs(args)

    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting Encode...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')

    # set seed for internal WM viz and prompt loading
    seed_everything(0) # should be 0 cause it gets set to 0 later in the loop

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

   # load the prompts
    if args.dataset_id == 'coco':
        with open('coco/captions_val2017.json') as f:
            all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
    elif args.dataset_id == 'sdprompts':
        all_prompts = [sample['Prompt'] for sample in load_dataset('Gustavosta/Stable-Diffusion-Prompts')['test']]
    elif args.dataset_id == 'mjprompts':
        all_prompts = [sample['caption'] for sample in load_dataset('bghira/mj-v52-redux')['Collection_20']]
    else:
        print2file(args.log_file, 'Invalid dataset_id')
        return
    # sample the prompts
    prompts = random.sample(all_prompts, args.num_images)
    print2file(args.log_file,  '\nPrompts:')
    for i, prompt in enumerate(prompts):
        print2file(args.log_file, f'{i}: {prompt}')

    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        os.makedirs(os.path.join(args.data_dir, 'wm'))
        os.makedirs(os.path.join(args.data_dir, 'nowm'))
    print2file(args.log_file, f'\nSaving original images to {args.data_dir}')
    # also save the config file and the prompts
    with open(os.path.join(args.data_dir, 'config.json'), 'w') as f:
        temp = vars(args).copy()
        temp.pop('log_file')
        json.dump(temp, f, indent=4)
    with open(os.path.join(args.data_dir, 'prompts.txt'), 'w') as f:
        for i, prompt in enumerate(prompts):
            f.write(f'{i}: {prompt}\n')
    

    # generate images
    print2file(args.log_file, '\n\nStarting to generate images...\n')
    for i in tqdm(range(args.num_images)):
    
        seed_everything(i)

        current_prompt = prompts[i]
            
        orig_image_wm = encoder.generate_img(current_prompt, 
                                            do_wm=True,  
                                            seed=i,
                                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT, 
                                            pattern_index=args.pattern_index if args.method == 'rid' else None)
        orig_image_nowm = encoder.generate_img(current_prompt, 
                                            do_wm=False, 
                                            seed=i,
                                            num_images_per_prompt=NUM_IMAGES_PER_PROMPT, 
                                            pattern_index=args.pattern_index if args.method == 'rid' else None)
        
        orig_image_wm.save(os.path.join(args.data_dir, 'wm', f'{i}.png'))
        orig_image_nowm.save(os.path.join(args.data_dir, 'nowm', f'{i}.png'))


    print2file(args.log_file, '\n' + '#'*100 + '\n')

    print2file(args.log_file, '\nFinished Encoding\n')



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

