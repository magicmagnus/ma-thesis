import os
import sys
import argparse
import torch
import pickle
import json
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
from datasets import load_dataset
from sklearn import metrics



# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prc'))
from prc.export import PRCWatermark
from prc.src.optim_utils import image_distortion

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussianshading'))
from gaussianshading.export import GSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
from treeringwatermark.export import TRWatermark



def main(args):

    

    print('\n\n\nStarting decode...')
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
                                     # nowm=args.nowm, 
                                     num_images=args.num_images, 
                                     guidance_scale=args.guidance_scale)
        
    elif args.method == 'gs':
        gs_watermark = GSWatermark(model_id=args.model_id,
                                      inf_steps=args.inf_steps,
                                      fpr=args.fpr,
                                      # nowm=args.nowm,
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
                                      # nowm=args.nowm,
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
    
    print('Loading imgs from', f'results/{exp_id}')

    

    def seed_everything(seed, workers=False):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
        return seed
    
    # generate images

    # for prc
    results_detect_wm = []
    results_detect_nowm = []
    results_decode_wm = []
    results_decode_nowm = []

    # for tr 
    
    no_w_metrics = []
    w_metrics = []


    print('\n\nStarting to decode...\n')
    for i in tqdm(range(args.num_images)):
        seed_everything(1)
        img_wm = Image.open(f'results/{exp_id}/wm/{i}.png')
        img_nowm = Image.open(f'results/{exp_id}/nowm/{i}.png')
        # distortion
        seed = 42
        img_wm_auged, img_nowm_auged = image_distortion(img_wm, img_nowm, seed, args)
        
            
        if args.method == 'prc':
            detect_res_nowm, decode_res_nowm = prc_watermark.detect_and_decode_watermark(img_nowm_auged, prompt='')
            detect_res_wm, decode_res_wm = prc_watermark.detect_and_decode_watermark(img_wm_auged, prompt='')
            results_detect_nowm.append(detect_res_nowm)
            results_detect_wm.append(detect_res_wm)
            results_decode_nowm.append(decode_res_nowm)
            results_decode_wm.append(decode_res_wm)
        elif args.method == 'gs':
            reversed_latents_nowm = gs_watermark.get_inversed_latents(img_nowm_auged, prompt='')
            reversed_latents_wm = gs_watermark.get_inversed_latents(img_wm_auged, prompt='')
            no_w_metric = gs_watermark.gs.eval_watermark(reversed_latents_nowm)
            w_metric = gs_watermark.gs.eval_watermark(reversed_latents_wm)
            no_w_metrics.append(no_w_metric)
            w_metrics.append(w_metric)
        elif args.method == 'tr':
            reversed_latents_nowm = tr_watermark.get_inversed_latents(img_nowm_auged, prompt='')
            reversed_latents_wm = tr_watermark.get_inversed_latents(img_wm_auged, prompt='')
            no_w_metric, w_metric = tr_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
            no_w_metrics.append(-no_w_metric) # negative, as the roc_curve function expects the higher value to be the more likely to be positive/wm
            w_metrics.append(-w_metric)
        else:
            print('Invalid method')
            return
        
    # compute the results
    if args.method == 'prc':
        tpr_detection = sum(results_detect_wm) / len(results_detect_wm)
        print(f'\nTPR Detection: \t{tpr_detection} at fpr {args.fpr}' )
        tpr_decode = sum(results_decode_wm) / len(results_decode_wm)
        print(f'\nTPR Decode: \t{tpr_decode} at fpr {args.fpr}' )
        comp_fpr_detect = sum(results_detect_nowm) / len(results_detect_nowm)
        print(f'\nFPR Detection: \t{comp_fpr_detect} at fpr {args.fpr}' )
    
    elif args.method == 'gs':
        preds = no_w_metrics +  w_metrics
        t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)

        # Find the TPR at the desired FPR
        index = np.where(fpr <= args.fpr)[0][-1]
        low = tpr[index]
        threshold = thresholds[index]

        print(f'\nTPR: {low} at fpr {args.fpr}')
        print(f'\n \tAUC: {auc}; ACC: {acc} at fpr {args.fpr}')
        print(f'\nw_metrics: {w_metrics}')
        print(f'\nno_w_metrics: {no_w_metrics}')
        print(f'\nThreshold: {threshold} with mean wm dist: {np.mean(w_metrics)} and mean no wm dist: {np.mean(no_w_metrics)}')
        print('\n\n')
        tpr_detection, tpr_traceability = gs_watermark.gs.get_tpr()
        tpr_detection = tpr_detection / args.num_images
        tpr_traceability = tpr_traceability / args.num_images
        print(f'results from gs_watermark.gs.get_tpr(): TPR Detection: {tpr_detection}; TPR Traceability: {tpr_traceability}')
        # Print all FPR, TPR, and thresholds
        print('\nDetailed ROC Curve Data:')
        for f, t, th in zip(fpr, tpr, thresholds):
            print(f'FPR: {f}; TPR: {t}; Threshold: {th}')
        
    if args.method == 'tr':
        preds = no_w_metrics +  w_metrics
        t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)

        # Find the TPR at the desired FPR
        index = np.where(fpr <= args.fpr)[0][-1]
        low = tpr[index]
        threshold = thresholds[index]

        print(f'\nTPR: {low} at fpr {args.fpr}')
        print(f'\n \tAUC: {auc}; ACC: {acc} at fpr {args.fpr}')
        print(f'\nw_metrics: {w_metrics}')
        print(f'\nno_w_metrics: {no_w_metrics}')
        print(f'\nThreshold: {threshold} with mean wm dist: {np.mean(w_metrics)} and mean no wm dist: {np.mean(no_w_metrics)}')
        # Print all FPR, TPR, and thresholds
        print('\nDetailed ROC Curve Data:')
        for f, t, th in zip(fpr, tpr, thresholds):
            print(f'FPR: {f}; TPR: {t}; Threshold: {th}')
            
        

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

    # for testing
    parser.add_argument('--test_path', type=str, default='original_images')

    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)
    
    


    args = parser.parse_args()

   
    
    main(args)