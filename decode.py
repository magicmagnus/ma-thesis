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
import datetime



# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prc'))
from prc.export import PRCWatermark
from prc.src.optim_utils import image_distortion, print2file

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussianshading'))
from gaussianshading.export import GSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
from treeringwatermark.export import TRWatermark




def main(args):

    HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'
    #HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    
    
    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting decode...')
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
        exp_id = exp_id + '_coco'

    
    print2file(args.log_file, '\nLoading imgs from', f'results/{exp_id}')

    

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
    thrshld_nowm = []
    thrshld_wm = []


    # for tr and gs
    no_wm_metrics = []
    wm_metrics = []

    results = []


    print2file(args.log_file, '\n\nStarting to decode...\n')

    
    # of all the distortion args, get the one that is not None and loop over it
    # if no attacks, placeholer attack
    attack_vals = [None]
    attack_name = None

    for arg in vars(args):
        if getattr(args, arg) is not None and arg in ['r_degree', 'jpeg_ratio', 'crop_scale', 'crop_ratio', 'gaussian_blur_r', 'gaussian_std', 'brightness_factor', ]:
            print2file(args.log_file, f'\nlooping over {arg}: {getattr(args, arg)}')
            attack_vals = getattr(args, arg)
            attack_name = arg
            break   

    for strength in range(len(attack_vals)):
        if attack_name is not None:
            print2file(args.log_file, f'\n Attack {attack_name}: {attack_vals[strength]}')
        else:
            print2file(args.log_file, f'\n No attack')



        for i in tqdm(range(args.num_images)):
            seed_everything(1)
            img_wm = Image.open(f'results/{exp_id}/wm/{i}.png')
            img_nowm = Image.open(f'results/{exp_id}/nowm/{i}.png')
            # distortion
            seed = 42
            img_wm_auged, img_nowm_auged = image_distortion(img_wm, img_nowm, seed, args, strength, i==0)
            
                
            if args.method == 'prc':
                # detect_res_nowm, decode_res_nowm = prc_watermark.detect_and_decode_watermark(img_nowm_auged, prompt='')
                # detect_res_wm, decode_res_wm = prc_watermark.detect_and_decode_watermark(img_wm_auged, prompt='')
                # results_detect_nowm.append(detect_res_nowm)
                # results_detect_wm.append(detect_res_wm)
                # results_decode_nowm.append(decode_res_nowm)
                # results_decode_wm.append(decode_res_wm)
                detected_nowm, metric_nowm, threshold_nowm = prc_watermark.detect_watermark(img_nowm_auged)
                detected_wm, metric_wm, threshold_wm = prc_watermark.detect_watermark(img_wm_auged)
                no_wm_metrics.append(metric_nowm) # probably also negative???
                wm_metrics.append(metric_wm)
                results_detect_nowm.append(detected_nowm)
                results_detect_wm.append(detected_wm)
                thrshld_nowm.append(threshold_nowm)
                thrshld_wm.append(threshold_wm)
                decoded_nowm = prc_watermark.decode_watermark(img_nowm_auged)
                decoded_wm = prc_watermark.decode_watermark(img_wm_auged)
                results_decode_nowm.append(decoded_nowm)
                results_decode_wm.append(decoded_wm)
            elif args.method == 'gs':
                reversed_latents_nowm = gs_watermark.get_inversed_latents(img_nowm_auged, prompt='')
                reversed_latents_wm = gs_watermark.get_inversed_latents(img_wm_auged, prompt='')
                no_w_metric = gs_watermark.gs.eval_watermark(reversed_latents_nowm)
                w_metric = gs_watermark.gs.eval_watermark(reversed_latents_wm)
                no_wm_metrics.append(no_w_metric)
                wm_metrics.append(w_metric)
            elif args.method == 'tr':
                reversed_latents_nowm = tr_watermark.get_inversed_latents(img_nowm_auged, prompt='')
                reversed_latents_wm = tr_watermark.get_inversed_latents(img_wm_auged, prompt='')
                no_w_metric, w_metric = tr_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
                no_wm_metrics.append(-no_w_metric) # negative, as the roc_curve function expects the higher value to be the more likely to be positive/wm
                wm_metrics.append(-w_metric)
            else:
                print2file(args.log_file, 'Invalid method')
                return
            
        # compute the results
        print2file(args.log_file, '''
                  ___                                      
                 / _ \                                     
                / /_\ \ ___ ___ _   _ _ __ __ _  ___ _   _ 
                |  _  |/ __/ __| | | | '__/ _` |/ __| | | |
                | | | | (_| (__| |_| | | | (_| | (__| |_| |
                \_| |_/\___\___|\__,_|_|  \__,_|\___|\__, |
                                                      __/ |
                                                     |___/ 
                ''')
        # if args.method == 'prc':
        #     tpr_detection = sum(results_detect_wm) / len(results_detect_wm)
        #     print2file(args.log_file, f'\nTPR Detection: \t{tpr_detection} at fpr {args.fpr}' )
        #     tpr_decode = sum(results_decode_wm) / len(results_decode_wm)
        #     print2file(args.log_file, f'\nTPR Decode: \t{tpr_decode} at fpr {args.fpr}' )
        #     comp_fpr_detect = sum(results_detect_nowm) / len(results_detect_nowm)
        #     print2file(args.log_file, f'\nFPR Detection: \t{comp_fpr_detect} at fpr {args.fpr}' )

        #     results.append({
        #         'attack': attack_name,
        #         'strength': attack_vals[strength],
        #         'tpr_detection': tpr_detection,
        #         'tpr_decode': tpr_decode,
        #         'fpr_detection': comp_fpr_detect
        #     })
        
        if args.method == 'gs' or args.method == 'tr' or args.method == 'prc':
            preds = no_wm_metrics +  wm_metrics
            t_labels = [0] * len(no_wm_metrics) + [1] * len(wm_metrics)

            fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            acc = np.max(1 - (fpr + (1 - tpr))/2)

            # Find the TPR at the desired FPR
            index = np.where(fpr <= args.fpr)[0][-1]
            low = tpr[index]
            threshold = thresholds[index]

            print2file(args.log_file, f'\n\tTPR: {low} at fpr {args.fpr} (empirical)')
            print2file(args.log_file, f'\n(AUC: {auc}; ACC: {acc} at fpr {args.fpr})')
            print2file(args.log_file, f'\nw_metrics: {wm_metrics}')
            print2file(args.log_file, f'no_w_metrics: {no_wm_metrics}')
            print2file(args.log_file, f'\nThreshold: {threshold} with mean wm dist: {np.mean(wm_metrics)} and mean no wm dist: {np.mean(no_wm_metrics)}')
            
            # Print all FPR, TPR, and thresholds
            print2file(args.log_file, '\nDetailed (empirical) ROC Curve Data:')
            for f, t, th in zip(fpr, tpr, thresholds):
                print2file(args.log_file, f'FPR: {f:.3f}; TPR: {t:.3f}; Threshold: {th:.3f}')

            if args.method == 'gs':
                print2file(args.log_file, f'\nAlternative Analytical Method (Built-in GS):')
                tpr_detection_count, tpr_traceability_count = gs_watermark.gs.get_tpr()
                tpr_detection = tpr_detection_count / args.num_images
                tpr_traceability = tpr_traceability_count / args.num_images
                print2file(args.log_file, f'\n\tTPR Detection: {tpr_detection} at fpr {args.fpr}' )
                print2file(args.log_file, f'\n\tTPR Traceability: {tpr_traceability} at fpr {args.fpr}' )

            if args.method == 'prc':
                print2file(args.log_file, f'\nAlternative Analytical Method (Built-in PRC):')
                tpr_detection = sum(results_detect_wm) / len(results_detect_wm)
                results_decode_wm_sum = [1 if x is not None else 0 for x in results_decode_wm]
                tpr_decode = results_decode_wm_sum / len(results_decode_wm)
                print2file(args.log_file, f'\n\tTPR Detection: \t{tpr_detection} at fpr {args.fpr}' )
                print2file(args.log_file, f'\n\tTPR Decode: \t{tpr_decode} at fpr {args.fpr}' )

                # print2file(args.log_file, f'\nThreshold no wm: {thrshld_nowm}')
                # print2file(args.log_file, f'Threshold wm: {thrshld_wm}')
                
            

            results.append({
                'attack': attack_name,
                'strength': attack_vals[strength],
                'tpr_detection': low,
                'auc': auc,
                'acc': acc,
                'threshold': threshold,
                'mean_wm_dist': np.mean(wm_metrics),
                'mean_no_wm_dist': np.mean(no_wm_metrics),
                'tpr_detection_gs': tpr_detection if args.method == 'gs' else None,
                'tpr_traceability_gs': tpr_traceability if args.method == 'gs' else None
            })
                
        
    # save results
    with open(f'{args.log_dir}/results.txt', 'wb') as f:
        pickle.dump(results, f)

    print2file(args.log_file, '\n' + '#'*100 + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')

    # ################### general #######################
    # parser.add_argument('--num_images', type=int, default=10)
    # parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
    # parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
    # parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts') # coco 
    # parser.add_argument('--inf_steps', type=int, default=50)
    # parser.add_argument('--fpr', type=float, default=0.00001)
    # parser.add_argument('--guidance_scale', type=float, default=3.0)
    # parser.add_argument('--num_images_per_prompt', type=int, default=1)
    # parser.add_argument('--run_name', type=str, default='test')

    # ################### for testing ###################
    # # for image distortion
    # parser.add_argument('--r_degree', default=None, type=float)
    # parser.add_argument('--jpeg_ratio', default=None, type=int)
    # parser.add_argument('--crop_scale', default=None, type=float)
    # parser.add_argument('--crop_ratio', default=None, type=float)
    # parser.add_argument('--gaussian_blur_r', default=None, type=int)
    # parser.add_argument('--gaussian_std', default=None, type=float)
    # parser.add_argument('--brightness_factor', default=None, type=float)
    # parser.add_argument('--rand_aug', default=0, type=int)
    
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

    # args = parser.parse_args()

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
    args.log_dir = f'./experiments/decode_{date}_{args.run_name}'
    os.makedirs(args.log_dir)

    

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_{args.run_name}'
    if args.dataset_id == 'coco':
        exp_id += '_coco'
    # create a log file
    args.log_file = open(f'{args.log_dir}/{exp_id}.txt', 'w', buffering=1)  # Use line buffering

    print2file(args.log_file, f'Experiment ID: {exp_id}')
    
    main(args)