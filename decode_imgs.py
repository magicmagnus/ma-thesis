import os
if "is/sg2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import json
import torch
import pickle
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt

import time 

from datasets import load_dataset

from utils import seed_everything, print2file, get_dirs

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prc'))
from prc.export import PRCWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussianshading'))
from gaussianshading.export import GSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'rid'))
from ringid.export import RingIDWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
from treeringwatermark.export import TRWatermark
from treeringwatermark.pytorch_fid.fid_score import calculate_fid_given_paths

sys.path.append(os.path.join(os.path.dirname(__file__), 'waves', 'adversarial'))
from waves.adversarial.embedding import adv_emb_attack_custom 



def main(args):

    if "is/sg2" in os.getcwd():
        HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    else:
        HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

     # paramters that could theoretically be moved to the config file, but are always the same
    REFERENCE_MODEL = 'ViT-g-14'
    REFERENCE_MODEL_PRETRAIN = 'laion2b_s12b_b42k'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    log_dir, args.data_dir = get_dirs(args, "decode_imgs")# , extra=args.run_name)
    args.log_dir = os.path.join(log_dir, f'{args.date}_{args.run_name}')
    os.makedirs(args.log_dir)
    
    logfile_name = f'{args.run_name}'
    args.log_file = open(os.path.join(args.log_dir, f'{logfile_name}.txt'), 'w', buffering=1)  # Use line buffering

    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting Decode...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')

    # first genrate all the keys per method
    if args.method == 'prc':
        prc_watermark = PRCWatermark(args, hf_cache_dir=HF_CACHE_DIR)
    elif args.method == 'gs':
        gs_watermark = GSWatermark(args, hf_cache_dir=HF_CACHE_DIR)
    elif args.method == 'tr':
        tr_watermark = TRWatermark(args, hf_cache_dir=HF_CACHE_DIR)
    elif args.method == 'rid':
        rid_watermark = RingIDWatermark(args, hf_cache_dir=HF_CACHE_DIR)
        
    else:
        print2file(args.log_file, 'Invalid method')
        return
    
    print2file(args.log_file, f'\nLoading attacked images from {args.data_dir}')


    distortions = ['r_degree', 'jpeg_ratio', 'crop_scale', 'crop_ratio', 'gaussian_blur_r', 'gaussian_std', 'brightness_factor', ]
    adversarial_embeds = ['adv_embed_resnet18', 'adv_embed_resnet50', 'adv_embed_klvae8', 'adv_embed_sdxlvae', 'adv_embed_klvae16']
    adversarial_surr = ['adv_surr_resnet18', 'adv_surr_resnet50']
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
            attack_type = None

    # start the decoding
    print2file(args.log_file, '\n\nStarting to decode...\n')
    for strength in range(len(attack_vals)):
        print2file(args.log_file, f'\nAttacktype "{attack_type}" with Attack "{attack_name}": {attack_vals[strength]}' if attack_name is not None else '\n\nNo attack')
        
        # clear the metrics before each attack
        no_wm_metrics = []
        wm_metrics = []
    
        t_labels = []
        preds = []
        tpr_detection = None
        tpr_decode = None

        if args.method == 'gs':
            gs_watermark.gs.clear_count()
        if args.method == 'prc':
            results_detect_wm = []
            results_detect_nowm = []
            results_decode_wm = []
            results_decode_nowm = []
            thrshld_nowm = []
            thrshld_wm = []
        
        
        for i in tqdm(range(args.num_images)):
            
            seed_everything(i)
        
            # load the attackeg images
            if attack_type == "adversarial_surr":
                img_wm_attacked = Image.open(os.path.join(args.data_dir, 'wm', args.run_name, str(attack_vals[strength]), f'{i}.png'))
                img_nowm_attacked = Image.open(os.path.join(args.data_dir, 'nowm', args.run_name, str(attack_vals[strength]), f'{i}.png'))
            elif attack_type is not None:
                img_wm_attacked = Image.open(os.path.join(args.data_dir, 'wm', attack_name, str(attack_vals[strength]), f'{i}.png'))
                img_nowm_attacked = Image.open(os.path.join(args.data_dir, 'nowm', attack_name, str(attack_vals[strength]), f'{i}.png'))
            else:
                img_wm_attacked = Image.open(os.path.join(args.data_dir, 'wm', f'{i}.png'))
                img_nowm_attacked = Image.open(os.path.join(args.data_dir, 'nowm', f'{i}.png'))


            # decode the images
            if args.method == 'prc':
                reversed_latents_nowm, true_latents_nowm = prc_watermark.get_inversed_latents(img_nowm_attacked, prompt='', do_wm=False, seed=i)
                reversed_latents_wm, true_latents_wm = prc_watermark.get_inversed_latents(img_wm_attacked, prompt='', do_wm=True, seed=i)
                if i == 0:
                    # prc_watermark.viz_reversed_latents(reversed_latents_nowm, reversed_latents_wm, attack_name, attack_vals, strength)
                    prc_watermark.viz_reversed_latents(true_latents_nowm=true_latents_nowm,
                                                       reversed_latents_nowm=reversed_latents_nowm,
                                                       true_latents_wm=true_latents_wm,
                                                       reversed_latents_wm=reversed_latents_wm,
                                                       attack_name=attack_name, attack_vals=attack_vals, strength=strength)
                detected_nowm, metric_nowm, threshold_nowm = prc_watermark.detect_watermark(reversed_latents_nowm)
                detected_wm, metric_wm, threshold_wm = prc_watermark.detect_watermark(reversed_latents_wm)
                no_wm_metrics.append(metric_nowm)
                wm_metrics.append(metric_wm)
                results_detect_nowm.append(detected_nowm)
                results_detect_wm.append(detected_wm)
                thrshld_nowm.append(threshold_nowm)
                thrshld_wm.append(threshold_wm)
                decoded_nowm = prc_watermark.decode_watermark(reversed_latents_nowm)
                decoded_wm = prc_watermark.decode_watermark(reversed_latents_wm)
                results_decode_nowm.append(decoded_nowm)
                results_decode_wm.append(decoded_wm)
            elif args.method == 'gs':
                reversed_latents_nowm, true_latents_nowm  = gs_watermark.get_inversed_latents(img_nowm_attacked, prompt='', do_wm=False, seed=i)
                reversed_latents_wm, true_latents_wm = gs_watermark.get_inversed_latents(img_wm_attacked, prompt='', do_wm=True, seed=i)
                if i == 0:
                    # gs_watermark.viz_reversed_latents(reversed_latents_nowm, reversed_latents_wm, attack_name, attack_vals, strength)
                    gs_watermark.viz_reversed_latents(true_latents_nowm=true_latents_nowm,
                                                        reversed_latents_nowm=reversed_latents_nowm,
                                                        true_latents_wm=true_latents_wm,
                                                        reversed_latents_wm=reversed_latents_wm,
                                                        attack_name=attack_name, attack_vals=attack_vals, strength=strength)
                no_w_metric = gs_watermark.eval_watermark(reversed_latents_nowm, count=False) # sometimes no-wm samples might be detected as wm, but we don't count them, as we calculate the fpr later
                w_metric = gs_watermark.eval_watermark(reversed_latents_wm)
                no_wm_metrics.append(no_w_metric)
                wm_metrics.append(w_metric)
            elif args.method == 'tr':
                reversed_latents_nowm, true_latents_nowm = tr_watermark.get_inversed_latents(img_nowm_attacked, prompt='', do_wm=False, seed=i)
                reversed_latents_wm, true_latents_wm = tr_watermark.get_inversed_latents(img_wm_attacked, prompt='', do_wm=True, seed=i)
                if i == 0:
                    #tr_watermark.viz_reversed_latents(reversed_latents_nowm, reversed_latents_wm, attack_name, attack_vals, strength)
                    #tr_watermark.viz_difference(true_latents_nowm, reversed_latents_nowm, true_latents_wm, reversed_latents_wm, attack_name, attack_vals, strength)
                    tr_watermark.viz_reversed_latents(true_latents_nowm=true_latents_nowm, 
                                                      reversed_latents_nowm=reversed_latents_nowm, 
                                                      true_latents_wm=true_latents_wm, 
                                                      reversed_latents_wm=reversed_latents_wm, 
                                                      attack_name=attack_name, attack_vals=attack_vals, strength=strength)
                no_w_metric, w_metric = tr_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
                no_wm_metrics.append(-no_w_metric) # negative, as the roc_curve function expects the higher value to be the more likely to be positive/wm
                wm_metrics.append(-w_metric)
            elif args.method == 'rid':
                reversed_latents_nowm, true_latents_nowm = rid_watermark.get_inversed_latents(img_nowm_attacked, prompt='', do_wm=False, seed=i, pattern_index=args.pattern_index)
                reversed_latents_wm, true_latents_wm = rid_watermark.get_inversed_latents(img_wm_attacked, prompt='', do_wm=True, seed=i, pattern_index=args.pattern_index)
                if i == 0:
                    #rid_watermark.viz_reversed_latents(reversed_latents_nowm, reversed_latents_wm, attack_name, attack_vals, strength)
                    rid_watermark.viz_reversed_latents(true_latents_nowm=true_latents_nowm,
                                                        reversed_latents_nowm=reversed_latents_nowm,
                                                        true_latents_wm=true_latents_wm,
                                                        reversed_latents_wm=reversed_latents_wm,
                                                        attack_name=attack_name, attack_vals=attack_vals, strength=strength)
                no_w_metric, w_metric = rid_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
                no_wm_metrics.append(-no_w_metric)
                wm_metrics.append(-w_metric)
                
            else:
                print2file(args.log_file, 'Invalid method')
                return
            
            
        # compute the results, the empirical ROC curve
        preds = no_wm_metrics +  wm_metrics
        t_labels = [0] * len(no_wm_metrics) + [1] * len(wm_metrics)
        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        # Find the TPR at the desired FPR
        index = np.where(fpr <= args.fpr)[0][-1]
        low = tpr[index]
        threshold = thresholds[index]

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
        print2file(args.log_file, f'\nTPR at fpr {args.fpr} (empirical)')
        print2file(args.log_file, f'\n\t{low}')
        print2file(args.log_file, f'\n(AUC: {auc}; ACC: {acc} at fpr {args.fpr})')
        print2file(args.log_file, f'\nw_metrics: {wm_metrics}')
        print2file(args.log_file, f'no_w_metrics: {no_wm_metrics}')
        print2file(args.log_file, f'\nThreshold: {threshold} with mean wm dist: {np.mean(wm_metrics)} and mean no wm dist: {np.mean(no_wm_metrics)}')
        
        # Print all FPR, TPR, and thresholds
        print2file(args.log_file, '\nDetailed (empirical) ROC Curve Data:')
        for f, t, th in zip(fpr, tpr, thresholds):
            print2file(args.log_file, f'FPR: {f:.3f}; TPR: {t:.3f}; Threshold: {th:.3f}')
        
        # Alternative analytical method
        if args.method == 'gs':
            print2file(args.log_file, f'\nAlternative Analytical Method (Built-in GS):')
            tpr_detection_count, tpr_traceability_count = gs_watermark.gs.get_tpr()
            tpr_detection = tpr_detection_count / args.num_images
            tpr_traceability = tpr_traceability_count / args.num_images
            print2file(args.log_file, f'\n\tTPR Detection: {tpr_detection} at fpr {args.fpr} (tpr_detected: {tpr_detection_count})' )
            print2file(args.log_file, f'\n\tTPR Traceability: {tpr_traceability} at fpr {args.fpr} (tpr_traceable: {tpr_traceability_count})' )

        if args.method == 'prc':
            print2file(args.log_file, f'\nAlternative Analytical Method (Built-in PRC):')
            tpr_detection = sum(results_detect_wm) / len(results_detect_wm)
            results_decode_wm_sum = [1 if x is not None else 0 for x in results_decode_wm]  
            tpr_decode = sum(results_decode_wm_sum) / len(results_decode_wm_sum)
            print2file(args.log_file, f'\n\tTPR Detection: \t{tpr_detection} at fpr {args.fpr}' )
            print2file(args.log_file, f'\n\tTPR Decode: \t{tpr_decode} at fpr {args.fpr}' )

        # plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr)
        plt.grid()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curve for {args.method} with {attack_name}={attack_vals[strength]}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.log_dir, f'roc_{attack_name}_{attack_vals[strength]}.png'))
        plt.close()
    
        # save the metrics in a pickle file
        with open(os.path.join(args.data_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump({
                f'tpr_{attack_name}_{attack_vals[strength]}': low,
                f'auc_{attack_name}_{attack_vals[strength]}': auc,
                f'acc_{attack_name}_{attack_vals[strength]}': acc,
                f'threshold_{attack_name}_{attack_vals[strength]}': threshold,
                f'mean_wm_dist_{attack_name}_{attack_vals[strength]}': np.mean(wm_metrics),
                f'mean_no_wm_dist_{attack_name}_{attack_vals[strength]}': np.mean(no_wm_metrics),
                f'tpr_analytical_{attack_name}_{attack_vals[strength]}': tpr_detection,
                f'tpr_decode_{attack_name}_{attack_vals[strength]}': tpr_decode,
            }, f)


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
    args.date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    main(args)