import os

from regex import T
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

sys.path.append(os.path.join(os.path.dirname(__file__), 'rid'))
from ringid.export import RingIDWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
from treeringwatermark.export import TRWatermark
from treeringwatermark.pytorch_fid.fid_score import calculate_fid_given_paths

sys.path.append(os.path.join(os.path.dirname(__file__), 'waves', 'adversarial'))
from waves.adversarial.embedding import adv_emb_attack_custom 

CALC_FID = True

def main(args):

    #HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'
    HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting decode...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}'

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

    print2file(args.log_file, '\n\nStarting to decode...\n')

    
    # of all the distortion args, get the one that is not None and loop over it
    # if no attacks, placeholer attack
    attack_vals = [None]
    attack_name = None
    distortions = ['r_degree', 'jpeg_ratio', 'crop_scale', 'crop_ratio', 'gaussian_blur_r', 'gaussian_std', 'brightness_factor', ]
    adversarial_embeds = ['adv_embed_resnet18', 'adv_embed_clip', 'adv_embed_klvae8', 'adv_embed_sdxlvae', 'adv_embed_klvae16']
    attack_type = None

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

    for strength in range(len(attack_vals)):
        if attack_name is not None:
            print2file(args.log_file, f'\n Attack {attack_name}: {attack_vals[strength]}')
        else:
            print2file(args.log_file, f'\n No attack')

        # clear the metrics before each attack
        results_detect_wm = []
        results_detect_nowm = []
        results_decode_wm = []
        results_decode_nowm = []
        thrshld_nowm = []
        thrshld_wm = []

        no_wm_metrics = []
        wm_metrics = []

        results = []

        t_labels = []
        preds = []

        tpr_detection = 0
        tpr_decode = 0

        if args.method == 'gs':
            gs_watermark.gs.clear_count()

        # saved attacked images in the results/attack_name/attack_val folder

        if args.overwrite_attacked_imgs:    
            os.makedirs(f'results/{exp_id}/wm/{attack_name}/{attack_vals[strength]}', exist_ok=True)
            os.makedirs(f'results/{exp_id}/nowm/{attack_name}/{attack_vals[strength]}', exist_ok=True)

        for i in tqdm(range(args.num_images)):
            seed_everything(1)
            img_wm = Image.open(f'results/{exp_id}/wm/{i}.png')
            img_nowm = Image.open(f'results/{exp_id}/nowm/{i}.png')
            # distortion
            seed = 42

            path_attack_wm = f'results/{exp_id}/wm/{attack_name}/{attack_vals[strength]}/{i}.png'
            path_attack_nowm = f'results/{exp_id}/nowm/{attack_name}/{attack_vals[strength]}/{i}.png'

            if args.overwrite_attacked_imgs or not os.path.exists(path_attack_wm):
                # if overwrite is true or file does not exist, distort the image and save it
                if attack_type == 'distortion' or attack_type is None:
                    img_wm_auged, img_nowm_auged = image_distortion(img_wm, img_nowm, i, args, strength, i==0)
                elif attack_type == 'adversarial_embed':
                    img_wm_auged = adv_emb_attack_custom(img_wm, attack_name, attack_vals[strength], device)
                    img_nowm_auged = adv_emb_attack_custom(img_nowm, attack_name, attack_vals[strength], device)
                img_wm_auged.save(path_attack_wm)
                img_nowm_auged.save(path_attack_nowm)
            else:
                # if not overwrite and file exists, open the existing image
                img_wm_auged = Image.open(path_attack_wm)
                img_nowm_auged = Image.open(path_attack_nowm)


            

            
                
            if args.method == 'prc':
                # detect_res_nowm, decode_res_nowm = prc_watermark.detect_and_decode_watermark(img_nowm_auged, prompt='')
                # detect_res_wm, decode_res_wm = prc_watermark.detect_and_decode_watermark(img_wm_auged, prompt='')
                # results_detect_nowm.append(detect_res_nowm)
                # results_detect_wm.append(detect_res_wm)
                # results_decode_nowm.append(decode_res_nowm)
                # results_decode_wm.append(decode_res_wm)


                # detected_nowm, metric_nowm, threshold_nowm = prc_watermark.detect_watermark(img_nowm_auged)
                # detected_wm, metric_wm, threshold_wm = prc_watermark.detect_watermark(img_wm_auged)
                # if i == 0:
                #     prc_watermark.viz_reversed_latents(img_nowm_auged, img_wm_auged, attack_name, attack_vals, strength)
                # no_wm_metrics.append(metric_nowm) # probably also negative???
                # wm_metrics.append(metric_wm)
                # results_detect_nowm.append(detected_nowm)
                # results_detect_wm.append(detected_wm)
                # thrshld_nowm.append(threshold_nowm)
                # thrshld_wm.append(threshold_wm)
                # decoded_nowm = prc_watermark.decode_watermark(img_nowm_auged)
                # decoded_wm = prc_watermark.decode_watermark(img_wm_auged)
                # results_decode_nowm.append(decoded_nowm)
                # results_decode_wm.append(decoded_wm)

                reversed_latents_nowm = prc_watermark.get_inversed_latents(img_nowm_auged, prompt='')
                reversed_latents_wm = prc_watermark.get_inversed_latents(img_wm_auged, prompt='')
                if i == 0:
                    prc_watermark.viz_reversed_latents(reversed_latents_nowm, reversed_latents_wm, attack_name, attack_vals, strength)

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
                reversed_latents_nowm = gs_watermark.get_inversed_latents(img_nowm_auged, prompt='')
                reversed_latents_wm = gs_watermark.get_inversed_latents(img_wm_auged, prompt='')
                if i == 0:
                    gs_watermark.viz_reversed_latents(reversed_latents_nowm, reversed_latents_wm, attack_name, attack_vals, strength)
                no_w_metric = gs_watermark.gs.eval_watermark(reversed_latents_nowm, count=False) # sometimes no-wm samples might be detected as wm, but we don't count them
                w_metric = gs_watermark.gs.eval_watermark(reversed_latents_wm)
                no_wm_metrics.append(no_w_metric)
                wm_metrics.append(w_metric)
            elif args.method == 'tr':
                reversed_latents_nowm = tr_watermark.get_inversed_latents(img_nowm_auged, prompt='')
                reversed_latents_wm = tr_watermark.get_inversed_latents(img_wm_auged, prompt='')
                if i == 0:
                    tr_watermark.viz_reversed_latents(reversed_latents_nowm, reversed_latents_wm, attack_name, attack_vals, strength)
                no_w_metric, w_metric = tr_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
                no_wm_metrics.append(-no_w_metric) # negative, as the roc_curve function expects the higher value to be the more likely to be positive/wm
                wm_metrics.append(-w_metric)
            elif args.method == 'rid':
                reversed_latents_nowm = rid_watermark.get_inversed_latents(img_nowm_auged, prompt='')
                reversed_latents_wm = rid_watermark.get_inversed_latents(img_wm_auged, prompt='')
                if i == 0:
                    rid_watermark.viz_reversed_latents(reversed_latents_nowm, reversed_latents_wm, attack_name, attack_vals, strength)
                no_w_metric, w_metric = rid_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
                no_wm_metrics.append(-no_w_metric)
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
        
        
        preds = no_wm_metrics +  wm_metrics
        t_labels = [0] * len(no_wm_metrics) + [1] * len(wm_metrics)

        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)

        # also, plot the roc curve
        import matplotlib.pyplot as plt
        plt.close()
        plt.plot(fpr, tpr)
        # add gridlines
        plt.grid()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curve for {args.method} with {attack_name}={attack_vals[strength]}')
        plt.tight_layout()
        plt.savefig(f'{args.log_dir}/roc_{attack_name}_{attack_vals[strength]}.png')
        plt.close()

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
            print2file(args.log_file, f'\n\tTPR Detection: {tpr_detection} at fpr {args.fpr} (tpr_detected: {tpr_detection_count})' )
            print2file(args.log_file, f'\n\tTPR Traceability: {tpr_traceability} at fpr {args.fpr} (tpr_traceable: {tpr_traceability_count})' )

        if args.method == 'prc':
            print2file(args.log_file, f'\nAlternative Analytical Method (Built-in PRC):')
            tpr_detection = sum(results_detect_wm) / len(results_detect_wm)
            results_decode_wm_sum = [1 if x is not None else 0 for x in results_decode_wm]  
            tpr_decode = sum(results_decode_wm_sum) / len(results_decode_wm_sum)
            print2file(args.log_file, f'\n\tTPR Detection: \t{tpr_detection} at fpr {args.fpr}' )
            print2file(args.log_file, f'\n\tTPR Decode: \t{tpr_decode} at fpr {args.fpr}' )

        fid_score_wm = None
        fid_score_nowm = None
        if CALC_FID:
            # measure the FID between original and attacked images, both with and without watermark
            print2file(args.log_file, '''
            ______ _____ _____  
            |  ____|_   _|  __ \ 
            | |__    | | | |  | |
            |  __|   | | | |  | |
            | |     _| |_| |__| |
            |_|    |_____|_____/ 
                        ''')
            if args.dataset_id == 'coco':
                fid_score_wm = calculate_fid_given_paths([f'results/{exp_id}/wm/{attack_name}/{attack_vals[strength]}', '/is/sg2/mkaut/ma-thesis/coco/val2017'], 
                                                        batch_size=50, 
                                                        device=device, 
                                                        dims=2048,
                                                        max_samples=args.num_images)
                print2file(args.log_file, f'\nFID score with watermark for attack {attack_name}={attack_vals[strength]} for {args.num_images} samples: \n\t{fid_score_wm}')
                fid_score_nowm = calculate_fid_given_paths([f'results/{exp_id}/nowm/{attack_name}/{attack_vals[strength]}', '/is/sg2/mkaut/ma-thesis/coco/val2017'], 
                                                        batch_size=50, 
                                                        device=device, 
                                                        dims=2048,
                                                        max_samples=args.num_images)
                print2file(args.log_file, f'\nFID score without watermark for attack {attack_name}={attack_vals[strength]} for {args.num_images} samples: \n\t{fid_score_nowm}')

            else:
                print2file(args.log_file, f'\nNo FID score for non-coco datasets')

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
            'tpr_traceability_gs': tpr_traceability if args.method == 'gs' else None,
            'fid_score_wm': fid_score_wm if args.dataset_id == 'coco' else None,
            'fid_score_nowm': fid_score_nowm if args.dataset_id == 'coco' else None,

        })
        
        # save results
        with open(f'{args.log_dir}/results_{attack_name}_{attack_vals[strength]}.pkl', 'wb') as f:
            pickle.dump(results, f)

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
    args.log_dir = f'./experiments/{date}_decode_{args.run_name}'
    os.makedirs(args.log_dir)

    exp_id = f'{args.method}_num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_{args.run_name}'
    if args.dataset_id == 'coco':
        exp_id += '_coco'
    # create a log file
    args.log_file = open(f'{args.log_dir}/{exp_id}.txt', 'w', buffering=1)  # Use line buffering
    
    main(args)