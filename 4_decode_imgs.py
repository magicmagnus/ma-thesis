import os
if 'is/sg2' in os.getcwd():
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
import json
import torch
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt

from datasets import load_dataset

from utils import seed_everything, print2file, get_dirs, load_prompts, get_pipe, bootstrap_tpr, bootstrap_grids_tpr, bootstrap_grids_dynamic_thresholds, plot_heatmaps

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'prc'))
from prc.export import PRCWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussianshading'))
from gaussianshading.export import GSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'rid'))
from ringid.export import RingIDWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'grids'))
from grids.export import GRIDSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'treeringwatermark'))
from treeringwatermark.export import TRWatermark
import treeringwatermark.open_clip as open_clip
from treeringwatermark.optim_utils import measure_similarity
from treeringwatermark.pytorch_fid.fid_score import calculate_fid_given_paths

sys.path.append(os.path.join(os.path.dirname(__file__), 'waves', 'adversarial'))
from waves.adversarial.embedding import adv_emb_attack_custom 



def main(args):

    if 'is/sg2' in os.getcwd():
        HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
    else:
        HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

     # paramters that could theoretically be moved to the config file, but are always the same
    REFERENCE_MODEL = 'ViT-g-14'
    REFERENCE_MODEL_PRETRAIN = 'laion2b_s12b_b42k'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # check if the latent_channels_wm is correct for the model
    if args.model_id == 'sd' and args.latent_channels_wm != 4:
        print('Warning: For the sd model, the latent_channels_wm should be 4\nSetting it to 4')
        args.latent_channels_wm = 4
    
    log_dir, args.data_dir = get_dirs(args, 'decode_imgs')# , extra=args.run_name)
    args.log_dir = os.path.join(log_dir, f'{args.date}_{args.run_name}')
    os.makedirs(args.log_dir)
    
    logfile_name = f'{args.run_name}'
    args.log_file = open(os.path.join(args.log_dir, f'{logfile_name}.txt'), 'w', buffering=1)  # Use line buffering

    print2file(args.log_file, '\n' + '#'*100 + '\n')
    print2file(args.log_file, '\nStarting Decode...')
    print2file(args.log_file, '\nArgs:\n')
    for arg in vars(args):
        print2file(args.log_file, f'{arg}: {getattr(args, arg)}')

    # set seed for internal WM viz and prompt loading
    seed_everything(0) # should be 0 cause it gets set to 0 later in the loop

    # create the pipe
    pipe = get_pipe(args, device, HF_CACHE_DIR)

    # first genrate all the keys per method
    if args.method == 'prc':
        prc_watermark = PRCWatermark(args, pipe)
    elif args.method == 'gs':
        gs_watermark = GSWatermark(args, pipe)
    elif args.method == 'tr':
        tr_watermark = TRWatermark(args, pipe)
    elif args.method == 'rid':
        rid_watermark = RingIDWatermark(args, pipe)
    elif args.method == 'grids':
        grids_watermark = GRIDSWatermark(args, pipe)
    else:
        print2file(args.log_file, 'Invalid method')
        return
    
    print2file(args.log_file, f'\nLoading attacked images from {args.data_dir}')

    # set seed for internal WM viz and prompt loading
    seed_everything(0) # should be 0 cause it gets set to 0 later in the loop

    # load the prompt dataset
    prompts = load_prompts(args)

    if args.calc_CLIP:
        # load the reference CLIP model
        print2file(args.log_file, f'\nLoading reference CLIP model {REFERENCE_MODEL}')
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            REFERENCE_MODEL, 
            pretrained=REFERENCE_MODEL_PRETRAIN, 
            device=device,
            cache_dir=HF_CACHE_DIR)
        ref_tokenizer = open_clip.get_tokenizer(REFERENCE_MODEL)

    # create the results dataframe
    results_df = pd.DataFrame(columns=['wm_method', 
                                       'model_id', 
                                       'dataset_id',
                                       'attack_type',
                                       'attack_name',
                                       'attack_strength',
                                       'tpr_empirical',
                                       'tpr_empirical_mean',
                                       'tpr_std_error',
                                       'tpr_ci_lower_percentile',
                                       'tpr_ci_upper_percentile',
                                        'auc',
                                       'acc',
                                       'tpr_analytical',
                                       'tpr_decode',
                                       'tpr_traceability',
                                       'threshold',
                                       'mean_wm_dist',
                                       'mean_no_wm_dist',
                                       'wm_diff',
                                       'nowm_diff',
                                       'clip_score_wm',
                                       'clip_score_nowm',
                                       'fid_wm_coco',
                                       'fid_nowm_coco',
                                       'fid_wm_nowm',
                                       'set_fpr',
                                       'wm_ch',
                                       'inf_steps',
                                       'test_inf_steps',
                                       'guidance_scale',])
    results_df = results_df.astype({
        'wm_method': 'string',
        'model_id': 'string', 
        'dataset_id': 'string',
        'attack_type': 'string',
        'attack_name': 'string',
        'attack_strength': 'float64',
        'tpr_empirical': 'float64',
        'tpr_empirical_mean': 'float64',
        'tpr_std_error': 'float64',
        'tpr_ci_lower_percentile': 'float64',
        'tpr_ci_upper_percentile': 'float64',
        'auc': 'float64', 
        'acc': 'float64',
        'tpr_analytical': 'float64',
        'tpr_decode': 'float64',
        'tpr_traceability': 'float64',
        'threshold': 'float64',
        'mean_wm_dist': 'float64',
        'mean_no_wm_dist': 'float64', 
        'wm_diff': 'float64',
        'nowm_diff': 'float64',
        'clip_score_wm': 'float64',
        'clip_score_nowm': 'float64',
        'fid_wm_coco': 'float64',
        'fid_nowm_coco': 'float64',
        'fid_wm_nowm': 'float64',
        'set_fpr': 'float64',
        'wm_ch': 'float64',
        'inf_steps': 'float64',
        'test_inf_steps': 'float64',
        'guidance_scale': 'float64',
    })

    # 
    distortions = ['r_degree', 'jpeg_ratio', 'crop_scale', 'crop_ratio', 'crop', 'gaussian_blur_r', 'gaussian_std', 'brightness_factor', ]
    adversarial_embeds = ['adv_embed_resnet18', 'adv_embed_resnet50', 'adv_embed_klvae8', 'adv_embed_sdxlvae', 'adv_embed_klvae16']
    adversarial_surr = ['adv_surr_resnet18', 'adv_surr_resnet50']
    attack_vals = ['no_attack']
    attack_name = 'no_attack'
    attack_type = 'no_attack'
    
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

    # start the decoding
    # each iteration is one 'decode run' with a different attack strength
    print2file(args.log_file, '\n\nStarting to decode...\n')
    for strength in range(len(attack_vals)):
        print2file(args.log_file, f'\nAttacktype "{attack_type}" with Attack "{attack_name}": {attack_vals[strength]}' if attack_name != 'no_attack' else '\nNo attack')
        
        # clear the metrics before each attack
        no_wm_metrics = []
        wm_metrics = []

        # total abs distances of the wm and no-wm samples to their respective true latents
        wm_diffs = []
        no_wm_diffs = []
    
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
        if args.method == 'grids':
            rid_nowm_metrics = []
            rid_wm_metrics = []
            gs_nowm_metrics = []
            gs_wm_metrics = []

        # get dirs of the attacked images, per attack type
        if attack_type == 'distortion' or attack_type == 'adversarial_embed':
            path_attacked_wm = os.path.join(args.data_dir, 'wm', attack_name, str(attack_vals[strength]))
            path_attacked_nowm = os.path.join(args.data_dir, 'nowm', attack_name, str(attack_vals[strength]))
        elif attack_type == 'adversarial_surr':
            path_attacked_wm = os.path.join(args.data_dir, 'wm', args.run_name, str(attack_vals[strength]))
            path_attacked_nowm = os.path.join(args.data_dir, 'nowm', args.run_name, str(attack_vals[strength]))
        elif attack_type == 'no_attack':
            path_attacked_wm = os.path.join(args.data_dir, 'wm')
            path_attacked_nowm = os.path.join(args.data_dir, 'nowm')
        else:
            RuntimeError('Invalid attack type')

        # loop over the images in both dirs
        for i in tqdm(range(args.num_images)):
            
            seed_everything(i)
        
            img_wm_attacked = Image.open(os.path.join(path_attacked_wm, f'{i}.png'))
            img_nowm_attacked = Image.open(os.path.join(path_attacked_nowm, f'{i}.png'))

            # decode the images
            if args.method == 'prc':
                reversed_latents_nowm, true_latents_nowm = prc_watermark.get_inversed_latents(img_nowm_attacked, prompt='', do_wm=False, seed=i)
                reversed_latents_wm, true_latents_wm = prc_watermark.get_inversed_latents(img_wm_attacked, prompt='', do_wm=True, seed=i)
                if i == 0:
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
                    rid_watermark.viz_reversed_latents(true_latents_nowm=true_latents_nowm,
                                                        reversed_latents_nowm=reversed_latents_nowm,
                                                        true_latents_wm=true_latents_wm,
                                                        reversed_latents_wm=reversed_latents_wm,
                                                        attack_name=attack_name, attack_vals=attack_vals, strength=strength)
                no_w_metric, w_metric = rid_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
                no_wm_metrics.append(-no_w_metric)
                wm_metrics.append(-w_metric)
            elif args.method == 'grids':
                reversed_latents_nowm, true_latents_nowm = grids_watermark.get_inversed_latents(img_nowm_attacked, prompt='', do_wm=False, seed=i, pattern_index=args.pattern_index)
                reversed_latents_wm, true_latents_wm = grids_watermark.get_inversed_latents(img_wm_attacked, prompt='', do_wm=True, seed=i, pattern_index=args.pattern_index)
                if i == 0:
                    grids_watermark.viz_reversed_latents(true_latents_nowm=true_latents_nowm,
                                                        reversed_latents_nowm=reversed_latents_nowm,
                                                        true_latents_wm=true_latents_wm,
                                                        reversed_latents_wm=reversed_latents_wm,
                                                        attack_name=attack_name, attack_vals=attack_vals, strength=strength)
                # no_w_metric, w_metric = grids_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
                rid_nowm_metric, rid_wm_metric, gs_nowm_metric, gs_wm_metric = grids_watermark.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
                #no_wm_metrics.append(no_w_metric) # until now, the metrics are the same as for gs
                #wm_metrics.append(w_metric)
                rid_nowm_metrics.append(-rid_nowm_metric) #the rids are negative, as the roc_curve function expects the higher value to be the more likely to be positive/wm
                rid_wm_metrics.append(-rid_wm_metric)
                gs_nowm_metrics.append(gs_nowm_metric) # but the gs are positive, 
                gs_wm_metrics.append(gs_wm_metric)
            else:
                RuntimeError('Invalid method')
            wm_diffs.append((reversed_latents_wm - true_latents_wm).abs().mean().item()) # im image domain, the mean of the abs differences of the latents is the mean per-pixel difference
            no_wm_diffs.append((reversed_latents_nowm - true_latents_nowm).abs().mean().item())


        
        
        # The whole tor/accuray calcualtion is differnt for the GRIDS WM, as it uses two metrics
        if args.method == 'grids':
            #######################################################

            gs_wm = np.array(gs_wm_metrics)
            gs_nowm = np.array(gs_nowm_metrics)
            rid_wm = np.array(rid_wm_metrics)
            rid_nowm = np.array(rid_nowm_metrics)

            # threshold ranges
            THRESHOLD_STEPS = 50 # max(50, args.num_images)
            gs_thresh_range = np.linspace(min(gs_nowm.min(), gs_wm.min()), max(gs_nowm.max(), gs_wm.max()), THRESHOLD_STEPS) # range 0 - 1
            rid_thresh_range = np.linspace(min(rid_nowm.min(), rid_wm.min()), max(rid_nowm.max(), rid_wm.max()), THRESHOLD_STEPS) # range 0 - 64

            FPR_grid = np.zeros((len(gs_thresh_range), len(rid_thresh_range)))
            TPR_grid = np.zeros((len(gs_thresh_range), len(rid_thresh_range)))

            # Grid search over thresholds
            for i, gs_thresh in enumerate(gs_thresh_range):
                for j, rid_thresh in enumerate(rid_thresh_range):

                    # Decisions using logical OR
                    nowm_decisions = (gs_nowm > gs_thresh) | (rid_nowm > rid_thresh)
                    wm_decisions = (gs_wm > gs_thresh) | (rid_wm > rid_thresh)

                    fpr = np.mean(nowm_decisions)  # False positive rate
                    tpr = np.mean(wm_decisions)    # True positive rate

                    FPR_grid[i, j] = fpr
                    TPR_grid[i, j] = tpr

            # Plot FPR, TPR with contour at FPR = 0.01
            plot_heatmaps(
                args=args,
                fpr_grid=FPR_grid,
                tpr_grid=TPR_grid,
                title_suffix=f'{attack_name}={attack_vals[strength]}',
                xticks=np.round(rid_thresh_range, 2), 
                yticks=np.round(gs_thresh_range, 2),
                levels=args.fpr,
            )

            # find max TPR where FPR <= args.fpr     
            mask = np.array(FPR_grid <= 0.01)
            masked_tpr = np.where(mask, TPR_grid, -np.inf)  # keep values >= 0, rest are not valid tpr values 

            if np.any(mask):
                best_idx = np.unravel_index(np.argmax(masked_tpr), masked_tpr.shape)
                best_tpr = TPR_grid[best_idx]
                best_gs_thresh = gs_thresh_range[best_idx[0]]
                best_rid_thresh = rid_thresh_range[best_idx[1]]
            else:
                print2file(args.log_file, "\nNo threshold pair found that achieves FPR <= 0.01")
                best_tpr = 0.0
                best_gs_thresh = gs_thresh_range[-1] # i.e ca 0.54, max(gs_thresh_range)?
                best_rid_thresh = rid_thresh_range[-1] # i.e ca -64, max(rid_thresh_range)?
                
            print2file(args.log_file, f"\nDual Metric TPR @ FPR <= 0.01: {best_tpr:.3f}")
            print2file(args.log_file, f"\nOptimal thresholds: GS={best_gs_thresh:.4f}, RID={best_rid_thresh:.4f}")

            ## UNTIL HERE: 
            # for best threshold pair calculation

            # renaming some vars to match the other methods (from below)
            low = best_tpr
            # Create binary decisions for TPR and FPR , only for the printing before the bootstrap
            no_wm_metrics = ((gs_nowm > best_gs_thresh) | (rid_nowm > best_rid_thresh)).astype(int).tolist()
            wm_metrics = ((gs_wm > best_gs_thresh) | (rid_wm > best_rid_thresh)).astype(int).tolist()
        

            #######################################################
        
            # ## calculate the TPR afor just RID metrids
            # preds_rid = rid_nowm_metrics + rid_wm_metrics
            # t_labels_rid = [0] * len(rid_nowm_metrics) + [1] * len(rid_wm_metrics)
            # fpr_rid, tpr_rid, thresholds = metrics.roc_curve(t_labels_rid, preds_rid, pos_label=1)
            # auc_rid = metrics.auc(fpr_rid, tpr_rid)
            # acc_rid = np.max(1 - (fpr_rid + (1 - tpr_rid))/2)
            # # Find the TPR at the desired FPR
            # index = np.where(fpr_rid <= args.fpr)[0][-1]
            # low_rid = tpr_rid[index]
            # threshold_rid = thresholds[index]
            # print2file(args.log_file, '\n' + '#'*10 + '\n')
            # print2file(args.log_file, f'\nTPR at fpr {args.fpr} (RID)')
            # print2file(args.log_file, f'\n\t{low_rid}')
            # print2file(args.log_file, f'\n(AUC: {auc_rid}; ACC: {acc_rid} at fpr {args.fpr})')
            # # Print all FPR, TPR, and thresholds
            # print2file(args.log_file, '\nDetailed (RID) ROC Curve Data:')
            # for f, t, th in zip(fpr_rid, tpr_rid, thresholds):
            #     print2file(args.log_file, f'FPR: {f:.3f}; TPR: {t:.3f}; Threshold: {th:.3f}')
            # print2file(args.log_file, '\n' + '#'*10 + '\n')
            # print2file(args.log_file, f'\nMean (RID) Metric for:')
            # print2file(args.log_file, f'\n\tWM: {np.mean(rid_wm_metrics)} vs NOWM: {np.mean(rid_nowm_metrics)}')
            # print2file(args.log_file, f'\nwith Threshold: {threshold_rid}')
            # print2file(args.log_file, f'\nstd WM: {np.std(rid_wm_metrics)} vs std NOWM: {np.std(rid_nowm_metrics)}')
            # print2file(args.log_file, f'\nWM metrics: {rid_wm_metrics}')
            # print2file(args.log_file, f'NOWM metrics: {rid_nowm_metrics}')

            # # calculate the TPR afor just GS metrids
            # preds = gs_nowm_metrics + gs_wm_metrics
            # t_labels = [0] * len(gs_nowm_metrics) + [1] * len(gs_wm_metrics)
            # fpr_gs, tpr_gs, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
            # auc_gs = metrics.auc(fpr_gs, tpr_gs)
            # acc_gs = np.max(1 - (fpr_gs + (1 - tpr_gs))/2)
            # # Find the TPR at the desired FPR
            # index = np.where(fpr_gs <= args.fpr)[0][-1]
            # low_gs = tpr_gs[index]
            # threshold_gs = thresholds[index]
            # print2file(args.log_file, '\n' + '#'*10 + '\n')
            # print2file(args.log_file, f'\nTPR at fpr {args.fpr} (GS)')
            # print2file(args.log_file, f'\n\t{low_gs}')
            # print2file(args.log_file, f'\n(AUC: {auc_gs}; ACC: {acc_gs} at fpr {args.fpr})')
            # # Print all FPR, TPR, and thresholds
            # print2file(args.log_file, '\nDetailed (GS) ROC Curve Data:')
            # for f, t, th in zip(fpr_gs, tpr_gs, thresholds):
            #     print2file(args.log_file, f'FPR: {f:.3f}; TPR: {t:.3f}; Threshold: {th:.3f}')
            # print2file(args.log_file, '\n' + '#'*10 + '\n')
            # print2file(args.log_file, f'\nMean (GS) Metric for:')
            # print2file(args.log_file, f'\n\tWM: {np.mean(gs_wm_metrics)} vs NOWM: {np.mean(gs_nowm_metrics)}')
            # print2file(args.log_file, f'\nwith Threshold: {threshold_gs}')
            # print2file(args.log_file, f'\nstd WM: {np.std(gs_wm_metrics)} vs std NOWM: {np.std(gs_nowm_metrics)}')
            # print2file(args.log_file, f'\nWM metrics: {gs_wm_metrics}')
            # print2file(args.log_file, f'NOWM metrics: {gs_nowm_metrics}')

            ####################################################################
        
        
        # compute the results, the empirical ROC curve
        preds = no_wm_metrics + wm_metrics
        t_labels = [0] * len(no_wm_metrics) + [1] * len(wm_metrics)
        fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        acc = np.max(1 - (fpr + (1 - tpr))/2)
        # Find the TPR at the desired FPR
        index = np.where(fpr <= args.fpr)[0][-1]
        low = tpr[index]
        threshold = thresholds[index]

        print2file(args.log_file, '\n' + '#'*10 + '\n')
        print2file(args.log_file, f'\nTPR at fpr {args.fpr} (empirical)')
        print2file(args.log_file, f'\n\t{low}')
        print2file(args.log_file, f'\n(AUC: {auc}; ACC: {acc} at fpr {args.fpr})')

        if args.method == 'grids':
            #tpr_mean, tpr_std, ci_normal, ci_percentile = bootstrap_grids_tpr(gs_nowm, gs_wm, rid_nowm, rid_wm, best_gs_thresh, best_rid_thresh, args.fpr)
            tpr_mean, tpr_std, ci_normal, ci_percentile = bootstrap_grids_dynamic_thresholds(gs_nowm, gs_wm, rid_nowm, rid_wm, args.fpr, n_thresholds=50)
        else:
            tpr_mean, tpr_std, ci_normal, ci_percentile = bootstrap_tpr(no_wm_metrics, wm_metrics, args.fpr)

        print2file(args.log_file, f'\nTPR at fpr {args.fpr} (empirical mean): {tpr_mean:.4f}')
        print2file(args.log_file, f'Standard Error: {tpr_std:.4f}')
        print2file(args.log_file, f'95% CI (Normal Approximation): [{ci_normal[0]:.4f}, {ci_normal[1]:.4f}]')
        print2file(args.log_file, f'95% CI (Percentile Method): [{ci_percentile[0]:.4f}, {ci_percentile[1]:.4f}]')
        
        # Print all FPR, TPR, and thresholds
        print2file(args.log_file, '\nDetailed (empirical) ROC Curve Data:')
        for f, t, th in zip(fpr, tpr, thresholds):
            print2file(args.log_file, f'FPR: {f:.3f}; TPR: {t:.3f}; Threshold: {th:.3f}')
        
        tpr_detection = None
        tpr_traceability = None
        tpr_decode = None
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

        # additional stats
        print2file(args.log_file, '\n' + '#'*10 + '\n')
        print2file(args.log_file, f'\nMean Metric for:')
        print2file(args.log_file, f'\n\tWM: {np.mean(wm_metrics)} vs NOWM: {np.mean(no_wm_metrics)}')
        print2file(args.log_file, f'\nwith Threshold: {threshold}')
        print2file(args.log_file, f'\nstd WM: {np.std(wm_metrics)} vs std NOWM: {np.std(no_wm_metrics)}')
        print2file(args.log_file, f'\nWM metrics: {wm_metrics}')
        print2file(args.log_file, f'NOWM metrics: {no_wm_metrics}')
        print2file(args.log_file, f'\n\nMean per-pixel difference between true and reversed latents for:')
        print2file(args.log_file, f'\n\tWM: {np.mean(wm_diffs)} vs NOWM: {np.mean(no_wm_diffs)}')
        print2file(args.log_file, f'\nstd WM: {np.std(wm_diffs)} vs std NOWM: {np.std(no_wm_diffs)}')

        # plot the ROC curve
        plt.figure()
        plt.plot(fpr, tpr)
        # mark empriical TPR at FPR=args.fpr
        plt.scatter(fpr[index], low, color='red', label=f'TPR at FPR={args.fpr}={low:.3f}')
        plt.grid()
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC Curve for {args.method} with {attack_name}={attack_vals[strength]} \n with TPR@FPR={args.fpr} = {low:.3f}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.log_dir, f'roc_{attack_name}_{attack_vals[strength]}.pdf'))
        plt.close()

        clip_scores_wm = []
        clip_scores_nowm = []
        if args.calc_CLIP:
            # also loop per-image, to calculate the clip scores
            for i in tqdm(range(args.num_images)):
                img_wm = Image.open(os.path.join(path_attacked_wm, f'{i}.png'))
                img_nowm = Image.open(os.path.join(path_attacked_nowm, f'{i}.png'))
                sims = measure_similarity([img_nowm, img_wm], prompts[i], ref_model, ref_clip_preprocess, ref_tokenizer, device)
                clip_scores_nowm.append(sims[0].item())
                clip_scores_wm.append(sims[1].item())

        fid_wm_coco = None
        fid_nowm_coco = None
        fid_wm_nowm = None
        clip_score_wm = None
        clip_score_nowm = None
        
        if args.calc_CLIP:
            # calculate CLIP score between the generated images with and without watermark to the prompt with the reference model
            clip_score_wm = np.mean(clip_scores_wm)
            clip_score_nowm = np.mean(clip_scores_nowm)
            print2file(args.log_file, '\n' + '#'*10 + '\n')
            print2file(args.log_file, f'\nCLIP scores for:')
            print2file(args.log_file, f'\n\tWM: {clip_score_wm:.4f} vs NOWM: {clip_score_nowm:.4f}')

        if args.calc_FID:
            # measure the FID between original and attacked images, both with and without watermark
            fid_wm_coco = calculate_fid_given_paths([path_attacked_wm, '/is/sg2/mkaut/ma-thesis/coco/val2017'], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048,
                                                    max_samples=args.num_images)
            fid_nowm_coco = calculate_fid_given_paths([path_attacked_nowm, '/is/sg2/mkaut/ma-thesis/coco/val2017'], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048,
                                                    max_samples=args.num_images)
            fid_wm_nowm = calculate_fid_given_paths([path_attacked_wm, path_attacked_nowm], 
                                                    batch_size=50, 
                                                    device=device, 
                                                    dims=2048,
                                                    max_samples=args.num_images)
            print2file(args.log_file, '\n' + '#'*10 + '\n')
            print2file(args.log_file, f'\nFID (distances) :')
            print2file(args.log_file, f'\n\tagainst COCO for WM: {fid_wm_coco:.4f} vs NOWM: {fid_nowm_coco:.4f}')
            print2file(args.log_file, f'\n\tWM against NOWM: {fid_wm_nowm:.4f}')
            
        # collect results of one 'decode run' in a dictionary
        results = {
            'wm_method': args.method,
            'model_id': args.model_id,
            'dataset_id': args.dataset_id,  
            'attack_type': attack_type,
            'attack_name': attack_name,
            'attack_strength': attack_vals[strength],
            'tpr_empirical': low,
            'tpr_empirical_mean': tpr_mean,
            'tpr_std_error': tpr_std,
            'tpr_ci_lower_percentile': ci_percentile[0],
            'tpr_ci_upper_percentile': ci_percentile[1], 
            'auc': auc,
            'acc': acc,
            'tpr_analytical': tpr_detection,
            'tpr_decode': tpr_decode,
            'tpr_traceability': tpr_traceability,
            'threshold': threshold,
            'mean_wm_dist': np.mean(wm_metrics),    
            'mean_no_wm_dist': np.mean(no_wm_metrics),
            'wm_diff': np.mean(wm_diffs),
            'nowm_diff': np.mean(no_wm_diffs),
            'clip_score_wm': clip_score_wm,
            'clip_score_nowm': clip_score_nowm,
            'fid_wm_coco': fid_wm_coco, # technically, FID is not a score, but a distance
            'fid_nowm_coco': fid_nowm_coco,
            'fid_wm_nowm': fid_wm_nowm,
            'set_fpr': args.fpr,
            'wm_ch': args.latent_channels_wm,
            'inf_steps': args.inf_steps,
            'test_inf_steps': args.test_inf_steps,
            'guidance_scale': args.guidance_scale,
        }

        # save the results to existing dataframe
        df_new = pd.DataFrame([results])
        results_df = pd.concat([results_df, df_new], ignore_index=True)
        results_df.to_csv(os.path.join(args.log_dir, f'results_{args.run_name}.csv'), index=False)

        print2file(args.log_file, '\n\n' + '#'*100 + '\n')

    print2file(args.log_file, '\nFINISHED JOB\n')




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