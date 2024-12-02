"""
For PRC watermarking Only, will add Tree-Ring and Gaussian Shading watermarking later
"""

import argparse
import os
import pickle
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from src.prc import Detect, Decode
import src.pseudogaussians as prc_gaussians
from src.baseline.treering_watermark import tr_detect
from src.optim_utils import set_random_seed, transform_img, get_dataset, image_distortion
from src.baseline.gs_watermark import Gaussian_Shading_chacha
from inversion import stable_diffusion_pipe, exact_inversion

parser = argparse.ArgumentParser('Args')
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc') # gs, tr, prc
parser.add_argument('--model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
parser.add_argument('--dataset_id', type=str, default='Gustavosta/Stable-Diffusion-Prompts')
parser.add_argument('--inf_steps', type=int, default=50)
parser.add_argument('--nowm', type=int, default=0)
parser.add_argument('--fpr', type=float, default=0.00001)
parser.add_argument('--prc_t', type=int, default=3)

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
parser.add_argument('--run_name', type=str, default='decode_test')
args = parser.parse_args()
print(args)

hf_cache_dir = '/home/mkaut/.cache/huggingface/hub'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n = 4 * 64 * 64  # the length of a PRC codeword
method = args.method
test_num = args.test_num
model_id = args.model_id
dataset_id = args.dataset_id
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
exp_id = f'{method}_num_{test_num}_steps_{args.inf_steps}_fpr_{fpr}_nowm_{nowm}'

if method == 'prc':
    with open(f'keys/{exp_id}.pkl', 'rb') as f:
        encoding_key, decoding_key = pickle.load(f)
    print(f'Loaded PRC keys from file keys/{exp_id}.pkl')
elif method == 'gs':
    gs_watermark = Gaussian_Shading_chacha(ch_factor=1, hw_factor=8, fpr=fpr, user_number=10000)
    
    with open(f'keys/{exp_id}.pkl', 'rb') as f:
        watermark_m, key, nonce, watermark = pickle.load(f)
        print(f'Loaded GS keys from file keys/{exp_id}.pkl')
    gs_watermark.watermark = watermark
    gs_watermark.key = key
    gs_watermark.nonce = nonce

elif method == 'tr':
    tr_key = 'ef913efe2dbcd0b23067e96dd115a88dcc5ac379748bdc2cab466a77bea8e458_2_10_ring'
    with open(os.path.join("keys", tr_key + '.pkl'), 'rb') as f:
        w_key, w_mask = pickle.load(f)

pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=hf_cache_dir)
pipe.set_progress_bar_config(disable=True)

#
print('Loading imgs from', f'results/{exp_id}/{args.test_path}')
#
print('\nArgs:\n')
for arg in vars(args):
    print(f'{arg}: {getattr(args, arg)}')

if method == 'prc':
    cur_inv_order = 0
    var = 1.5
    combined_results = []
   
    for i in tqdm(range(test_num)):
        img = Image.open(f'results/{exp_id}/{args.test_path}/{i}.png')
        
        # manual detection

        # distortion
        seed = 42
        img_w_auged, img_w_auged2 = image_distortion(img, img, seed, args)

        reversed_latents = exact_inversion(img_w_auged,
                                        prompt='',
                                        test_num_inference_steps=args.inf_steps,
                                        inv_order=cur_inv_order,
                                        pipe=pipe
                                        )
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(var)).flatten().cpu()
        detection_result = Detect(decoding_key, reversed_prc)
        decoding_result = (Decode(decoding_key, reversed_prc) is not None)
        combined_result = detection_result or decoding_result
        combined_results.append(combined_result)
        print(f'image {i:03d}: Detection: {detection_result}; Decoding: {decoding_result}; Combined: {combined_result}')

    with open('decoded.txt', 'w') as f:
        for result in combined_results:
            f.write(f'{result}\n')

    print(f'Decoded results saved to decoded.txt')

    tpr_detection = sum(combined_results) / len(combined_results)
    print(f'TPR Detection: {tpr_detection} at fpr {fpr}' )

elif method == 'gs':
    
    for i in tqdm(range(test_num)):
        img = Image.open(f'results/{exp_id}/{args.test_path}/{i}.png')
        
        # manual detection
        tester_prompt = ''
        text_embeddings = pipe.get_text_embedding(tester_prompt)
        # reverse img with watermarking
        # distortion
        seed = 42
        img_w_auged, img_w_auged2 = image_distortion(img, img, seed, args)


        img_w = transform_img(img_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.inf_steps,
        )

        acc = gs_watermark.eval_watermark(reversed_latents_w)

        tp_onebit_count, tp_bits_count = gs_watermark.get_tpr()

        print(f'image {i:03d}: Acc: {acc}')
    
    print(f'TPR Onebit: {tp_onebit_count / test_num}; TPR Bits: {tp_bits_count / test_num}')
    print(f'at fpr {fpr}')



            
elif method == 'tr':
    threshold = 72
    combined_results = []
    w_metrics = []
    # for i in tqdm(range(test_num)):
    for i in tqdm(range(10)):
        img = Image.open(f'results/{exp_id}/{args.test_path}/{i}.png')
        
        # manual detection
        tester_prompt = '' # assume at the detection time, the original prompt is unknown
        text_embeddings = pipe.get_text_embedding(tester_prompt)

        # distortion
        seed = 42
        img_w_auged, img_w_auged2 = image_distortion(img, img, seed, args)

        # reverse img with watermarking
        img_w = transform_img(img_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.inf_steps,
        )

        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2)).cpu()
        
        dist = torch.abs(reversed_latents_w_fft[w_mask] - w_key[w_mask]).mean().item()
        w_metrics.append(dist)

        if dist <= threshold:
            result = True
        else:
            result = False

        combined_results.append(result)

        print(f'image {i:03d}: Dist: {dist}; Detection: {result}')

    # tp = sum(combined_results)
    # tn = test_num - tp
    # tnr = tn / test_num
    # tpr = tp / test_num
    # fpr = 1 - tnr

    preds = w_metrics
    t_labels = [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<args.fpr)[0][-1]]
    threshold = thresholds[np.where(fpr<args.fpr)[0][-1]]

    print(f'TPR: {low}; AUC: {auc}; ACC: {acc} at fpr {args.fpr}')
    print(f'Average Dist: {np.mean(w_metrics)}')
    print(f'threshold: {threshold}')
    


    ### implement testing againt noWM images in all 3 types of watermarking, to test 
    ### the actual false positive rate FPR, which should be close to the specified FPR in the 
    ### th PRC and GS watermarking methods, and we need to calculate the actual FPR in the TR watermarking method.