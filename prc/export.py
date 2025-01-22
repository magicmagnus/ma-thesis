# here we define a class that contains all important funtions to encode and decode images with watermarks, 
# so that we can compare it better to other methods on a higher level

# with it, we should be able to perfome everything that we did in the encode.py and decode.py files

import os
import argparse
import torch
import pickle
import json
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

from src.prc import KeyGen, Encode, str_to_bin, bin_to_str, Detect, Decode
import src.pseudogaussians as prc_gaussians
from src.optim_utils import set_random_seed, transform_img, get_dataset, image_distortion
from inversion import stable_diffusion_pipe, exact_inversion, generate

class PRCWatermark():
    def __init__(self,
                args,
                hf_cache_dir='/home/mkaut/.cache/huggingface/hub'
    ):
        self.model_id = args.model_id
        self.inf_steps = args.inf_steps
        self.test_inf_steps = args.test_inf_steps
        self.fpr = args.fpr
        self.prc_t = args.prc_t
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hf_cache_dir = hf_cache_dir
        self.method = 'prc' # only for path in exp_id
        self.num_images = args.num_images # only for path in exp_id
        self.n = 4 * 64 * 64  # the length of a PRC codeword
        self.cur_inv_order = 0
        self.var = 1.5
        self.guidance_scale = args.guidance_scale
        # the exp_id is used to save the keys or to load existing keys (if object is only created for decoding)
        self.exp_id = f'{self.method}_num_{self.num_images}_steps_{self.inf_steps}_fpr_{self.fpr}'
        self.encoding_key = None
        self.decoding_key = None
        self.args = args

        self.generate_keys()
        self.pipe = stable_diffusion_pipe(solver_order=1, model_id=self.model_id, cache_dir=self.hf_cache_dir)
        self.pipe.set_progress_bar_config(disable=True)

        self.visualize_watermark_pattern()

    def visualize_watermark_pattern(self):
        set_random_seed(1)
        init_latents = self.get_encoded_latents()

        fig, ax = plt.subplots(1, 4, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(4):
            ax[i].axis('off')
        for i in range(4):
            ax[i].imshow(init_latents[0, i].real.cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
        ax[0].set_title('Watermark pattern', loc='left', fontsize=10)
        fig.suptitle(f'PRC Watermark pattern', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_wm_only.png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)


    def generate_keys(self):
        if not os.path.exists(f'keys/{self.exp_id}.pkl'):
            (self.encoding_key_ori, self.decoding_key_ori) = KeyGen(self.n, false_positive_rate=self.fpr, t=self.prc_t)
            with open(f'keys/{self.exp_id}.pkl', 'wb') as f:
                pickle.dump((self.encoding_key_ori, self.decoding_key_ori), f)
            with open(f'keys/{self.exp_id}.pkl', 'rb') as f:
                self.encoding_key, self.decoding_key = pickle.load(f)
            assert self.encoding_key[0].all() == self.encoding_key_ori[0].all()
            print(f'Generated PRC keys and saved to file keys/{self.exp_id}.pkl')
        else:
            with open(f'keys/{self.exp_id}.pkl', 'rb') as f:
                self.encoding_key, self.decoding_key = pickle.load(f)
            print(f'Loaded PRC keys from file keys/{self.exp_id}.pkl')
    
    ############################# ENCODING ########################################
    def get_encoded_latents(self, message=None):
        self.prc_codeword = Encode(self.encoding_key, message)
        init_latents = prc_gaussians.sample(self.prc_codeword).reshape(1, 4, 64, 64).to(self.device)
        return init_latents
    
    def generate_img(self, prompt, nowm, message=None, num_images_per_prompt=1):
        if nowm:
            init_latents_np = np.random.randn(1, 4, 64, 64)
            init_latents = torch.from_numpy(init_latents_np).to(torch.float64).to(self.device)
        else:
            init_latents = self.get_encoded_latents(message)
        
        img, _, _ = generate(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            init_latents=init_latents,
            num_inference_steps=self.inf_steps,
            solver_order=1,
            pipe=self.pipe,
            guidance_scale=self.guidance_scale
        )
        return img
    
    ############################# DECODING ########################################
    def get_inversed_latents(self, img, prompt=''):
        reversed_latents = exact_inversion(img, 
                                           prompt, 
                                           test_num_inference_steps=self.test_inf_steps,
                                           inv_order=self.cur_inv_order,
                                           pipe=self.pipe)
        
        
        return reversed_latents
    
    def detect_watermark(self, reversed_latents):
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(self.var)).flatten().cpu()
        return Detect(self.decoding_key, reversed_prc)
    
    def decode_watermark(self, reversed_latents):
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(self.var)).flatten().cpu()
        return Decode(self.decoding_key, reversed_prc)
    
    def viz_reversed_latents(self, reversed_latents_no_w, reversed_latents_w, attack_name=None, attack_vals=[None], strength=0):
        set_random_seed(1)
        init_latents = self.get_encoded_latents()

        diff_wm = reversed_latents_w - init_latents
        diff_no_wm = reversed_latents_no_w - init_latents

        _, metric_wm, _ = self.detect_watermark(reversed_latents_w)
        _, metric_no_wm, _ = self.detect_watermark(reversed_latents_no_w)


        fig, ax = plt.subplots(2, 4, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        for i in range(4):
            ax[0, i].imshow(reversed_latents_no_w[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[1, i].imshow(diff_no_wm[0, i].cpu().numpy(), cmap='RdBu', vmin=-10, vmax=10)
        ax[0, 0].set_title('Reversed latents', loc='left', fontsize=10)
        ax[1, 0].set_title(f'abs. avg. diff to WM pattern: {diff_no_wm.abs().mean():.3f} with metric {metric_no_wm:.2f}', loc='left', fontsize=10)
        fig.suptitle(f'PRC Watermark decoding without watermark', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_reversed_no_wm_latents_{attack_name}_{attack_vals[strength]}.png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

        fig, ax = plt.subplots(2, 4, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        for i in range(4):
            ax[0, i].imshow(reversed_latents_w[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[1, i].imshow(diff_wm[0, i].cpu().numpy(), cmap='RdBu', vmin=-10, vmax=10)
        ax[0, 0].set_title('Reversed latents', loc='left', fontsize=10)
        ax[1, 0].set_title(f'abs. avg. diff to WM pattern: {diff_wm.abs().mean():.3f} with metric {metric_wm:.2f}', loc='left', fontsize=10)
        fig.suptitle(f'PRC Watermark decoding with watermark', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_reversed_wm_latents_{attack_name}_{attack_vals[strength]}.png', bbox_inches='tight', pad_inches=0.2)

        plt.close(fig)

    