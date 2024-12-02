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
                model_id='stabilityai/stable-diffusion-2-1-base',
                inf_steps=50,
                test_num_inference_steps=50,
                fpr=0.01,
                prc_t=3,
                num_images=10,
                guidance_scale=3.0
    ):
        self.model_id = model_id
        self.inf_steps = inf_steps
        self.test_num_inference_steps = test_num_inference_steps
        self.fpr = fpr
        self.prc_t = prc_t
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hf_cache_dir = '/home/mkaut/.cache/huggingface/hub'
        self.method = 'prc' # only for path in exp_id
        self.num_images = num_images # only for path in exp_id
        self.n = 4 * 64 * 64  # the length of a PRC codeword
        self.cur_inv_order = 0
        self.var = 1.5
        self.guidance_scale = guidance_scale
        # the exp_id is used to save the keys or to load existing keys (if object is only created for decoding)
        self.exp_id = f'{self.method}_num_{self.num_images}_steps_{self.inf_steps}_fpr_{self.fpr}'
        self.encoding_key = None
        self.decoding_key = None

        self.generate_keys()
        self.pipe = stable_diffusion_pipe(solver_order=1, model_id=model_id, cache_dir=self.hf_cache_dir)
        self.pipe.set_progress_bar_config(disable=True)

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
            print('Encoding without watermark')
        else:
            init_latents = self.get_encoded_latents(message)
            print('Encoding with watermark')
        
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
                                           test_num_inference_steps=self.test_num_inference_steps,
                                           inv_order=self.cur_inv_order,
                                           pipe=self.pipe)
        
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(self.var)).flatten().cpu()
        return reversed_prc
    
    def detect_watermark(self, latents):
        return Detect(self.decoding_key, latents)
    
    def decode_watermark(self, latents):
        return (Decode(self.decoding_key, latents) is not None)
    
    def detect_and_decode_watermark(self, img, prompt=''):
        reversed_prc = self.get_inversed_latents(img, prompt)
        return (self.detect_watermark(reversed_prc), self.decode_watermark(reversed_prc))