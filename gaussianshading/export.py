# here we define a class that contains all important funtions to encode and decode images with watermarks, 
# so that we can compare it better to other methods on a higher level

import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from .watermark import Gaussian_Shading_chacha, Gaussian_Shading
from .inverse_stable_diffusion import InversableStableDiffusionPipeline
from .image_utils import transform_img, set_random_seed
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

class GSWatermark:
    def __init__(self, 
                 args,
                 hf_cache_dir='/home/mkaut/.cache/huggingface/hub'
    ):
        
        self.model_id = args.model_id
        self.inf_steps = args.inf_steps
        self.test_inf_steps = args.test_inf_steps
        self.fpr = args.fpr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hf_cache_dir = hf_cache_dir
        self.method = 'gs'
        self.num_images = args.num_images
        self.guidance_scale = args.guidance_scale
        self.watermark_m = None
        self.chacha = args.gs_chacha
        self.ch_factor = args.gs_ch_factor
        self.hw_factor = args.gs_hw_factor
        self.user_number = args.gs_user_number
        self.args = args

        # the exp_id is used to save the keys or to load existing keys (if object is only created for decoding)
        self.exp_id = f'{self.method}_num_{self.num_images}_steps_{self.inf_steps}_fpr_{self.fpr}'
        
        if self.chacha: # if we use chacha20 encryption, we get a nonce
            self.gs = Gaussian_Shading_chacha(self.ch_factor, self.hw_factor, self.fpr, self.user_number)
            if not os.path.exists(f'keys/{self.exp_id}.pkl'):
                watermark_m_ori, key_ori, nonce_ori, watermark_ori = self.gs.create_watermark_and_return_w()
                with open(f'keys/{self.exp_id}.pkl', 'wb') as f:
                    pickle.dump((watermark_m_ori, key_ori, nonce_ori, watermark_ori), f)
                with open(f'keys/{self.exp_id}.pkl', 'rb') as f:
                    watermark_m, key, nonce, watermark = pickle.load(f)
                assert watermark_m.all() == watermark_m_ori.all()
                self.watermark_m = watermark_m
                self.gs.watermark = watermark
                self.gs.key = key
                self.gs.nonce = nonce
                print(f"Generated watermark and saved it to file keys/{self.exp_id}.pkl")
            else:
                with open(f'keys/{self.exp_id}.pkl', 'rb') as f:
                    watermark_m, key, nonce, watermark = pickle.load(f)
                self.watermark_m = watermark_m
                self.gs.watermark = watermark
                self.gs.key = key
                self.gs.nonce = nonce
                print(f"Loaded watermark from file keys/{self.exp_id}.pkl")
        else: 
            self.gs = Gaussian_Shading(self.ch_factor, self.hw_factor, self.fpr, self.user_number)
            if not os.path.exists(f'keys/{self.exp_id}.pkl'):
                watermark_m_ori, key_ori, watermark_ori = self.gs.create_watermark_and_return_w()
                with open(f'keys/{self.exp_id}.pkl', 'wb') as f:
                    pickle.dump((watermark_m_ori, key_ori, watermark_ori), f)
                with open(f'keys/{self.exp_id}.pkl', 'rb') as f:
                    watermark_m, key, watermark = pickle.load(f)
                assert watermark_m.all() == watermark_m_ori.all()
                self.watermark_m = watermark_m
                self.gs.watermark = watermark
                self.gs.key = key
                print(f"Generated watermark and saved it to file keys/{self.exp_id}.pkl")
            else:
                with open(f'keys/{self.exp_id}.pkl', 'rb') as f:
                    watermark_m, key, watermark = pickle.load(f)
                self.watermark_m = watermark_m
                self.gs.watermark = watermark
                self.gs.key = key
                print(f"Loaded watermark from file keys/{self.exp_id}.pkl")
                

        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                self.model_id,
                scheduler=scheduler,
                # torch_dtype=torch.float16,
                # revision='fp16',
                torch_dtype=torch.float32,
                cache_dir=self.hf_cache_dir,

        ).to(self.device)
        # self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

        self.visualize_watermark_pattern()

    def visualize_watermark_pattern(self):
        
        set_random_seed(1) # will generata the same latents as the first image will have
        init_latents = self.gs.truncSampling(self.watermark_m)

        print("min max watermark pattern", init_latents.min(), init_latents.max())

        fig, ax = plt.subplots(1, 4, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(4):
            ax[i].axis('off')
        for i in range(4):
            ax[i].imshow(init_latents[0, i].real.cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
        ax[0].set_title('Watermark pattern', loc='left', fontsize=10)
        fig.suptitle(f'GS Watermark pattern', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_wm_only.png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)


    ############################# ENCODING ########################################
    def generate_img(self, prompt, nowm, num_images_per_prompt=1):
        if nowm:
            init_latents_np = np.random.randn(1, 4, 64, 64)
            init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        else:
            init_latents = self.gs.truncSampling(self.watermark_m) # inside this, is specifically set to .half()
            init_latents = init_latents.to(torch.float32).to(self.device)
    

        outputs = self.pipe(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.inf_steps,
            height=512,
            width=512,
            latents=init_latents,
        )
        image_w = outputs.images[0]
        return image_w
    
    ############################# DECODING ########################################
    def get_inversed_latents(self, img, prompt=''):
        embedded_prompt = self.pipe.get_text_embedding(prompt)

        img = transform_img(img).unsqueeze(0).to(embedded_prompt.dtype).to(self.device)
        img_latents = self.pipe.get_image_latents(img, sample=False)
        reversed_latents = self.pipe.forward_diffusion(
            latents=img_latents,
            text_embeddings=embedded_prompt,
            guidance_scale=1,
            num_inference_steps=self.test_inf_steps,
        )
        return reversed_latents
    
    def viz_reversed_latents(self, reversed_latents_no_w, reversed_latents_w, attack_name=None, attack_vals=[None], strength=0):
        set_random_seed(1)
        pattern_latents = self.gs.truncSampling(self.watermark_m)
        
        diff_wm = reversed_latents_w - pattern_latents
        diff_no_wm = reversed_latents_no_w - pattern_latents

    

        
        
        no_w_metric = self.gs.eval_watermark(reversed_latents_no_w, count=False)
        w_metric = self.gs.eval_watermark(reversed_latents_w, count=False)

        
        fig, ax = plt.subplots(2, 4, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        for i in range(4):
            ax[0, i].imshow(reversed_latents_no_w[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[1, i].imshow(diff_no_wm[0, i].cpu().numpy(), cmap='RdBu',vmin=-10, vmax=10)
        ax[0, 0].set_title('Reversed latents', loc='left', fontsize=10)
        ax[1, 0].set_title(f'abs. avg. diff to WM pattern: {diff_no_wm.abs().mean()} with metric {no_w_metric:.2f}', loc='left', fontsize=10)
        fig.suptitle(f'Reversed no_wm latents of {self.method} with attack {attack_name}: {attack_vals[strength]}', fontsize=12)
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
            ax[1, i].imshow(diff_wm[0, i].cpu().numpy(), cmap='RdBu',vmin=-10, vmax=10)
        ax[0, 0].set_title('Reversed latents', loc='left', fontsize=10)
        ax[1, 0].set_title(f'abs. avg. diff to WM pattern: {diff_wm.abs().mean()} with metric {w_metric:.2f}', loc='left', fontsize=10)
        fig.suptitle(f'Reversed WM latents of {self.method} with attack {attack_name}: {attack_vals[strength]}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_reversed_wm_latents_{attack_name}_{attack_vals[strength]}.png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)