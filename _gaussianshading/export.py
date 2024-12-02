# here we define a class that contains all important funtions to encode and decode images with watermarks, 
# so that we can compare it better to other methods on a higher level

import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
import pickle
import os
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
from .watermark import Gaussian_Shading_chacha, Gaussian_Shading
from .inverse_stable_diffusion import InversableStableDiffusionPipeline
from .image_utils import transform_img
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler

class GSWatermark:
    def __init__(self, 
                 model_id='stabilityai/stable-diffusion-2-1-base',
                 inf_steps=50,
                 test_num_inference_steps=50,
                 fpr=0.01,
                 num_images=10,
                 chacha=True,
                 ch_factor=1,
                 hw_factor=8,
                 user_number=10000,
                 guidance_scale=3.0
    ):
        
        self.model_id = model_id
        self.inf_steps = inf_steps
        self.test_num_inference_steps = test_num_inference_steps
        self.fpr = fpr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hf_cache_dir = '/home/mkaut/.cache/huggingface/hub'
        self.method = 'gs'
        self.num_images = num_images
        self.guidance_scale = guidance_scale
        self.watermark_m = None

        # the exp_id is used to save the keys or to load existing keys (if object is only created for decoding)
        self.exp_id = f'{self.method}_num_{self.num_images}_steps_{self.inf_steps}_fpr_{self.fpr}'
        
        if chacha: # if we use chacha20 encryption, we get a nonce
            self.gs = Gaussian_Shading_chacha(ch_factor, hw_factor, fpr, user_number)
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
            self.gs = Gaussian_Shading(ch_factor, hw_factor, fpr, user_number)
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
                

        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                model_id,
                scheduler=scheduler,
                # torch_dtype=torch.float16,
                # revision='fp16',
                torch_dtype=torch.float32,
                cache_dir=self.hf_cache_dir,

        ).to(self.device)
        # self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)


    ############################# ENCODING ########################################
    def generate_img(self, prompt, nowm, num_images_per_prompt=1):
        if nowm:
            init_latents_np = np.random.randn(1, 4, 64, 64)
            init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
            print('Encoding without watermark')
        else:
            init_latents = self.gs.truncSampling(self.watermark_m) # inside this, is specifically set to .half()
            init_latents = init_latents.to(torch.float32).to(self.device)
            print('Encoding with watermark')
    

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
            num_inference_steps=self.test_num_inference_steps,
        )
        return reversed_latents
    
    