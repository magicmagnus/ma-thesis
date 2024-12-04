# here we define a class that contains all important funtions to encode and decode images with watermarks, 
# so that we can compare it better to other methods on a higher level

import torch
import copy
import numpy as np
import pickle
import os

from diffusers import DPMSolverMultistepScheduler
from .inverse_stable_diffusion import InversableStableDiffusionPipeline
from .optim_utils import set_random_seed, circle_mask, transform_img


class TRWatermark():
    def __init__(self, 
                 model_id='stabilityai/stable-diffusion-2-1-base',
                 inf_steps=50,
                 test_num_inference_steps=50,
                 fpr=0.01,
                 num_images=10,
                 guidance_scale=3.0,
                 w_seed=0,
                 w_channel=0,
                 w_pattern='rand',
                 w_mask_shape='circle',
                 w_radius=10,
                 w_measurement='l1_complex',
                 w_injection='complex',
                 w_pattern_const=0,
                 hf_cache_dir='/home/mkaut/.cache/huggingface/hub'
    ):
        self.model_id = model_id
        self.inf_steps = inf_steps
        self.test_num_inference_steps = test_num_inference_steps
        self.fpr = fpr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hf_cache_dir = hf_cache_dir
        self.method = 'tr'
        self.num_images = num_images
        self.guidance_scale = guidance_scale
        self.shape = (1, 4, 64, 64)

        self.exp_id = f'{self.method}_num_{self.num_images}_steps_{self.inf_steps}_fpr_{self.fpr}'
        scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            # torch_dtype=torch.float16,
            # revision='fp16',
            torch_dtype=torch.float32,
            cache_dir=self.hf_cache_dir,

            ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.w_seed = w_seed
        self.w_channel = w_channel
        self.w_pattern = w_pattern
        self.w_mask_shape = w_mask_shape
        self.w_radius = w_radius
        self.w_measurement = w_measurement
        self.w_injection = w_injection
        self.w_pattern_const = w_pattern_const

        self.gt_patch = None
        self.watermarking_mask = None

        # if the key/pattern (and mask) already exists, we load them, otherwise we generate them
        if not os.path.exists(f'keys/{self.exp_id}.pkl'):
            self.generate_watermarking_pattern()
            self.get_watermarking_mask()
            with open(f'keys/{self.exp_id}.pkl', 'wb') as f:
                pickle.dump((self.gt_patch, self.watermarking_mask), f)
            print(f'Generated TR keys and saved to file keys/{self.exp_id}.pkl')
        else:
            with open(f'keys/{self.exp_id}.pkl', 'rb') as f:
                self.gt_patch, self.watermarking_mask = pickle.load(f)
            print(f'Loaded TR keys from file keys/{self.exp_id}.pkl')
        
    ############################# ENCODING ########################################
    def generate_watermarking_pattern(self):
        set_random_seed(self.w_seed)
        if self.shape is not None:
            gt_init = torch.randn(*self.shape, device=self.device)
        else:
            gt_init = self.pipe.get_random_latents()

        if 'seed_ring' in self.w_pattern:
            gt_patch = gt_init

            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(self.device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
        elif 'seed_zeros' in self.w_pattern:
            gt_patch = gt_init * 0
        elif 'seed_rand' in self.w_pattern:
            gt_patch = gt_init
        elif 'rand' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
            gt_patch[:] = gt_patch[0]
        elif 'zeros' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        elif 'const' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
            gt_patch += self.w_pattern_const
        elif 'ring' in self.w_pattern:
            gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

            gt_patch_tmp = copy.deepcopy(gt_patch)
            for i in range(self.w_radius, 0, -1):
                tmp_mask = circle_mask(gt_init.shape[-1], r=i)
                tmp_mask = torch.tensor(tmp_mask).to(self.device)
                
                for j in range(gt_patch.shape[1]):
                    gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

        self.gt_patch = gt_patch

    def get_watermarking_mask(self):
        watermarking_mask = torch.zeros(self.shape, dtype=torch.bool).to(self.device)

        if self.w_mask_shape == 'circle':
            np_mask = circle_mask(self.shape[-1], r=self.w_radius)
            torch_mask = torch.tensor(np_mask).to(self.device)

            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :] = torch_mask
            else:
                watermarking_mask[:, self.w_channel] = torch_mask
        elif self.w_mask_shape == 'square':
            anchor_p = self.shape[-1] // 2
            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :, anchor_p-self.w_radius:anchor_p+self.w_radius, anchor_p-self.w_radius:anchor_p+self.w_radius] = True
            else:
                watermarking_mask[:, self.w_channel, anchor_p-self.w_radius:anchor_p+self.w_radius, anchor_p-self.w_radius:anchor_p+self.w_radius] = True
        elif self.w_mask_shape == 'no':
            pass
        else:
            raise NotImplementedError(f'w_mask_shape: {self.w_mask_shape}')
        
        self.watermarking_mask = watermarking_mask

    def inject_watermark(self, init_latents):
        init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        if self.w_injection == 'complex':
            init_latents_w_fft[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
        elif self.w_injection == 'seed':
            init_latents[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()
            return init_latents
        else:
            NotImplementedError(f'w_injection: {self.w_injection}')

        init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

        return init_latents
    
    def generate_img(self, current_prompt, nowm, num_images_per_prompt=1):
        # set_random_seed(self.w_seed)
        
        init_latents_np = np.random.randn(1, 4, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        
        if not nowm: # if we want to inject watermark
            init_latents = self.inject_watermark(init_latents)

        outputs = self.pipe(
            current_prompt,
            num_images_per_prompt=num_images_per_prompt, 
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.inf_steps,
            height=512,
            width=512,
            latents=init_latents,
            )
        orig_image = outputs.images[0]

        return orig_image

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
    
    def eval_watermark(self, reversed_latents_no_w, reversed_latents_w):
        if 'complex' in self.w_measurement:
            reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
            reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
            target_patch = self.gt_patch
        elif 'seed' in self.w_measurement:
            reversed_latents_no_w_fft = reversed_latents_no_w
            reversed_latents_w_fft = reversed_latents_w
            target_patch = self.gt_patch
        else:
            NotImplementedError(f'w_measurement: {self.w_measurement}')

        if 'l1' in self.w_measurement:
            no_w_metric = torch.abs(reversed_latents_no_w_fft[self.watermarking_mask] - target_patch[self.watermarking_mask]).mean().item()
            w_metric = torch.abs(reversed_latents_w_fft[self.watermarking_mask] - target_patch[self.watermarking_mask]).mean().item()
        else:
            NotImplementedError(f'w_measurement: {self.w_measurement}')

        return no_w_metric, w_metric


                 