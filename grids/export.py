import os
import sys
import copy
import torch
import pickle
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything, transform_img, plot_wm_pattern_fft, plot_wm_latents_fft, visualize_reversed_latents_spatial_domain

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipes.inverse_flux_pipeline import InversableFluxPipeline
from pipes.inverse_stable_diffusion import InversableStableDiffusionPipeline

sys.path.append(os.path.join(os.path.dirname(__file__), 'gaussianshading'))
from gaussianshading.export import GSWatermark

sys.path.append(os.path.join(os.path.dirname(__file__), 'rid'))
from ringid.export import RingIDWatermark

class GRIDSWatermark():
    def __init__(self, 
                 args,
                 pipe, 
                 #hf_cache_dir='/home/mkaut/.cache/huggingface/hub'
    ):
        
        self.model_id = args.model_id
        self.inf_steps = args.inf_steps
        self.test_inf_steps = args.test_inf_steps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.hf_cache_dir = hf_cache_dir
        self.method = 'grids'
        self.num_images = args.num_images
        self.guidance_scale = args.guidance_scale
        self.args = args

        self.latent_channels = 4 if args.model_id == 'sd' else 16
        self.latent_channels_wm = args.latent_channels_wm
        
        self.pipe = pipe

        # 1. create the RID watermark
        self.rid_wm = RingIDWatermark(self.args, pipe)
        
        # 2. create the GS watermark 
        # the RID watermark will be on self.rid_wm.WATERMARK_CHANNEL, 
        # e.g. [0, 3] of 4 available wm_channels
        # so create gs_watermark_channel for the opposite channels
        # in this case [1, 2] of the 4 available wm_channels
        temp_latent_channels_wm = self.latent_channels_wm
        args.latent_channels_wm = int(self.latent_channels_wm / 2) # e.g. 8 instead of 16 or 2 instead of 4
        self.gs_watermark_channel = [i for i in range(self.latent_channels_wm) if i not in self.rid_wm.watermark_channel]
        print(f'[GRIDSWatermark] gs_watermark_channel: {self.gs_watermark_channel}')
        self.gs_wm = GSWatermark(self.args, pipe)
        
        args.latent_channels_wm = temp_latent_channels_wm # reset the args object to the original value for the RID watermark
        
        # generate single watermark pattern for visualization
        self.visualize_watermark_pattern()

    def visualize_watermark_pattern(self):

        # 1. plot the watermark pattern in the frequency domain
        # has latent_channels_wm channels, not neccessarily the full latent_channels
        pattern = self.rid_wm.Fourier_watermark_pattern_list[self.args.pattern_index]
        
        title_pattern = (
            f'GRIDS Watermark RID pattern in Frequency Domain, pattern index {self.args.pattern_index},\n'
            f'channel {self.rid_wm.watermark_channel}, radius {self.rid_wm.RADIUS}, radius_cutoff {self.rid_wm.RADIUS_CUTOFF},\n'
            f'timeshift {self.args.time_shift}, tsfactor {self.args.time_shift_factor}, fix_gt {self.args.fix_gt}'
        )
        save_path = f'{self.args.log_dir}/{self.method}_wm_only.pdf'

        plot_wm_pattern_fft(num_channels=self.latent_channels_wm,
                            pattern=pattern,
                            title=title_pattern,
                            save_path=save_path)
        
        # 2. inject watermark and plot the latents in the frequency domain
        seed_everything(0) # to be same as first iteration in encode.py
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64) # 16 for flux, 4 for sd
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        init_latents_original = copy.deepcopy(init_latents) # for before/after visualization
        
        # imprint both watermarks in the latents
        init_latents_gs_wm = self.gs_wm.gs.truncSampling(self.gs_wm.watermark_m).to(torch.float32).to(self.device) # has latent_channels_wm / 2 channels, [1, 2, 64, 64]
        print(f'[viz_pattern] init_latents_gs_wm shape: {init_latents_gs_wm.shape}')
        init_latents[:, self.gs_watermark_channel, ...] = init_latents_gs_wm
        print(f'[viz_pattern] init_latents shape: {init_latents.shape}')
        init_latents_watermarked = self.rid_wm.inject_watermark(init_latents, self.args.pattern_index)
        print(f'[viz_pattern] init_latents_watermarked shape: {init_latents_watermarked.shape}')
       
        diff = init_latents_watermarked - init_latents_original # only for before/after visualization

        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        init_latents_watermarked_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_watermarked), dim=(-1, -2))

        title_latents = (
            f'GRIDS Watermark latents in Frequency and image Domain, pattern index={self.args.pattern_index},\n'
            f'RID channel={self.rid_wm.watermark_channel}\n'
            f'GS channel={self.gs_watermark_channel},\n'
        )
        save_path = f'{self.args.log_dir}/{self.method}_wm_latents.pdf'

        plot_wm_latents_fft(num_channels=self.latent_channels,
                            init_latents_fft=init_latents_fft,
                            init_latents_watermarked_fft=init_latents_watermarked_fft,
                            init_latents_watermarked=init_latents_watermarked,
                            diff=diff,
                            title=title_latents,
                            save_path=save_path)
   
    ############################# ENCODING ########################################
    def generate_img(self, prompt, do_wm, seed, pattern_index=0, num_images_per_prompt=1, **kwargs):
        
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64) # 16 for flux, 4 for sd
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)

        if do_wm:
            init_latents_gs_wm = self.gs_wm.gs.truncSampling(self.gs_wm.watermark_m).to(torch.float32).to(self.device) # has latent_channels_wm / 2 channels, [1, 2, 64, 64]
            init_latents[:, self.gs_watermark_channel, ...] = init_latents_gs_wm # (1, 4, 64, 64)
            init_latents = self.rid_wm.inject_watermark(init_latents, pattern_index) # (1, 4, 64, 64)
            
    
        seed_everything(seed)
        if isinstance(self.pipe, InversableFluxPipeline):
            ## (1, self.latent_channels, 64, 64) --> (1, 1024, 64)
            #print(f'\n[generate_img] init_latents min/max before reshape_latents_SD_to_flux: {init_latents.min().item()}/{init_latents.max().item()}')
            init_latents = self.pipe.reshape_latents_SD_to_flux(wm_latents=init_latents,
                                                                batch_size=num_images_per_prompt,
                                                                num_channels_latents=16,
                                                                height=512, # full height and width before
                                                                width=512,
                                                                dtype=torch.float32,
                                                                device=self.device,
                                                                generator=None,)
        
        outputs = self.pipe(
            prompt,
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
    def get_inversed_latents(self, img, prompt='', do_wm=False, seed=None, pattern_index=None):
        
        
        true_latents_np = np.random.randn(1, self.latent_channels, 64, 64) 
        true_latents = torch.from_numpy(true_latents_np).to(torch.float32).to(self.device)
        if do_wm:
            true_latents_gs_wm = self.gs_wm.gs.truncSampling(self.gs_wm.watermark_m).to(torch.float32).to(self.device) #  (1, 2, 64, 64)
            true_latents[:, self.gs_watermark_channel, ...] = true_latents_gs_wm # (1, 16, 64, 64)
            true_latents = self.rid_wm.inject_watermark(true_latents, pattern_index)  # (1, 16, 64, 64)
        seed_everything(seed) 
        #same from here on
        
        dtype = self.pipe.text_encoder.dtype
        img = transform_img(img).unsqueeze(0).to(dtype).to(self.device) 

        img_latents = self.pipe.get_image_latents(img, 
                                                  sample=False, 
                                                  batch_size=1,
                                                  num_channels_latents=16, 
                                                  height=512, # full height and width before
                                                  width=512,)

        reversed_latents = self.pipe.forward_diffusion(latents=img_latents, 
                                                       prompt=prompt, 
                                                       guidance_scale=1,
                                                       num_inference_steps=self.test_inf_steps,
                                                       device=self.device,)
        
        if isinstance(self.pipe, InversableFluxPipeline):
            reversed_latents = self.pipe._unpack_latents(latents=reversed_latents, 
                                                         height=512, 
                                                         width=512, 
                                                         vae_scale_factor=self.pipe.vae_scale_factor)
            
            reversed_latents = reversed_latents.to(torch.float32) # always (1, 16, 64, 64)
            
        return reversed_latents, true_latents
    
    def eval_watermark(self, reversed_latents_nowm, reversed_latents_wm):
        # 1. get the RID watermark metrics
        rid_nowm_metric, rid_wm_metric = self.rid_wm.eval_watermark(reversed_latents_nowm, reversed_latents_wm)
        
        # 2. get the GS watermark metrics, select only the channels of the watermark
        gs_nowm_metric = self.gs_wm.gs.eval_watermark(reversed_latents_nowm[:, self.gs_watermark_channel, ...], count=False)
        gs_wm_metric = self.gs_wm.gs.eval_watermark(reversed_latents_wm[:, self.gs_watermark_channel, ...], count=False)
       
        return rid_nowm_metric, rid_wm_metric, gs_nowm_metric, gs_wm_metric



    
    def viz_reversed_latents(self, true_latents_nowm, reversed_latents_nowm, true_latents_wm, reversed_latents_wm, attack_name, attack_vals, strength):
        
        self.rid_wm.viz_reversed_latents(true_latents_nowm, 
                                         reversed_latents_nowm, 
                                         true_latents_wm, 
                                         reversed_latents_wm, 
                                         attack_name, attack_vals, strength)
        # for the GS watermark, we need to select the channels where the watermark is
        self.gs_wm.viz_reversed_latents(true_latents_nowm, 
                                        reversed_latents_nowm, 
                                        true_latents_wm, 
                                        reversed_latents_wm, 
                                        attack_name, attack_vals, strength, 
                                        gs_watermark_channel=self.gs_watermark_channel)