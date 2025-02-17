import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything, transform_img
from .watermark import Gaussian_Shading_chacha, Gaussian_Shading

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# flux
from pipes.inverse_flux_pipeline import InversableFluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# sd
from pipes.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler


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
        self.latent_channels = 4 if args.model_id == 'sd' else 16
        self.latent_channels_wm = args.gs_latent_channels_wm # whether to fill all the channels, or only the forst 4 ones
        self.args = args

        if args.model_id == 'sd' and self.latent_channels_wm != 4:
            raise ValueError("SD model only supports 4 latent channels")
        
        # Load or generata watermark
        key_id = f'{self.method}_chacha_{self.chacha}_ch_{self.ch_factor}_hw_{self.hw_factor}_fpr_{self.fpr}_user_{self.user_number}_channels_{self.latent_channels_wm}'
        key_path = f'keys/{key_id}.pkl'
        
        if self.chacha: # if we use chacha20 encryption, we get a nonce
            self.gs = Gaussian_Shading_chacha(self.ch_factor, self.hw_factor, self.fpr, self.user_number, self.latent_channels_wm)
            if not os.path.exists(key_path):
                # key does not exist yet, generate watermark and save it
                watermark_m_ori, key_ori, nonce_ori, watermark_ori = self.gs.create_watermark_and_return_w()
                with open(key_path, 'wb') as f:
                    pickle.dump((watermark_m_ori, key_ori, nonce_ori, watermark_ori), f)
                with open(key_path, 'rb') as f:
                    watermark_m, key, nonce, watermark = pickle.load(f)
                assert watermark_m.all() == watermark_m_ori.all()
                self.watermark_m = watermark_m
                self.gs.watermark = watermark
                self.gs.key = key
                self.gs.nonce = nonce
                print(f"\nGenerated GaussianShading watermark and saved to file {key_path}")
            else:
                # load the existing keys
                with open(key_path, 'rb') as f:
                    watermark_m, key, nonce, watermark = pickle.load(f)
                self.watermark_m = watermark_m
                self.gs.watermark = watermark
                self.gs.key = key
                self.gs.nonce = nonce
                print(f"\nLoaded GaussianShading watermark from file {key_path}")
        else: 
            self.gs = Gaussian_Shading(self.ch_factor, self.hw_factor, self.fpr, self.user_number, self.latent_channels_wm)
            if not os.path.exists(key_path):
                # key does not exist yet, generate watermark and save it
                watermark_m_ori, key_ori, watermark_ori = self.gs.create_watermark_and_return_w()
                with open(key_path, 'wb') as f:
                    pickle.dump((watermark_m_ori, key_ori, watermark_ori), f)
                with open(key_path, 'rb') as f:
                    watermark_m, key, watermark = pickle.load(f)
                assert watermark_m.all() == watermark_m_ori.all()
                self.watermark_m = watermark_m
                self.gs.watermark = watermark
                self.gs.key = key
                print(f"\nGenerated GaussianShading watermark and saved to file {key_path}")
            else:
                # load the existing keys
                with open(key_path, 'rb') as f:
                    watermark_m, key, watermark = pickle.load(f)
                self.watermark_m = watermark_m
                self.gs.watermark = watermark
                self.gs.key = key
                print(f"\nLoaded GaussianShading watermark from file {key_path}")

        # which Model to use
        if args.model_id == 'sd':
            print("\nUsing SD model")
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                'stabilityai/stable-diffusion-2-1-base', 
                subfolder='scheduler')
            self.pipe = InversableStableDiffusionPipeline.from_pretrained(
                'stabilityai/stable-diffusion-2-1-base',
                scheduler=scheduler,
                torch_dtype=torch.float32,
                cache_dir=self.hf_cache_dir,
                ).to(self.device)
        elif args.model_id == 'flux':
            print("\nUsing FLUX model")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                'black-forest-labs/FLUX.1-dev',
                subfolder='scheduler'
            )
            self.pipe = InversableFluxPipeline.from_pretrained(
                'black-forest-labs/FLUX.1-dev',
                scheduler=scheduler,
                torch_dtype=torch.bfloat16,
                cache_dir=self.hf_cache_dir,
            ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        # generate single watermark pattern for visualization
        self.visualize_watermark_pattern()

    def visualize_watermark_pattern(self):

        init_latents = self.gs.truncSampling(self.watermark_m) # has latent_channels_wm channels

        fig, ax = plt.subplots(1, self.latent_channels_wm, figsize=(self.latent_channels_wm*2, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(self.latent_channels_wm):
            ax[i].axis('off')
        for i in range(self.latent_channels_wm):
            ax[i].imshow(init_latents[0, i].real.cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
        ax[0].set_title('Watermark pattern', loc='left', fontsize=10)
        fig.suptitle(f'GS Watermark pattern', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_wm_only.png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)


    ############################# ENCODING ########################################
    def generate_img(self, prompt, do_wm, seed, num_images_per_prompt=1, **kwargs):
       
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64)                  # channesl fixed by model type, 4 or 16
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        if do_wm:
            init_latents_wm = self.gs.truncSampling(self.watermark_m).to(torch.float32).to(self.device) #  e.g. (1, 4, 64, 64)
            init_latents[:, :self.latent_channels_wm, ...] = init_latents_wm # e.g. (1, 16, 64, 64)
            
        seed_everything(seed)
        if isinstance(self.pipe, InversableFluxPipeline):
            ## (1, self.latent_channels, 64, 64) --> (1, 1024, 64)
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
        image_w = outputs.images[0]
        return image_w
    
    ############################# DECODING ########################################
    def get_inversed_latents(self, img, prompt='', do_wm=False, seed=None):
        
        
        true_latents_np = np.random.randn(1, self.latent_channels, 64, 64) 
        true_latents = torch.from_numpy(true_latents_np).to(torch.float32).to(self.device)
        if do_wm:
            true_latents_wm = self.gs.truncSampling(self.watermark_m).to(torch.float32).to(self.device) #  e.g. (1, 4, 64, 64)
            true_latents[:, :self.latent_channels_wm, ...] = true_latents_wm # e.g. (1, 16, 64, 64)
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
    
    def eval_watermark(self, latents, count=True):
        latents = latents[:, :self.latent_channels_wm, ...] # only check the wm channels
        return self.gs.eval_watermark(latents, count=count)
    
    def viz_reversed_latents(self, true_latents_nowm, reversed_latents_nowm, true_latents_wm, reversed_latents_wm, attack_name, attack_vals, strength):
        
        metric_wm = self.eval_watermark(reversed_latents_wm, count=False)
        metric_nowm = self.eval_watermark(reversed_latents_nowm, count=False)

        masked_diff_wm = reversed_latents_wm - true_latents_wm # both again the same wm pattern
        masked_diff_nowm = reversed_latents_nowm - true_latents_wm

        diff_wm = reversed_latents_wm - true_latents_wm
        diff_nowm = reversed_latents_nowm - true_latents_nowm 

        abs_diff_wm = masked_diff_wm.abs()
        abs_diff_nowm = masked_diff_nowm.abs()

        # visualize the reversed wm_latents against the watermark pattern
        fig = plt.figure(figsize=(22, 8), constrained_layout=True)
        gs = fig.add_gridspec(3, 8)
        ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(8)] for i in range(3)])
        plt.subplots_adjust(wspace=0.1, hspace=0.01)

        for i in range(3):
            for j in range(8):
                    ax[i, j].axis('off') 

        for i in range(4):
            ax[0, i].imshow(reversed_latents_wm[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[1, i].imshow(np.abs(masked_diff_wm[0, i].cpu().numpy()), cmap='gray', vmin=0, vmax=10)
            ax[2, i].imshow(np.abs(diff_wm[0, i].cpu().numpy()), cmap='gray', vmin=0, vmax=10)

            ax[0, i+4].imshow(reversed_latents_nowm[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[1, i+4].imshow(np.abs(masked_diff_nowm[0, i].cpu().numpy()), cmap='gray', vmin=0, vmax=10)
            ax[2, i+4].imshow(np.abs(diff_nowm[0, i].cpu().numpy()), cmap='gray', vmin=0, vmax=10)

        divider_line = plt.Line2D([0.5, 0.5], [0, 0.9], transform=fig.transFigure, color='black', linewidth=1)
        fig.add_artist(divider_line)

        ax[0, 0].set_title('Reversed latents with WM', loc='left', fontsize=16)
        ax[1, 0].set_title(f'Abs. Diff. to WM pattern:\n{abs_diff_wm.sum():.3f} (mean {abs_diff_wm.mean():.3f})\nwith metric {metric_wm:.2f}', loc='left', fontsize=16) 
        ax[2, 0].set_title('Abs. Diff. to ground truth WM latents', loc='left', fontsize=16)

        ax[0, 4].set_title('Reversed latents with NOWM', loc='left', fontsize=16)
        ax[1, 4].set_title(f'Abs. Diff. to WM pattern:\n{abs_diff_nowm.sum():.3f} (mean {abs_diff_nowm.mean():.3f})\nwith metric {metric_nowm:.2f}', loc='left', fontsize=16)
        ax[2, 4].set_title('Abs. Diff. to ground truth NOWM latents', loc='left', fontsize=16)

        fig.suptitle(f'PRC Watermark decoding with and without watermark', fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_reversed_latents_{attack_name}_{attack_vals[strength]}.png',
            bbox_inches='tight', 
            pad_inches=0.1,
            dpi=150)
        plt.close(fig)
