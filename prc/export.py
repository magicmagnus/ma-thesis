import os
import sys
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything, transform_img
import src.pseudogaussians as prc_gaussians
from src.optim_utils import get_dataset, image_distortion #, transform_img,
from src.prc import KeyGen, Encode, str_to_bin, bin_to_str, Detect, Decode
#from inversion import exact_inversion, generate, stable_diffusion_pipe

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# flux
from pipes.inverse_flux_pipeline import InversableFluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# sd
from pipes.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

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
        self.method = 'prc' 
        self.num_images = args.num_images # 
        self.latent_channels = 4 if args.model_id == 'sd' else 16
        self.latent_channels_wm = args.latent_channels_wm # whether to fill all the channels, or only the forst 4 ones
        self.n = self.latent_channels_wm * 64 * 64  # the length of a PRC codeword
        self.inv_order = args.inv_order
        self.decoder_inv = args.decoder_inv
        self.var = args.var
        self.guidance_scale = args.guidance_scale
        self.args = args

        if args.model_id == 'sd' and self.latent_channels_wm != 4:
            raise ValueError("SD model only supports 4 latent channels")
       
        self.encoding_key = None
        self.decoding_key = None
        

        # Load or generate watermark patterns
        key_id = f'{self.method}_prct_{self.prc_t}_fpr_{self.fpr}_ch_{self.latent_channels_wm}'
        key_path = f'keys/{key_id}.pkl'

        if not os.path.exists(key_path):
            # key does not exist yet, generate watermark and save it
            (self.encoding_key_ori, self.decoding_key_ori) = KeyGen(self.n, false_positive_rate=self.fpr, t=self.prc_t)
            with open(key_path, 'wb') as f:
                pickle.dump((self.encoding_key_ori, self.decoding_key_ori), f)
            with open(key_path, 'rb') as f:
                self.encoding_key, self.decoding_key = pickle.load(f)
            assert self.encoding_key[0].all() == self.encoding_key_ori[0].all()
            print(f'\nGenerated PRC keys and saved to file {key_path}')
        else:
            # load the existing keys
            with open(key_path, 'rb') as f:
                self.encoding_key, self.decoding_key = pickle.load(f)
            print(f'\nLoaded PRC keys from file {key_path}')

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
            # self.pipe = stable_diffusion_pipe(solver_order=1, 
            #                                   model_id='stabilityai/stable-diffusion-2-1-base', 
            #                                   cache_dir=self.hf_cache_dir)
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

        self.visualize_watermark_pattern()

    def visualize_watermark_pattern(self):

        init_latents = self.get_encoded_latents()

        fig, ax = plt.subplots(1, self.latent_channels_wm, figsize=(self.latent_channels_wm*2, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(self.latent_channels_wm):
            ax[i].axis('off')
        for i in range(self.latent_channels_wm):
            ax[i].imshow(init_latents[0, i].real.cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
        ax[0].set_title('Watermark pattern', loc='left', fontsize=10)
        fig.suptitle(f'PRC Watermark pattern', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_wm_only.png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
    
    ############################# ENCODING ########################################
    def get_encoded_latents(self, message=None):
        self.prc_codeword = Encode(self.encoding_key, message)
        init_latents = prc_gaussians.sample(self.prc_codeword).reshape(1, self.latent_channels_wm, 64, 64).to(self.device)
        return init_latents
    
    def generate_img(self, prompt, do_wm, seed, message=None, num_images_per_prompt=1, **kwargs):
        
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        if do_wm:
            init_latents_wm = self.get_encoded_latents(message).to(torch.float32).to(self.device) # e.g. (1, 4, 64, 64)
            init_latents[:, :self.latent_channels_wm, ...] = init_latents_wm # e.g. (1, 16, 64, 64)

        seed_everything(seed)
        if isinstance(self.pipe, InversableFluxPipeline):
            ## (1, self.latent_channel, 64, 64) --> (1, 1024, 64)
            init_latents = self.pipe.reshape_latents_SD_to_flux(wm_latents=init_latents,
                                                                batch_size=num_images_per_prompt,
                                                                num_channels_latents=16, # later try to set it to 4
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
        img = outputs.images[0]
        # else:
        #     img, _, _ = generate(
        #         prompt=prompt,
        #         num_images_per_prompt=num_images_per_prompt,
        #         init_latents=init_latents,
        #         num_inference_steps=self.inf_steps,
        #         solver_order=1,
        #         pipe=self.pipe,
        #         guidance_scale=self.guidance_scale
        #     )
        return img
  
    ############################# DECODING ########################################
    def get_inversed_latents(self, img, prompt='', do_wm=False, seed=None, message=None):
        
        true_latents_np = np.random.randn(1, self.latent_channels, 64, 64)
        true_latents = torch.from_numpy(true_latents_np).to(torch.float32).to(self.device)
        if do_wm:
            true_latents_wm = self.get_encoded_latents(message).to(torch.float32).to(self.device) # e.g. (1, 4, 64, 64)
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
            

        # else:
        #     reversed_latents = exact_inversion(image=img, 
        #                                     prompt=prompt, 
        #                                     test_num_inference_steps=self.test_inf_steps,
        #                                     inv_order=self.inv_order,
        #                                     pipe=self.pipe,
        #                                     guidance_scale=1, # was 3.0, but in non-exact inversion it is 1.0
        #                                     decoder_inv=self.decoder_inv, # by default True for exact inversion
        #                                     solver_order=1, # by default 1 for exact inversion
        #                                     )
        
        
        return reversed_latents, true_latents
    
    def detect_watermark(self, reversed_latents):
        reversed_latents = reversed_latents[:, :self.latent_channels_wm, ...] # only check the wm channels
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(self.var)).flatten().cpu()
        return Detect(self.decoding_key, reversed_prc)
    
    def decode_watermark(self, reversed_latents):
        reversed_latents = reversed_latents[:, :self.latent_channels_wm, ...] # only check the wm channels
        reversed_prc = prc_gaussians.recover_posteriors(reversed_latents.to(torch.float64).flatten().cpu(), variances=float(self.var)).flatten().cpu()
        return Decode(self.decoding_key, reversed_prc)
    
    def viz_reversed_latents(self, true_latents_nowm, reversed_latents_nowm, true_latents_wm, reversed_latents_wm, attack_name, attack_vals, strength):

        _, metric_wm, _ = self.detect_watermark(reversed_latents_wm)
        _, metric_nowm, _ = self.detect_watermark(reversed_latents_nowm)

        masked_diff_wm = reversed_latents_wm - true_latents_wm # in the wm_latents the watermark is encoded
        masked_diff_nowm = reversed_latents_nowm - true_latents_wm # in the no_wm_latents the watermark is not encoded

        diff_wm = reversed_latents_wm - true_latents_wm # 
        diff_nowm = reversed_latents_nowm - true_latents_nowm

        abs_diff_wm = masked_diff_wm.abs()
        abs_diff_nowm = masked_diff_nowm.abs()

        # visualize the reversed wm_latents against the watermark pattern
        fig = plt.figure(figsize=(22, 8), constrained_layout=True)
        gs = fig.add_gridspec(3, 8)
        ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(8)] for i in range(3)])
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

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

        