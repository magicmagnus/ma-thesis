import os
import sys
import copy
import torch
import pickle
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything, transform_img
from .utils import ring_mask, make_Fourier_ringid_pattern, fft, ifft
#from .io_utils import *

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# flux
from pipes.inverse_flux_pipeline import InversableFluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# sd
from pipes.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

class RingIDWatermark():
    def __init__(self, 
                 args,
                 hf_cache_dir='/home/mkaut/.cache/huggingface/hub'
    ):
        self.model_id = args.model_id
        self.inf_steps = args.inf_steps
        self.test_inf_steps = args.test_inf_steps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hf_cache_dir = hf_cache_dir
        self.method = 'rid'
        self.num_images = args.num_images
        self.guidance_scale = args.guidance_scale
        self.args = args

        self.latent_channels = 4 if args.model_id == 'sd' else 16
        
        self.RADIUS = args.radius
        self.RADIUS_CUTOFF = args.radius_cutoff
        self.RING_WATERMARK_CHANNEL = args.ring_watermark_channel
        self.HETER_WATERMARK_CHANNEL = args.heter_watermark_channel
        self.WATERMARK_CHANNEL = sorted(self.HETER_WATERMARK_CHANNEL + self.RING_WATERMARK_CHANNEL)

        self.watermark_channel = self.WATERMARK_CHANNEL  # Set self.watermark_channel


    	# Load or generate watermark patterns
        key_id = f'{self.method}_seed_{args.watermark_seed}_ringwidth_{args.ring_width}_timeshift_{args.time_shift}_fixgt_{args.fix_gt}_channelmin_{args.channel_min}_assignedkeys_{args.assigned_keys}'
        key_path = f'keys/{key_id}.pkl'

        if not os.path.exists(key_path):
            # key does not exist yet, generate watermark and save it
            self.generate_watermark_patterns()
            with open(key_path, 'wb') as f:
                pickle.dump([self.Fourier_watermark_pattern_list, self.watermark_region_mask], f)
            print(f'\nGenerated RingID keys and saved to file {key_path}')
        else:
            # load the existing keys
            with open(key_path, 'rb') as f:
                self.Fourier_watermark_pattern_list, self.watermark_region_mask = pickle.load(f)
            print(f'\nLoaded RingID keys from file {key_path}')


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

        # TODO maybe set random seed here, cause duplicated watermark patterns???

        init_latents_np = np.random.randn(1, 4, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        
        pattern = self.Fourier_watermark_pattern_list[self.args.pattern_index]

        # first plot only the pattern
        fig, ax = plt.subplots(2, 4, figsize=(10, 6))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        for i in range(4):
            ax[0, i].imshow(pattern[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[1, i].imshow(pattern[0, i].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
        ax[0, 0].set_title('Watermark pattern (real part)', loc='left', fontsize=10)
        ax[1, 0].set_title('Watermark pattern (imaginary part)', loc='left', fontsize=10)
        # draw a line for x=32 and y=32
        for i in range(2):
            for j in range(4):
                ax[i, j].axvline(32, color='r', linestyle='--', linewidth=1)
                ax[i, j].axhline(32, color='r', linestyle='--', linewidth=1)
        fig.suptitle(f'RingID Watermark, pattern index {self.args.pattern_index}, channel {self.watermark_channel}, radius {self.RADIUS}, radius_cutoff {self.RADIUS_CUTOFF}, timeshift {self.args.time_shift}, fix_gt {self.args.fix_gt}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_wm_only.png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

        
        init_latents_watermarked = self.inject_watermark(init_latents, self.args.pattern_index)

        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        init_latents_watermarked_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_watermarked), dim=(-1, -2))

        diff = init_latents_watermarked - init_latents # only for visualization

        # plot the watermark pattern applied to the latents
        fig, ax = plt.subplots(5, 4, figsize=(10, 12))
        fig.subplots_adjust(hspace=0.3, wspace=0.1)
        for i in range(5):
            for j in range(4):
                ax[i, j].axis('off')
        for i in range(4):
            ax[0, i].imshow(init_latents_fft[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[1, i].imshow(init_latents_watermarked_fft[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[2, i].imshow(init_latents_watermarked_fft[0, i].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[3, i].imshow(init_latents_watermarked[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[4, i].imshow(diff[0, i].cpu().numpy(), cmap='RdBu', vmin=-4, vmax=4)
        ax[0, 0].set_title('Original FFT of init_latents (real part)', loc='left', fontsize=10)
        ax[1, 0].set_title('FFT after watermarking (real part)', loc='left', fontsize=10)
        ax[2, 0].set_title('FFT after watermarking (imaginary part)', loc='left', fontsize=10)
        ax[3, 0].set_title('init_latents after watermarking (spatial domain)', loc='left', fontsize=10)
        ax[4, 0].set_title('Difference (watermarked - original)', loc='left', fontsize=10)
        fig.suptitle(f'RingID Watermark, pattern index {self.args.pattern_index}, channel {self.watermark_channel}, radius {self.RADIUS}, radius_cutoff {self.RADIUS_CUTOFF}, timeshift {self.args.time_shift}, fix_gt {self.args.fix_gt}', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.args.log_dir}/{self.method}_wm_latents.png', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        
    # only called once in the beginning
    def generate_watermark_patterns(self):
                
        base_latents_np = np.random.randn(1, 4, 64, 64)
        base_latents = torch.from_numpy(base_latents_np).to(torch.float64).to(self.device)
        self.original_latents_shape = base_latents.shape

        # Generate watermark masks as in utils.py
        self.single_channel_ring_watermark_mask = torch.tensor(
            ring_mask(
                size=self.original_latents_shape[-1], 
                r_out=self.RADIUS, 
                r_in=self.RADIUS_CUTOFF)
        )

        # Get heterogeneous watermark mask
        if len(self.HETER_WATERMARK_CHANNEL) > 0:
            single_channel_heter_watermark_mask = torch.tensor(
                ring_mask(
                    size = self.original_latents_shape[-1], 
                    r_out = self.RADIUS, 
                    r_in = self.RADIUS_CUTOFF)  
                )
            heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(self.HETER_WATERMARK_CHANNEL), 1, 1).to(self.device)
        else:
            heter_watermark_region_mask = None

        self.watermark_region_mask = [self.single_channel_ring_watermark_mask for _ in self.WATERMARK_CHANNEL]
        self.watermark_region_mask = torch.stack(self.watermark_region_mask).to(self.device)

        # Generate key value combinations
        single_channel_num_slots = self.RADIUS - self.RADIUS_CUTOFF
        value_range = np.linspace(-self.args.ring_value_range, self.args.ring_value_range, self.args.quantization_levels).tolist()
        key_value_list = [[list(combo) for combo in itertools.product(value_range, repeat=len(self.RING_WATERMARK_CHANNEL))] for _ in range(single_channel_num_slots)]
        key_value_combinations = list(itertools.product(*key_value_list))
        
        # Limit the number of assigned keys if specified
        if self.args.assigned_keys > 0:
            key_value_combinations = random.sample(key_value_combinations, self.args.assigned_keys)
            
        self.Fourier_watermark_pattern_list = []
        for _, combo in enumerate(key_value_combinations):
            pattern = make_Fourier_ringid_pattern(
                self.device,
                list(combo),
                base_latents,
                radius=self.RADIUS,
                radius_cutoff=self.RADIUS_CUTOFF,
                ring_width=self.args.ring_width,
                ring_watermark_channel=self.RING_WATERMARK_CHANNEL,
                heter_watermark_channel=self.HETER_WATERMARK_CHANNEL,
                heter_watermark_region_mask=heter_watermark_region_mask if len(self.HETER_WATERMARK_CHANNEL) > 0 else None
            )
            self.Fourier_watermark_pattern_list.append(pattern)
        print(f'Generated {len(self.Fourier_watermark_pattern_list)} watermark patterns')

        if self.args.fix_gt == 1:
            # deletes the ring pattern in the imaginary part of the watermark pattern
            self.Fourier_watermark_pattern_list = [fft(ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in self.Fourier_watermark_pattern_list]

        if self.args.time_shift == 1:
            # from "concentric cirlces" to "cross-like" pattern
            for Fourier_watermark_pattern in self.Fourier_watermark_pattern_list:
                # use_time_shift_factor to suppress the attern in the middle a bit more
                Fourier_watermark_pattern[:, self.RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, self.RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * self.args.time_shift_factor)
                # Fourier_watermark_pattern[:, self.RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, self.RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)))

   
    ############################# ENCODING ########################################
    def inject_watermark(self, init_latents, pattern_index=0):
        #######
        # the whole process is based on 4 channel latent space, so make a copy of the latents
        # and inject the watermark pattern into the 4 channels, then return the watermarked latents
        # Inject the specified watermark pattern into the latents
        latents_orig = copy.deepcopy(init_latents)
        init_latents = init_latents[:, :4, ...] # only take the first 4 of (4 or 16 ) channels
        #######

        pattern = self.Fourier_watermark_pattern_list[pattern_index]
        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents.to(torch.float64)), dim=(-1, -2))

        # Replace values in the masked region with the watermark pattern
        assert len(self.watermark_channel) == len(self.watermark_region_mask)
        for idx, channel in enumerate(self.watermark_channel):
            channel_mask = self.watermark_region_mask[idx].unsqueeze(0).to(self.device).bool()
            # Apply the mask to replace FFT coefficients
            init_latents_fft[:, channel, :, :][channel_mask] = pattern[:, channel, :, :][channel_mask]

        # Inverse FFT to get watermarked latents
        init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real.to(torch.float32)

        #######
        # after the process, we have to merge the original latents with the watermarked latents
        # to get the full 16 channel latent space
        latents_orig[:, :4, ...] = init_latents # move back to first 4 of (4 or 16) channels	
        init_latents = latents_orig
        #######

        return init_latents


    def generate_img(self, prompt, do_wm, seed, pattern_index=0, num_images_per_prompt=1, **kwargs):
        
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64) # 16 for flux, 4 for sd
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)

        if do_wm:
            init_latents = self.inject_watermark(init_latents, pattern_index) 
            
        
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
        orig_image = outputs.images[0]
        return orig_image

    ############################# DECODING ########################################
    def get_inversed_latents(self, img, prompt='', do_wm=False, seed=None, pattern_index=None):

        # get the true latents from the image, as a comparison
        true_latents_np = np.random.randn(1, self.latent_channels, 64, 64)
        true_latents = torch.from_numpy(true_latents_np).to(torch.float32).to(self.device)
        
        if do_wm: 
            true_latents = self.inject_watermark(true_latents, pattern_index)
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
            
            reversed_latents = reversed_latents.to(torch.float32) # from flux, always have 16 channels
            
        return reversed_latents, true_latents
    
    

    def eval_watermark(self, reversed_latents_nowm, reversed_latents_wm):
        
        reversed_latents_nowm = reversed_latents_nowm[:, :4, ...] # only take the first 4 of (4 or 16 ) channels
        reversed_latents_wm = reversed_latents_wm[:, :4, ...] # only take the first 4 of (4 or 16 ) channels

        pattern = self.Fourier_watermark_pattern_list[self.args.pattern_index]
        watermark_region_mask = self.watermark_region_mask.unsqueeze(0)
        
        reversed_latents_nowm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_nowm), dim=(-1, -2))
        reversed_latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_wm), dim=(-1, -2))


        if not self.args.channel_min:
            # average over all channels
            diff_nowm = torch.abs(pattern[:, self.watermark_channel][watermark_region_mask] - reversed_latents_nowm_fft[:, self.watermark_channel][watermark_region_mask]).mean().item()
            diff_wm = torch.abs(pattern[:, self.watermark_channel][watermark_region_mask] - reversed_latents_wm_fft[:, self.watermark_channel][watermark_region_mask]).mean().item()
        
            return diff_nowm, diff_wm
        else:
            # return only the min distance of the channels
            diffs_nowm = []
            diffs_wm = []
            # the self.watermark_channel already selects only the watermarke channels
            diff_nowm = torch.abs(pattern[:, self.watermark_channel] - reversed_latents_nowm_fft[:, self.watermark_channel])
            diff_wm = torch.abs(pattern[:, self.watermark_channel] - reversed_latents_wm_fft[:, self.watermark_channel])
            for c_idx in range(len(self.watermark_channel)):
                channel_mask = watermark_region_mask[:, c_idx, ...]
                diffs_nowm.append(diff_nowm[:, c_idx, ...][channel_mask].mean().item())
                diffs_wm.append(diff_wm[:, c_idx, ...][channel_mask].mean().item())

            return min(diffs_nowm), min(diffs_wm)
        
    def viz_reversed_latents(self, true_latents_nowm, reversed_latents_nowm, true_latents_wm, reversed_latents_wm, attack_name, attack_vals, strength):
        
        true_latents_nowm_fft = torch.fft.fftshift(torch.fft.fft2(true_latents_nowm), dim=(-1, -2))
        reversed_latents_nowm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_nowm), dim=(-1, -2))
        true_latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(true_latents_wm), dim=(-1, -2))
        reversed_latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_wm), dim=(-1, -2))

        pattern = self.Fourier_watermark_pattern_list[self.args.pattern_index]
        watermark_region_mask = self.watermark_region_mask.unsqueeze(0)

        # return only the min distance of the channels
        wm_channel_idx = 0
        abs_diff_nowm_fft = []
        abs_diff_wm_fft = []

        # these are for the abs. sum differene values
        masked_diff_nowm_fft = torch.abs(pattern[:, self.watermark_channel] - reversed_latents_nowm_fft[:, self.watermark_channel]) # this already only has 2 channels
        masked_diff_wm_fft = torch.abs(pattern[:, self.watermark_channel] - reversed_latents_wm_fft[:, self.watermark_channel])
        for c_idx in range(reversed_latents_nowm_fft.shape[1]): # over 4 channels
            if c_idx in self.watermark_channel:
                channel_mask = watermark_region_mask[:, wm_channel_idx, ...]
                abs_diff_nowm_fft.append(masked_diff_nowm_fft[:, wm_channel_idx, ...][channel_mask].sum().item())
                abs_diff_wm_fft.append(masked_diff_wm_fft[:, wm_channel_idx, ...][channel_mask].sum().item())
                wm_channel_idx += 1
            else:
                abs_diff_nowm_fft.append(0)
                abs_diff_wm_fft.append(0)
            
        # while these are only for the visualization
        masked_diff_wm_fft = torch.zeros_like(reversed_latents_wm_fft)
        masked_diff_nowm_fft = torch.zeros_like(reversed_latents_nowm_fft)
        for i in range(len(self.watermark_channel)):
            masked_diff_wm_fft[:, self.watermark_channel[i], ...] = pattern[:, self.watermark_channel[i], ...] - reversed_latents_wm_fft[:, self.watermark_channel[i], ...]
            masked_diff_nowm_fft[:, self.watermark_channel[i], ...] = pattern[:, self.watermark_channel[i], ...] - reversed_latents_nowm_fft[:, self.watermark_channel[i], ...]
            masked_diff_wm_fft[:, self.watermark_channel[i], ...] = masked_diff_wm_fft[:, self.watermark_channel[i], ...] * watermark_region_mask[:, i, ...]
            masked_diff_nowm_fft[:, self.watermark_channel[i], ...] = masked_diff_nowm_fft[:, self.watermark_channel[i], ...] * watermark_region_mask[:, i, ...]
    
        # difference to their own ground truth latents
        diff_wm_fft = reversed_latents_wm_fft - true_latents_wm_fft # here, in true, the gt_patch should already be injected
        diff_nowm_fft = reversed_latents_nowm_fft - true_latents_nowm_fft
        

        fig = plt.figure(figsize=(24, 18), constrained_layout=True)
        gs = fig.add_gridspec(6, 8)
        ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(8)] for i in range(6)])

        for i in range(6):
            for j in range(8):
                ax[i, j].axis('off') 

        for i in range(4):
            ax[0, i].imshow(reversed_latents_wm_fft[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[1, i].imshow(reversed_latents_wm_fft[0, i].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[2, i].imshow(np.abs(masked_diff_wm_fft[0, i].real.cpu().numpy()), cmap='gray',vmin=0, vmax=150)
            ax[2, i].set_title(f'Abs. Diff. (real):\n{abs_diff_wm_fft[i]:.2f}', loc='left', fontsize=16)
            ax[3, i].imshow(np.abs(masked_diff_wm_fft[0, i].imag.cpu().numpy()), cmap='gray',vmin=0, vmax=150)
            ax[4, i].imshow(np.abs(diff_wm_fft[0, i].real.cpu().numpy()), cmap='gray',vmin=0, vmax=150)
            ax[5, i].imshow(np.abs(diff_wm_fft[0, i].imag.cpu().numpy()), cmap='gray',vmin=0, vmax=150)


            ax[0, i+4].imshow(reversed_latents_nowm_fft[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[1, i+4].imshow(reversed_latents_nowm_fft[0, i].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[2, i+4].imshow(np.abs(masked_diff_nowm_fft[0, i].real.cpu().numpy()), cmap='gray',vmin=0, vmax=150)
            ax[2, i+4].set_title(f'Abs. Diff. (real):\n{abs_diff_nowm_fft[i]:.2f}', loc='left', fontsize=16)
            ax[3, i+4].imshow(np.abs(masked_diff_nowm_fft[0, i].imag.cpu().numpy()), cmap='gray',vmin=0, vmax=150)
            ax[4, i+4].imshow(np.abs(diff_nowm_fft[0, i].real.cpu().numpy()), cmap='gray',vmin=0, vmax=150)
            ax[5, i+4].imshow(np.abs(diff_nowm_fft[0, i].imag.cpu().numpy()), cmap='gray',vmin=0, vmax=150)
            
        ax[0, 0].set_title('Reversed FFT of WM latents (real)', loc='left', fontsize=16)
        ax[1, 0].set_title('Reversed FFT of WM latents (imag)', loc='left', fontsize=16)
        ax[2, 0].set_title(f'Abs. Diff. to WM Patch (real):\n{abs_diff_wm_fft[0]:.2f}', loc='left', fontsize=16)
        ax[3, 0].set_title(f'(imag)', loc='left', fontsize=16)
        ax[4, 0].set_title(f'Abs. Diff. to ground truth WM latents (real)', loc='left', fontsize=16)
        ax[5, 0].set_title(f'(imag)', loc='left', fontsize=16)
        #
        ax[0, 4].set_title('Reversed FFT of NOWM latents (real)', loc='left', fontsize=16)
        ax[1, 4].set_title('Reversed FFT of NOWM latents (imag)', loc='left', fontsize=16)
        ax[2, 4].set_title(f'Abs. Diff. to WM Patch (real):\n{abs_diff_nowm_fft[0]:.2f}', loc='left', fontsize=16)
        ax[3, 4].set_title(f'(imag)', loc='left', fontsize=16)
        ax[4, 4].set_title(f'Abs. Diff. to ground truth NOWM latents (real)', loc='left', fontsize=16)
        ax[5, 4].set_title(f'(imag)', loc='left', fontsize=16)

        fig.colorbar(ax[3, 0].imshow(masked_diff_nowm_fft[0, 0].imag.cpu().numpy(), cmap='gray',vmin=0, vmax=150), ax=ax[3, 0], orientation='horizontal', fraction=0.046, pad=0.04)
        fig.suptitle(f'Reversed latents of method "{self.method}", model "{self.args.model_id}" with attack "{attack_name}"={attack_vals[strength]}', 
            fontsize=22, y=1.02)
        plt.savefig(f'{self.args.log_dir}/{self.method}_reversed_latents_{attack_name}_{attack_vals[strength]}.png', 
        bbox_inches='tight', 
        pad_inches=0.1,
        dpi=300)
        plt.close(fig)





        