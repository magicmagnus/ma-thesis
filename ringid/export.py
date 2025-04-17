import os
import sys
import copy
import torch
import pickle
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything, transform_img, plot_wm_pattern_fft, plot_wm_latents_fft, visualize_reversed_latents_fft
from .utils import ring_mask, make_Fourier_ringid_pattern, fft, ifft, create_diverse_pattern_list


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
                 pipe,
    ):
        self.model_id = args.model_id
        self.inf_steps = args.inf_steps
        self.test_inf_steps = args.test_inf_steps
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.method = 'rid'
        self.num_images = args.num_images
        self.guidance_scale = args.guidance_scale
        self.args = args

        self.latent_channels = 4 if args.model_id == 'sd' else 16
        self.latent_channels_wm = args.latent_channels_wm
        
        self.RADIUS = args.radius
        self.RADIUS_CUTOFF = args.radius_cutoff
        self.RING_WATERMARK_CHANNEL = args.ring_watermark_channel
        self.HETER_WATERMARK_CHANNEL = args.heter_watermark_channel
        self.WATERMARK_CHANNEL = sorted(self.HETER_WATERMARK_CHANNEL + self.RING_WATERMARK_CHANNEL)

        self.watermark_channel = self.WATERMARK_CHANNEL  # Set self.watermark_channel

        self.clone_patterns = args.clone_patterns
        self.shuffle_patterns = args.shuffle_patterns
        

    	# Load or generate watermark patterns
        key_id = f'{self.method}_seed_{args.watermark_seed}_patidx_{args.pattern_index}_timeshift_{args.time_shift}_tsfactor_{args.time_shift_factor}_fixgt_{args.fix_gt}_channels_{self.latent_channels_wm}_assignedkeys_{args.assigned_keys}_clone_{self.clone_patterns}_shuffle_{self.shuffle_patterns}'
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

        self.pipe = pipe

        # for the potential case of having the watermark duplicated over the 16 channels
        watermark_channel = []
        for i in range(self.latent_channels_wm // 4):
            for channel in self.watermark_channel:
                watermark_channel.append(channel + i*4)
        self.watermark_channel = watermark_channel
        self.watermark_region_mask = self.watermark_region_mask.repeat(self.latent_channels_wm // 4, 1, 1)

        # generate single watermark pattern for visualization
        self.visualize_watermark_pattern()

    def visualize_watermark_pattern(self):

        # 1. plot the watermark pattern in the frequency domain
        # has latent_channels_wm channels, not neccessarily the full latent_channels
        pattern = self.Fourier_watermark_pattern_list[self.args.pattern_index]
        
        title = (
            f'RingID Watermark in Frequency Domain, pattern index {self.args.pattern_index},\n'
            f'channel {self.watermark_channel}, radius {self.RADIUS}, radius_cutoff {self.RADIUS_CUTOFF},\n'
            f'timeshift {self.args.time_shift}, tsfactor {self.args.time_shift_factor}, fix_gt {self.args.fix_gt}'
        )
        save_path = f'{self.args.log_dir}/{self.method}_wm_only.pdf'

        plot_wm_pattern_fft(num_channels=self.latent_channels_wm,
                            pattern=pattern,
                            title=title,
                            save_path=save_path)
        
        # 2. inject watermark and plot the latents in the frequency domain
        seed_everything(0) # to be same as first iteration in encode.py
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        
        init_latents_watermarked = self.inject_watermark(init_latents, self.args.pattern_index)
        # print(f"init_latents RID dtype: {init_latents_watermarked.dtype}") # torch.float32
        diff = init_latents_watermarked - init_latents # only for before/after visualization

        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        init_latents_watermarked_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_watermarked), dim=(-1, -2))

        save_path = f'{self.args.log_dir}/{self.method}_wm_latents.pdf'

        plot_wm_latents_fft(num_channels=self.latent_channels,
                            init_latents_fft=init_latents_fft,
                            init_latents_watermarked_fft=init_latents_watermarked_fft,
                            init_latents_watermarked=init_latents_watermarked,
                            diff=diff,
                            title=title,
                            save_path=save_path)

        
    # only called once in the beginning
    def generate_watermark_patterns(self):
                
        seed_everything(self.args.watermark_seed)
        base_latents_np = np.random.randn(1, 4, 64, 64)                                        # here 4 channels, else takes forever
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
        print(f'\nGenerated {len(self.Fourier_watermark_pattern_list)} watermark patterns')

        if self.args.fix_gt == 1:
            # deletes the ring pattern in the imaginary part of the watermark pattern
            self.Fourier_watermark_pattern_list = [fft(ifft(Fourier_watermark_pattern).real) for Fourier_watermark_pattern in self.Fourier_watermark_pattern_list]

        if self.args.time_shift == 1:
            # from "concentric cirlces" to "cross-like" pattern
            for Fourier_watermark_pattern in self.Fourier_watermark_pattern_list:
                # use_time_shift_factor to suppress the attern in the middle a bit more
                Fourier_watermark_pattern[:, self.RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, self.RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)) * self.args.time_shift_factor)
                # Fourier_watermark_pattern[:, self.RING_WATERMARK_CHANNEL, ...] = fft(torch.fft.fftshift(ifft(Fourier_watermark_pattern[:, self.RING_WATERMARK_CHANNEL, ...]), dim = (-1, -2)))

        # extend the watermark pattern to the full number of channels	
        num_repeats = self.latent_channels_wm // 4
        if num_repeats > 1:
            if self.clone_patterns:
                # repeat the pattern across all channels, 16 or 12 or 8 instead of 4
                self.Fourier_watermark_pattern_list = [pattern.repeat(1, num_repeats, 1, 1) for pattern in self.Fourier_watermark_pattern_list]
            else: # 4 different patterns for each block of 4 channels
                original_patterns = self.Fourier_watermark_pattern_list.copy()
                if self.shuffle_patterns:
                    random.shuffle(original_patterns)
                
                # draw different patterns for each block of 4 channels, instead of repeating the same pattern
                self.Fourier_watermark_pattern_list = create_diverse_pattern_list(original_patterns, num_repeats)
     
   
    ############################# ENCODING ########################################
    def inject_watermark(self, init_latents, pattern_index=0):
        
        # has latent_channels_wm channels, not neccessarily the full latent_channels
        pattern = self.Fourier_watermark_pattern_list[pattern_index]
        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents.to(torch.float64)), dim=(-1, -2))

        assert len(self.watermark_channel) == len(self.watermark_region_mask)
    
        # Replace values in the masked region with the watermark pattern
        for idx, channel in enumerate(self.watermark_channel):
            channel_mask = self.watermark_region_mask[idx].unsqueeze(0).to(self.device).bool()
            init_latents_fft[:, channel, :, :][channel_mask] = pattern[:, channel, :, :][channel_mask]

        # Inverse FFT to get watermarked latents
        init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real.to(torch.float32)

        return init_latents


    def generate_img(self, prompt, do_wm, seed, pattern_index=0, num_images_per_prompt=1, **kwargs):
        
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64) # 16 for flux, 4 for sd
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)

        if do_wm:
            init_latents = self.inject_watermark(init_latents, pattern_index) 
            
        
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
        
        #print(f'\n[generate_img] init_latents min/max before pipe call: {init_latents.min().item()}/{init_latents.max().item()}')
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
        #print(f'\n[get_inversed_latents] img min/max after transform: {img.min().item()}/{img.max().item()}') # -1, 1

        img_latents = self.pipe.get_image_latents(img, 
                                                  sample=False, 
                                                  batch_size=1,
                                                  num_channels_latents=16, 
                                                  height=512, # full height and width before
                                                  width=512,)

        #print(f'\n[get_inversed_latents] img_latents min/max after get_image_latents: {img_latents.min().item()}/{img_latents.max().item()}') # -4, 4
        reversed_latents = self.pipe.forward_diffusion(latents=img_latents, 
                                                       prompt=prompt, 
                                                       guidance_scale=1,
                                                       num_inference_steps=self.test_inf_steps,
                                                       device=self.device,)
        
        #print(f'\n[get_inversed_latents] reversed_latents min/max after forward_diffusion: {reversed_latents.min().item()}/{reversed_latents.max().item()}') # -3,5 , 3,5

        if isinstance(self.pipe, InversableFluxPipeline):
            reversed_latents = self.pipe._unpack_latents(latents=reversed_latents, 
                                                         height=512, 
                                                         width=512, 
                                                         vae_scale_factor=self.pipe.vae_scale_factor)
            
            reversed_latents = reversed_latents.to(torch.float32) # from flux, always have 16 channels
            
        #print(f'\n[get_inversed_latents] reversed_latents min/max after _unpack_latents: {reversed_latents.min().item()}/{reversed_latents.max().item()}')
        return reversed_latents, true_latents
    
    

    def eval_watermark(self, reversed_latents_nowm, reversed_latents_wm):
        

        pattern = self.Fourier_watermark_pattern_list[self.args.pattern_index]
        watermark_region_mask = self.watermark_region_mask.unsqueeze(0) # from 8, 64, 64 to 1, 8, 64, 64
        
        reversed_latents_nowm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_nowm), dim=(-1, -2))
        reversed_latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_wm), dim=(-1, -2))

    
        if not self.args.channel_min:
            # average over all channels
            #print("\n[ev] pattern[:, self.watermark_channel].shape: ", pattern[:, self.watermark_channel].shape) # 1, 8, 64, 64
            #print("[ev] reversed_latents_nowm_fft[:, self.watermark_channel].shape: ", reversed_latents_nowm_fft[:, self.watermark_channel].shape) # 1, 8, 64, 64
            #print("[ev] pattern[:, self.watermark_channel][watermark_region_mask].shape: ", pattern[:, self.watermark_channel][watermark_region_mask].shape) # 1, 8, 64, 64
            #print("[ev] reversed_latents_nowm_fft[:, self.watermark_channel][watermark_region_mask].shape: ", reversed_latents_nowm_fft[:, self.watermark_channel][watermark_region_mask].shape) # 1, 8, 64, 64
            diff_nowm = torch.abs(pattern[:, self.watermark_channel][watermark_region_mask] - reversed_latents_nowm_fft[:, self.watermark_channel][watermark_region_mask]).mean().item()
            diff_wm = torch.abs(pattern[:, self.watermark_channel][watermark_region_mask] - reversed_latents_wm_fft[:, self.watermark_channel][watermark_region_mask]).mean().item()
        
            return diff_nowm, diff_wm
        else:
            # return only the min distance of the channels
            #print("\n[ev] pattern[:, self.watermark_channel].shape: ", pattern[:, self.watermark_channel].shape) # 1, 8, 64, 64
            #print("[ev] reversed_latents_nowm_fft[:, self.watermark_channel].shape: ", reversed_latents_nowm_fft[:, self.watermark_channel].shape) # 1, 8, 64, 64
            diffs_nowm = []
            diffs_wm = []
            # the self.watermark_channel already selects only the watermarke channels
            diff_nowm = torch.abs(pattern[:, self.watermark_channel] - reversed_latents_nowm_fft[:, self.watermark_channel])
            diff_wm = torch.abs(pattern[:, self.watermark_channel] - reversed_latents_wm_fft[:, self.watermark_channel])
            #print("[ev] diff_nowm.shape: ", diff_nowm.shape) # 1, 8, 64, 64
            #print("[ev] diff_wm.shape: ", diff_wm.shape) # 1, 8, 64, 64
            for c_idx in range(len(self.watermark_channel)): # over the 8 channels
                channel_mask = watermark_region_mask[:, c_idx, ...]
                #print(f"\n[ev] channel: {c_idx}") # 1 to 8
                #print("\n[ev] channel_mask.shape: ", channel_mask.shape) # 1, 64, 64
                #print("[ev] diff_nowm[:, c_idx, ...].shape: ", diff_nowm[:, c_idx, ...].shape) # 1, 64, 64
                #print("[ev] diff_wm[:, c_idx, ...].shape: ", diff_wm[:, c_idx, ...].shape) # 1, 64, 64
                #print("[ev] diff_nowm[:, c_idx, ...][channel_mask].shape: ", diff_nowm[:, c_idx, ...][channel_mask].shape) # 556, cause channel_mask selects only the watermark region (the circle inside the 64x64 image)
                #print("[ev] diff_wm[:, c_idx, ...][channel_mask].shape: ", diff_wm[:, c_idx, ...][channel_mask].shape) # 556
                #print("[ev] diff_nowm[:, c_idx, ...][channel_mask].mean().item(): ", diff_nowm[:, c_idx, ...][channel_mask].mean().item()) # ca. 50-70
                #print("[ev] diff_wm[:, c_idx, ...][channel_mask].mean().item(): ", diff_wm[:, c_idx, ...][channel_mask].mean().item()) # ca 20-30 
                diffs_nowm.append(diff_nowm[:, c_idx, ...][channel_mask].mean().item())
                diffs_wm.append(diff_wm[:, c_idx, ...][channel_mask].mean().item())

            #print(f"\n[ev] diffs_nowm: {diffs_nowm}")
            #print(f"[ev] diffs_wm: {diffs_wm}")
            return min(diffs_nowm), min(diffs_wm)
        
    def viz_reversed_latents(self, true_latents_nowm, reversed_latents_nowm, true_latents_wm, reversed_latents_wm, attack_name, attack_vals, strength):
        
        true_latents_nowm_fft = torch.fft.fftshift(torch.fft.fft2(true_latents_nowm), dim=(-1, -2))
        reversed_latents_nowm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_nowm), dim=(-1, -2))
        true_latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(true_latents_wm), dim=(-1, -2))
        reversed_latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_wm), dim=(-1, -2))

        pattern = self.Fourier_watermark_pattern_list[self.args.pattern_index]
        watermark_region_mask = self.watermark_region_mask.unsqueeze(0) # from 8, 64, 64 to 1, 8, 64, 64

        # return only the min distance of the channels
        wm_channel_idx = 0
        ch_mean_abs_diff_wm_nowm_fft = []
        ch_mean_abs_diff_wm_wm_fft = []

        # these are for the abs. mean differene values
        abs_diff_wm_nowm_fft = torch.abs(pattern[:, self.watermark_channel] - reversed_latents_nowm_fft[:, self.watermark_channel]) # this already only has 2 channels (or 8, when 16 channels)
        abs_diff_wm_wm_fft = torch.abs(pattern[:, self.watermark_channel] - reversed_latents_wm_fft[:, self.watermark_channel])
        for c_idx in range(reversed_latents_nowm_fft.shape[1]): # over 4 (or 16) channels
            if c_idx in self.watermark_channel:
                channel_mask = watermark_region_mask[:, wm_channel_idx, ...]
                ch_mean_abs_diff_wm_nowm_fft.append(abs_diff_wm_nowm_fft[:, wm_channel_idx, ...][channel_mask].mean().item())
                ch_mean_abs_diff_wm_wm_fft.append(abs_diff_wm_wm_fft[:, wm_channel_idx, ...][channel_mask].mean().item())
                wm_channel_idx += 1
            else:
                ch_mean_abs_diff_wm_nowm_fft.append(0)
                ch_mean_abs_diff_wm_wm_fft.append(0)
           
        # while these are only for the visualization
        diff_wm_wm_fft = torch.zeros_like(reversed_latents_wm_fft)
        diff_wm_nowm_fft = torch.zeros_like(reversed_latents_nowm_fft)
        for i in range(len(self.watermark_channel)):
            channel_mask = watermark_region_mask[:, i, ...]
            diff_wm_wm_fft[:, self.watermark_channel[i], ...] = (pattern[:, self.watermark_channel[i], ...] - reversed_latents_wm_fft[:, self.watermark_channel[i], ...]) * channel_mask
            diff_wm_nowm_fft[:, self.watermark_channel[i], ...] = (pattern[:, self.watermark_channel[i], ...] - reversed_latents_nowm_fft[:, self.watermark_channel[i], ...]) * channel_mask
           
        # difference to their own ground truth latents
        diff_wm_true_fft = reversed_latents_wm_fft - true_latents_wm_fft # here, in true, the gt_patch should already be injected
        diff_nowm_true_fft = reversed_latents_nowm_fft - true_latents_nowm_fft
        mean_abs_diff_wm_true_fft = torch.abs(diff_wm_true_fft).mean().item()
        mean_abs_diff_nowm_true_fft = torch.abs(diff_nowm_true_fft).mean().item()

        title = (f'Reversed latents of method "{self.method}", model "{self.args.model_id}"\n'
                f'with attack "{attack_name}"={attack_vals[strength]}\n'
                f'wm_channels={self.latent_channels_wm} out of {self.latent_channels} channels'
        )
        save_path = f'{self.args.log_dir}/{self.method}_reversed_latents_{attack_name}_{attack_vals[strength]}.pdf'

        visualize_reversed_latents_fft(num_channels=self.latent_channels,
                                   reversed_latents_wm_fft=reversed_latents_wm_fft,
                                   reversed_latents_nowm_fft=reversed_latents_nowm_fft,
                                   diff_wm_wm_fft=diff_wm_wm_fft,
                                   diff_wm_nowm_fft=diff_wm_nowm_fft,
                                   diff_wm_true_fft=diff_wm_true_fft,
                                   diff_nowm_true_fft=diff_nowm_true_fft,
                                   ch_mean_abs_diff_wm_wm_fft=ch_mean_abs_diff_wm_wm_fft,
                                   ch_mean_abs_diff_wm_nowm_fft=ch_mean_abs_diff_wm_nowm_fft,
                                   mean_abs_diff_wm_true_fft=mean_abs_diff_wm_true_fft,
                                   mean_abs_diff_nowm_true_fft=mean_abs_diff_nowm_true_fft,
                                   title=title,
                                   save_path=save_path)





        