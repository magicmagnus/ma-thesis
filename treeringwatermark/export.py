import os
import sys
import copy
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import seed_everything, transform_img, plot_wm_pattern_fft, plot_wm_latents_fft, visualize_reversed_latents_fft
from .optim_utils import circle_mask# , transform_img

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# flux
from pipes.inverse_flux_pipeline import InversableFluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# sd
from pipes.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler


class TRWatermark():
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
        self.method = 'tr'
        self.num_images = args.num_images
        self.guidance_scale = args.guidance_scale
        self.args = args

        self.latent_channels = 4 if args.model_id == 'sd' else 16
        self.latent_channels_wm = args.latent_channels_wm
        self.shape = (1, self.latent_channels_wm, 64, 64)

        # wm specific args
        self.w_seed = args.w_seed
        self.w_channel = args.w_channel
        self.w_pattern = args.w_pattern
        self.w_mask_shape = args.w_mask_shape
        self.w_radius = args.w_radius
        self.w_measurement = args.w_measurement
        self.w_injection = args.w_injection
        self.w_pattern_const = args.w_pattern_const

        self.gt_patch = None
        self.watermarking_mask = None
        w_channel = []
        for i in range(self.latent_channels_wm // 4):
            w_channel.append(self.w_channel + i * 4)
        self.w_channel = w_channel

        # Load or generate watermark patterns
        key_id = f'{self.method}_ch_{self.w_channel}_r_{self.w_radius}_p_{self.w_pattern}_seed_{self.w_seed}_wmch_{self.latent_channels_wm}'
        key_path = f'keys/{key_id}.pkl'

        if not os.path.exists(key_path):
            # key does not exist yet, generate watermark and save it
            self.generate_watermarking_pattern()
            self.get_watermarking_mask()
            with open(key_path, 'wb') as f:
                pickle.dump((self.gt_patch, self.watermarking_mask), f)
            print(f'Generated TR keys and saved to file {key_path}')
        else:
            # load the existing keys
            with open(key_path, 'rb') as f:
                self.gt_patch, self.watermarking_mask = pickle.load(f)
            print(f'Loaded TR keys from file {key_path}')
        

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
        elif args.model_id == 'flux_s':
            print("\nUsing FLUX schnell model")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                'black-forest-labs/FLUX.1-schnell',
                subfolder='scheduler'
            )
            self.pipe = InversableFluxPipeline.from_pretrained(
                'black-forest-labs/FLUX.1-schnell',
                scheduler=scheduler,
                torch_dtype=torch.bfloat16,
                cache_dir=self.hf_cache_dir,
            ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.visualize_watermarking_pattern()

    def visualize_watermarking_pattern(self):
        
        pattern_plt = self.gt_patch.clone() * 0
        pattern_plt[self.watermarking_mask] = self.gt_patch[self.watermarking_mask].clone()

        
        title = (f'Tree-Ring Watermark in Frequency Domain\n'
                f'w_injection={self.w_injection}, w_channel={self.w_channel}\n'
                 f'w_radius={self.w_radius}, w_pattern={self.w_pattern}'
        )
        save_path = f'{self.args.log_dir}/{self.method}_wm_only.pdf'
        plot_wm_pattern_fft(num_channels=self.latent_channels_wm,
                            pattern=pattern_plt,
                            title=title,
                            save_path=save_path)
                        

        seed_everything(0) # to be same as first iteration in encode.py
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        
        init_latents_watermarked = self.inject_watermark(init_latents) 
        
        # perform FFT on the initial latents and apply the watermark pattern in the frequency domain
        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        init_latents_watermarked_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_watermarked), dim=(-1, -2))
            
        diff = init_latents_watermarked - init_latents # only for visualization, im image space

        title = (f'Tree-Ring Watermark in Frequency Domain\n'
                f'w_injection={self.w_injection}, w_channel={self.w_channel}\n'
                    f'w_radius={self.w_radius}, w_pattern={self.w_pattern}'
        )
        save_path = f'{self.args.log_dir}/{self.method}_wm_latents.pdf'
        plot_wm_latents_fft(num_channels=self.latent_channels,  
                            init_latents_fft=init_latents_fft,
                            init_latents_watermarked_fft=init_latents_watermarked_fft,
                            init_latents_watermarked=init_latents_watermarked,
                            diff=diff,
                            title=title,
                            save_path=save_path)
        
        
    # only called once in the beginning
    def generate_watermarking_pattern(self):
        
        seed_everything(self.w_seed)
        
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
        print(f'[gen] Generated watermarking pattern with shape {self.gt_patch.shape}')
    
    # only called once in the beginning
    def get_watermarking_mask(self):
        watermarking_mask = torch.zeros(self.shape, dtype=torch.bool).to(self.device)

        if self.w_mask_shape == 'circle':
            np_mask = circle_mask(self.shape[-1], r=self.w_radius)
            torch_mask = torch.tensor(np_mask).to(self.device)

            if self.w_channel == -1:
                # all channels
                watermarking_mask[:, :] = torch_mask
            else:
                for i in range(len(self.w_channel)):
                    watermarking_mask[:, self.w_channel[i]] = torch_mask
                #watermarking_mask[:, self.w_channel] = torch_mask
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
        print(f'[gen] Generated watermarking mask with shape {self.watermarking_mask.shape}')

    ############################# ENCODING ########################################
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
    
    def generate_img(self, current_prompt, do_wm, seed, num_images_per_prompt=1, **kwargs):
        
        init_latents_np = np.random.randn(1, self.latent_channels, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        
        if do_wm: 
            init_latents = self.inject_watermark(init_latents) 

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
    def get_inversed_latents(self, img, prompt='', do_wm=False, seed=None):
        
        # get the true latents from the image, as a comparison
        true_latents_np = np.random.randn(1, self.latent_channels, 64, 64)
        true_latents = torch.from_numpy(true_latents_np).to(torch.float32).to(self.device)
        
        if do_wm: 
            true_latents = self.inject_watermark(true_latents)
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

    def viz_reversed_latents(self, true_latents_nowm, reversed_latents_nowm, true_latents_wm, reversed_latents_wm, attack_name, attack_vals, strength):
        
        if 'complex' in self.w_measurement:
            true_latents_nowm_fft = torch.fft.fftshift(torch.fft.fft2(true_latents_nowm), dim=(-1, -2))
            reversed_latents_nowm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_nowm), dim=(-1, -2))
            true_latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(true_latents_wm), dim=(-1, -2))
            reversed_latents_wm_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_wm), dim=(-1, -2))
            target_patch = self.gt_patch
        elif 'seed' in self.w_measurement:
            true_latents_nowm_fft = true_latents_nowm
            reversed_latents_nowm_fft = reversed_latents_nowm
            true_latents_wm_fft = true_latents_wm
            reversed_latents_wm_fft = reversed_latents_wm
            target_patch = self.gt_patch
        else:
            NotImplementedError(f'w_measurement: {self.w_measurement}')

        # 1. first the reversed latents against the true WM patch,  to see if watermark is still “more clearly” visible in the WM than in the NOWM
        ch_mean_abs_diff_wm_wm_fft = []
        ch_mean_abs_diff_wm_nowm_fft = []

        for i in range(reversed_latents_nowm_fft.shape[1]): # over 4 (or 16) channels
            if i in self.w_channel: # todo multiple channels
                channel_mask = self.watermarking_mask[0, i]
                ch_mean_abs_diff_wm_wm_fft.append(torch.abs(target_patch[0, i][channel_mask] - reversed_latents_wm_fft[0, i][channel_mask]).mean().item())
                ch_mean_abs_diff_wm_nowm_fft.append(torch.abs(target_patch[0, i][channel_mask] - reversed_latents_nowm_fft[0, i][channel_mask]).mean().item())
            else:
                ch_mean_abs_diff_wm_wm_fft.append(0)
                ch_mean_abs_diff_wm_nowm_fft.append(0)

        
        diff_wm_wm_fft = (target_patch - reversed_latents_wm_fft) * self.watermarking_mask # only inside the watermarking mask the eval is done
        diff_wm_nowm_fft = (target_patch - reversed_latents_nowm_fft) * self.watermarking_mask
     
    
        diff_wm_true_fft = reversed_latents_wm_fft - true_latents_wm_fft # here, in true, the gt_patch should already be injected
        diff_nowm_true_fft = reversed_latents_nowm_fft - true_latents_nowm_fft
        mean_abs_diff_wm_true_fft = torch.abs(diff_wm_true_fft).mean().item()
        mean_abs_diff_nowm_true_fft = torch.abs(diff_nowm_true_fft).mean().item()

        
        
        if 'complex' in self.w_measurement: 
            
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

        else: # seed, in spatial domain
            pass

   


            

            
