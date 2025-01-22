# here we define a class that contains all important funtions to encode and decode images with watermarks, 
# so that we can compare it better to other methods on a higher level

import torch
import copy
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from diffusers import DPMSolverMultistepScheduler
from .inverse_stable_diffusion import InversableStableDiffusionPipeline
from .optim_utils import set_random_seed, circle_mask, transform_img


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
        self.shape = (1, 4, 64, 64)
        self.args = args

        self.exp_id = f'{self.method}_num_{self.num_images}_steps_{self.inf_steps}_fpr_{self.fpr}'
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder='scheduler')
        self.pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            # torch_dtype=torch.float16,
            # revision='fp16',
            torch_dtype=torch.float32,
            cache_dir=self.hf_cache_dir,

            ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

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

        # if the key/pattern (and mask) already exists, we load them, otherwise we generate them
        key_id = f'{self.method}_ch_{self.w_channel}_r_{self.w_radius}_p_{self.w_pattern}_seed_{self.w_seed}'
        key_path = f'keys/{key_id}.pkl'
        if not os.path.exists(key_path):
            self.generate_watermarking_pattern()
            self.get_watermarking_mask()
            with open(key_path, 'wb') as f:
                pickle.dump((self.gt_patch, self.watermarking_mask), f)
            print(f'Generated TR keys and saved to file {key_path}')
        else:
            with open(key_path, 'rb') as f:
                self.gt_patch, self.watermarking_mask = pickle.load(f)
            print(f'Loaded TR keys from file {key_path}')
        # if not os.path.exists(f'keys/{self.exp_id}.pkl'):
        #     self.generate_watermarking_pattern()
        #     self.get_watermarking_mask()
        #     with open(f'keys/{self.exp_id}.pkl', 'wb') as f:
        #         pickle.dump((self.gt_patch, self.watermarking_mask), f)
        #     print(f'Generated TR keys and saved to file keys/{self.exp_id}.pkl')
        # else:
        #     with open(f'keys/{self.exp_id}.pkl', 'rb') as f:
        #         self.gt_patch, self.watermarking_mask = pickle.load(f)
        #     print(f'Loaded TR keys from file keys/{self.exp_id}.pkl')
    	
        self.visualize_watermarking_pattern()

        
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

    def visualize_watermarking_pattern(self):
        set_random_seed(1)
        init_latents_np = np.random.randn(1, 4, 64, 64)
        init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to(self.device)
        init_latents_orig = init_latents.clone()
    
        pattern = self.gt_patch

        if self.w_injection == 'complex':
            # first plot only the masked pattern
            fig, ax = plt.subplots(2, 4, figsize=(10, 6))
            fig.subplots_adjust(hspace=0.3, wspace=0.1)
            for i in range(2):
                for j in range(4):
                    ax[i, j].axis('off')
            pattern_plt = pattern * 0
            pattern_plt[self.watermarking_mask] = pattern[self.watermarking_mask].clone()
            for i in range(4):
                ax[0, i].imshow(pattern_plt[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
                ax[1, i].imshow(pattern_plt[0, i].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[0, 0].set_title('Watermark pattern (real part)', loc='left', fontsize=10)
            ax[1, 0].set_title('Watermark pattern (imaginary part)', loc='left', fontsize=10)
            fig.suptitle(f'Tree-Ring Watermarking with w_injection={self.w_injection}, w_channel={self.w_channel}, w_radius={self.w_radius}, w_pattern={self.w_pattern}', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{self.args.log_dir}/{self.args.method}_wm_only.png', bbox_inches='tight', pad_inches=0.2)

            # perform FFT on the initial latents and apply the watermark pattern in the frequency domain
            init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
            init_latents_fft_watermarked = init_latents_fft.clone() # clone only for visualization
            init_latents_fft_watermarked[self.watermarking_mask] = pattern[self.watermarking_mask].clone()
            init_latents_watermarked = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft_watermarked, dim=(-1, -2))).real
            diff = init_latents_watermarked - init_latents_orig # only for visualization

            # plot the watermark pattern applied to the latents
            fig, ax = plt.subplots(5, 4, figsize=(10, 12))
            fig.subplots_adjust(hspace=0.3, wspace=0.1)
            for i in range(5):
                for j in range(4):
                    ax[i, j].axis('off')
            for i in range(4):
                ax[0, i].imshow(init_latents_fft[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
                ax[1, i].imshow(init_latents_fft_watermarked[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
                ax[2, i].imshow(init_latents_fft_watermarked[0, i].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
                ax[3, i].imshow(init_latents_watermarked[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
                ax[4, i].imshow(diff[0, i].cpu().numpy(), cmap='RdBu', vmin=-4, vmax=4)
            ax[0, 0].set_title('Original FFT of init_latents (real part)', loc='left', fontsize=10)
            ax[1, 0].set_title('FFT after watermarking (real part)', loc='left', fontsize=10)
            ax[2, 0].set_title('FFT after watermarking (imaginary part)', loc='left', fontsize=10)
            ax[3, 0].set_title('init_latents after watermarking (spatial domain)', loc='left', fontsize=10)
            ax[4, 0].set_title('Difference (watermarked - original)', loc='left', fontsize=10)

            fig.suptitle(f'Tree-Ring Watermarking with w_injection={self.w_injection}, w_channel={self.w_channel}, w_radius={self.w_radius}, w_pattern={self.w_pattern}', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{self.args.log_dir}/{self.args.method}_wm_latents.png', bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)
        elif self.w_injection == 'seed':	
            # first plot only the masked pattern (this time in the spatial domain)
            fig, ax = plt.subplots(1, 4, figsize=(10, 3))
            fig.subplots_adjust(hspace=0.3, wspace=0.1)
            for i in range(1):
                for j in range(4):
                    ax[j].axis('off')
            pattern_plt = pattern * 0
            pattern_plt[self.watermarking_mask] = pattern[self.watermarking_mask].clone()
            for i in range(4):
                ax[i].imshow(pattern_plt[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[0].set_title('Watermark pattern (spatial domain)', loc='left', fontsize=10)
            fig.suptitle(f'Tree-Ring Watermarking with w_injection={self.w_injection}, w_channel={self.w_channel}, w_radius={self.w_radius}, w_pattern={self.w_pattern}', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{self.args.log_dir}/{self.args.method}_wm_only.png', bbox_inches='tight', pad_inches=0.2)

            # apply the watermark pattern in the spatial domain
            init_latents_watermarked = init_latents.clone() # clone only for visualization
            init_latents_watermarked[self.watermarking_mask] = pattern[self.watermarking_mask].clone()
            diff = init_latents_watermarked - init_latents_orig # only for visualization

            # plot the watermark pattern applied to the latents
            fig, ax = plt.subplots(3, 4, figsize=(10, 9))
            fig.subplots_adjust(hspace=0.3, wspace=0.1)
            for i in range(3):
                for j in range(4):
                    ax[i, j].axis('off')
            for i in range(4):
                ax[0, i].imshow(init_latents[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
                ax[1, i].imshow(init_latents_watermarked[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
                ax[2, i].imshow(diff[0, i].cpu().numpy(), cmap='RdBu', vmin=-4, vmax=4)
            ax[0, 0].set_title('Original init_latents (spatial domain)', loc='left', fontsize=10)
            ax[1, 0].set_title('init_latents after watermarking', loc='left', fontsize=10)
            ax[2, 0].set_title('Difference (watermarked - original)', loc='left', fontsize=10)

            fig.suptitle(f'Tree-Ring Watermarking with w_injection={self.w_injection}, w_channel={self.w_channel}, w_radius={self.w_radius}, w_pattern={self.w_pattern}', fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{self.args.log_dir}/{self.args.method}_wm_latents.png', bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)
    
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
            num_inference_steps=self.test_inf_steps,
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

    def viz_reversed_latents(self, reversed_latents_no_w, reversed_latents_w, attack_name=None, attack_vals=[None], strength=0):
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
            abs_diff_per_channel_no_wm = []
            abs_diff_per_channel_wm = []
            
            for i in range(4): # indiv over the channels
                channel_mask = self.watermarking_mask[0, i]
                diff_wm = reversed_latents_w_fft[0, i] - target_patch[0, i]
                diff_no_wm = reversed_latents_no_w_fft[0, i] - target_patch[0, i]
                abs_diff_per_channel_wm.append(torch.abs(diff_wm[channel_mask]).mean().item())
                abs_diff_per_channel_no_wm.append(torch.abs(diff_no_wm[channel_mask]).mean().item())

            diff_wm = reversed_latents_w_fft - target_patch
            diff_wm = diff_wm * self.watermarking_mask
            
            diff_no_wm = reversed_latents_no_w_fft - target_patch
            diff_no_wm = diff_no_wm * self.watermarking_mask
           
            if 'complex' in self.w_measurement: 
                # visualize the reversed wm_latents against the watermark pattern
                fig, ax = plt.subplots(4, 4, figsize=(10, 12))
                fig.subplots_adjust(hspace=0.3, wspace=0.1)
                for i in range(4):
                    for j in range(4):
                        ax[i, j].axis('off') 
                for i in range(4):
                    ax[0, i].imshow(reversed_latents_w_fft[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
                    ax[1, i].imshow(reversed_latents_w_fft[0, i].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
                    ax[2, i].imshow(diff_wm[0, i].real.cpu().numpy(), cmap='RdBu',vmin=-150, vmax=150)
                    ax[2, i].set_title(f'abs. avg. diff: {abs_diff_per_channel_wm[i]:.2f} (real)', loc='left', fontsize=10)
                    ax[3, i].imshow(diff_wm[0, i].imag.cpu().numpy(), cmap='RdBu',vmin=-150, vmax=150)
                    ax[3, i].set_title(f'(imag)', loc='center', fontsize=10)
                ax[0, 0].set_title('Reversed FFT of wm_latents (real part)', loc='left', fontsize=10)
                ax[1, 0].set_title('Reversed FFT of wm_latents (imaginary part)', loc='left', fontsize=10)
                fig.colorbar(ax[3, 0].imshow(diff_wm[0, 0].imag.cpu().numpy(), cmap='RdBu',vmin=-150, vmax=150), ax=ax[3, 0], orientation='horizontal', fraction=0.046, pad=0.04)
                fig.suptitle(f'Reversed WM latents of {self.method} with attack {attack_name}: {attack_vals[strength]}', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{self.args.log_dir}/{self.method}_reversed_wm_latents_{attack_name}_{attack_vals[strength]}.png', bbox_inches='tight', pad_inches=0.2)
                plt.close(fig)
                
                # visualize the reversed no_wm_latents against the watermark pattern
                fig, ax = plt.subplots(4, 4, figsize=(10, 12))
                fig.subplots_adjust(hspace=0.3, wspace=0.1)
                for i in range(4):
                    for j in range(4):
                        ax[i, j].axis('off')
                for i in range(4):
                    ax[0, i].imshow(reversed_latents_no_w_fft[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
                    ax[1, i].imshow(reversed_latents_no_w_fft[0, i].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
                    ax[2, i].imshow(diff_no_wm[0, i].real.cpu().numpy(), cmap='RdBu',vmin=-150, vmax=150)
                    ax[2, i].set_title(f'abs. avg. diff: {abs_diff_per_channel_no_wm[i]:.2f} (real)', loc='left', fontsize=10)
                    ax[3, i].imshow(diff_no_wm[0, i].imag.cpu().numpy(), cmap='RdBu',vmin=-150, vmax=150)
                    ax[3, i].set_title(f'(imag)', loc='center', fontsize=10)
                ax[0, 0].set_title('Reversed FFT of no_wm_latents (real part)', loc='left', fontsize=10)
                ax[1, 0].set_title('Reversed FFT of no_wm_latents (imaginary part)', loc='left', fontsize=10)
                fig.colorbar(ax[3, 0].imshow(diff_no_wm[0, 0].imag.cpu().numpy(), cmap='RdBu',vmin=-150, vmax=150), ax=ax[3, 0], orientation='horizontal', fraction=0.046, pad=0.04)
                fig.suptitle(f'Reversed no_wm latents of {self.method} with attack {attack_name}: {attack_vals[strength]}', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{self.args.log_dir}/{self.method}_reversed_no_wm_latents_{attack_name}_{attack_vals[strength]}.png', bbox_inches='tight', pad_inches=0.2)
                plt.close(fig)
            else: # seed, in spatial domain
                pass


        else:
            NotImplementedError(f'w_measurement: {self.w_measurement}')