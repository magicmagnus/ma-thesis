# modified_flux.py
from turtle import width
from typing import Callable, List, Optional, Union, Any, Dict
import torch
import PIL
import numpy as np

from diffusers import FluxPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.flux.pipeline_flux_control_img2img import retrieve_latents

# adapted from
# diffusers.pipelines.flux.pipeline_flux.py
# and the InversableStableDiffusionPipeline from
# https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/modified_stable_diffusion.py

class ModifiedFluxPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    init_latents: Optional[torch.FloatTensor]

class ModifiedFluxPipeline(FluxPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        scheduler,
        image_encoder=None,
        feature_extractor=None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor):
        scaled_latents = 1 / 0.18215 * latents
        image = self.vae.decode(scaled_latents).sample
        return image
    
    
    # prepare for direction image -> latent_noise
    @torch.inference_mode()
    def get_image_latents(self, 
                          image, 
                          sample, 
                          height, 
                          width, 
                          num_channels_latents, 
                          batch_size, 
                          generator=None, 
                          **kwargs):

        if sample == False:
            sample_mode = "argmax"
        else:
            sample_mode = "sample"

        # first encode the image to latent space of the VAE
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode=sample_mode)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor  # (1, 16, 64, 64)

        # then transform the latents to the shape of the Flux model (1, 1024, 64)
        batch_size = image_latents.shape[0]
        num_channels_latents = image_latents.shape[1]
        
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)  # (1, 1024, 64)

        return image_latents
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None, # (1, 1024, 64)
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width) # (1, 16, 64, 64)

        if latents is not None:
            # print("[prepare latents] using custom latents")
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) # (1, 16, 64, 64)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width) # (1, 1024, 64)
        
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids
    
    # prepare latents for direction latent_noise -> image, (1, 16, 64, 64) -> (1, 1024, 64)
    def reshape_latents_SD_to_flux(self, 
                                   wm_latents, # (1, 16, 64, 64)
                                   batch_size,
                                   num_channels_latents, # default 16
                                   height,
                                   width,
                                   dtype,
                                   device,
                                   generator):
        
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        #shape = (batch_size, num_channels_latents, height, width) # (1, 16, 64, 64)

        # if isinstance(generator, list) and len(generator) != batch_size:
        #     raise ValueError(
        #         f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
        #         f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        #     )
        
        #raw_latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) # (1, 16, 64, 64)
        
        # if wm_latents.shape[1] < raw_latents.shape[1]: # then wm_latents will have 4 channels
        #     # for Flux model, RID and TR watermark latents and their non-watermarked latents still only have 4 channels, (2 + 2 cases)
        #     print("imprinting watermark to the first 4 channels of the raw latents")
        #     raw_latents[:, :wm_latents.shape[1], :, :] = wm_latents # merge the wm_latens into the first 4 channels of the raw_latents
        # elif wm_latents.shape[1] == raw_latents.shape[1]:
        #     # for Flux model, PRC or GS watermarked and their non-watermarked latents have 16 channels (2 + 2 cases)
        #     print("the watermark latents have the same number of channels as the raw latents")
        #     raw_latents = wm_latents                                # for watermarks PRC and GaussianShading, that have a variable number of channels/length of the WM                               
        # else:
        #     RuntimeError("The watermark latents have more channels than the raw latents.")
        # raw_latents[:, :4, :, :] = wm_latents     # merge the wm_latens into the first 4 channels of the raw_latents
        #latents = self._pack_latents(raw_latents, batch_size, num_channels_latents, height, width)   # (1, 1024, 64)
        latents = self._pack_latents(wm_latents, batch_size, num_channels_latents, height, width)   # (1, 1024, 64)

        # latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents #, latent_image_ids

