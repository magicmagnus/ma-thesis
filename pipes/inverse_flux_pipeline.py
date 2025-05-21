# inversable_flux.py
from functools import partial
import torch
import numpy as np
from pipes.modified_flux_pipeline import ModifiedFluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
import time 


def flux_forward(x_t, sigma_t, sigma_next, flow_output):
    """Forward process using FlowMatch equations"""
    return x_t + (sigma_next - sigma_t) * flow_output

def flux_backward(x_t, sigma_t, sigma_prev, flow_output):
    """Backward process using FlowMatch equations"""
    return x_t + (sigma_prev - sigma_t) * flow_output

# adapted from
# diffusers.pipelines.flux.pipeline_flux.py, __version__ = "0.32.2"
# and the InversableStableDiffusionPipeline from
# https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/inverse_stable_diffusion.py

class InversableFluxPipeline(ModifiedFluxPipeline):
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
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

        
    
    @torch.inference_mode()
    def backward_diffusion(
        self,
        prompt=None,
        latents=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        reverse_process: bool = False,
        device: torch.device = None,
        height: int = 512,
        width: int = 512,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        """Bidirectional diffusion process"""
        do_classifier_free_guidance = guidance_scale > 1.0
        #t0 = time.time()
        # 1. prepare the prompt
        (   prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(prompt=prompt,
                                prompt_2=None,
                            prompt_embeds=None,
                            pooled_prompt_embeds=None,
                            device=None,
                            num_images_per_prompt=1,
                            max_sequence_length=512,
                            lora_scale=None)
        #t1 = time.time()
        #print(f"Prompt encoding took {t1 - t0} seconds")

        #t0 = time.time()
        # 2. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1] 
        mu = calculate_shift(
            image_seq_len,
            base_seq_len=self.scheduler.config.base_image_seq_len,
            max_seq_len=self.scheduler.config.max_image_seq_len,
            base_shift=self.scheduler.config.base_shift,
            max_shift=self.scheduler.config.max_shift
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        #t1 = time.time()
        #print(f"Preparing timesteps took {t1 - t0} seconds")

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # Prepare latent image ids
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        #t0 = time.time()
        # reverse the necessary tensors
        # sigmas_orig = self.scheduler.sigmas # high to low, float32
        if reverse_process:
            print("[FluxPipe] Forward process")
            timesteps = torch.flip(timesteps, [0])
            self.scheduler.sigmas = torch.flip(self.scheduler.sigmas, [0])  # Reverse sigmas, float32
            self.scheduler._step_index = 0  # Start at the beginning of reversed sigmas
            print("[FluxPipe] reversed sigmas:", self.scheduler.sigmas) # [0.00, 0.25, 0.50, 0.75, 1.00], increasing
            print("[FluxPipe] reversed timesteps", timesteps) # [ 250.,  500.,  750., 1000.], increasing
            print("[FluxPipe] step index", self.scheduler._step_index) # 0 -> 1000, increasing
        else:
            # not printed, as normal backward is called from orig flux_pipeline.py
            print("[FluxPipe] Backward process")
            print("[FluxPipe] original sigmas:", self.scheduler.sigmas) # [1.00, 0.75, 0.50, 0.25, 0.00], decreasing
            print("[FluxPipe] original timesteps", timesteps) # [1000.,  750.,  500.,  250.]
            print("[FluxPipe] step index", self.scheduler._step_index) # None, increasing

        #t1 = time.time()

        #t0 = time.time()
        # previously in the loop:
        expanded_timesteps = torch.cat([
            torch.full((latents.shape[0],), t.to(latents.dtype).item(), device=latents.device, dtype=latents.dtype) 
            for t in timesteps
        ]).unsqueeze(1)
        #print("expanded_timesteps", expanded_timesteps.shape) # torch.Size([50, 1])
        #expanded_timesteps = expanded_timesteps.view(len(timesteps), -1)  # Shape: [num_steps, batch_size]
        #t1 = time.time()
        #print(f"Expanding timesteps took {t1 - t0} seconds") # too long

        # t0_loop = time.time()
        # model_time = 0
        # stepping_time = 0
        #print(f'\n\t[fowardcall] latents min/max before denoising loop: {latents.min().item()}/{latents.max().item()}')
        for i, t in enumerate(timesteps):
            #t0 = time.time()
            # Expand latents for guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            #t1 = time.time()
            #print(f"cat latents took {t1 - t0} seconds") # too long

            #t0 = time.time()
            # Create 1D timestep tensor that matches the batch size
            #timestep = t.to(latents.dtype)
            #timestep = torch.full((latent_model_input.shape[0],), timestep.item(), device=latent_model_input.device, dtype=latents.dtype)
            timestep = expanded_timesteps[i]
            #t1 = time.time()
            #print(f"Expanding latents took {t1 - t0} seconds") # too long
            #print("timestep", timestep) # torch.Size([50])
            

            # t0 = time.time()
            # Get model output
            model_output = self.transformer(
                hidden_states=latent_model_input.to(device),
                timestep=timestep / 1000, 
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            # t1 = time.time()
            # model_time += t1 - t0

            # Apply guidance
            # TODO: for backward pass, need guidance
            
            
            # Apply forward or backward step
            if reverse_process:
                # FORWARD, image to noise, x_0 --> ...  x_[t-1] --> x_t --> x_[t+1] --> ... --> x_T
                # t0 = time.time()
                # EQUIVALANT TO STEP FUNCTION
                # sample = latents.to(torch.float32)
                # sigma = sigmas_orig[- (i + 2)]    # sigmas_orig are still high to low
                # sigma_next = sigmas_orig[-(i + 1)]
                # # # x_t = x_[t-1] + (sigma - sigma-next) * pred_noise(x_t, t)
                # next_sample = sample + (sigma - sigma_next) * model_output
                # latents = next_sample.to(model_output.dtype)

                # print("[bwdiff] timestep", timestep) # 37
                # print("[bwdiff] sigma (manual):", sigmas_orig[-(i + 2)]) # 0.0369
                # print("[bwdiff] sigma_next (manual):", sigmas_orig[-(i + 1)]) # 0
                # print("[bwdiff] sigma (step function):", self.scheduler.sigmas[i]) # 0.0369
                # print("[bwdiff] sigma_next (step function):", self.scheduler.sigmas[i + 1]) # 0
                latents = self.scheduler.step(model_output=model_output,
                                              timestep=t,
                                              sample=latents,
                                              return_dict=False)[0]
                
                # t1 = time.time()
                # stepping_time += t1 - t0

            else:
                # BACKWARD, normal noise to image diffusion, x_T --> ...  x_[t+1] --> x_t --> x_[t-1] --> ... --> x_0	
                
                # EQUIVALANT TO STEP FUNCTION
                # sample = latents.to(torch.float32)
                # sigma = self.scheduler.sigmas[i]
                # sigma_next = self.scheduler.sigmas[i + 1]
                # # x_[t-1] = x_t + (sigma_next - sigma) * pred_noise(x_t, t)
                # prev_sample = sample + (sigma_next - sigma) * model_output
                # latents = prev_sample

                latents = self.scheduler.step(model_output=model_output,
                                              timestep=t,
                                              sample=latents,
                                              return_dict=False)[0] 
                
        # t1_loop = time.time()
        # print(f"diffusion loop {t1_loop - t0_loop} seconds") # 10.841530323028564 seconds
        # print(f"model time {model_time} seconds") # 8.0813729763031 seconds
        # print(f"stepping time {stepping_time} seconds")

        #print(f'\n\t[fowardcall] latents min/max after loop: {latents.min().item()}/{latents.max().item()}')

        return latents