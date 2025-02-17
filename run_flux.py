import torch
import os
if "is/sg2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from diffusers import FluxPipeline, StableDiffusionPipeline, DiffusionPipeline
from utils import seed_everything
import numpy as np

from pipes.inverse_flux_pipeline import InversableFluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from pipes.inverse_stable_diffusion import InversableStableDiffusionPipeline

if "is/sg2" in os.getcwd():
    HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
else:
    HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

# pipe = InversableFluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", 
#                                     torch_dtype=torch.bfloat16,
#                                     cache_dir=HF_CACHE_DIR)

pipe = StableDiffusionPipeline.from_pretrained(
                'stabilityai/stable-diffusion-2-1-base',
                torch_dtype=torch.float32,
                cache_dir=HF_CACHE_DIR)
pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "a gorgeous woman, photo by peter lindbergh, realistic, natural sunlight, smooth face, perfect eyes, symmetrical, full body shot, wide angle, sharp focus, 8 k high definition, insanely detailed, intricate, elegant, art by artgerm"

resolution = 512
seed = 13

seed_everything(seed)
# init_latents_np = np.random.randn(1, 4, resolution//8, resolution//8)
# init_latents = torch.from_numpy(init_latents_np).to(torch.float32).to("cuda")
# latents = pipe.reshape_latents_SD_to_flux(init_latents,
                                        #   batch_size=1,
                                        #   num_channels_latents=16, # later try to set it to 4
                                        #   height=512, # full height and width before
                                        #   width=512,
                                        #   dtype=torch.float32,
                                        #   device="cuda",
                                        #   generator=None,)
# seed_everything(seed)
# # 

prompts = [
    "dog in the park",
    "a beautiful sunset",
    "a cute cat",
    "a beautiful painting",
     "dog in the park",
    "a beautiful sunset",
    "a cute cat",
    "a beautiful painting",
     "dog in the park",
    "a beautiful sunset",
    "a cute cat",
    "a beautiful painting",
     "dog in the park",
    "a beautiful sunset",
    "a cute cat",
    "a beautiful painting",
]
images = pipe(prompts, 
            num_inference_steps=50, 
            guidance_scale=2.0,
            height=resolution,# ,
            width=resolution,
            #latents=latents,
            ).images

import matplotlib.pyplot as plt

fig, ax = plt.subplots(int(np.ceil(np.sqrt(len(prompts)))), int(np.ceil(np.sqrt(len(prompts)))), figsize=(20, 20))
for i in range(len(prompts)):
    ax[i // int(np.ceil(np.sqrt(len(prompts))) ), i % int(np.ceil(np.sqrt(len(prompts))) )].imshow(images[i])
    ax[i // int(np.ceil(np.sqrt(len(prompts))) ), i % int(np.ceil(np.sqrt(len(prompts))) )].axis("off")

plt.savefig("flux_prompts.png")
#