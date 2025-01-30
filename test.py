import torch
import os
if "is/sg2" in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from diffusers import FluxPipeline

if "is/sg2" in os.getcwd():
    HF_CACHE_DIR = '/is/sg2/mkaut/.cache/huggingface/hub'
else:
    HF_CACHE_DIR = '/home/mkaut/.cache/huggingface/hub'

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", 
                                    torch_dtype=torch.bfloat16,
                                    cache_dir=HF_CACHE_DIR)
# pipe.enable_model_cpu_offload()
pipe.to("cuda")

prompt = "Two kids are playing baseball in Wii Sports"


image = pipe(prompt, 
            num_inference_steps=50, 
            guidance_scale=3.5,
            height=512,
            width=512
            ).images[0]

image.save("flux.png")