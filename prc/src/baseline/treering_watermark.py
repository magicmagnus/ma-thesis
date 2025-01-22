import numpy as np
import torch
from torchvision import transforms
import PIL
import pickle
from diffusers import DDIMInverseScheduler
from typing import Union, List, Tuple
import hashlib
import os

#
import matplotlib.pyplot as plt 


def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0) ** 2 + (y - y0) ** 2) <= r ** 2


def _get_pattern(shape, w_pattern='ring', generator=None):
    gt_init = torch.randn(shape, generator=generator)

    if 'rand' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'ring' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = gt_patch.clone().detach()
        for i in range(shape[-1] // 2, 0, -1):
            tmp_mask = _circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


# def get_noise(shape: Union[torch.Size, List, Tuple], model_hash: str) -> torch.Tensor:
def tr_get_noise(shape: Union[torch.Size, List, Tuple], keys_path, from_file: str = None, generator=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not from_file:
        # for now we hard code all hyperparameters
        w_channel = 2#0  # id for watermarked channel
        w_radius = 10  # watermark radius
        w_pattern = 'ring' #'rand'  # watermark pattern

        # get watermark key and mask
        np_mask = _circle_mask(shape[-1], r=w_radius)
        torch_mask = torch.tensor(np_mask)
        w_mask = torch.zeros(shape, dtype=torch.bool)
        w_mask[:, w_channel] = torch_mask

        w_key = _get_pattern(shape, w_pattern=w_pattern, generator=generator)

        # inject watermark
        assert len(shape) == 4, f"Make sure you pass a `shape` tuple/list of length 4 not {len(shape)}"
        assert shape[0] == 1, f"For now only batch_size=1 is supported, not {shape[0]}."

        # visualize some before and after
        # we want to see the watermark pattern in the frequency domain, 
        # and the init_latents in the spatial domain before and after watermarking
        

        init_latents = torch.randn(shape, generator=generator)
        init_latents_orig = init_latents.clone()
        # 1 plot the init_latents
        fig, ax = plt.subplots(5, 4)
        for i in range(4):
            im = ax[0, i].imshow(init_latents[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[0, i].axis('off')
            if i == 0:
                fig.colorbar(im, ax=ax[0, i], orientation='vertical', label='init_latents', location='left')
        ax[0, 0].set_title('init_latents, spatial domain')
    
        # 2 plot the watermark pattern which is the w_key, already in the frequency domain
        for i in range(4):
            im = ax[1, i].imshow(w_key[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[1, i].axis('off')
            if i == 0:
                fig.colorbar(im, ax=ax[1, i], orientation='vertical', label='w_key', location='left')
        ax[1, 0].set_title('w_key, frequency domain, (real part)')

        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        init_latents_fft[w_mask] = w_key[w_mask].clone()
        init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

        # 3 plot the init_latents_fft after watermarking
        for i in range(4):
            im = ax[2, i].imshow(init_latents_fft[0, i].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
            ax[2, i].axis('off')
            if i == 0:
                fig.colorbar(im, ax=ax[2, i], orientation='vertical', label='init_latents_fft', location='left')
        ax[2, 0].set_title('init_latents_fft after watermarking, frequency domain, (real part)')
    
        # 4 plot the init_latents after watermarking
        for i in range(4):
            im = ax[3, i].imshow(init_latents[0, i].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
            ax[3, i].axis('off')
            if i == 0:
                fig.colorbar(im, ax=ax[3, i], orientation='vertical', label='init_latents', location='left')
        ax[3, 0].set_title('init_latents after watermarking, spatial domain')

        # 6 plot the difference between the init_latents before and after watermarking
        for i in range(4):
            im = ax[4, i].imshow((init_latents - init_latents_orig)[0, i].cpu().numpy(), cmap='RdBu', vmin=-4, vmax=4)
            ax[4, i].axis('off')
            if i == 0:
                fig.colorbar(im, ax=ax[4, i], orientation='vertical', label='init_latents - init_latents_orig', location='left')
        ax[4, 0].set_title('init_latents - init_latents_orig')	

        fig.suptitle(f'Tree-Ring Watermarking with w_channel={w_channel}, w_radius={w_radius}, w_pattern={w_pattern}')

        plt.savefig('watermarking.png')
        plt.close()
        
        print("min(init_latents):", init_latents.min())
        print("max(init_latents):", init_latents.max())


        # convert the tensor to bytes
        tensor_bytes = init_latents.numpy().tobytes()

        # generate a hash from the bytes
        hash_object = hashlib.sha256(tensor_bytes)
        hex_dig = hash_object.hexdigest()

        file_name = "_".join([hex_dig, str(w_channel), str(w_radius), w_pattern]) + ".pkl"
        file_path = os.path.join(keys_path, file_name)
        print(f"Saving watermark key to {file_path}")
        with open(f'{file_path}', 'wb') as f:
            pickle.dump((w_key, w_mask), f)

    else:
        file_name = f"{from_file}.pkl"
        file_path = os.path.join(keys_path, file_name)

        with open(f'{file_path}', 'rb') as f:
            w_key, w_mask = pickle.load(f)
        init_latents = torch.randn(shape, generator=generator)

        init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
        init_latents_fft[w_mask] = w_key[w_mask].clone()
        init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    return init_latents, w_key, w_mask


def _transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


# def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], model_hash: str):
def tr_detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], pipe, keys_path, model_hash):
    detection_time_num_inference = 50
    threshold = 72

    file_name = f"{model_hash}.pkl"
    file_path = os.path.join(keys_path, file_name)

    with open(f'{file_path}', 'rb') as f:
        w_key, w_mask = pickle.load(f)

    # ddim inversion
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    img = _transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.18215
    inverted_latents = pipe(
        prompt='',
        latents=image_latents,
        guidance_scale=1,
        num_inference_steps=detection_time_num_inference,
        output_type='latent',
    )
    inverted_latents = inverted_latents.images.float().cpu()

    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
    dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

    if dist <= threshold:
        pipe.scheduler = curr_scheduler
        return dist, True

    return dist, False
