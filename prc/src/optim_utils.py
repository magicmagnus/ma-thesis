import torch
from torchvision import transforms
from datasets import load_dataset
import random
import numpy as np

from PIL import Image, ImageFilter


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def get_dataset(datasets):
    if 'laion' in datasets:
        dataset = load_dataset(datasets)['train']
        prompt_key = 'TEXT'
    else:
        dataset = load_dataset(datasets)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key

def print2file(logfile, *args):
    print(*args)
    print(file=logfile, *args)

def image_distortion(img1, img2, seed, args, i, print_args=True):
    if print_args:
        save_name = 'example'

    if hasattr(args, 'r_degree'): # number between 0 and 360
        img1 = transforms.RandomRotation((args.r_degree[i], args.r_degree[i]))(img1)
        img2 = transforms.RandomRotation((args.r_degree[i], args.r_degree[i]))(img2)
        if print_args: 
            print2file(args.log_file, f"Rotating images by {args.r_degree[i]} degrees")
            save_name += f"_rot{args.r_degree[i]}"

    if hasattr(args, 'jpeg_ratio'): # number between 0 and 100
        img1.save(f"tmp_{args.jpeg_ratio[i]}.jpg", quality=args.jpeg_ratio[i])
        img1 = Image.open(f"tmp_{args.jpeg_ratio[i]}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio[i]}.jpg", quality=args.jpeg_ratio[i])
        img2 = Image.open(f"tmp_{args.jpeg_ratio[i]}.jpg")
        if print_args: 
            print2file(args.log_file, f"Compressing images with JPEG quality {args.jpeg_ratio[i]}")
            save_name += f"_jpeg{args.jpeg_ratio[i]}"

    if hasattr(args, 'crop_scale') and hasattr(args, 'crop_ratio'): # scale between 0 and 1, ratio between 0 and 1
        set_random_seed(seed)
        ratio = (args.crop_ratio[i], (1+(1-args.crop_ratio[i])))	
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale[i], args.crop_scale[i]), ratio=ratio)(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale[i], args.crop_scale[i]), ratio=ratio)(img2)
        if print_args: 
            print2file(args.log_file, f"Cropping images with scale {args.crop_scale[i]} and ratio {args.crop_ratio[i]}")
            save_name += f"_crop{args.crop_scale[i]}_{args.crop_ratio[i]}"
        
    if hasattr(args, 'gaussian_blur_r'):# radius between 0 and inf (ca. 50)
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r[i]))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r[i]))
        if print_args: 
            print2file(args.log_file, f"Applying Gaussian blur with radius {args.gaussian_blur_r[i]}")
            save_name += f"_blur{args.gaussian_blur_r[i]}"

    if hasattr(args, 'gaussian_std'): # standard deviation between 0 and inf (ca. 1)
        set_random_seed(seed)
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std[i], img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))
        if print_args: 
            print2file(args.log_file, f"Adding Gaussian noise with standard deviation {args.gaussian_std[i]}")
            save_name += f"_noise{args.gaussian_std[i]}"

    if hasattr(args, 'brightness_factor'): # factor between 0 and inf (ca. 20)
        img1 = transforms.ColorJitter(brightness=args.brightness_factor[i])(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor[i])(img2)
        if print_args: 
            print2file(args.log_file, f"Adjusting brightness with factor {args.brightness_factor[i]}")
            save_name += f"_bright{args.brightness_factor[i]}"

    if print_args:
        img1.save(f"{args.log_dir}/{save_name}_img1.png")

    return img1, img2