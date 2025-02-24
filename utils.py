import os
import random
import numpy as np
import torch
import json
from PIL import Image, ImageFilter
from torchvision import transforms

def print2file(logfile, *args):
    print(*args)
    print(file=logfile, *args)

def seed_everything(seed, workers=False):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Added this line to avoid cuBLAS memory error
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        return seed

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

def image_distortion(img1, img2, seed, args, i, print_args=True):
    if print_args:
        save_name = 'example'

    if hasattr(args, 'r_degree'): # number between 0 and 360
        img1 = transforms.RandomRotation((args.r_degree[i], args.r_degree[i]))(img1)
        img2 = transforms.RandomRotation((args.r_degree[i], args.r_degree[i]))(img2)
        if print_args: 
            #print2file(args.log_file, f"Rotating images by {args.r_degree[i]} degrees")
            save_name += f"_rot{args.r_degree[i]}"

    if hasattr(args, 'jpeg_ratio'): # number between 0 and 100
        img1.save(f"tmp_{args.jpeg_ratio[i]}.jpg", quality=args.jpeg_ratio[i])
        img1 = Image.open(f"tmp_{args.jpeg_ratio[i]}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio[i]}.jpg", quality=args.jpeg_ratio[i])
        img2 = Image.open(f"tmp_{args.jpeg_ratio[i]}.jpg")
        if print_args: 
            #print2file(args.log_file, f"Compressing images with JPEG quality {args.jpeg_ratio[i]}")
            save_name += f"_jpeg{args.jpeg_ratio[i]}"

    if hasattr(args, 'crop_scale') and hasattr(args, 'crop_ratio'): # scale between 0 and 1, ratio between 0 and 1
        seed_everything(seed)
        ratio = (args.crop_ratio[i], (1+(1-args.crop_ratio[i])))	
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale[i], args.crop_scale[i]), ratio=ratio)(img1)
        seed_everything(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale[i], args.crop_scale[i]), ratio=ratio)(img2)
        if print_args: 
            #print2file(args.log_file, f"Cropping images with scale {args.crop_scale[i]} and ratio {args.crop_ratio[i]}")
            save_name += f"_crop{args.crop_scale[i]}_{args.crop_ratio[i]}"
        
    if hasattr(args, 'gaussian_blur_r'):# radius between 0 and inf (ca. 50)
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r[i]))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r[i]))
        if print_args: 
            #print2file(args.log_file, f"Applying Gaussian blur with radius {args.gaussian_blur_r[i]}")
            save_name += f"_blur{args.gaussian_blur_r[i]}"

    if hasattr(args, 'gaussian_std'): # standard deviation between 0 and inf (ca. 1)
        seed_everything(seed)
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std[i], img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))
        if print_args: 
            #print2file(args.log_file, f"Adding Gaussian noise with standard deviation {args.gaussian_std[i]}")
            save_name += f"_noise{args.gaussian_std[i]}"

    if hasattr(args, 'brightness_factor'): # factor between 0 and inf (ca. 20)
        img1 = transforms.ColorJitter(brightness=[args.brightness_factor[i], args.brightness_factor[i]])(img1)
        img2 = transforms.ColorJitter(brightness=[args.brightness_factor[i], args.brightness_factor[i]])(img2)
        if print_args: 
            #print2file(args.log_file, f"Adjusting brightness with factor {args.brightness_factor[i]}")
            save_name += f"_bright{args.brightness_factor[i]}"

    if print_args:
        img1.save(f"{args.log_dir}/{save_name}_img1.png")

    return img1, img2

def get_dirs(args, script_type, extra=None):

    base_dir = os.path.join("experiments", 
                            args.method,
                            args.model_id,
                            args.dataset_id,
                            f"num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_gdscale_{args.guidance_scale}"
                            )

    # the dir in which the logs will be saved
    log_dir = os.path.join(base_dir,
                           script_type,
                           "logs",
                           )
    # the dir in which the encoded images will be saved/loaded from 
    data_dir = os.path.join(base_dir,
                            "encoded_imgs", # data is always saved in the encoded_imgs folder
                            "data",
                            )
    
    return log_dir, data_dir
    
def create_and_save_decode_confs(args):

    # in args are args.log_dir and args.data_dir

    # goal: load the templates, merge with current args, and save in the log_dir
    # load the templates
    # copy the args
    args_copy = args

    templates_dir = os.path.join('confs', 'decode_templates')
    # replace encoded_imgs with decoded_imgs in "data_dir"
    output_conf_dir = os.path.join("experiments", 
                            args.method,
                            args.model_id,
                            args.dataset_id,
                            f"num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_gdscale_{args.guidance_scale}",
                            "decode_imgs",
                            "confs"
                            )
    output_jobs_dir = os.path.join("experiments", 
                            args.method,
                            args.model_id,
                            args.dataset_id,
                            f"num_{args.num_images}_steps_{args.inf_steps}_fpr_{args.fpr}_gdscale_{args.guidance_scale}",
                            "decode_imgs",
                            "jobs"
                            )
                                   
    os.makedirs(output_conf_dir, exist_ok=True)
    os.makedirs(output_jobs_dir, exist_ok=True)
    os.makedirs(os.path.join(output_jobs_dir, "logs"), exist_ok=True)

    templates = [t for t in os.listdir(templates_dir) if t.endswith('.json')]
    # order alphabetically
    templates = sorted(templates)
    template_job_bash = os.path.join(templates_dir, 'jobs', 'decode.sh')
    template_job_sub = os.path.join(templates_dir, 'jobs', 'decode.sub')

    # so basically, in the templates are the decode.json files, and we wanna 
    # take our source (only one) encode.json that in args.config, and merge it with the decode.json files
    # so we first load the source, 
    # then for all the templates, we load them, merge them with the source (double attributes are
    # therefore taken from the templates), and save them in the output_conf_dir

    with open(args.config, 'r') as f:
        source = json.load(f)

    # open the job templates
    with open(template_job_bash, 'r') as f:
        job_bash_all = f.read()
    with open(template_job_sub, 'r') as f:
        job_sub_all = f.read()

    job_bash_decode_all = job_bash_all
    job_bash_attack_all = job_bash_all

    for template in templates:
        with open(os.path.join(templates_dir, template), 'r') as f:
            decode = json.load(f)

        # merge the two
        run_name = decode['run_name']
        decode.update(source)
        decode['run_name'] = run_name

        # save the merged
        with open(os.path.join(output_conf_dir, template), 'w') as f:
            json.dump(decode, f, indent=4)

         # open the job templates
        with open(template_job_bash, 'r') as f:
            job_bash = f.read()
        with open(template_job_sub, 'r') as f:
            job_sub = f.read()

        # the job files, decode
        job_bash_decode = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python decode_imgs.py --config {output_conf_dir}/{template}"
        job_bash_decode_all += f"\n/is/sg2/mkaut/miniconda3/bin/python decode_imgs.py --config {output_conf_dir}/{template}"
        # attack
        job_bash_attack = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python attack_imgs.py --config {output_conf_dir}/{template}"
        job_bash_attack_all += f"\n/is/sg2/mkaut/miniconda3/bin/python attack_imgs.py --config {output_conf_dir}/{template}"

        

        # save the job files as .sh files
        os.makedirs(os.path.join(output_jobs_dir, "decode"), exist_ok=True)
        os.makedirs(os.path.join(output_jobs_dir, "attack"), exist_ok=True)
        template_name = template.split(".")[0]
        with open(os.path.join(output_jobs_dir, "decode", f"{template_name}.sh"), 'w') as f:
            f.write(job_bash_decode)
        with open(os.path.join(output_jobs_dir, "attack", f"{template_name}.sh"), 'w') as f:
            f.write(job_bash_attack)
        if 'surr' in template:
            job_bash_train = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python attack_train_surrogate.py --config {output_conf_dir}/{template}"
            with open(os.path.join(output_jobs_dir, f"train_{template_name}.sh"), 'w') as f:
                f.write(job_bash_train)


        # to the sub file, add the final lines 
        job_sub_decode = job_sub + f"\narguments = /fast/mkaut/ma-thesis/{output_jobs_dir}/decode/{template_name}.sh"
        job_sub_decode += f"\nerror = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).err"
        job_sub_decode += f"\noutput = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).out"
        job_sub_decode += f"\nlog = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).log"
        # mem requirements: sd = ca 8G, flux ca 33G, clip ca 6G
        job_sub_decode += f"\nrequest_memory = {18432 if args.model_id == 'sd' else 46080}" 
        job_sub_decode += f"\nqueue"

        job_sub_attack = job_sub + f"\narguments = /fast/mkaut/ma-thesis/{output_jobs_dir}/attack/{template_name}.sh"
        job_sub_attack += f"\nerror = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).err"
        job_sub_attack += f"\noutput = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).out"
        job_sub_attack += f"\nlog = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).log"
        job_sub_attack += f"\nrequest_memory = {18432}"
        job_sub_attack += f"\nqueue"

        # save the job files as .sub files
        with open(os.path.join(output_jobs_dir, "decode", f"{template_name}.sub"), 'w') as f:
            f.write(job_sub_decode)
        with open(os.path.join(output_jobs_dir, "attack", f"{template_name}.sub"), 'w') as f:
            f.write(job_sub_attack)
        if 'surr' in template:
            job_sub_train = job_sub + f"\narguments = /fast/mkaut/ma-thesis/{output_jobs_dir}/{template_name}.sh"
            job_sub_train += f"\nerror = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).err"
            job_sub_train += f"\noutput = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).out"
            job_sub_train += f"\nlog = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/{template_name}.$(Process).log"
            job_sub_train += f"\nrequest_memory = {18432}"
            job_sub_train += f"\nqueue"
            with open(os.path.join(output_jobs_dir, f"train_{template_name}.sub"), 'w') as f:
                f.write(job_sub_train)

    # save the job files as .sh files
    with open(os.path.join(output_jobs_dir, "decode_all.sh"), 'w') as f:
        f.write(job_bash_decode_all)
    with open(os.path.join(output_jobs_dir, "attack_all.sh"), 'w') as f:
        f.write(job_bash_attack_all)

    # to the sub file, add the final lines
    job_sub_decode_all = job_sub + f"\narguments = /fast/mkaut/ma-thesis/{output_jobs_dir}/decode_all.sh"
    job_sub_decode_all += f"\nerror = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/decode_all.$(Process).err"
    job_sub_decode_all += f"\noutput = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/decode_all.$(Process).out"
    job_sub_decode_all += f"\nlog = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/decode_all.$(Process).log"
    job_sub_decode_all += f"\nrequest_memory = {18432 if args.model_id == 'sd' else 18432 *2}"
    job_sub_decode_all += f"\nqueue"

    job_sub_attack_all = job_sub + f"\narguments = /fast/mkaut/ma-thesis/{output_jobs_dir}/attack_all.sh"
    job_sub_attack_all += f"\nerror = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/attack_all.$(Process).err"
    job_sub_attack_all += f"\noutput = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/attack_all.$(Process).out"
    job_sub_attack_all += f"\nlog = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/attack_all.$(Process).log"
    job_sub_attack_all += f"\nrequest_memory = {18432 if args.model_id == 'sd' else 18432 *2}"
    job_sub_attack_all += f"\nqueue"

    # save the job files as .sub files
    with open(os.path.join(output_jobs_dir, "decode_all.sub"), 'w') as f:
        f.write(job_sub_decode_all)
    with open(os.path.join(output_jobs_dir, "attack_all.sub"), 'w') as f:
        f.write(job_sub_attack_all)




