import os
import random
import numpy as np
import torch
import json
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from sklearn import metrics

 # flux
from pipes.inverse_flux_pipeline import InversableFluxPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
# sd
from pipes.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

from datasets import load_dataset

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
    # print(f'[transform_img] image min/max: {image.min().item()}/{image.max().item()}')
    return 2.0 * image - 1.0

def load_prompts(args):
    # load the prompts
    if args.dataset_id == 'coco':
        with open('coco/captions_val2017.json') as f:
            all_prompts = [ann['caption'] for ann in json.load(f)['annotations']]
    elif args.dataset_id == 'sdprompts':
        all_prompts = [sample['Prompt'] for sample in load_dataset('Gustavosta/Stable-Diffusion-Prompts')['test']]
    elif args.dataset_id == 'mjprompts':
        all_prompts = [sample['caption'] for sample in load_dataset('bghira/mj-v52-redux')['Collection_5']]
    else:
        print2file(args.log_file, 'Invalid dataset_id')
        return
    # sample the prompts
    seed_everything(30) # should be 0 cause it gets set to 0 later in the loop
    prompts = random.sample(all_prompts, args.num_images)
    seed_everything(0) # should be 0 cause it gets set to 0 later in the loop
    print2file(args.log_file,  '\nPrompts:')
    for i, prompt in enumerate(prompts):
        print2file(args.log_file, f'{i}: {prompt}')

    return prompts

def get_pipe(args, device, HF_CACHE_DIR):
    # which Model to use
    if args.model_id == 'sd':
        print("\nUsing SD model")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base', 
            subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base',
            scheduler=scheduler,
            torch_dtype=torch.float32,
            cache_dir=HF_CACHE_DIR,
            ).to(device)
    elif args.model_id == 'flux':
        print("\nUsing FLUX model")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            subfolder='scheduler'
        )
        pipe = InversableFluxPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-dev',
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
            cache_dir=HF_CACHE_DIR,
        ).to(device)
    elif args.model_id == 'flux_s':
        print("\nUsing FLUX schnell model")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            'black-forest-labs/FLUX.1-schnell',
            subfolder='scheduler'
        )
        pipe = InversableFluxPipeline.from_pretrained(
            'black-forest-labs/FLUX.1-schnell',
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
            cache_dir=HF_CACHE_DIR,
        ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def bootstrap_tpr(no_wm_metrics, wm_metrics, fpr_target, n_bootstraps=1000):
    
    tpr_samples = []
    n_wm = len(wm_metrics)
    n_nowm = len(no_wm_metrics)

    for _ in range(n_bootstraps):
        # Bootstrap resample
        wm_resample = np.random.choice(wm_metrics, size=n_wm, replace=True)
        nowm_resample = np.random.choice(no_wm_metrics, size=n_nowm, replace=True)

        # Compute ROC curve
        preds = list(nowm_resample) + list(wm_resample)
        labels = [0] * len(nowm_resample) + [1] * len(wm_resample)
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)

        # Find TPR at the given FPR
        idx = np.where(fpr <= fpr_target)[0][-1]  # Last index where fpr <= target
        tpr_samples.append(tpr[idx])

    # Compute statistics
    tpr_samples = np.array(tpr_samples)
    tpr_mean = np.mean(tpr_samples)
    tpr_std = np.std(tpr_samples)  # Standard error estimate

    # Confidence intervals (normal approximation)
    ci_lower = tpr_mean - 1.96 * tpr_std
    ci_upper = tpr_mean + 1.96 * tpr_std

    # OR Percentile confidence intervals
    ci_lower_perc, ci_upper_perc = np.percentile(tpr_samples, [2.5, 97.5])

    return tpr_mean, tpr_std, (ci_lower, ci_upper), (ci_lower_perc, ci_upper_perc)

def bootstrap_grids_tpr(gs_nowm, gs_wm, rid_nowm, rid_wm, best_gs_thresh, best_rid_thresh, fpr_target, n_bootstraps=1000):

    tpr_samples = []
    n_wm = len(gs_wm)
    n_nowm = len(gs_nowm)

    for _ in range(n_bootstraps):
        # Bootstrap resample
        idx_wm = np.random.choice(n_wm, size=n_wm, replace=True)
        idx_nowm = np.random.choice(n_nowm, size=n_nowm, replace=True)

        gs_wm_resample = gs_wm[idx_wm]
        gs_nowm_resample = gs_nowm[idx_nowm]
        rid_wm_resample = rid_wm[idx_wm]
        rid_nowm_resample = rid_nowm[idx_nowm]

        # Apply OR rule
        preds_wm = ((gs_wm_resample > best_gs_thresh) | (rid_wm_resample > best_rid_thresh)).astype(int)
        preds_nowm = ((gs_nowm_resample > best_gs_thresh) | (rid_nowm_resample > best_rid_thresh)).astype(int)

        preds = np.concatenate([preds_nowm, preds_wm])
        labels = np.array([0] * len(preds_nowm) + [1] * len(preds_wm))

        fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)

        # Find TPR at given FPR
        valid = np.where(fpr <= fpr_target)[0]
        if len(valid) > 0:
            tpr_samples.append(tpr[valid[-1]])
        else:
            tpr_samples.append(0.0)  # no threshold achieving target FPR

    # Calculate statistics
    tpr_samples = np.array(tpr_samples)
    tpr_mean = np.mean(tpr_samples)
    tpr_std = np.std(tpr_samples)

    ci_normal = (tpr_mean - 1.96 * tpr_std, tpr_mean + 1.96 * tpr_std)
    ci_percentile = np.percentile(tpr_samples, [2.5, 97.5])

    return tpr_mean, tpr_std, ci_normal, ci_percentile

def bootstrap_grids_dynamic_thresholds(
    gs_nowm, gs_wm,
    rid_nowm, rid_wm,
    fpr_target=0.01,
    n_bootstraps=1000,
    n_thresholds=50
):

    tpr_samples = []
    n_wm = len(gs_wm)
    n_nowm = len(gs_nowm)

    # Define full threshold ranges (same every time)
    gs_thresh_range = np.linspace(min(np.min(gs_nowm), np.min(gs_wm)), max(np.max(gs_nowm), np.max(gs_wm)), n_thresholds)
    rid_thresh_range = np.linspace(min(np.min(rid_nowm), np.min(rid_wm)), max(np.max(rid_nowm), np.max(rid_wm)), n_thresholds)

    for _ in range(n_bootstraps):
        # Resample with replacement
        idx_wm = np.random.choice(n_wm, size=n_wm, replace=True)
        idx_nowm = np.random.choice(n_nowm, size=n_nowm, replace=True)

        gs_wm_boot = gs_wm[idx_wm]
        gs_nowm_boot = gs_nowm[idx_nowm]
        rid_wm_boot = rid_wm[idx_wm]
        rid_nowm_boot = rid_nowm[idx_nowm]

        # Grid search to find best threshold pair
        best_tpr = 0.0
        best_gs_thresh = 0.0
        best_rid_thresh = 0.0

        for gs_thresh in gs_thresh_range:
            for rid_thresh in rid_thresh_range:
                preds_wm = ((gs_wm_boot > gs_thresh) | (rid_wm_boot > rid_thresh)).astype(int)
                preds_nowm = ((gs_nowm_boot > gs_thresh) | (rid_nowm_boot > rid_thresh)).astype(int)

                tpr = np.mean(preds_wm)
                fpr = np.mean(preds_nowm)

                if fpr <= fpr_target and tpr > best_tpr: # get best TPR thats below the FPR threshold
                    best_tpr = tpr
                    best_gs_thresh = gs_thresh
                    best_rid_thresh = rid_thresh

        tpr_samples.append(best_tpr)
        print(f"[bootstrapping] Best TPR: {best_tpr}, GS Threshold: {best_gs_thresh}, RID Threshold: {best_rid_thresh}")

    # Convert to numpy array
    tpr_samples = np.array(tpr_samples)

    # Compute statistics
    tpr_mean = np.mean(tpr_samples)
    tpr_std = np.std(tpr_samples)
    ci_normal = (tpr_mean - 1.96 * tpr_std, tpr_mean + 1.96 * tpr_std)
    ci_percentile = np.percentile(tpr_samples, [2.5, 97.5])

    return tpr_mean, tpr_std, ci_normal, ci_percentile

def image_distortion(img1, img2, seed, args, i, print_args=True):
    if print_args:
        save_name = 'example'

    if hasattr(args, 'r_degree'): # number between 0 and 360
        img1 = transforms.RandomRotation((args.r_degree[i], args.r_degree[i]), interpolation=InterpolationMode.BICUBIC)(img1)
        img2 = transforms.RandomRotation((args.r_degree[i], args.r_degree[i]), interpolation=InterpolationMode.BICUBIC)(img2)
        if print_args: 
            #print2file(args.log_file, f"Rotating images by {args.r_degree[i]} degrees")
            save_name += f"_rot{args.r_degree[i]}"

    if hasattr(args, 'jpeg_ratio'): # number between 0 and 100
        path = os.path.join(args.log_dir, f"tmp_{args.jpeg_ratio[i]}.jpg")
        img1.save(path, quality=args.jpeg_ratio[i])
        img1 = Image.open(path)
        img2.save(path, quality=args.jpeg_ratio[i])
        img2 = Image.open(path)
        if print_args: 
            #print2file(args.log_file, f"Compressing images with JPEG quality {args.jpeg_ratio[i]}")
            save_name += f"_jpeg{args.jpeg_ratio[i]}"

    if hasattr(args, 'crop_scale') and hasattr(args, 'crop_ratio'): # scale between 0 and 1, ratio between 0 and 1
        scale = (args.crop_scale[i], args.crop_scale[i]) # e.g. exact 50% amount of the area
        ratio = (1 - args.crop_ratio[i], 1 + args.crop_ratio[i])  # e.g. (0.5, 1.5)
        
        seed_everything(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BICUBIC)(img1)
        seed_everything(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BICUBIC)(img2)
        if print_args: 
            #print2file(args.log_file, f"Cropping images with scale {args.crop_scale[i]} and ratio {args.crop_ratio[i]}")
            save_name += f"_cropscale{args.crop_scale[i]}_{args.crop_ratio[i]}"
     
    if hasattr(args, 'crop'): # scale between 0 and 1
        scale = (args.crop[i], args.crop[i]) # e.g. exact 50% amount of the area
        ratio = (1.0, 1.0)  # e.g. (0.5, 1.5)
        
        seed_everything(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BICUBIC)(img1)
        seed_everything(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BICUBIC)(img2)
        if print_args: 
            #print2file(args.log_file, f"Cropping images with scale {args.crop[i]}}")
            save_name += f"_crop{args.crop[i]}"
        
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
        #g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255).astype(np.uint8))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255).astype(np.uint8))
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

def get_dataset_id(args):

    return f"num_{args.num_images}_fpr_{args.fpr}_cfg_{args.guidance_scale}_wmch_{args.latent_channels_wm}_infsteps_{args.test_inf_steps}"

def get_dirs(args, script_type, extra=None):

    base_dir = os.path.join("experiments",
                            args.exp_name, 
                            args.method,
                            args.model_id,
                            args.dataset_id,
                            get_dataset_id(args)
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
                            args.exp_name, 
                            args.method,
                            args.model_id,
                            args.dataset_id,
                            get_dataset_id(args),
                            "decode_imgs",
                            "confs"
                            )
    output_jobs_dir = os.path.join("experiments",
                            args.exp_name, 
                            args.method,
                            args.model_id,
                            args.dataset_id,
                            get_dataset_id(args),
                            "decode_imgs",
                            "jobs"
                            )
                                   
    os.makedirs(output_conf_dir, exist_ok=True)
    os.makedirs(output_jobs_dir, exist_ok=True)
    os.makedirs(os.path.join(output_jobs_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_jobs_dir, "decode"), exist_ok=True)
    os.makedirs(os.path.join(output_jobs_dir, "attack"), exist_ok=True)

    templates = [t for t in os.listdir(templates_dir) if t.endswith('.json')]
    # order alphabetically
    templates = sorted(templates)
    template_job_bash = os.path.join(templates_dir, 'jobs', 'decode.sh')
    template_job_sub = os.path.join(templates_dir, 'jobs', 'decode.sub')

    # so basically, in the templates are the decode.json files, and we wanna 
    # take our source (only one) encode.json that in args.config, and merge it with the decode.json files
    # so we first load the source, 
    
    # the source is the encode.json file
    with open(args.config, 'r') as f:
        source = json.load(f)

    # open the job templates
    # the decode.sh that calls the python script with the config file as argument
    with open(template_job_bash, 'r') as f:
        job_bash_all = f.read()
    # and the decode.sub that submits the decode.sh to the cluster
    with open(template_job_sub, 'r') as f:
        job_sub_all = f.read()



    # these will by .sh files that submit all individual decode.sub (or attack.sub) files as separate jobs
    submit_decode_all = ''
    submit_attack_all = ''
    

    # per attack type, we need:
    # (0. load the template and merge with source)
    # 1. create new files in the decode_imgs folder:
    #   1.1 create the .json file
    #       - confs/[attack_name].json 
    #   1.2 create the .sh files (one for decode, one for attack)
    #       1.2.1 jobs/decode/[attack_name].sh
    #       1.2.2 jobs/attack/[attack_name].sh
    #       1.2.3 jobs/train_[attack_name].sh (only for 1 attack type)
    #   1.3 create the .sub files (one for decode, one for attack)
    #       1.3.1 jobs/decode/[attack_name].sub
    #       1.3.2 jobs/attack/[attack_name].sub
    #       1.3.3 jobs/train_[attack_name].sub (only for 1 attack type)
    # 2. add line for that attack type to submit_decode_all and submit_attack_all

    decode_mem = 16000 if args.model_id == 'sd' else 50000 # in MB
    attack_mem = 20000 # in MB
    job_bid = 100 # cluster money amount

    for template in templates: # template has format "[attack_name].json"
        
        template_name = template.split(".")[0] # has format "[attack_name]"

        # 0. load the template and merge with source
        with open(os.path.join(templates_dir, template), 'r') as f:
            decode = json.load(f)

        # merge the two
        run_name = decode['run_name']
        decode.update(source)
        decode['run_name'] = run_name

        # 1. create new files in the decode_imgs folder:
        # 1.1 create the .json file
        # save the merged
        with open(os.path.join(output_conf_dir, template), 'w') as f:
            json.dump(decode, f, indent=4)

        # 1.2 create the .sh files (one for decode, one for attack)
        # open the .sh templates
        with open(template_job_bash, 'r') as f:
            job_bash = f.read()
         
        # 1.2.1 create the decode.sh file
        job_bash_decode = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python 4_decode_imgs.py --config {output_conf_dir}/{template}"
        # 1.2.2 create the attack.sh file
        job_bash_attack = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python 3_attack_imgs.py --config {output_conf_dir}/{template}"
        
        # save .sh files
        with open(os.path.join(output_jobs_dir, "decode", f"{template_name}.sh"), 'w') as f:
            f.write(job_bash_decode)
        with open(os.path.join(output_jobs_dir, "attack", f"{template_name}.sh"), 'w') as f:
            f.write(job_bash_attack)

    	# 1.2.3 create the train.sh file
        if 'surr' in template:
            job_bash_train = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python 2_attack_train_surrogate.py --config {output_conf_dir}/{template}"
            with open(os.path.join(output_jobs_dir, f"train_{template_name}.sh"), 'w') as f:
                f.write(job_bash_train)

        # 1.3 create the .sub files (one for decode, one for attack)
        with open(template_job_sub, 'r') as f:
            job_sub = f.read()
        # 1.3.1 create the decode.sub file
        job_sub_decode = job_sub + f"\narguments = /fast/mkaut/ma-thesis/{output_jobs_dir}/decode/{template_name}.sh"
        job_sub_decode += f"\nerror = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/decode_{template_name}.$(Process).err"
        job_sub_decode += f"\noutput = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/decode_{template_name}.$(Process).out"
        job_sub_decode += f"\nlog = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/decode_{template_name}.$(Process).log"
        job_sub_decode += f"\nrequest_memory = {decode_mem}" 
        job_sub_decode += f"\nrequirements = TARGET.CUDAGlobalMemoryMb > {decode_mem}"
        job_sub_decode += f"\nqueue"

        # 1.3.2 create the attack.sub file
        job_sub_attack = job_sub + f"\narguments = /fast/mkaut/ma-thesis/{output_jobs_dir}/attack/{template_name}.sh"
        job_sub_attack += f"\nerror = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/attack_{template_name}.$(Process).err"
        job_sub_attack += f"\noutput = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/attack_{template_name}.$(Process).out"
        job_sub_attack += f"\nlog = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/attack_{template_name}.$(Process).log"
        if 'adv' in template:
            # if its an adversarial attack, we need a GPU and more memory
            job_sub_attack += f"\nrequest_memory = {attack_mem}"
            job_sub_attack += f"\nrequirements = TARGET.CUDAGlobalMemoryMb > {attack_mem}"
        else:
            # if its not an adversarial attack, we can use the CPU and don't need a GPU
            job_sub_attack += f"\nrequest_memory = 1000"
            job_sub_attack += f"\nrequest_gpus = 0"

        job_sub_attack += f"\nqueue"

        # save .sub files
        with open(os.path.join(output_jobs_dir, "decode", f"{template_name}.sub"), 'w') as f:
            f.write(job_sub_decode)
        with open(os.path.join(output_jobs_dir, "attack", f"{template_name}.sub"), 'w') as f:
            f.write(job_sub_attack)

        # 1.3.3 create the train.sub file
        if 'surr' in template:
            job_sub_train = job_sub + f"\narguments = /fast/mkaut/ma-thesis/{output_jobs_dir}/train_{template_name}.sh"
            job_sub_train += f"\nerror = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/train_{template_name}.$(Process).err"
            job_sub_train += f"\noutput = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/train_{template_name}.$(Process).out"
            job_sub_train += f"\nlog = /fast/mkaut/ma-thesis/{output_jobs_dir}/logs/train_{template_name}.$(Process).log"
            job_sub_train += f"\nrequest_memory = {10000}"
            job_sub_train += f"\nrequirements = TARGET.CUDAGlobalMemoryMb > {10000}"
            job_sub_train += f"\nqueue"
            with open(os.path.join(output_jobs_dir, f"train_{template_name}.sub"), 'w') as f:
                f.write(job_sub_train)

        # 2. add line for that attack type to submit_decode_all and submit_attack_all
        if not "default" in template_name:
            # we only need to attack the non-default ones, the default is not a real attack
            submit_attack_all += f"condor_submit_bid {job_bid} /fast/mkaut/ma-thesis/{output_jobs_dir}/attack/{template_name}.sub\n"
        submit_decode_all += f"condor_submit_bid {job_bid} /fast/mkaut/ma-thesis/{output_jobs_dir}/decode/{template_name}.sub\n"

    # save the .sh submit files
    with open(os.path.join(output_jobs_dir, "submit_decode_all.sh"), 'w') as f:
        f.write(submit_decode_all)
    with open(os.path.join(output_jobs_dir, "submit_attack_all.sh"), 'w') as f:
        f.write(submit_attack_all)





        