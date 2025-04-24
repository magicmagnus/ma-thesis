import os
import random
import numpy as np
import torch
import json
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

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
    seed_everything(43) # should be 0 cause it gets set to 0 later in the loop
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
    from sklearn import metrics
    from scipy.stats import norm
    
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

def plot_heatmaps(args, fpr_grid, tpr_grid, title_suffix, xticks, yticks, levels=0.01, contour_label='FPR = 0.01'):
    xlabel = 'RID Thresholds (higher = more likely WM)'
    ylabel = 'GS Thresholds (higher = more likely WM)'

    title_FPR = f'FPR Grid for {title_suffix}'
    title_TPR = f'TPR Grid for {title_suffix}' 
    contour_label = f'FPR={levels}'

    # Create 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Use imshow for heatmaps ---
    # Define extent for imshow: [left, right, bottom, top]
    # Coordinates correspond to the *edges* of the pixels/cells
    

    # FPR plot using imshow
    im1 = ax1.imshow(fpr_grid, cmap="viridis", vmin=0, vmax=1, aspect='auto', origin='lower')
    fig.colorbar(im1, ax=ax1, label='False Positive Rate (↓)')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title_FPR, fontsize=20)
    # Set ticks manually for imshow
    ax1.set_xticks(np.arange(len(xticks)))
    ax1.set_yticks(np.arange(len(yticks)))
    ax1.set_xticklabels(xticks, rotation=90)
    ax1.set_yticklabels(yticks)
    # ax1.invert_yaxis() # origin='lower' handles the y-axis direction

    # TPR plot using imshow
    im2 = ax2.imshow(tpr_grid, cmap="viridis", vmin=0, vmax=1, aspect='auto', origin='lower')
    fig.colorbar(im2, ax=ax2, label='True Positive Rate (↑)')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(title_TPR, fontsize=20)
    # Set ticks manually for imshow
    ax2.set_xticks(np.arange(len(xticks)))
    ax2.set_yticks(np.arange(len(yticks)))
    ax2.set_xticklabels(xticks, rotation=90)
    ax2.set_yticklabels(yticks)
    # ax2.invert_yaxis() # origin='lower' handles the y-axis direction

    # --- Contour plot ---
    # Create coordinates for contour centers (matching imshow pixel centers)
    x_centers = np.arange(fpr_grid.shape[1]) - 0.5
    y_centers = np.arange(fpr_grid.shape[0]) - 0.5
    X_centers, Y_centers = np.meshgrid(x_centers, y_centers)

    # Generate FPR contour on FPR plot
    fpr_CS = ax1.contour(X_centers, Y_centers, fpr_grid, levels=[levels], colors='red', linewidths=1.5, linestyles='-')
    ax1.clabel(fpr_CS, fmt={levels: contour_label}, inline=True, fontsize=10)

    # Display the same FPR contour on TPR plot
    fpr_contour_on_tpr = ax2.contour(X_centers, Y_centers, fpr_grid, levels=[levels], colors='red', linewidths=1.5, linestyles='-')
    ax2.clabel(fpr_contour_on_tpr, fmt={levels: contour_label}, inline=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, f'grids_{title_suffix}_heatmaps.pdf'))
    plt.show()






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

    return f"num_{args.num_images}_fpr_{args.fpr}_cfg_{args.guidance_scale}_wmch_{args.latent_channels_wm}"

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
    attack_mem = 10000 # in MB

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
        job_bash_decode = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python decode_imgs.py --config {output_conf_dir}/{template}"
        # 1.2.2 create the attack.sh file
        job_bash_attack = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python attack_imgs.py --config {output_conf_dir}/{template}"
        
        # save .sh files
        with open(os.path.join(output_jobs_dir, "decode", f"{template_name}.sh"), 'w') as f:
            f.write(job_bash_decode)
        with open(os.path.join(output_jobs_dir, "attack", f"{template_name}.sh"), 'w') as f:
            f.write(job_bash_attack)

    	# 1.2.3 create the train.sh file
        if 'surr' in template:
            job_bash_train = job_bash + f"\n/is/sg2/mkaut/miniconda3/bin/python attack_train_surrogate.py --config {output_conf_dir}/{template}"
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
            submit_attack_all += f"condor_submit_bid 20 /fast/mkaut/ma-thesis/{output_jobs_dir}/attack/{template_name}.sub\n"
        submit_decode_all += f"condor_submit_bid 20 /fast/mkaut/ma-thesis/{output_jobs_dir}/decode/{template_name}.sub\n"

    # save the .sh submit files
    with open(os.path.join(output_jobs_dir, "submit_decode_all.sh"), 'w') as f:
        f.write(submit_decode_all)
    with open(os.path.join(output_jobs_dir, "submit_attack_all.sh"), 'w') as f:
        f.write(submit_attack_all)


def setup_gridspec_figure(nrows, ncols, fs, title, fs_title, y_adj, title_height_ratio, sp_width=1.5, sp_height=1.5):
    """Create figure with gridspec layout and title rows."""
    # Basic layout parameters
    data_height_ratio = 1.0
    height_ratios = [title_height_ratio if i % 2 == 0 else data_height_ratio 
                    for i in range(2 * nrows)]
    wspace = 0.05
    hspace = 0.05

    # Calculate figure size
    subplot_width = sp_width
    subplot_height = sp_height
    fig_width = subplot_width * ncols + (ncols - 1) * wspace * subplot_width
    fig_height = subplot_height * sum(height_ratios) + (2 * nrows - 1) * hspace * subplot_height

    # Create figure and gridspec
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(2 * nrows, ncols, figure=fig, 
                          wspace=wspace, hspace=hspace, 
                          height_ratios=height_ratios)
    
    # Create title axes
    title_axes = []
    for i in range(0, 2 * nrows, 2):
        ax = fig.add_subplot(gs[i, :])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)  # Hide axis lines
        title_axes.append(ax)
    
    fig.suptitle(title, fontsize=fs_title, fontweight="bold", y=y_adj)
    
    return fig, gs, title_axes

# for RID and TR
def plot_wm_pattern_fft(num_channels, pattern, title, save_path):
    
    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 1.2
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 10
        y_adj = 1.05
        title_height_ratio = 0.15
    """Plot watermark pattern FFT."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=2, ncols=num_channels, 
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )
    
    # Add row titles
    title_axes[0].text(0.5, 0.25, 'Watermark pattern (real)', 
                      fontsize=fs, fontweight="bold", ha="center", va="center")
    title_axes[1].text(0.5, 0.25, 'Watermark pattern (imag)', 
                      fontsize=fs, fontweight="bold", ha="center", va="center")
    
    # Plot data
    for col in range(num_channels):
        ax1 = fig.add_subplot(gs[1, col])
        ax3 = fig.add_subplot(gs[3, col])
        
        ax1.imshow(pattern[0, col].real.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
        ax3.imshow(pattern[0, col].imag.cpu().numpy(), cmap='GnBu', vmin=-50, vmax=50)
        
        for ax in [ax1, ax3]:
            ax.axvline(32, color='r', linestyle='--', linewidth=1)
            ax.axhline(32, color='r', linestyle='--', linewidth=1)
            ax.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# for PRC and GS
def plot_wm_pattern_spatial_domain(num_channels, pattern, title, save_path):
    
    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 1.2
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 10
        y_adj = 1.05
        title_height_ratio = 0.15

    """Plot watermark pattern in spatial domain."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=1, ncols=num_channels, 
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )

    # Add row title
    title_axes[0].text(0.5, 0.25, 'Watermark pattern (spatial domain)', 
                      fontsize=fs, fontweight="bold", ha="center", va="center")
    
    # Plot data
    for col in range(num_channels):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(pattern[0, col].cpu().numpy(), cmap='OrRd', vmin=-4, vmax=4)
        ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# for RID and TR
def plot_wm_latents_fft(num_channels, init_latents_fft, init_latents_watermarked_fft,
                        init_latents_watermarked, diff, title, save_path):
    
    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 1.0
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 10
        y_adj = 0.95
        title_height_ratio = 0.15

    """Plot watermarked latents FFT and spatial domain."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=5, ncols=num_channels,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )
    
    # Add row titles
    row_titles = [
        'FFT of original latents (real)',
        'FFT after watermarking (real)',
        'FFT after watermarking (imag)',
        'Latents after watermarking (spatial domain)',
        'Abs. Difference (spatial domain)'
    ]
    for ax, title_text in zip(title_axes, row_titles):
        ax.text(0.5, 0.25, title_text, fontsize=fs, fontweight="bold",
                ha="center", va="center")
    
    # Plot data
    for col in range(num_channels):
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(5)]
        plots = [
            (init_latents_fft[0, col].real.cpu().numpy(), 'GnBu', -50, 50),
            (init_latents_watermarked_fft[0, col].real.cpu().numpy(), 'GnBu', -50, 50),
            (init_latents_watermarked_fft[0, col].imag.cpu().numpy(), 'GnBu', -50, 50),
            (init_latents_watermarked[0, col].cpu().numpy(), 'OrRd', -4, 4),
            (np.abs(diff[0, col].cpu().numpy()), 'gray', -4, 4)
        ]
        
        for ax, (data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# for RID and TR
def visualize_reversed_latents_fft(num_channels, 
                               reversed_latents_wm_fft,
                               reversed_latents_nowm_fft,
                               diff_wm_wm_fft,
                               diff_wm_nowm_fft,
                               diff_wm_true_fft,
                               diff_nowm_true_fft,
                               ch_mean_abs_diff_wm_wm_fft,
                               ch_mean_abs_diff_wm_nowm_fft,
                               mean_abs_diff_wm_true_fft,
                               mean_abs_diff_nowm_true_fft,
                               title, save_path):

    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 0.95
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 9
        y_adj = 0.91
        title_height_ratio = 0.15

    """Visualize reversed latents in frequency domain."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=12, ncols=num_channels,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )

    # Add row titles
    row_titles = [
        'FFT of Reversed WM latents (real)',
        'FFT of Reversed WM latents (imag)',
        '|WM latents - masked WM| (real)',
        '|WM latents - masked WM| (imag)',
        f'|WM latents - true WM latents| (real) (per-pixel mean={mean_abs_diff_wm_true_fft:.2f})',
        '|WM latents - true WM latents (imag)',
        'FFT of Reversed NoWM latents (real)',
        'FFT of Reversed NoWM latents (imag)',
        '|NoWM latents - masked WM| (real)',
        '|NoWM latents - masked WM| (imag)',
        f'|NoWM latents - true NoWM latents| (real) (per-pixel mean={mean_abs_diff_nowm_true_fft:.2f})',
        '|NoWM latents - true NoWM latents| (imag)'
    ]

    for ax, title_text in zip(title_axes, row_titles):
        ax.text(0.5, 0.25, title_text, fontsize=fs, fontweight="bold",
                ha="center", va="center")
        
    # Plot data
    for col in range(num_channels):
        # first all for WM
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(6)]
        plots = [
            (0 , reversed_latents_wm_fft[0, col].real.cpu().numpy(), 'GnBu', -50, 50),
            (1 , reversed_latents_wm_fft[0, col].imag.cpu().numpy(), 'GnBu', -50, 50),
            (2 , np.abs(diff_wm_wm_fft[0, col].real.cpu().numpy()), 'gray', 0, 150),
            (3 , np.abs(diff_wm_wm_fft[0, col].imag.cpu().numpy()), 'gray', 0, 150),
            (4 , np.abs(diff_wm_true_fft[0, col].real.cpu().numpy()), 'gray', 0, 150),
            (5 , np.abs(diff_wm_true_fft[0, col].imag.cpu().numpy()), 'gray', 0, 150)
        ]
        
        for ax, (idx, data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            if idx == 2:
                # add text to top rght corner, the abs_diff_wm_fft[col]
                ax.text(0.02, 0.98, f"{ch_mean_abs_diff_wm_wm_fft[col]:.2f}", 
                        fontsize=fs, ha="left", va="top",
                        transform=ax.transAxes, color='white')
                
        # then for NoWM
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(6, 12)]
        plots = [
            (0 , reversed_latents_nowm_fft[0, col].real.cpu().numpy(), 'GnBu', -50, 50),
            (1 , reversed_latents_nowm_fft[0, col].imag.cpu().numpy(), 'GnBu', -50, 50),
            (2 , np.abs(diff_wm_nowm_fft[0, col].real.cpu().numpy()), 'gray', 0, 150),
            (3 , np.abs(diff_wm_nowm_fft[0, col].imag.cpu().numpy()), 'gray', 0, 150),
            (4 , np.abs(diff_nowm_true_fft[0, col].real.cpu().numpy()), 'gray', 0, 150),
            (5 , np.abs(diff_nowm_true_fft[0, col].imag.cpu().numpy()), 'gray', 0, 150)
        ]

        for ax, (idx, data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            if idx == 2:
                # add text to top rght corner, the abs_diff_nowm_fft[col]
                ax.text(0.02, 0.98, f"{ch_mean_abs_diff_wm_nowm_fft[col]:.2f}", 
                        fontsize=fs, ha="left", va="top",
                        transform=ax.transAxes, color='white')
                
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# for PRC and GS
def visualize_reversed_latents_spatial_domain(num_channels, 
                               reversed_latents_wm,
                               reversed_latents_nowm,
                               diff_wm_wm,
                               diff_nowm_wm,
                               diff_wm_true,
                               diff_nowm_true,
                               metric_wm,
                               metric_nowm,
                               mean_abs_diff_wm_wm,
                               mean_abs_diff_nowm_wm,
                               mean_abs_diff_wm_true,
                               mean_abs_diff_nowm_true,
                               title, save_path):

    if num_channels == 16: # for flux model
        fs_title = 20
        fs = 12
        y_adj = 0.95
        title_height_ratio = 0.2 
    else: # 4, for SD model
        fs_title = 10
        fs = 8
        y_adj = 0.91
        title_height_ratio = 0.15

    """Visualize reversed latents in spatial domain."""
    fig, gs, title_axes = setup_gridspec_figure(
        nrows=6, ncols=num_channels,
        fs=fs, title=title, fs_title=fs_title,
        y_adj=y_adj, title_height_ratio=title_height_ratio
    )

    # Add row titles
    row_titles = [
        'Reversed WM latents (spatial domain)',
        f'|WM latents - WM pattern| (per-pixel mean={mean_abs_diff_wm_wm:.2f}) with metric {metric_wm:.2f}',
        f'|WM latents - true WM latents| (mean per-pixel={mean_abs_diff_wm_true:.2f})',
        'Reversed NoWM latents (spatial domain)',
        f'|NoWM latents - WM pattern| (per-pixel mean={mean_abs_diff_nowm_wm:.2f}) with metric {metric_nowm:.2f}',
        f'|NoWM latents - true NoWM latents| (per-pixel mean={mean_abs_diff_nowm_true:.2f})'
    ]

    for ax, title_text in zip(title_axes, row_titles):
        ax.text(0.5, 0.25, title_text, fontsize=fs, fontweight="bold",
                ha="center", va="center")
        
    # Plot data
    for col in range(num_channels):
        # first all for WM
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(3)]
        plots = [
            (0 , reversed_latents_wm[0, col].cpu().numpy(), 'OrRd', -4, 4),
            (1 , np.abs(diff_wm_wm[0, col].cpu().numpy()), 'gray', 0, 4),
            (2 , np.abs(diff_wm_true[0, col].cpu().numpy()), 'gray', 0, 4)

        ]

        for ax, (idx, data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
                  
                
        # then for NoWM
        axes = [fig.add_subplot(gs[2*i+1, col]) for i in range(3, 6)]
        plots = [
            (0 , reversed_latents_nowm[0, col].cpu().numpy(), 'OrRd', -4, 4),
            (1 , np.abs(diff_nowm_wm[0, col].cpu().numpy()), 'gray', 0, 4),
            (2 , np.abs(diff_nowm_true[0, col].cpu().numpy()), 'gray', 0, 4)
        ]

        for ax, (idx, data, cmap, vmin, vmax) in zip(axes, plots):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            
                
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)



        