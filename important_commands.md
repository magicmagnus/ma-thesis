# starting a job
- first create a job file, e.g. test.sub
- then submit the job using condor_submit_bid <number of bids> <job file>
- monitor the job using condor_ssh_to_job <job id>
- check the job status using condor_q <username>



# start a job 
condor_submit_bid 20 run_job.sub 

# monitor the job
condor_ssh_to_job 15905476.0
prints PID of the job
tail -f /proc/[PID]/fd/1

# check the job status
condor_q mkaut

# stop a job
condor_rm 15905476.0


## encodeing, decodeing and attacking
- encode.py
    - gets encoding/conf.yml file with watermark parameters
    - generates 2 sets of images, wm and no_wm
    - saves both sets for decoding or attacking

- attack_train_surrogate.py
    - gets same decoding/conf.yml file with watermark parameters, and attack parameters
    - gets 2 sets of images, wm and no_wm, OR wm and real images
    - trains surrogate model on wm and no_wm images, OR wm and real images
    - saves the surrogate model for attacking, in same directory as original images

- attack_images.py
    - gets decoding/conf.yml file with watermark parameters, and attack parameters
    - 2 modes, either load pre-attacked images or attack raw images and save them
        either "overwrite_attacked_imgs" = False or True
        if False:
        - loads already attacked images, to wm_attacked and no_wm_attacked
        if True:
        - loads 2 sets of raw images, 
        - generates 2 sets of attacked of images, to wm_attacked and no_wm_attacked
        - saves both attacked sets for decoding
    - calculates FID and CLIP score metrics for wm_attacked and no_wm_attacked, 

- decode.py:
    - gets same decoding/conf.yml file with watermark parameters, and attack parameters
    - gets 2 sets of attacked images, wm_attacked and no_wm_attacked
    - calculates inversed latents from given images,
    - checks for watermark, calculates metric