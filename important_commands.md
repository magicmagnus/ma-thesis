# starting a job
- first create a job file, e.g. test.sub
- then submit the job using condor_submit_bid <number of bids> <job file>
- monitor the job using condor_ssh_to_job <job id>
- check the job status using condor_q <username>



# start a job 
condor_submit_bid 50 test.sub 

# monitor the job
condor_ssh_to_job 15905476.0
tail -f /proc/1258692/fd/1

# check the job status
condor_q mkaut

# stop a job
condor_rm 15905476.0