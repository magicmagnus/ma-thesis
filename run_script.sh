# Create a new virtual environment in /tmp and reinstall the required packages
/usr/bin/python3 -m venv /tmp/venv
echo "created new venv in /tmp"
source /tmp/venv/bin/activate
echo "activated new venv"
cd /fast/mkaut/ma-thesis
pip install -r prc/requirements.txt
echo "reinstalled required packages"

# Run the Python script
cd /fast/mkaut/ma-thesis/


# python encode_imgs.py --config "confs/encode/gs/encode.json"
# python attack_train_surrogate.py --config "experiments/gs/flux/mjprompts/num_50_steps_50_fpr_0.01_gdscale_3.0/decode_imgs/confs/decode.json"
# python attack_imgs.py --config "experiments/gs/flux/mjprompts/num_50_steps_50_fpr_0.01_gdscale_3.0/decode_imgs/confs/decode.json"
python decode_imgs.py --config "experiments/gs/flux/mjprompts/num_50_steps_50_fpr_0.01_gdscale_3.0/decode_imgs/confs/decode.json" 



