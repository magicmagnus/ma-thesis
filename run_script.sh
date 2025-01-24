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

# encode.py
# python encode.py --config "config_runs/encode/rid/rid_coco_default.json"
# python attack_train_surrogate.py --config "config_runs/decode/rid/rid_coco_adv_surr_resnet18_wm_nowm.json"
python attack_images.py --config "config_runs/decode/rid/rid_coco_adv_surr_resnet18_wm_nowm.json"
# python decode.py --config "config_runs/decode/gs/gs_coco_jpeg.json" 



