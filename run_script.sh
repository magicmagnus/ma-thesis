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
python encode.py --config "config_runs/encode/prc_coco_default.json"
python decode.py --config "config_runs/decode/prc_coco_jpeg.json" 



