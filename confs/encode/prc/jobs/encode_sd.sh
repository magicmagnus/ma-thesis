# # Create a new virtual environment in /tmp and reinstall the required packages
# /usr/bin/python3 -m venv /tmp/venv
# echo "created new venv in /tmp"
# source /tmp/venv/bin/activate
# echo "activated new venv"
# cd /fast/mkaut/ma-thesis
# pip install -r prc/requirements.txt
# echo "reinstalled required packages"
cd /fast/mkaut/ma-thesis/


source /is/sg2/mkaut/miniconda3/bin/activate
echo "activated conda env"
# Disable user site-packages
export PYTHONNOUSERSITE=1

# echo "PYTHONPATH=$PYTHONPATH"
# echo "PATH=$PATH"
# which python
# python -c "import sys; print(sys.path)"

# Run the Python script with user site-packages disabled
/is/sg2/mkaut/miniconda3/bin/python -s 1_encode_imgs.py --config "confs/encode/prc/encode_sd.json"

