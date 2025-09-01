# Create a new virtual environment in /tmp and reinstall the required packages
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

# Run the Python script
/is/sg2/mkaut/miniconda3/bin/python 8_analyze_dataset.py