# Create a new virtual environment in /tmp
python3 -m venv /tmp/venv

echo "created new venv in /tmp"

# Activate the new virtual environment
source /tmp/venv/bin/activate

echo "activated new venv"


# Navigate to the project directory
cd /fast/mkaut/ma-thesis/PRC-Watermark

# Reinstall the required packages
pip install -r requirements.txt

echo "reinstalled required packages"



echo "executing encode.py"

# Run the Python script
python encode.py --method "tr"
