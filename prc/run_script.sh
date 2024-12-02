# Create a new virtual environment in /tmp and reinstall the required packages
python3 -m venv /tmp/venv
echo "created new venv in /tmp"
source /tmp/venv/bin/activate
echo "activated new venv"
pip install -r requirements.txt
echo "reinstalled required packages"

# Run the Python script
cd /fast/mkaut/ma-thesis/PRC-Watermark

# encode.py
python -u encode.py --test_num 50 --method "prc" --fpr 0.01 --nowm 1 
python -u encode.py --test_num 50 --method "gs"  --fpr 0.01 --nowm 1
python -u encode.py --test_num 50 --method "tr"  --fpr 0.01 --nowm 1

# decode.py
#python -u decode.py --test_num 10 --method "prc" --r_degree 1 ## at 6 all 10 were incorrectly decoded


#python -u decode.py --test_num 10 --method "prc" --fpr 0.001 --r_degree 1

# python -u decode.py --test_num 50 --method "prc" --fpr 0.01 --gaussian_blur_r 6  ## tpr = 0.28
# python -u decode.py --test_num 50 --method "gs"  --fpr 0.01 --gaussian_blur_r 6  ## tpr = 1.0
# python -u decode.py --test_num 50 --method "tr"  --fpr 0.01 --gaussian_blur_r 6  ## tpr = ???

# python -u decode.py --test_num 20 --method "tr"  --fpr 0.01 --gaussian_blur_r 6


# TODO tomorrow:
# verschiedene Qualitätsstufen durchtesten für PRC, wie wirkt sich auf Accuracy aus?