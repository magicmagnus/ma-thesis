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
#python -u encode.py --num_images 15 --method "tr" --fpr 0.01 
python -u decode.py --num_images 15 --method "tr" --fpr 0.01 --gaussian_blur_r 4
python -u decode.py --num_images 15 --method "gs" --fpr 0.01 --gaussian_blur_r 4
python -u decode.py --num_images 15 --method "prc" --fpr 0.01 --gaussian_blur_r 4

#
# gaussian_std 30
#
# decode.py
#python -u decode.py --test_num 10 --method "prc" --r_degree 1 ## at 6 all 10 were incorrectly decoded


#python -u decode.py --test_num 10 --method "prc" --fpr 0.001 --r_degree 1

# python -u decode.py --test_num 50 --method "prc" --fpr 0.01 --gaussian_blur_r 6  ## tpr = 0.28
# python -u decode.py --test_num 50 --method "gs"  --fpr 0.01 --gaussian_blur_r 6  ## tpr = 1.0
# python -u decode.py --test_num 50 --method "tr"  --fpr 0.01 --gaussian_blur_r 6  ## tpr = ???

# python -u decode.py --test_num 20 --method "tr"  --fpr 0.01 --gaussian_blur_r 6

