# first move the venv to /tmp
cp -r /fast/mkaut/ma-thesis/PRC-Watermark/venv /tmp

echo "copied venv to /tmp"

cd /tmp/venv

echo "echoing TMP"
echo $TMP
echo $TMPDIR
ls -la

# activate the venv
source bin/activate

echo "activated venv"

#reinstall galois
pip install galois

echo "reinstalled galois"

cd /fast/mkaut/ma-thesis/PRC-Watermark

echo "executing install_req.sh"


python /fast/mkaut/ma-thesis/PRC-Watermark/encode.py --method "tr"