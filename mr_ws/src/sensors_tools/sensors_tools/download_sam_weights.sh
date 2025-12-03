# Print usage message
if [ "$#" -ne 1 ]; then
    echo "Usage: download_sam_weights.sh <path>"
    exit 1
fi

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O $1/sam_vit_b_01ec64.pth