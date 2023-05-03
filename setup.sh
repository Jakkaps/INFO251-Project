# Read necessary flags
while getopts d flag
do
    case "${flag}" in
        d) download="True";
    esac
done

# Make necessary folders
mkdir -p data
mkdir -p models
mkdir -p results

# Make all scripts runnable
chmod +x train.sh
chmod +x test.sh
chmod +x data_augmentations.sh
chmod +x hyperparam_tuning.sh

# Create pip env with proper packages installed
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download artifacts. These include 
# - SAM for segmentation
# - Pretrained NTS model weights for analysis
# - Datasets w/ augmentations
if [ "$download" = "True" ]; then
    # gdown -O models/sam_vit_h_4b8939.pth https://drive.google.com/uc\?export\=download\&id\=1nIASaHlEGShjZxxA1T_3L-Qs_wBk0GPE
    gdown -O data/fgvc-aircraft-augmented.zip  "1W4sai3IsRadDMTzD-zLbQlQFIo5K79of" && unzip data/fgvc-aircraft-augmented.zip -d data/

    # TODO: Model weights
    # gdown -O models/nts_weights.pth  
fi