# EHMCANet
This repo holds code for "Efficient Hierarchical Multiscale Convolutional Attention for Accurate Medical Image Segmentation"
## Usage
### 1. Prepare data
### 2. Environment
Please prepare an environment with python>=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
### 3. Train/Test
* Train
  

EHMCANet:
python train.py --root_path ./EHMCANet/train --batch_size 4 --img_size 256 --module networks.EHMCANet --output_dir './MoNuSeg/EHMCANet/Test_session_06.25_23h28/models'

