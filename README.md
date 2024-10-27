# EHMCANet
This repo holds code for "Efficient Hierarchical Multiscale Convolutional Attention for Accurate Medical Image Segmentation"
## Usage
### 1. Prepare data
* Download the MoNuSeg dataset from the official [website](https://monuseg.grand-challenge.org/Data/).
Then prepare the datasets in the following format for easy use of the code:
  
```
  ├── MoNuSeg
    ├── Train_Folder
    │    └──img
    │          ├── TCGA-21-5784-01Z-00-DX1.png
    │          ├── TCGA-21-5786-01Z-00-DX1.png
    │          └── ......
    │    └──labelcol
    │          ├── TCGA-21-5784-01Z-00-DX1.png
    │          ├── TCGA-21-5786-01Z-00-DX1.png
    │          └── ......
    ├── Val_Folder
    │    └──img
    │          ├── TCGA-18-5592-01Z-00-DX1.png
    │          ├── TCGA-AY-A8YK-01A-01-TS1.png
    │          └── ......
    │    └──labelcol
    │          ├── TCGA-18-5592-01Z-00-DX1.png
    │          ├── TCGA-AY-A8YK-01A-01-TS1.png
    │          └── ......
    └── Test_Folder
    │    └──img
    │          ├── TCGA-2Z-A9J9-01A-01-TS1.png
    │          ├── TCGA-44-2665-01B-06-BS6.png
    │          └── ......
    │    └──labelcol
    │          ├── TCGA-2Z-A9J9-01A-01-TS1.png
    │          ├── TCGA-44-2665-01B-06-BS6.png
    │          └── ......

```
        
### 2. Environment
Please prepare an environment with python>=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
### 3. Train/Test
* Train
```
EHMCANet:
python EHMCANet/train.py --root_path ./EHMCANet/train.py --batch_size 4 --img_size 256 
```
* Test
```
EHMCANet:
python EHMCANet/test.py --root_path ./EHMCANet/test.py --img_size 256 
```
You can download the weights of our network from the [link](https://drive.google.com/drive/folders/1eQBm9W30OCnbQVLN2YaxWCTcWSR7SQxD?usp=drive_link).

## Reference

## Citations
