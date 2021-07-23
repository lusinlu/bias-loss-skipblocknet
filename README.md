# SkipNet
Demo for SkipNet architecture presented in ICCV 2021 paper of "Bias Loss for Mobile Neural Networks".

## Requirements
for installing required packages run
` pip install -r requirements.txt`

## Usage
Pretrained models are available from [Google Drive](https://drive.google.com/drive/folders/1G3UR8wtTFB8S-9Fp6sRtfn9Vtfb6XcTU?usp=sharing).

`python validate.py --data path/to/the/dataset`

## Introduction to SkipNet
"Bias Loss for Mobile Neural Networks"

By Lusine Abrahamyan, Valentin Ziatchin, Yiming Chen and Nikos Deligiannis.
### Approach
<img src="https://github.com/lusinlu/skipnet_evaluation/blob/main/figures/architecture.png" width="500" height="500">

### Performance
SkipNet beats other SOTA lightweight CNNs such as MobileNetV3 and FBNet.

<img src="https://github.com/lusinlu/skipnet_evaluation/blob/main/figures/flops_vs_top1.png" width="450" height="400"> |
<img src="https://github.com/lusinlu/skipnet_evaluation/blob/main/figures/params_vs_top1.png" width="450" height="400">

## Citation


