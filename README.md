# Bias Loss & SkipblockNet
[ICCV 2021]Demo for the bias loss and SkipblockNet architecture presented in the [paper](https://arxiv.org/pdf/2107.11170.pdf) "Bias Loss for Mobile Neural Networks".

## Requirements
for installing required packages run
` pip install -r requirements.txt`

## Usage (SkipblockNet)
Pretrained SkipblockNet-m is available from [Google Drive](https://drive.google.com/drive/folders/1G3UR8wtTFB8S-9Fp6sRtfn9Vtfb6XcTU?usp=sharing). For the testing please download and place the model in the same directory as the validation script.

`python validate.py --data path/to/the/dataset`

## Usage (Bias loss)
Training and testing codes are available for DenseNet121, ShuffleNet V2 0.5x and ResNet18. To test the pretrained models please download corresponding model from the [Google Drive](https://drive.google.com/drive/u/0/folders/1G3UR8wtTFB8S-9Fp6sRtfn9Vtfb6XcTU) and run the testing script in the bias loss directory

`python test.py --checkpoint 'path to the checkpoint' --model 'name of the model' --data_path 'path to the cifar-100 dataset'`

To train the models run the training script in the bias loss directory as follows:

`python train.py --model 'name of the model to be trained' --data_path 'path to the cifar-100 dataset'`

## Introduction
"Bias Loss for Mobile Neural Networks"

By Lusine Abrahamyan, Valentin Ziatchin, Yiming Chen and Nikos Deligiannis.
### Approach (SkipblockNet)
<img src="https://github.com/lusinlu/skipnet_evaluation/blob/main/figures/architecture.png" width="300" height="300">

### Performance (SkipblockNet)
SkipNet beats other SOTA lightweight CNNs such as MobileNetV3 and FBNet.

<img src="https://github.com/lusinlu/skipnet_evaluation/blob/main/figures/flops_vs_top1.png" width="300" height="250"> |
<img src="https://github.com/lusinlu/skipnet_evaluation/blob/main/figures/params_vs_top1.png" width="300" height="250">

### Approach (Bias loss)
The bias loss is a dynamically scaled cross-entropy loss, where the scale decays as the variance of data point decreases.
<img src="https://github.com/lusinlu/skipnet_evaluation/blob/main/figures/biasloss.png" width="300" height="250">

### Performance (Bias loss)
Bellow is the results of the pretrained models that can be found in the [Google Drive](https://drive.google.com/drive/u/0/folders/1G3UR8wtTFB8S-9Fp6sRtfn9Vtfb6XcTU)

| Model         | Top-1 bias loss | Top-1 CE |
| :------------ |:---------------:| -----:|
| ResNet18            | 75.51%    |   74.33% |
| DenseNet121         | 77.83%    |   75.98% |
| ShuffleNet V2 0.5x  | 72.00%    |   71.55% |

## Citation
If you find the code useful for your research, please consider citing our works

```
@article{abrahamyanbias,
  title={Bias Loss for Mobile Neural Networks},
  author={Lusine, Abrahamyan and  Valentin, Ziatchin and Yiming, Chen and Nikos, Deligiannis},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  publisher = {IEEE},
  year={2021}
}
```

## Acknowledgement
Codes is heavily modified from [pytorch-vision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) and [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100). 



