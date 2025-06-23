# RCOC-Distill

Official PyTorch implementation of the paper “RCOC-Distill: Boosting 4D Radar-Camera Online Calibration with Knowledge Distillation from LiDAR Features”.

**Note: We have released our code, datasets and weitghts.**

## Table of Contents

- [Requirements](#Requirements)
- [Pre-trained model](#Pre-trained_model)
- [Datasets](#Datasets)
- [Evaluation](#Evaluation)
- [Train](#Train)
- [Citation](#Citation)



## Requirements

* python 3.6 (recommend to use [Anaconda](https://www.anaconda.com/))
* PyTorch==1.0.1.post2
* Torchvision==0.2.2
* Install requirements and dependencies
```commandline
pip install -r requirements.txt
```

## Pre-trained model

Pre-trained models can be downloaded from [baidu drive](https://pan.baidu.com/s/1ER60IJ0e-qLCCCtKcTIl4A?pwd=1234)

## Datasets

We have produced calibration datasets based on the original published [NTU4DRaDLM](https://github.com/junzhang2016/NTU4DRadLM) and [Dual-Radar](https://github.com/adept-thu/Dual-Radar) datasets.

You can download the [NTU4DRaDLM-Calib](https://pan.baidu.com/s/16onSWtdY8XkLgsl1-hE_7Q?pwd=1234) and [Dual-Radar-Calib](https://pan.baidu.com/s/1DArDg0ThXGHK9lt11VyAfg?pwd=1234) datasets here.

## Evaluation

1. Change the path to the dataset in [evaluate_calib_RCOCDistill.py](evaluate_calib_RCOCDistill.py).
```python
dataset = 'NTU4DRadLM'
data_folder = '/path/to/the/Datasets/NTU4DRadLM-Calib'
weights = '/path/to/Pre-trained'
```
2. Run evaluation.
```commandline
python evaluate_calib_RCOCDistill.py
```

## Train
1. Change the path to the dataset in [train_with_sacred_RCOCDistill.py](train_with_sacred_RCOCDistill.py).
```python
dataset = 'NTU4DRadLM'
data_folder = '/path/to/the/Datasets/NTU4DRadLM-Calib'
val_sequence = 6     # NTU4DRadLM: 6 7 8    Dual_Radar: 0
```
2. Run train.
```commandline
python train_with_sacred_RCOCDistill.py
```

## Citation
 
Thank you for citing our paper if you use any of this code or datasets.


### Acknowledgments
 We are grateful to [LCCNet](https://github.com/IIPCVLAB/LCCNet). We use it as our initial code base.