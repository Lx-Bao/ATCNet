# ATCNet
**Aggregating Transformers and CNNs for Salient Object Detection in Optical Remote Sensing Images.** 
 Liuxin Bao, Xiaofei Zhou, Bolun Zheng, Haibing Yin, Zunjie Zhu, Jiyong Zhang, and Chenggang Yan. Neurocomputing, 2023, 553: 126560.

## Usage

### 1. Clone the repository
```
https://github.com/Lx-Bao/ATCNet.git
```
### 2. Training
Download the pretrained model **swin_base_patch4_window12_384_22k.pth** and **resnet50-19c8e357.pth**, and put them on ./pretrained/
```
python train.py
```
### 3. Testing
```
python test.py
```

### 4. Evaluation

- We provide [saliency maps](https://pan.baidu.com/s/1AEoMaddDCn6CobGiUb5uVg?pwd=stvm) (fetch code: stvm) of our ATCNet on EORSSD and ORSSD dataset.


## Architecture
![ATCNet architecture](Fig/fig_framework.png)

## Qualitative Comparison
### ORSSD
![EORSSD](Fig/fig_comparison.png)

## Citation
```
@article{bao2023aggregating,
  title={Aggregating transformers and CNNs for salient object detection in optical remote sensing images},
  author={Bao, Liuxin and Zhou, Xiaofei and Zheng, Bolun and Yin, Haibing and Zhu, Zunjie and Zhang, Jiyong and Yan, Chenggang},
  journal={Neurocomputing},
  volume={553},
  pages={126560},
  year={2023},
  publisher={Elsevier}
}
```

- If you have any questions, feel free to contact me via: `lxbao@hdu.edu.cn` or `zxforchid@outlook.com`.
