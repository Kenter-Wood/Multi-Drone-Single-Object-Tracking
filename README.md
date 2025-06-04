# Multi-Drone-Single-Object-Tracking
This repository contains the experimental code from my undergraduate thesis project.

# 使用说明
这一仓库包含了“多无人机协同目标跟踪”任务的实验代码。为了方便阅读和开发，我将目标跟踪领域一些使用较多的代码框架和模型仓库进行了一定的改写，使逻辑上较为清晰。当前框架包含7个主要分区，我会在各部分展开介绍其作用，并在此处介绍代码运行方法。

方法上，我的实验参考了一部分开源方法的代码和思想，包括Aba-ViTrack的轻量化思路（[原始仓库](https://github.com/xyyang317/Aba-ViTrack)）、ODTrack的时序建模方案（[原始仓库](https://github.com/GXNU-ZhongLab/ODTrack)），以及TransMDOT的多无人机协同跟踪框架（[原始仓库](https://github.com/cgjacklin/transmdot)）。模型的训练、测试流程，以及模型搭建思路，则是参考了OSTrack（[原始仓库](https://github.com/botaoye/OSTrack)）和Pytracking架构。（[原始仓库](https://github.com/visionml/pytracking)）。
## 运行方案
### 环境配置和运行
该仓库代码依赖以下主要库运行：
```
torch
torchvision
torchscale
safetensors
timm
transformers
numpy
scipy
opencv-python

```
同时包含了一些用于数据处理和分析的其它库，如matplotlib、tensorboard等。

当前使用环境的所有库已导出为`requirements.txt`，可通过以下指令安装环境。运行时采用的cuda版本为11.5，而torch使用了能够与其兼容的11.6版本。
```
conda create -n Pytracking python=3.9
pip install -r requirements.txt
```
我采用了LaSOT、GOT10K、MDOT的训练集，以及UAVDT、DTB70的测试集，这些数据集均有官方提供的免费下载方式，本仓库也包含了对它们的图像、标注读取方案。如果需要使用其它数据集，需要按照类似的方式处理。

为了使环境正确找到对应的数据、模型位置，你应当在`train`和`test`文件夹中的`paths.py`文件分别修改对应的路径，使它们指向对应位置。


### 训练
训练模型的关键代码在train文件夹中。我采用的模型参数和ViT-Tiny相当，可以直接加载预训练模型。我采用的是[MAE-Lite](https://github.com/wangsr126/MAE-Lite)中提供的mae_tiny_400e.pth.tar模型。具体的训练流程见train文件夹。
### 测试 
测试模型的关键代码在test文件夹中。测试的流程是，输入第一帧的图像和边界框标注进行初始化，之后在跟踪过程中，每次送入新的一帧图像，模型会给出标注结果，以此类推。具体的测试流程见test文件夹。

### 声明
