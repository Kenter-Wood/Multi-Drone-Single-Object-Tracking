# Multi-Drone-Single-Object-Tracking
This repository contains the experimental code from my undergraduate thesis project.

# 使用说明
这一仓库包含了“多无人机协同目标跟踪”任务的实验代码。为了方便阅读和开发，我将目标跟踪领域一些使用较多的代码框架和模型仓库进行了一定的改写，使逻辑上较为清晰。当前框架包含7个主要分区，我会在各部分展开介绍其作用，并在此处介绍代码运行方法。

方法上，我的实验参考了一部分开源方法的代码和思想，包括Aba-ViTrack的轻量化思路（[原始仓库](https://github.com/xyyang317/Aba-ViTrack)）、ODTrack的时序建模方案（[原始仓库](https://github.com/GXNU-ZhongLab/ODTrack)），以及TransMDOT的多无人机协同跟踪框架（[原始仓库](https://github.com/cgjacklin/transmdot)）。模型的训练、测试流程，以及模型搭建思路，则是参考了OSTrack（[原始仓库](https://github.com/botaoye/OSTrack)）和Pytracking架构。（[原始仓库](https://github.com/visionml/pytracking)）。
## 运行方案
### 环境配置和运行
该仓库代码依赖以下库运行：
```
torch
```
搭建环境通过requirements文件即可
```
pip install -r requirements.txt
```

### 训练
训练模型的关键代码在train文件夹中。我采用的模型参数和ViT-Tiny相当，可以直接加载预训练模型。我采用的是[MAE-Lite](https://github.com/wangsr126/MAE-Lite)中提供的mae_tiny_400e.pth.tar模型。具体的训练流程见train文件夹。
### 测试 
测试模型的关键代码在test文件夹中。测试的流程是，输入第一帧的图像和边界框标注进行初始化，之后在跟踪过程中，每次送入新的一帧图像，模型会给出标注结果，以此类推。具体的测试流程见test文件夹。

### 声明
