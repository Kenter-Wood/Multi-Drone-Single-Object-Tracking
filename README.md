# Multi-Drone-Single-Object-Tracking
This repository contains the experimental code from my undergraduate thesis project.

# 使用说明：
这一仓库包含了“多无人机协同目标跟踪”任务的实验代码。为了方便阅读和开发，我将目标跟踪领域一些使用较多的代码框架和模型仓库进行了一定的改写，使逻辑上较为清晰。当前框架包含7个主要分区，我会在各部分展开介绍其作用，并在此处介绍代码运行方法。

方法上，我的实验参考了一部分开源方法的代码和思想，包括Aba-ViTrack的轻量化思路（[原始仓库](https://github.com/xyyang317/Aba-ViTrack)）、ODTrack的时序建模方案（[原始仓库](https://github.com/GXNU-ZhongLab/ODTrack)），以及TransMDOT的多无人机协同跟踪框架（[原始仓库](https://github.com/cgjacklin/transmdot)）。模型的训练、测试流程，以及模型搭建思路，则是参考了OSTrack（[原始仓库](https://github.com/botaoye/OSTrack)）和Pytracking架构。（[原始仓库](https://github.com/visionml/pytracking)）。
## 运行方案
