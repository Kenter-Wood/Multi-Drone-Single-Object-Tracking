通过这个文件夹里的代码，可以训练跟踪模型
# 训练指令
训练的起始文件是train.py，它会接受一系列参数，生成训练的运行指令，并调用其它文件。因此，训练过程应当在终端运行train.py，并传入一些关键参数。
能够从终端传入的参数包括：

`--script`: 它应当对应`scripts`文件夹里的一个子文件夹，代表训练采用的模型。

`--config`: 它对应上述子文件夹下的一个yaml文件，其中给定了本次训练应采用的超参数。

`--save_dir`: 它指定了训练结果的保留路径，通常来说，可以设置为项目根目录下的一个outputs文件夹，此后就可以按此次的模型、超参配置，保留此次训练的权重、tensorboard和日志信息。

`--mode`: 指定是否采用分布式训练。可以选择“single”或“multiple”模式。

`--nproc_per_node`: 指定参加训练的节点，即GPU数量。采用multiple模式时，应当指定使用多少GPU进行训练。可以在`run_training.py`里面设置`os.environ["CUDA_VISIBLE_DEVICES"]`,使一部分显卡在训练中被调用。当使用multiple模式时，会启用DDP的分布式训练。这会将数据分布到多个GPU上，并对梯度进行累积，所以不会降低单GPU显存占用，但提高整体训练效率。

除此之外，你也可以指定程序编号和是否采用蒸馏方案等。最后，你可以运行这样的指令开始一次训练：

首先激活conda环境，进入项目根目录，并将根目录设为PYTHONPATH(因为代码中很多路径关系是以项目根目录出发的)。
```
conda activate Pytracking
cd MultiTrack
export PYTHONPATH=/data/yakun/MultiTrack:$PYTHONPATH
```
然后运行多卡训练。单卡模式将`--mode`设置为`single`,且不设置`--nproc_per_node`：
```
python train/train.py --script multitrack --config multitrack --save_dir output --mode multiple --nproc_per_node 4
```

# 关键代码及修改指南
当需要对训练的流程进行修改，例如改变优化器、损失函数、记录间隔配置时，需要在其它文件中修改。首先介绍训练的流程：

1. 在`train.py`中，根据终端给定的参数生成相应的训练指令，指令会运行`run_training.py`
2. `run_training.py`会接受上述参数，设定初始种子，并导向`train_script.py`中的run逻辑
3. `train_script.py`负责多个模块的初始化设定，包括配置超参数、初始化日志、构建dataloader、构建模型网络、设置DDP训练、设置损失函数、优化器，以及构建一个actor
4. 这里定义的actor可以将模型进行前向传递，并计算损失，
