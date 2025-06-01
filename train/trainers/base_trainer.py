import os
import glob
import torch
import torch.nn as nn
import traceback
from torch.utils.data.distributed import DistributedSampler

def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.parallel.distributed.DistributedDataParallel))


class MultiGPU(nn.parallel.distributed.DistributedDataParallel):
    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)

def freeze_layers(model, freeze_keywords, exclude_layers):
    """
    冻结模型中包含特定关键词的参数，但排除特定层。

    Args:
        model (torch.nn.Module): 模型对象。
        freeze_keyword (list): 要冻结的层名称中包含的关键词（如 ["backbone"]）。
        exclude_layers (list): 要排除冻结的层名称列表（如 ["backbone.block.11"]）。
    """
    for name, param in model.named_parameters():
        # 如果参数名称包含冻结关键词，但不在排除列表中，则冻结
        if any(keyword in name for keyword in freeze_keywords):
            param.requires_grad = False
        if any(exclude in name for exclude in exclude_layers):
            print('yes')
            param.requires_grad = True

def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.parallel.distributed.DistributedDataParallel))      



class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir'''
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

# 对train接受的参数做了一些调整，从接受伦次，到接受整个训练参数
    def train(self, cfg, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """


        max_epochs = cfg.TRAIN.EPOCH
        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                # freeze_layers(self.actor.net, ["backbone"], ["backbone.blocks.10", "backbone.blocks.11"])
                # print("Actual learnable parameters: ")
                # for name, param in self.actor.net.named_parameters():
                #     if param.requires_grad:
                #         print(name)
                # print(f"Reinitializing optimizer at epoch {epoch}...")
                    
                # trainable_params = filter(lambda p: p.requires_grad, self.actor.net.parameters())
                # if "differentiable" in self.optimizer.defaults:
                #     del self.optimizer.defaults["differentiable"]
                # self.optimizer = self.optimizer.__class__(trainable_params, **self.optimizer.defaults)
                if load_latest:
                    self.load_checkpoint()
                if load_previous_ckpt:
                    directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_prv)
                    self.load_state_dict(directory)
                if distill:
                    directory_teacher = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path_teacher)
                    self.load_state_dict(directory_teacher, distill=True)
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch
                    

                    # # 2025.3.31, 加入一个训练过程中逐渐解冻层的函数
                    # ###########
                    # if epoch <= 15:
                    #     freeze_layers(self.actor.net, ["backbone.blocks", 'backbone.patch_embed'], ["backbone.block.11"])
                    # elif epoch <= 30:
                    #     freeze_layers(self.actor.net, ["backbone.blocks", 'backbone.patch_embed'], ["backbone.block.10", "backbone.block.11"])  
                    # elif epoch <= 45:
                    #     freeze_layers(self.actor.net, ["backbone.blocks", 'backbone.patch_embed'], ["backbone.block.9", "backbone.block.10", "backbone.block.11"]) 
                    # elif epoch <= 60:
                    #     freeze_layers(self.actor.net, ["backbone.blocks", 'backbone.patch_embed'], ["backbone.block.8", "backbone.block.9", 
                    #                                                "backbone.block.10", "backbone.block.11"])
                    # else:
                    #     freeze_layers(self.actor.net, ["backbone.blocks", 'backbone.patch_embed'], ["backbone.block.7", "backbone.block.8", 
                    #                                                "backbone.block.9", "backbone.block.10", 
                    #                                                "backbone.block.11"])
                    
                    # print("Actual learnable parameters: ")
                    # for name, param in self.actor.net.named_parameters():
                    #     if param.requires_grad:
                    #         print(name)
                    # # 优化器的参数更新
                    # if epoch in [1, 16, 31, 46, 61]:
                    #     print(f"Reinitializing optimizer at epoch {epoch}...")
                    #     trainable_params = filter(lambda p: p.requires_grad, self.actor.net.parameters())
                    #     self.optimizer = self.optimizer.__class__(trainable_params, **self.optimizer.defaults)
                    #     if epoch == 1:
                    #         for param_group in self.optimizer.param_groups:
                    #             param_group['lr'] = 8e-5
                    #     else:
                    #         param_group['lr'] = 1e-5

                    ###########

                    self.train_epoch()

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)
                    # only save the last 10 checkpoints
                    save_every_epoch = getattr(self.settings, "save_every_epoch", False)
                    save_epochs = [79, 159, 239]
                    if epoch > (max_epochs - 1) or save_every_epoch or epoch % 1 == 0 or epoch in save_epochs or epoch > (max_epochs - 5):
                    # if epoch > (max_epochs - 10) or save_every_epoch or epoch % 100 == 0:
                        if self._checkpoint_dir:
                            if self.settings.local_rank in [-1, 0]:
                                self.save_checkpoint()
            except Exception as e:
                print('Training crashed at epoch {}'.format(epoch))
                print('Error message: {}'.format(e))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            # 'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            # 'settings': self.settings
        }

        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        print(directory)
        if not os.path.exists(directory):
            print("directory doesn't exist. creating...")
            os.makedirs(directory)

        # First save as a tmp file
        tmp_file_path = '{}/{}_ep{:04d}.tmp'.format(directory, net_type, self.epoch)
        torch.save(state, tmp_file_path)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)

        # Now rename to actual checkpoint. os.rename seems to be atomic if files are on same filesystem. Not 100% sure
        os.rename(tmp_file_path, file_path)

    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        # print(checkpoint_path)
        # print(f"WAWAWA! {net_type}, {checkpoint_dict['net_type']}")
        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
        # 2021.1.10 Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def load_state_dict(self, checkpoint=None, distill=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        if distill:
            net = self.actor.net_teacher.module if is_multi_gpu(self.actor.net_teacher) \
                else self.actor.net_teacher
        else:
            net = self.actor.net.module if is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True
