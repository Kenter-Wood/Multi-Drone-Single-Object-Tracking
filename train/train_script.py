import copy
import os
import sys
# loss function related
import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.box_ops import giou_loss
from utils.focal_loss import FocalLoss
# train pipeline related
from trainers.ltr_trainer import LTRTrainer
# distributed training related

# some more advanced functions
from settings import update_settings
from trainers.optimizer import get_optimizer_scheduler
# network related
from models.mytrack import build_mytrack
from models.mythreetrack import build_track_three
from models.multiaba.multiaba import build_multi_aba
from models.multitemp.multitemptrack import build_multitemptrack
# forward propagation related
from actors.mytrack import MyTrackActor
from actors.threetrack import ThreeTrackActor
from actors.multiabatrack import MultiAbaActor
from actors.multitemptrack import MultiTempActor
from dataloader.build_dataloader import build_dataloaders, build_dataloaders_threemdot
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file config给出参数列表，yaml则更新参数到指定值
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("configs.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log 日志记录
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders,这里包含了构建loader和数据集，此项目更改了每个样本加载的模板和搜索个数
    if settings.script_name in ["multitrack", "threetrack_distill", "multiaba", "multitemp"]:
        loader_train, loader_val = build_dataloaders_threemdot(cfg, settings)
    else:
        loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "mytrack":
        # ===== distillation ====
        is_distill_training = cfg.MODEL['IS_DISTILL']
        if is_distill_training:
            cfg_teacher.MODEL['BACKBONE']['TYPE'] = 'vit_tiny_patch16_224'
            cfg_teacher['MODEL']['BACKBONE']['TYPE'] = 'vit_tiny_patch16_224'
            net_teacher = build_mytrack(cfg_teacher)
            cur_path = os.path.abspath(__file__)
            pro_path = os.path.abspath(os.path.join(cur_path, '../../..'))
            checkpoint = torch.load(
                os.path.join(pro_path, 'teacher_model/vit_tiny_patch16_224/MyTrack_ep0300.pth.tar'),
                map_location="cpu")
            missing_keys, unexpected_keys = net_teacher.load_state_dict(checkpoint["net"], strict=False)
            net_teacher.cuda()
            print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
            cfg.MODEL['BACKBONE']['TYPE'] = 'deit_tiny_distilled_patch16_224'
            cfg['MODEL']['BACKBONE']['TYPE'] = 'deit_tiny_distilled_patch16_224'
            net = build_mytrack(cfg)
        else:
        # 这里构建的mytrack就是之后进行训练前向传递的mytrack
        # 所以net是一个MyTrack类型的Module
        # 它在actor里面进行了前传
            net = build_mytrack(cfg)
    elif settings.script_name == "multitrack":
        net = build_track_three(cfg)
    elif settings.script_name == "multiaba":
        net = build_multi_aba(cfg)
    elif settings.script_name == "multitemp":
        net = build_multitemptrack(cfg)
    elif settings.script_name == "threetrack_distill":
        is_distill_training = cfg.MODEL['IS_DISTILL']
        if is_distill_training:
            # 获取teacher的参数，并定义teacher网
            cfg_teacher = copy.deepcopy(cfg)
            cfg_teacher.MODEL['BACKBONE']['TYPE'] = 'vit_base_patch16_224_trackthree'
            cfg_teacher['MODEL']['BACKBONE']['TYPE'] = 'vit_base_patch16_224_trackthree'
            cfg_teacher.MODEL['IS_DISTILL'] = False
            net_teacher = build_track_three(cfg_teacher)
            # 从路径读取teacher的参数
            cur_path = os.path.abspath(__file__)
            pro_path = os.path.abspath(os.path.join(cur_path, '../../..'))
            checkpoint = torch.load(
                os.path.join(pro_path, 'teacher_model/vit_base_patch16_224_trackthree/ThreeTrack_ep0122.pth.tar'),
                map_location="cpu")
            missing_keys, unexpected_keys = net_teacher.load_state_dict(checkpoint["net"], strict=False)
            net_teacher.cuda()
            print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE_STUDENT)
            cfg.MODEL['BACKBONE']['TYPE'] = 'vit_tiny_patch16_224_trackthree'
            cfg['MODEL']['BACKBONE']['TYPE'] = 'vit_tiny_patch16_224_trackthree'

            net = build_track_three(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        # net_teacher = DDP(net_teacher, device_ids=[settings.local_rank], find_unused_parameters=True)
        # net_teacher2 = DDP(net_teacher2, device_ids=[settings.local_rank], find_unused_parameters=True)
        # net_teacher3 = DDP(net_teacher3, device_ids=[settings.local_rank], find_unused_parameters=True)
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        if cfg.MODEL['IS_DISTILL']:
            net_teacher = DDP(net_teacher, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "mytrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0,'mse':1.0}
        actor = MyTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        if is_distill_training:
            actor.net_teacher = net_teacher
    elif settings.script_name == "multitrack":
        is_distill_training = cfg.MODEL['IS_DISTILL']
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0,'mse':1.0}
        actor = ThreeTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        if is_distill_training:
            actor.net_teacher = net_teacher
    elif settings.script_name == "threetrack_distill":
        is_distill_training = cfg.MODEL['IS_DISTILL']
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0,'mse':1.0}
        actor = ThreeTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        if is_distill_training:
            actor.net_teacher = net_teacher    
    elif settings.script_name == "multiaba":
        is_distill_training = cfg.MODEL['IS_DISTILL']
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0,'mse':1.0}
        actor = MultiAbaActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        if is_distill_training:
            actor.net_teacher = net_teacher
    elif settings.script_name == "multitemp":
        is_distill_training = cfg.MODEL['IS_DISTILL']
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0,'mse':1.0}
        actor = MultiTempActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
        if is_distill_training:
            actor.net_teacher = net_teacher              
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    

    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    if is_distill_training:
        trainer.actor.net.is_distill_training = True
    else:
        trainer.actor.net.is_distill_training = False

    # train process
    trainer.train(cfg, load_latest=True, fail_safe=True)