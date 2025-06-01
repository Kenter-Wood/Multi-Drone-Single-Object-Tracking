"""
Basic model for Mytrack.
"""
import math
import os

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from torch.nn.functional import l1_loss

from ..layers.head import build_box_head
from .vision_transformer import vit_tiny_patch16_224
from utils.box_ops import box_xyxy_to_cxcywh

from .loss_functions import DJSLoss
from .statistics_network import (
    GlobalStatisticsNetwork,
)


class MyTrack(nn.Module):
    """ This is the base class for MyTrack """
    def __init__(self, backbone, head, aux_loss=False, head_type="CORNER", temp_token_len=1):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        # 本文选了三种vision transformer作为backbone
        self.backbone = backbone
        self.head = head
        self.temp_token_len = temp_token_len
        self.aux_loss = aux_loss
        self.head_type = head_type
        # 在定义模型的时候，首先初始化维持一个temp_token,
        # 这个token代表一个初始值，它对所有视频生效。这是一个尝试，后面可能换成具有视频特异性的量
        # 这个token并不是直接送到模型当中初始化的那个，
        self.temp_token = nn.Parameter(
            torch.zeros(1, temp_token_len, self.backbone.embed_dim)
        )
        nn.init.normal_(self.temp_token, mean=0, std=0.02)
        self.temp_token_update = None
        # 对角点和中心检测器，需要计算之后热力图的尺寸
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(head.feat_sz)
            self.feat_len_s = int(head.feat_sz ** 2)
            self.feat_sz_t = int(head.feat_sz_t)
            self.feat_len_t = int(head.feat_sz_t ** 2)
        if self.aux_loss:
            self.head = _get_clones(self.head, 6)

        self.djs_loss = DJSLoss()
        self.l1_loss = l1_loss

        self.feature_map_size = 8  #128x128
        self.feature_map_channels = backbone.embed_dim
        self.num_ch_coding =self.backbone.embed_dim
        self.coding_size =8
        self.global_stat_x = GlobalStatisticsNetwork(
            feature_map_size=self.feature_map_size,
            feature_map_channels=self.feature_map_channels,
            coding_channels=self.num_ch_coding,
            coding_size=self.coding_size,
        )
        


#         out_dict = self.net(template=template_list,search=search_img, 
# template_anno=template_anno, search_anno=search_anno, is_distill=self.net.is_distill_training)
# 这是actor中使用MyTrack的Forward的方式
# 需要改成让其接受一个序列的形式

    def forward(self, template: torch.Tensor,
                    search: torch.Tensor,
                    template_annos: torch.Tensor,
                    search_annos: torch.Tensor,
                    return_last_attn=False,
                    is_distill=False,
                ):
        if self.training:
            self.temp_token_update = None
        out_dict = []
        for i in range(len(search)):
            
            if self.training and not is_distill:
                template_anno = torch.round(template_annos[i:i+1]*8).int()
                template_anno[template_anno<0]=0
                search_anno = torch.round(search_annos[i:i+1]*16).int()
                search_anno[search_anno<0]=0
            else:
                template_anno = template_annos
                search_anno = search_annos
            # 这种情况下，每次temp_token需要更新的时候，它就会先赋予None，再赋予初始值。
            # 训练和测试阶段，需要初始化的位置不一样，所以分开设置None
            if self.temp_token_update == None:
                self.temp_token_update = (self.temp_token.clone()).detach()
            x, aux_dict = self.backbone(z=template[i:i+1] if isinstance(template, torch.Tensor) else template[i],
                                        x=search[i:i+1]if isinstance(search, torch.Tensor) else search[i], 
                                        return_last_attn=return_last_attn, is_distill=is_distill, 
                                        temp_token=self.temp_token_update,
                                        is_training = self.training)
            # 这个模式下，会把一个初始化的token与迭代后的temp_token相加
            self.temp_token_update = (x[:, :self.temp_token_len].clone()).detach() + self.temp_token
            # print("2334523445")
            if self.training and not is_distill:
                prob_active_m = torch.cat(aux_dict['probs_active'],dim=1).mean(dim=1)
                prob_active_m = prob_active_m.reshape(len(prob_active_m),1)
                expected_active_ratio = 0.7 * torch.ones(prob_active_m.shape)
                activeness_loss = self.l1_loss(prob_active_m ,expected_active_ratio.to(prob_active_m.device))
            else:
                activeness_loss = 0

            feat_last = x[:, self.temp_token_len:]
            if isinstance(x, list):
                feat_last = x[-1, :, self.temp_token_len:]
            out = self.forward_head(feat_last, None, template_anno=template_anno, search_anno=search_anno, is_distill=is_distill)
            # print(out)
            out.update(aux_dict)
            out['backbone_feat'] = x
            out['activeness_loss'] = activeness_loss
            out_dict.append(out)
        return out_dict
    #             search: torch.Tensor,
    #             template_anno: torch.Tensor,
    #             search_anno: torch.Tensor,
    #             return_last_attn=False,
    #             is_distill=False,
    #             ):

    #     if self.training and not is_distill:
    #         template_anno = torch.round(template_anno*8).int()
    #         template_anno[template_anno<0]=0
    #         search_anno = torch.round(search_anno*16).int()
    #         search_anno[search_anno<0]=0
    #     # backbone类型是如何接受z和x的？特别是Transformer
    #     # 作者在vision transformer里面做了修改，forward可以接受这个列表了
    #     x, aux_dict = self.backbone(z=template, x=search, 
    #                                 return_last_attn=return_last_attn, is_distill=is_distill, temp_token=self.temp_token)
        

    #     if self.training and not is_distill:
    #         prob_active_m = torch.cat(aux_dict['probs_active'],dim=1).mean(dim=1)
    #         prob_active_m = prob_active_m.reshape(len(prob_active_m),1)
    #         expected_active_ratio = 0.7 * torch.ones(prob_active_m.shape)
    #         activeness_loss = self.l1_loss(prob_active_m ,expected_active_ratio.to(prob_active_m.device))
    #     else:
    #         activeness_loss = 0

    #     # Forward head 现在这个头还是只用空间特征来回归，时序的token会传递到样本中的下一帧
    #     feat_temp = x[:, :self.temp_token_len, :]
    #     feat_last = x[:, self.temp_token_len:, :]
    #     if isinstance(x, list):
    #         feat_last = x[-1]
    #     out = self.forward_head(feat_last, None, template_anno=template_anno, search_anno=search_anno, is_distill=is_distill)

    #     out.update(aux_dict)
    #     out['backbone_feat'] = x
    #     out['activeness_loss'] = activeness_loss
    #     return out

    def forward_head(self, cat_feature, gt_score_map=None, template_anno=None, search_anno=None, is_distill=False):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if self.training:
            feat_len_t = cat_feature.shape[1]-self.feat_len_s
            feat_sz_t = int(math.sqrt(feat_len_t))
            enc_opt_z = cat_feature[:, 0:feat_len_t]
            opt = (enc_opt_z.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat_z = opt.view(-1, C, feat_sz_t, feat_sz_t)

        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)


        global_mutual_loss=0
        if self.training and not is_distill:
            # opt_feat_mask=torch.zeros(32,192,8,8)
            # opt_feat_x=torch.zeros(32,192,8,8)
            opt_feat_mask = torch.zeros(cat_feature.shape[0], cat_feature.shape[2], 8, 8)
            opt_feat_x = torch.zeros(cat_feature.shape[0], cat_feature.shape[2], 8, 8)
            for i in range(opt_feat.shape[0]):
                # 这里本来需要把第一维去掉，但新的条件下送进来了5个模板，所以squeeze不掉
                bbox = template_anno.squeeze()[i]
                bbox = torch.tensor([bbox[0], bbox[1], min([bbox[2], 8]), min([bbox[3], 8])])
                x_t = bbox[0]
                y_t = bbox[1]

                target_sz_t = opt_feat_mask[i,:, y_t:y_t+bbox[3], x_t:x_t+bbox[2]].shape

                bbox = search_anno.squeeze()[i]
                bbox = torch.tensor([bbox[0], bbox[1], min([bbox[2], 8]), min([bbox[3], 8])])

                target_sz_s = opt_feat[i,:,bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]].shape
                h = min([target_sz_t[1], target_sz_s[1]])
                w = min([target_sz_t[2], target_sz_s[2]])
                opt_feat_x[i,:, y_t:y_t+h, x_t:x_t+w] = opt_feat[i,:,bbox[1]:bbox[1]+h,bbox[0]:bbox[0]+w]
                opt_feat_mask[i,:, y_t:y_t+h, x_t:x_t+w] = 1


            opt_feat_z = opt_feat_z*opt_feat_mask.to(opt_feat_z.device)

            x = opt_feat_z.to(opt_feat.device)
            y = opt_feat_x.to(opt_feat.device)
            x_shuffled = torch.cat([x[1:], x[0].unsqueeze(0)], dim=0)

            # Global mutual information estimation
            global_mutual_M_R_x = self.global_stat_x(x, y)  # positive statistic
            global_mutual_M_R_x_prime = self.global_stat_x(x_shuffled, y)
            global_mutual_loss = self.djs_loss(
                T=global_mutual_M_R_x,
                T_prime=global_mutual_M_R_x_prime,
            )


        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,

                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            # print(outputs_coord.shape)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'mine_loss': global_mutual_loss,
                   }
            return out
        else:
            raise NotImplementedError


def build_mytrack(cfg, training=True):
    # 预训练的模型从这里加载
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    # 存在预训练的模型，预训练的不是MyTrack本身，并且是在训练的话，就加载模型
    if cfg.MODEL.PRETRAIN_FILE and ('MyTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    # 六种不同的backbone，都属于vit系列
    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224':
        backbone = deit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny':
        backbone = vit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224':
        backbone = eva02_tiny_patch14_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    # 由于这里的模型调用的transformer经过修改，forward已经适配了tracking，所以不用finetune
    # backbone的前传方法到visiontranformer里面的forward改
    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224'\
           or cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224'\
            or cfg.MODEL.BACKBONE.TYPE == 'vit_tiny':
        pass
    else:
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    head = build_box_head(cfg, hidden_dim)

    model = MyTrack(
        backbone,
        head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
    # 如果以MyTrack为pretrain，就直接加载上次的训练参数。
    if 'MyTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
