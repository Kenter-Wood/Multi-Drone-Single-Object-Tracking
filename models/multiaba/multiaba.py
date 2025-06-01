import os
import math
import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones
from utils.box_ops import box_xyxy_to_cxcywh
from models.layers.head import build_box_head
from models.multiaba.AbaViT import abavit_patch16_384

class MultiAbaTrack(nn.Module):
    '''
        This module combines Aba-ViT,an Efficient ViT model and the 
        multi-drone tracking task
    '''
    def __init__(self, backbone, head, aux_loss=False, head_type="CENTER"):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.aux_loss = aux_loss
        self.head_type = head_type

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(head.feat_sz)
            self.feat_len_s = int(head.feat_sz ** 2)

        if self.aux_loss:
            self.head = _get_clones(self.head, 6)
    
    def forward(self, 
                template1: torch.Tensor,
                template2: torch.Tensor,
                template3: torch.Tensor,
                search:torch.Tensor):
        
        x, aux_dict = self.backbone(z1=template1,
                                    z2=template2,
                                    z3=template3,
                                    x=search)
        
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        out = self.forward_head(feat_last, None)
        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

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
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                'score_map': score_map_ctr,
                'size_map': size_map,
                'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_multi_aba(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../train/pretrained')

    if cfg.MODEL.PRETRAIN_FILE and ('MultiAbaTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        if cfg.MODEL['IS_DISTILL']:
            pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE_STUDENT)
        else:
            pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = None

    if cfg.MODEL.BACKBONE.TYPE == 'abavit_patch16_384':
        backbone = abavit_patch16_384(pretrained)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError
    
    head = build_box_head(cfg, hidden_dim)

    model = MultiAbaTrack(
        backbone,
        head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE
    )

    if 'MultiAbaTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        # print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print(missing_keys, unexpected_keys)

    return model        