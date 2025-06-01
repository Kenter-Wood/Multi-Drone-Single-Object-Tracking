"""
@ author: Yakun Wu
@ version: 2025/03/20 
@ description:   
    三模板

"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from .base_actor import BaseActor
from utils.misc import NestedTensor
from utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from utils.merge import merge_template_search
from utils.heatmap_utils import generate_heatmap
from utils.ce_utils import generate_mask_cond, adjust_keep_rate

from models.mythreetrack.statistics_network import GlobalStatisticsNetwork
from models.mythreetrack.loss_functions import DJSLoss



class ThreeTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.djs_loss = DJSLoss()
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.feature_map_size = 24
        # 为什么是这样？
        self.feature_map_channels = 840
        self.coding_size = 8
        self.num_ch_coding = self.net.module.backbone.embed_dim
        
        self.global_stat_x = GlobalStatisticsNetwork(
            feature_map_size=self.feature_map_size,
            feature_map_channels=self.feature_map_channels,
            coding_channels=self.num_ch_coding,
            coding_size=self.coding_size,
        ).cuda()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        # assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        # print(len(data['template_images']))
        for i in range(3):      # 三机三个模板
            # print('template_size', data['template_images'].shape)
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        # if len(template_list) == 1:
        #     template_list = template_list[0]
        out_dict = self.net(template1=template_list[0],
                            template2=template_list[1],      # B机模板
                            template3=template_list[2],      # C机模板
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        if self.net.is_distill_training:
            with torch.no_grad():
                out_dict_teacher = self.net_teacher(template1=template_list[0],
                            template2=template_list[1],      # B机模板
                            template3=template_list[2],      # C机模板
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
            feat_teacher = out_dict_teacher['backbone_feat']
            feat_student = out_dict['backbone_feat']
            t = nn.functional.softmax(feat_teacher.div(2), dim=-1)
            s = nn.functional.log_softmax(feat_student.div(2), dim=-1)
            t_shuffled = torch.cat([t[1:], t[0].unsqueeze(0)], dim=0)

            # Global mutual information estimation
            global_mutual_M_R_x = self.global_stat_x(t, s)  # positive statistic
            global_mutual_M_R_x_prime = self.global_stat_x(t_shuffled, s)
            global_mutual_loss = self.djs_loss(
                T=global_mutual_M_R_x,
                T_prime=global_mutual_M_R_x_prime,
            )
            distill_loss = global_mutual_loss

            out_dict['distill_loss'] = distill_loss

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            # print("check score map", pred_dict['score_map'].shape, torch.min(pred_dict['score_map']).item(), torch.max(pred_dict['score_map']).item())
            # score map: B * C * H * W
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        if self.net.is_distill_training:
            distill_loss = pred_dict['distill_loss']
            loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss \
                     + 0.0001 * distill_loss


        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
