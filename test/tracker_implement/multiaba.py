import math
import torch

from PIL import Image
from models.multiaba.multiaba import build_multi_aba
from .basetracker import BaseTracker
from utils.vis_utils import gen_visualization
from utils.hann import hann2d
from utils.data_processing import sample_target
# for debug
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.data_utils import Preprocessor
from utils.box_ops import clip_box
from utils.ce_utils import generate_mask_cond
import time

class MultiAbaTrack(BaseTracker):
    def __init__(self, params):
        super(MultiAbaTrack, self).__init__(params)
        network = build_multi_aba(params.cfg, training=False)
        state_dict = torch.load(self.params.checkpoint, map_location='cpu')['net']
        network.load_state_dict(state_dict, strict=True)

        # ######
        #     #2025.4.3，适配作者模型临时修改
        # state_dict = torch.load(self.params.checkpoint, map_location='cpu')['net']
        # state_dict = {k.replace('box_head', 'head'): v for k, v in state_dict.items()}
        # network.load_state_dict(state_dict, strict=True)
        # ######

        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
#        self.debug = params.debug
        self.debug = False
#        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.z_dict2 = {}
        self.z_dict3 = {}

        self.z_dict_list = []   # 模板列表



    def initialize(self, image1, image2, image3, info1: dict, info2: dict, info3: dict):
        """
        优化后的 initialize 函数。
        """
        # 处理 A 机模板
        self.z_dict1, self.box_mask_z, self.z_patch_arr1, resize_factor1 = process_template(
            image1, info1, self.params.template_factor, self.params.template_size,
            self.preprocessor, self.cfg, self.transform_bbox_to_crop
        )

        # 处理 B 机模板
        self.z_dict2, self.box_mask_z2, self.z_patch_arr2, resize_factor2 = process_template(
            image2, info2, self.params.template_factor, self.params.template_size,
            self.preprocessor, self.cfg, self.transform_bbox_to_crop
        )

        # 处理 C 机模板
        self.z_dict3, self.box_mask_z3, self.z_patch_arr3, resize_factor3 = process_template(
            image3, info3, self.params.template_factor, self.params.template_size,
            self.preprocessor, self.cfg, self.transform_bbox_to_crop
        )

        # 保存状态
        self.state = info1['init_bbox']
        self.frame_id = 0

        if self.save_all_boxes:
            '''保存所有预测框'''
            all_boxes_save = info1['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}


    def track(self,drone_id, image1, image2, image3, info1: dict = None, info2: dict = None, info3: dict = None):
        """
        更改后的 track 函数。
        """
        # 预处理
        t1 = time.time()

        H, W, _ = image1.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image1, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)


        # 模型推理
        t2 = time.time()
        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer

            out_dict = self.network.forward(                  # 在forward这里要把三个模板传入模型
                template1=self.z_dict1.tensors, template2=self.z_dict2.tensors,template3=self.z_dict3.tensors, search=x_dict.tensors)
        
        time_net = time.time()
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        ######################测试阶段可视化
        # save_dir = "/data/yakun/MultiTrack/output/debug/redetect_vis"
    #     heatmap = response.squeeze().cpu().numpy()
    #     print(self.z_dict1.tensors.shape)
    #     print(x_dict.tensors.shape)

    #     os.makedirs(save_dir, exist_ok=True)
    #     save_path = os.path.join(save_dir, f"heatmap_drone{drone_id}_frame{self.frame_id}.png")

    #             # 创建热力图可视化
    #     plt.figure(figsize=(8, 6))
    #     plt.imshow(heatmap, cmap='jet')  # 使用'jet'色彩映射以获得更好的可视化效果
    #     plt.colorbar()  # 添加颜色条
    #     plt.title(f'Response Heatmap (Frame {self.frame_id})')
    #     plt.axis('off')  # 隐藏坐标轴
        
    #     # 保存图像
    #     plt.savefig(save_path, bbox_inches='tight', dpi=150)
    #     plt.close()  # 关闭图形以释放内存
        

    # 2. 保存模板图像
        # template_tensor = self.z_dict1.tensors.clone().cpu()
        # template_vis = template_tensor.squeeze(0).permute(1, 2, 0).numpy()
        # if template_vis.dtype == np.float32 or template_vis.dtype == np.float64:
        #     template_vis = ((template_vis - np.min(template_vis))/(np.max(template_vis)-np.min(template_vis))*255).astype(np.uint8)
        # print("template_vis", template_vis.shape)
        # tempale_img = Image.fromarray(template_vis)
        # tempale_img.save(os.path.join(save_dir, f"template1_drone{drone_id}_frame{self.frame_id}.png"))


        # template_tensor = self.z_dict2.tensors.clone().cpu()
        # template_vis = template_tensor.squeeze(0).permute(1, 2, 0).numpy()
        # if template_vis.dtype == np.float32 or template_vis.dtype == np.float64:
        #     template_vis = ((template_vis - np.min(template_vis))/(np.max(template_vis)-np.min(template_vis))*255).astype(np.uint8)
        # print("template_vis", template_vis.shape)
        # tempale_img = Image.fromarray(template_vis)
        # tempale_img.save(os.path.join(save_dir, f"template2_drone{drone_id}_frame{self.frame_id}.png"))

        # template_tensor = self.z_dict3.tensors.clone().cpu()
        # template_vis = template_tensor.squeeze(0).permute(1, 2, 0).numpy()
        # if template_vis.dtype == np.float32 or template_vis.dtype == np.float64:
        #     template_vis = ((template_vis - np.min(template_vis))/(np.max(template_vis)-np.min(template_vis))*255).astype(np.uint8)
        # print("template_vis", template_vis.shape)
        # tempale_img = Image.fromarray(template_vis)
        # tempale_img.save(os.path.join(save_dir, f"template3_drone{drone_id}_frame{self.frame_id}.png"))

        # search_tensor = x_dict.tensors.clone().cpu()
        # search_vis = search_tensor.squeeze(0).permute(1, 2, 0).numpy()
        # if search_vis.dtype == np.float32 or search_vis.dtype == np.float64:
        #     search_vis = ((search_vis - np.min(search_vis))/(np.max(search_vis)-np.min(search_vis))*255).astype(np.uint8)
        # print("template_vis", search_vis.shape)
        # search_img = Image.fromarray(search_vis)
        # search_img.save(os.path.join(save_dir, f"search_drone{drone_id}_frame{self.frame_id}.png"))

    #     time.sleep(5)
        ######################



        response_APCE = self.calAPCE_optimized(response)                # 计算平均峰值能量APCE

        pred_boxes, max_score = self.network.head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)   # 获得最大score
        # 后处理
        t3 = time.time()
        # print("头部推理：", t3-time_net)
        # print("骨干网络推理：", time_net-t2)

        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        t4 = time.time()
        # print('预处理时间：', t2-t1)
        # print('推理时间：', t3-t2)
        # print('后处理时间：', t4-t3)



        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image1, info1['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking'+drone_id)

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region'+drone_id)
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template'+drone_id)
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map'+drone_id)
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann'+drone_id)

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search'+drone_id)

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}, max_score, response_APCE




# 将B机搜索区域在A机中查询，然后重检测
    def three_search_redetect(self, image_a, image_b, drone_id, state_b, tmp_factor = 4.0, tmp_s_factor = 12.0, info_a: dict = None, info_b: dict = None):       

        # print(drone_id, "机丢失， cross redetect")

        H, W, _ = image_a.shape
        #self.frame_id += 1
        
        
        # (x1, y1, w, h)     # 将B的搜索区域裁出来，然后在A中查询
        # 这个sample将B的检测框扩大4倍，但是又会处理到原始模板大小
        x_patch_arr_b, resize_factor_b, x_amask_arr_b = sample_target(image_b, state_b, tmp_factor,
                                                                output_sz=self.params.template_size)  
        
        search_b = self.preprocessor.process(x_patch_arr_b, x_amask_arr_b)

        #print("before: ", self.state)
        # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
        # 这个sample将A的检测框扩大12倍，但是又会处理到原始搜索区域大小
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_a, self.state, tmp_s_factor,
                                                                output_sz=self.params.search_size)  

        search_a = self.preprocessor.process(x_patch_arr, x_amask_arr)
        # print(self.state)
        # save_dir = "/data/yakun/MultiTrack/output/debug/redetect_vis"
        # search_tensor = search_a.tensors.clone().cpu()
        # search_vis = search_tensor.squeeze(0).permute(1, 2, 0).numpy()
        # if search_vis.dtype == np.float32 or search_vis.dtype == np.float64:
        #     search_vis = ((search_vis - np.min(search_vis))/(np.max(search_vis)-np.min(search_vis))*255).astype(np.uint8)
        # print("template_vis", search_vis.shape)
        # search_img = Image.fromarray(search_vis)
        # search_img.save(os.path.join(save_dir, f"candidate_{drone_id}_frame{self.frame_id}_{tmp_s_factor}.png"))

        # search_tensor = search_b.tensors.clone().cpu()
        # search_vis = search_tensor.squeeze(0).permute(1, 2, 0).numpy()
        # if search_vis.dtype == np.float32 or search_vis.dtype == np.float64:
        #     search_vis = ((search_vis - np.min(search_vis))/(np.max(search_vis)-np.min(search_vis))*255).astype(np.uint8)
        # print("template_vis", search_vis.shape)
        # search_img = Image.fromarray(search_vis)
        # search_img.save(os.path.join(save_dir, f"reference_{drone_id}_frame{self.frame_id}_{tmp_factor}.png"))


        # 以B的搜索区域为模板，A的搜索区域为搜索区域，进行重检测
        # B是信息完善的，A需要B的信息重新定位自己的搜索区域，所以是对A的重检测
        with torch.no_grad():
            #x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(                  # 在forward这里三个模板都是B的
                template1=search_b.tensors, template2=search_b.tensors, template3=search_b.tensors, search=search_a.tensors)

        # add hann windows
        pred_score_map = out_dict['score_map']
        #response = self.output_window * pred_score_map

        #response_APCE = self.calAPCE(response)                # 计算平均峰值能量APCE

        pred_boxes, max_score = self.network.head.cal_bbox(pred_score_map, out_dict['size_map'], out_dict['offset_map'], return_score=True)   # 获得最大score
        # print("pred_boxes: ", pred_boxes)
        # print("max score: ", max_score)



        pred_boxes = pred_boxes.view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

####
        # x_patch_arr, resize_factor, x_amask_arr = sample_target(image_a, self.state, tmp_s_factor,
        #                                                         output_sz=self.params.search_size)  
        
        # search_new = self.preprocessor.process(x_patch_arr, x_amask_arr)
        # search_tensor = search_new.tensors.clone().cpu()
        # search_vis = search_tensor.squeeze(0).permute(1, 2, 0).numpy()
        # if search_vis.dtype == np.float32 or search_vis.dtype == np.float64:
        #     search_vis = ((search_vis - np.min(search_vis))/(np.max(search_vis)-np.min(search_vis))*255).astype(np.uint8)
        # print("template_vis", search_vis.shape)
        # search_img = Image.fromarray(search_vis)
        # search_img.save(os.path.join(save_dir, f"new_{drone_id}_frame{self.frame_id}_{tmp_s_factor}.png"))
####
# 通过可视化发现，redetect预测出来的框可能产生很大的偏差。后面会纠正嘛？    




        #print("after: ", self.state)
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_a, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image_a, info_a['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking'+drone_id)

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region'+drone_id)
                self.visdom.register(torch.from_numpy(x_patch_arr_b).permute(2, 0, 1), 'image', 1, 'template'+drone_id)
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map'+drone_id)
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann'+drone_id)

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search'+drone_id)

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        self.frame_id -= 1
        out_t, max_score_t, response_APCE_t = self.track(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)
        # out_t, max_score_t, response_APCE_t = self.three_nomulti_Fusetrack(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)       # 用于无多模板


        tmp_state = self.state.copy()         # 保存中间状态



        ############################# 在center中框个Box ##################################
        if tmp_s_factor > 8:
            # 将目标状态复制，强制将图像的中心点设置为搜索区域的中心点
            center_state = tmp_state.copy()
            center_state[1] , center_state[0] = H/2 , W/2
            # 创建新的搜索区域
            x_patch_arr_ctr, resize_factor_ctr, x_amask_arr_ctr = sample_target(image_a, center_state, tmp_s_factor,
                                                                    output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
            search_ctr = self.preprocessor.process(x_patch_arr_ctr, x_amask_arr_ctr)

            # with torch.no_grad():
            #     out_dict_ctr_z = self.network.forward(
            #         template=self.z_dict1.tensors, template2=self.z_dict2.tensors, search=search_ctr.tensors, ce_template_mask=self.box_mask_z)   # 在中间用模板检测

            # pred_score_map_ctr_z = out_dict_ctr_z['score_map']
            # pred_boxes_ctr_z, max_score_ctr_z = self.network.box_head.cal_bbox(pred_score_map_ctr_z, out_dict_ctr_z['size_map'], out_dict_ctr_z['offset_map'], return_score=True)   # 获得最大score

            # 中心区域
            with torch.no_grad():
                out_dict_ctr = self.network.forward(                 
                    template1=search_b.tensors, template2=search_b.tensors, template3=search_b.tensors, search=search_ctr.tensors)   # 在中间用搜索区域检测
            pred_score_map_ctr = out_dict_ctr['score_map']
            pred_boxes_ctr, max_score_ctr = self.network.head.cal_bbox(pred_score_map_ctr, out_dict_ctr['size_map'], out_dict_ctr['offset_map'], return_score=True)   # 获得最大score

            
            # 两种方案，以中心划定搜索区域，或者以bbox划定。
            # 判断中心框需确保其有足够的优势
            if (max_score_ctr - max_score) > 0.1:
                print("使用了中心框")
                # print(max_score_ctr)
                # print(max_score)
                # if (max_score_ctr - max_score_ctr_z) > 0:

                pred_boxes_ctr = pred_boxes_ctr.view(-1, 4)
                pred_box_ctr = (pred_boxes_ctr.mean(
                    dim=0) * self.params.search_size / resize_factor_ctr).tolist()  # (cx, cy, w, h) [0,1]
                self.state = clip_box(self.map_box_back(pred_box_ctr, resize_factor_ctr), H, W, margin=10)

                self.frame_id -= 1
                #out_ctr, max_score_ctr, response_APCE_ctr = self.three_multi_Fusetrack(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)
                # out_ctr, max_score_ctr, response_APCE_ctr = self.three_nomulti_Fusetrack(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)
                out_ctr, max_score_ctr, response_APCE_ctr = self.track(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)

                if max_score_ctr > max_score_t:
                    return out_ctr, max_score_ctr, response_APCE_ctr
                else:
                    self.state = tmp_state

        #########################一段可视化代码##########################
        # # 将所有预测框映射回原图坐标
        # # all_pred_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
        # all_pred_boxes = [self.state]

        # # 创建图像副本用于可视化
        # vis_image = image_a.copy()
            
        # # 绘制所有预测框
        # for i, box in enumerate(all_pred_boxes):
        #     x1, y1, w, h = [int(b) for b in box]
        #     # x1, y1, w, h = [int(b) for b in box.tolist()]
        #     # 使用不同颜色区分不同的预测框
        #     color = (0, 255, 0) if i == 0 else (255, 0, 0)  # 第一个框为绿色，其他为红色
        #     thickness = 2 if i == 0 else 1  # 第一个框线条更粗
        #     cv2.rectangle(vis_image, (x1, y1), (x1+w, y1+h), color, thickness)
        #     # 可选：显示预测分数
        #     # cv2.putText(vis_image, f"{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        # save_dir = "/data/yakun/Mytrack/debug/vis2"  # 可视化结果保存路径
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = os.path.join(save_dir, f"{drone_id}_frame{self.frame_id}_preds.jpg")
        # cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
        return out_t, max_score_t, response_APCE_t


    def one_time_redetect(self, image_a, image_b, drone_id, state_b, factor_list, info_a: dict = None, info_b: dict = None):
        """
            减少重检测对跟踪调用的次数
        """
        H, W, _ = image_a.shape
        assert len(factor_list) == 3
        search_list = []
        for factor in factor_list:
            x_patch_arr_b, resize_factor_b, x_amask_arr_b = sample_target(image_b, state_b, factor[0],
                                                                output_sz=self.params.template_size)  
        
            search_b = self.preprocessor.process(x_patch_arr_b, x_amask_arr_b)
            search_list.append(search_b.tensors.clone().detach())

        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_a, self.state, max([factor[1] for factor in factor_list]), output_sz=self.params.search_size)  
        
        search_a = self.preprocessor.process(x_patch_arr, x_amask_arr)        
        with torch.no_grad():
            #x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(                  # 在forward这里三个模板都是B的
                template1=search_list[0], template2=search_list[1], template3=search_list[2], search=search_a.tensors)
            
        pred_score_map = out_dict['score_map']

        pred_boxes, max_score = self.network.head.cal_bbox(pred_score_map, out_dict['size_map'], out_dict['offset_map'], return_score=True)   # 获得最大score

        pred_boxes = pred_boxes.view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)   

        self.frame_id -= 1
        out_t, max_score_t, response_APCE_t = self.track(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)


        tmp_state = self.state.copy()         # 保存中间状态

        # 将目标状态复制，强制将图像的中心点设置为搜索区域的中心点
        center_state = tmp_state.copy()
        center_state[1] , center_state[0] = H/2 , W/2
        # 创建新的搜索区域
        x_patch_arr_ctr, resize_factor_ctr, x_amask_arr_ctr = sample_target(image_a, center_state, max([factor[1] for factor in factor_list]), output_sz=self.params.search_size)  # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
        search_ctr = self.preprocessor.process(x_patch_arr_ctr, x_amask_arr_ctr)


            # 中心区域
        with torch.no_grad():
            out_dict_ctr = self.network.forward(                 
                template1=search_list[0], template2=search_list[1], template3=search_list[2], search=search_ctr.tensors)   # 在中间用搜索区域检测
        pred_score_map_ctr = out_dict_ctr['score_map']
        pred_boxes_ctr, max_score_ctr = self.network.head.cal_bbox(pred_score_map_ctr, out_dict_ctr['size_map'], out_dict_ctr['offset_map'], return_score=True)   # 获得最大score

            
            # 两种方案，以中心划定搜索区域，或者以bbox划定。
        # 判断中心框需确保其有足够的优势
        if (max_score_ctr - max_score) > 0.1:
            print("使用了中心框")

            pred_boxes_ctr = pred_boxes_ctr.view(-1, 4)
            pred_box_ctr = (pred_boxes_ctr.mean(
                dim=0) * self.params.search_size / resize_factor_ctr).tolist()  # (cx, cy, w, h) [0,1]
            self.state = clip_box(self.map_box_back(pred_box_ctr, resize_factor_ctr), H, W, margin=10)

            self.frame_id -= 1
            #out_ctr, max_score_ctr, response_APCE_ctr = self.three_multi_Fusetrack(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)
            # out_ctr, max_score_ctr, response_APCE_ctr = self.three_nomulti_Fusetrack(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)
            out_ctr, max_score_ctr, response_APCE_ctr = self.track(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)
            #中心框需要同时满足划定搜索域的分数合理，以及跟踪分数合理
            if max_score_ctr > max_score_t:
                return out_ctr, max_score_ctr, response_APCE_ctr
            else:
                self.state = tmp_state

        return out_t, max_score_t, response_APCE_t


    # 上述的重搜索方法存在的问题：
    # 为了一帧的重匹配，需要运行9次network(算上中心点)，计算密集
    # 相比图像匹配的优势在于：我们在目标跟踪中通常也会应对视角变换问题，如果用深度模型，也许
    # 方案一：既然要多尺度搜索，多个尺度的b一起送进去，直接在最大图搜索，效果会变差嘛？
    # 方案二：如果基于特征匹配，不用深度学习去找对应区域，受到
    # 关键困境：当前信息下，图像变换是三维的，但方法是二维的

    def feature_matching_redetect(self, image_a, image_b, drone_id, state_b, tmp_factor=4.0, tmp_s_factor=12.0, info_a: dict = None, info_b: dict = None):
        """
        基于特征匹配的重检测方法。
        """
        # print(drone_id, "机丢失， cross redetect")
        H, W, _ = image_a.shape
        #self.frame_id += 1
        
        # (x1, y1, w, h)     # 将B的搜索区域裁出来，然后在A中查询
        # 这个sample将B的检测框扩大4倍，但是又会处理到原始模板大小
        x_patch_arr_b, resize_factor_b, x_amask_arr_b = sample_target(image_b, state_b, tmp_factor,
                                                                output_sz=self.params.template_size)  
        #print("before: ", self.state)
        # (x1, y1, w, h)     # 以目标框为中心划定A的搜索区域
        # 这个sample将A的检测框扩大12倍，但是又会处理到原始搜索区域大小
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image_a, self.state, tmp_s_factor,
                                                                output_sz=self.params.search_size)  

        # 转化成灰度图。ref是作为参考的模板，find是搜寻范围，同样按一定放大比例划定
        # print("What is search_b", type(search_temp), search_temp.shape)
        # print("What is image_a", type(image_a), image_a.shape)
        gray_ref = cv2.cvtColor(x_patch_arr_b, cv2.COLOR_RGB2GRAY)
        gray_find = cv2.cvtColor(x_patch_arr, cv2.COLOR_RGB2GRAY)
        if gray_ref.dtype != np.uint8:
            gray_ref = (gray_ref * 255).astype(np.uint8)  # 如果值在 [0, 1] 范围内
        if gray_find.dtype != np.uint8:
            gray_find = (gray_find * 255).astype(np.uint8)  # 如果值在 [0, 1] 范围内
        # 建立orb，计算关键点keypoint和描述子descriptor
        orb = cv2.ORB_create(nfeatures=500)
        sift = cv2.SIFT_create(nfeatures=500)

        # kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
        # kp_find, des_find = orb.detectAndCompute(gray_find, None)
        kp_ref, des_ref = sift.detectAndCompute(gray_ref, None)
        kp_find, des_find = sift.detectAndCompute(gray_find, None)

        if des_ref is None or des_find is None or len(kp_ref) < 10 or len(kp_find) < 10:
            print("特征点不足，无法完成匹配")
            return {"target_bbox": self.state}, 0, 0

        # 匹配过程,依据汉明距离暴力匹配
        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # sift匹配过程，依据L2距离
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des_ref, des_find)
        # 排序并保留一定数量的优质匹配
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:min(50, len(matches))]
        
        if len(good_matches) < 10:
            print("有效匹配点不足，无法可靠估计变换")
            return {"target_bbox": self.state}, 0, 0

        # 提取特征点坐标
        ref_points = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        find_points = np.float32([kp_find[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC估计变换矩阵   
        M, mask = cv2.findHomography(ref_points, find_points, cv2.RANSAC, 5.0)
        if M is None:
            print("无法计算有效的变换矩阵")
            return {"target_bbox": self.state}, 0, 0
        
        # 计算B的模板在A的搜索区域中的位置
        h, w = gray_ref.shape
        center_ref = np.array([w/2, h/2, 1]).reshape(3, 1)

        if M.shape[0] == 3:  # 如果是单应性矩阵
            center_a_homog = M.dot(center_ref)
            center_find = (center_a_homog / center_a_homog[2])[:2].flatten()
        else:  # 如果是仿射变换
            center_find = M.dot(center_ref[:2]).flatten()

        size_factor = np.sqrt(np.abs(np.linalg.det(M[:2, :2])))
        template_width, template_height = state_b[2], state_b[3]

        new_width = template_width * size_factor
        new_height = template_height * size_factor


        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        half_side_a = 0.5 * self.params.search_size / resize_factor
        # 计算在原图中的实际位置
        cx_real = center_find[0] / self.params.search_size * (2 * half_side_a) + (cx_prev - half_side_a)
        cy_real = center_find[1] / self.params.search_size * (2 * half_side_a) + (cy_prev - half_side_a)
        # 更新状态
        new_state = [cx_real - 0.5 * new_width, cy_real - 0.5 * new_height, new_width, new_height]
        prev_state = self.state
        self.state = clip_box(new_state, H, W, margin=10)
        ####一段可视化代码

        # 确保 search_temp 是 uint8 类型

        # 确保 image_a 是 uint8 类型
        if image_a.dtype != np.uint8:
            image_a = (image_a * 255).astype(np.uint8)

        x1, y1, w, h = [int(coord) for coord in self.state]  # 将边界框坐标转换为整数
        image_with_bbox = image_a.copy()  # 创建图像副本用于绘制
        cv2.rectangle(image_with_bbox, (x1, y1), (x1 + w, y1 + h), color=(0, 255, 0), thickness=2)  # 绘制绿色边界框
        x1, y1, w, h = [int(coord) for coord in prev_state]
        cv2.rectangle(image_with_bbox, (x1, y1), (x1 + w, y1 + h), color=(255, 255, 0), thickness=2)  # 绘制黄色边界框

        matchesMask = mask.ravel().tolist() if mask is not None else None

        # 使用 matchesMask 参数来只绘制 RANSAC 认为有效的匹配点
        match_img = cv2.drawMatches(
            x_patch_arr_b, kp_ref, x_patch_arr, kp_find, 
            good_matches, None, 
            matchesMask=matchesMask,  # 添加这个参数来筛选匹配点
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        debug_dir = "/data/yakun/MultiTrack/output/debug/redetect"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{drone_id}_frame{self.frame_id}_testresult_{tmp_factor}.jpg"), 
                   cv2.cvtColor(image_with_bbox, cv2.COLOR_RGB2BGR))
        
        debug_dir = "/data/yakun/MultiTrack/output/debug/matching"
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{drone_id}_frame{self.frame_id}_matching_{tmp_factor}.jpg"), 
                   cv2.cvtColor(match_img, cv2.COLOR_RGB2BGR))
        
        ###


        out_t, max_score_t, response_APCE_t = self.track(image_a, image_b, image_b, drone_id, info_a, info_b, info_b)
        # print(max_score_t)
    
        return out_t, max_score_t, response_APCE_t


    # 计算平均峰值相关能量
    def calAPCE(self, response):
        faltted_response = response.flatten(1)
        max_score, idx_max = torch.max(faltted_response, dim=1, keepdim=True)
        min_score, idx_min = torch.min(faltted_response, dim=1, keepdim=True)

        _, response_len = faltted_response.shape

        tmp_sum = 0
        for i, score in enumerate(faltted_response.squeeze()):         # squeeze是把维度为1的去掉
            tmp_sum += (score - min_score) ** 2

        bottom = tmp_sum / (i+1)

        APEC = ((max_score - min_score) ** 2) / bottom

        # print("APCE: ", APEC)

        return APEC
    
    def calAPCE_optimized(self, response):
        faltted_response = response.flatten(1)
        max_score, _ = torch.max(faltted_response, dim=1, keepdim=True)
        min_score, _ = torch.min(faltted_response, dim=1, keepdim=True)
        
        # 向量化操作，完全在GPU上执行
        tmp_sum = torch.sum((faltted_response - min_score) ** 2, dim=1)
        
        bottom = tmp_sum / faltted_response.shape[1]
        APEC = ((max_score - min_score) ** 2) / bottom
        
        return APEC

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return MultiAbaTrack

def process_template(image, info, template_factor, template_size, preprocessor, cfg, transform_bbox_to_crop):
    """
    辅助函数：处理模板的裁剪、预处理和生成掩码。
    """
    z_patch_arr, resize_factor, z_amask_arr = sample_target(
        image, info['init_bbox'], template_factor, output_sz=template_size
    )
    template = preprocessor.process(z_patch_arr, z_amask_arr)
    with torch.no_grad():
        z_dict = template

    
    box_mask_z = None

    if cfg.MODEL.BACKBONE.CE_LOC:
        template_bbox = transform_bbox_to_crop(info['init_bbox'], resize_factor, template.tensors.device).squeeze(1)
        box_mask_z = generate_mask_cond(cfg, 1, template.tensors.device, template_bbox)

    return z_dict, box_mask_z, z_patch_arr, resize_factor