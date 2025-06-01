import numpy as np
from test.sequence import Sequence, SequenceList
from test.data.datasets import BaseDataset
from utils.load_text import load_text

# 被调用的方法是在测试/训练过程中直接指定--dataset_name，就会调用对应的数据集类
class ThreeMDOTTestDataset(BaseDataset):
    """
        TCSVT 2020 MDOT Dataset, test_split
    """
    def __init__(self):
        # super()继承了父类也就是BaseDataset的方法，进行初始化，这样就保留了基础配置。
        super().__init__()
        # 同时加上了自己的逻辑
        self.base_path = self.env_settings.threemdot_test_path
        # 直接从下方获取这个数据集的序列列表
        self.sequence_list = self._get_sequence_list()
        # 获取没有标号的序列列表
        self.clean_list = self.clean_seq_list()
    # 把序列的后缀去掉-1/-2/-3
    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst
    # 从上述的所有sequence列表中，逐一进行序列构建
    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])
    # 功能是通过给出的序列名称，读取完整的图像和标注信息，返回Sequence?
    def _construct_sequence(self, sequence_name):
        # class代表属于哪个场景，如md3001
        class_name = sequence_name.split('-')[0]
        # 标注文件路径
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)
        # 读取标注保存为列表，表里是float64
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        # 读取遮挡标注。遮挡情况下视为没有标注框
        # occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)
        occlusion_label_path = '{}/{}/{}/occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')
        # 没有遮挡且没有超出视野，判断条件
        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)
        # 获取图像的路径，以及图像名称的列表
        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        # nlpPath = '{}/{}/{}/nlp.txt'.format(self.base_path, class_name, sequence_name)
        # nlpPath = None
        target_class = class_name
        return Sequence(sequence_name, frames_list, 'threemdottest', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)


    # 从数据集的基类里面继承，但必须实现其中的下面两个方法。所以这里给出了具体的序列编号(测试)，以及sequence长度。
    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):             # 必须按顺序放，因为代码里给排序了
        sequence_list = [
                        "md3001-1",
                        "md3002-1",
                        "md3003-1",
                        "md3004-1",
                        "md3006-1",
                        "md3007-1",
                        "md3009-1",
                        "md3010-1",
                        "md3011-1",
                        "md3012-1",
                        "md3014-1",
                        "md3015-1",
                        "md3021-1",
                        "md3022-1",
                        "md3023-1",
                        "md3024-1",
                        "md3025-1",
                        "md3028-1",
                        "md3029-1",
                        "md3033-1",
                        "md3037-1",
                        "md3039-1",
                        "md3041-1",
                        "md3042-1",
                        "md3043-1",
                        "md3045-1",
                        "md3046-1",
                        "md3047-1",
                        "md3049-1",
                        "md3052-1",
                        "md3053-1",
                        "md3056-1",
                        "md3057-1",
                        "md3061-1",
                        "md3063-1",
                        "md3001-2",
                        "md3002-2",
                        "md3003-2",
                        "md3004-2",
                        "md3006-2",
                        "md3007-2",
                        "md3009-2",
                        "md3010-2",
                        "md3011-2",
                        "md3012-2",
                        "md3014-2",
                        "md3015-2",
                        "md3021-2",
                        "md3022-2",
                        "md3023-2",
                        "md3024-2",
                        "md3025-2",
                        "md3028-2",
                        "md3029-2",
                        "md3033-2",
                        "md3037-2",
                        "md3039-2",
                        "md3041-2",
                        "md3042-2",
                        "md3043-2",
                        "md3045-2",
                        "md3046-2",
                        "md3047-2",
                        "md3049-2",
                        "md3052-2",
                        "md3053-2",
                        "md3056-2",
                        "md3057-2",
                        "md3061-2",
                        "md3063-2",
                        "md3001-3",
                        "md3002-3",
                        "md3003-3",
                        "md3004-3",
                        "md3006-3",
                        "md3007-3",
                        "md3009-3",
                        "md3010-3",
                        "md3011-3",
                        "md3012-3",
                        "md3014-3",
                        "md3015-3",
                        "md3021-3",
                        "md3022-3",
                        "md3023-3",
                        "md3024-3",
                        "md3025-3",
                        "md3028-3",
                        "md3029-3",
                        "md3033-3",
                        "md3037-3",
                        "md3039-3",
                        "md3041-3",
                        "md3042-3",
                        "md3043-3",
                        "md3045-3",
                        "md3046-3",
                        "md3047-3",
                        "md3049-3",
                        "md3052-3",
                        "md3053-3",
                        "md3056-3",
                        "md3057-3",
                        "md3061-3",
                        "md3063-3",
                        ]
        return sequence_list