import numpy as np
from test.sequence import Sequence, SequenceList
from test.data.datasets import BaseDataset
from utils.load_text import load_text
import os


class UAVDTDatasetThree(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self):
        super().__init__()

        self.base_path = os.path.join(self.env_settings.uavdt_path)
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/{}_gt.txt'.format(self.base_path, 'anno',sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/{}'.format(self.base_path, 'sequences',sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[3:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        return Sequence(sequence_name, frames_list, 'uavdt', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        seq_path = os.path.join(self.base_path,'sequences')
        seqs = os.listdir(seq_path)
        #####
        seqs = seqs * 3
        return seqs
