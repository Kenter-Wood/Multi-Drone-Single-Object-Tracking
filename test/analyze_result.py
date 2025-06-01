import os
import csv
import torch
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
from test.data.datasets import get_dataset
from test.tracker_manager import trackerlist
from test.settings import env_settings
from utils.analysis_three import print_results, plot_results, get_plot_draw_styles


trackers = []
# dataset_name = 'dtb70three'
dataset_name = 'threemdottest'
# dataset_name = 'uavdtthree'
# dataset_name = 'visdrone2018'
# dataset_name = 'uav123'
# dataset_name = 'uav123_10fps'
method = 'MultiTempTrack'
file_path = "/data/yakun/MultiTrack/output/results.csv"


"""mytrack"""
trackers.extend(trackerlist(name='vit_base', parameter_name='vit_base_new_mdot_25_51', dataset_name=dataset_name,
                            run_ids=None, display_name='multitemptrack'))

dataset = get_dataset(dataset_name)
score = print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# **********************************
score['dataset'] = dataset_name
score['method'] = method

priority_keys = ["dataset", "method"]

# 构建结果列表
result = []
for key in priority_keys:
    if key in score:
        result.append(score[key])

# 添加剩余键的值（按原顺序）
for key in score:
    if key not in priority_keys:
        result.append(round(score[key].item(), 2))

print(result)

# 追加到 CSV 文件
with open(file_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(result)