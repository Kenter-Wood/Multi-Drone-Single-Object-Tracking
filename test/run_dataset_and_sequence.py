import time
from datetime import timedelta
import torch
import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from .sequence import Sequence
from .tracker_manager import Tracker, trackerlist



def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    #=====+++++
    if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
        print("create tracking result dir:", os.path.join(tracker.results_dir, seq.dataset))
        os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    if seq.dataset in ['trackingnet', 'got10k']:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    '''2021.1.5 create new folder for these two datasets'''
    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset,seq.name)
    print(base_results_path)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # print(key, data[1])
        # If data is empty
        if not data:
            print("No data!")
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key == 'APCE':    # 保存APCE
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_APCE.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving APCE...")
                bbox_file = '{}_APCE.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key == 'max_score':    # 保存score
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_max_score.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_max_score.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)


# # 在run_sequence中，有将一个片段的输出保存到文件的逻辑
# # 这里我让每个片段进行一次评估，输出对应的结果
# def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
#     """Runs a tracker on a sequence."""
#     '''2021.1.2 Add multiple gpu support'''
#     try:
#         worker_name = multiprocessing.current_process().name
#         worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
#         gpu_id = worker_id % num_gpu
#         torch.cuda.set_device(gpu_id)
#     except:
#         pass

#     def _results_exist():
#         if seq.object_ids is None:
#             if seq.dataset in ['trackingnet', 'got10k']:
#                 base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
#                 bbox_file = '{}.txt'.format(base_results_path)
#             else:
#                 bbox_file = '{}/{}/{}.txt'.format(tracker.results_dir, seq.dataset,seq.name)
#             return os.path.isfile(bbox_file)
#         else:
#             bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
#             missing = [not os.path.isfile(f) for f in bbox_files]
#             return sum(missing) == 0

#     if _results_exist() and not debug:
#         print('FPS: {}'.format(-1))
#         return

#     print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

#     if debug:
#         output = tracker.run_sequence(seq, debug=debug)
#     else:
#         try:
#             output = tracker.run_sequence(seq, debug=debug)
#         except Exception as e:
#             print(e)
#             return
#     sys.stdout.flush()
#     if isinstance(output['time'][0], (dict, OrderedDict)):
#         exec_time = sum([sum(times.values()) for times in output['time']])
#         num_frames = len(output['time'])
#     else:
#         exec_time = sum(output['time'])
#         num_frames = len(output['time'])

#     print('FPS: {}'.format(num_frames / exec_time))

#     if not debug:
#         _save_tracker_output(seq, tracker, output)

#     # # 在保留结果后，对应的位置已经出现了当前seq的预测结果，
#     # # os.path.join(tracker.results_dir, seq.dataset)得到的是结果文件夹
#     # result_pth = os.path.join(tracker.results_dir, seq.dataset,seq.name)
#     # trackers = []
#     # trackers.extend(trackerlist(name='mytrack', parameter_name='vit_tiny_060', dataset_name=seq.dataset,
#     #                         run_ids=None, display_name='mytrack'))
#     # print_results(trackers, [seq.name], seq.dataset, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
#     ########


# def run_dataset(dataset, trackers, debug=True, threads=0, num_gpus=8):
#     """Runs a list of trackers on a dataset.
#     args:
#         dataset: List of Sequence instances, forming a dataset.
#         trackers: List of Tracker instances.
#         debug: Debug level.
#         threads: Number of threads to use (default 0).
#     """
#     multiprocessing.set_start_method('spawn', force=True)

#     print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))
#     dataset_start_time = time.time()

#     multiprocessing.set_start_method('spawn', force=True)

#     if threads == 0:
#         mode = 'sequential'
#     else:
#         mode = 'parallel'

#     if mode == 'sequential':
#         for seq in dataset:
#             for tracker_info in trackers:
#                 run_sequence(seq, tracker_info, debug=debug)
#     elif mode == 'parallel':
#         param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
#         with multiprocessing.Pool(processes=threads) as pool:
#             pool.starmap(run_sequence, param_list)
#     # print("Done, total time: {}".format(str(timedelta(seconds=(time.time() - dataset_start_time)))))
#     print(123)

def run_sequence_three(seq1: Sequence, seq2: Sequence, seq3: Sequence, tracker: Tracker, debug=False, num_gpu=4):
    '''
        当接收到三个并行序列的时候需要一起进行跟踪
    '''
    # 这一部分是为了多gpu支持，保持原状
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except Exception as e:
        print(e)
        pass    
    
    # 这个内部函数是为了判断是否已经存在结果文件，区分单目标和多目标
    def _results_exist():
        if seq1.object_ids is None:
            bbox_file = '{}/{}/{}.txt'.format(tracker.results_dir, seq1.dataset, seq1.name)
            
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq1.name, obj_id) for obj_id in seq1.object_ids]
            
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print(f'You have already tested on this sequence,{seq1}, {seq2}, {seq3}')
        return
    
    # 在上面处理后，才开始真正的跟踪
    print('Tracker: {} {} {} ,  Sequence: {} {} {}'.format(tracker.name, tracker.config_name, tracker.run_id, 
                                                           seq1.name, seq2.name, seq3.name))
    
    try:
        output1, output2, output3 = tracker.run_sequence_three(seq1, seq2, seq3, debug=debug)
    except Exception as e:
        print(e)
        return
    sys.stdout.flush()

    if isinstance(output1['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output1['time']])
        num_frames = len(output1['time'])
    else:
        exec_time = sum(output1['time'])
        num_frames = len(output1['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        # print(output1.keys(), output2.keys(), output3.keys())
        _save_tracker_output(seq1, tracker, output1)
        _save_tracker_output(seq2, tracker, output2)
        _save_tracker_output(seq3, tracker, output3)


def run_dataset_three(dataset, trackers, debug=False, threads=0, num_gpus=4):
    '''
        接受的dataset每个样本包含3个序列，需要一起跟踪的情况下
        单独定义了一个rundatset方法， 重点是划分序列。
        在dataset定义的时候，1，2，3的序列是按照顺序排列的，所以这里直接划分即可
    '''
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))
    dataset_start_time = time.time()

    multiprocessing.set_start_method('spawn', force=True)

    # 实际的长度是dataset的三分之一，因为每三个序列是一组
    len_data = len(dataset)//3
    dataset1 = dataset[:len_data]
    dataset2 = dataset[len_data:2*len_data]
    dataset3 = dataset[2*len_data:]
    # print('datasets', dataset1, dataset2, dataset3)
    ############################
    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq1, seq2, seq3 in zip(dataset1, dataset2, dataset3):
            for tracker_info in trackers:
                run_sequence_three(seq1, seq2, seq3, tracker_info, debug=debug)
    elif mode == 'parallel':
#         param_list = [
#     (seq1, seq2, seq3, tracker_info, debug, num_gpus) 
#     for seq1, seq2, seq3, tracker_info in product(dataset1, dataset2, dataset3, trackers)  # 错误根源
# ]
        param_list = [(seq_a, seq_b, seq_c, tracker_info, debug, num_gpus) 
        for (seq_a, seq_b, seq_c), tracker_info in product(zip(dataset1, dataset2, dataset3), trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence_three, param_list)
    print('Tracking Test accomplished')