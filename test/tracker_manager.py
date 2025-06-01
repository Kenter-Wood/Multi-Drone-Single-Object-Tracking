import importlib
import os
from collections import OrderedDict
from .settings import env_settings
import time
import cv2 as cv

from utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import copy
import torch


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, script: str, config: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = script
        self.config_name = config
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.config_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.config_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                            'tracker_implement', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('test.tracker_implement.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            print("No Tracker Class Found!")
            self.tracker_class = None
    # tracker从lib/test/tracker/mytrack.py中构造，
    # 而这个文件会build mytrack并载入参数，所以tracker是MyTrack类的
    def create_tracker(self, params):
        tracker = self.tracker_class(params)
        return tracker
    # 测试阶段先将数据分成序列，然后通过这个函数，先产生tracker，然后执行序列操作。
    # 对于dataset当中的每个序列，都会重新建造一个tracker，所以所有tracker的寿命只有一个序列
    # 那么如果把初始化的temp_token作为模型init的一部分，似乎也可以
    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_test_config()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()
        # lib/test/tracker/mytrack.py, 这个tracker的定义
        tracker = self.create_tracker(params)
        # 给定tracker，序列，对单个序列时维护一个temp_token
        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])
        # seq.frames[0]这里是一个指向特定图片的地址
        # print(seq.frames[0])
        start_time = time.time()
        # 在这里定义一个时序的token
        # 目前假设对每个序列，有一个固定的初始时序token，这里的初始化就从Mytrack里面引入
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)
        # 这个循环包含了一个序列的每一帧
        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):

            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            #########这里调用lib/test/tracker当中的逻辑进行track，也就调用网络前传，执行一次
            ## 在这里的for循环维护一个序列的temp_token变量
            out = tracker.track(image, info)
            # print(temp_token[0, 0, 0])
            #########
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_test_config()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.config_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_test_config(self):
        """Get test_config."""
        test_config_module = importlib.import_module('configs.{}.config_test'.format(self.name))
        params = test_config_module.parameters(self.config_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


    def run_sequence_three(self, seq1, seq2, seq3, debug=None):
        """
        Run tracker on three sequences.
        """
        params = self.get_test_config()
        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info1 = seq1.init_info()
        init_info2 = seq2.init_info()
        init_info3 = seq3.init_info()
        # 按照相同的参数构建三个tracker
        tracker = self.create_tracker(params)
        tracker2 = self.create_tracker(params)
        tracker3 = self.create_tracker(params)

        # 需要一个函数融合三个序列的跟踪信息，并输出三个结果
        output1, output2, output3 = self._track_sequence_three(tracker, tracker2, tracker3, seq1, seq2, seq3,
                                                               init_info1, init_info2, init_info3)
        print('yesyes')
        return output1, output2, output3
    

    def _track_sequence_three(self, tracker, tracker2, tracker3, seq1, seq2, seq3, init_info1, init_info2, init_info3):
        """
        Input: 3 trackers, 3 sequences, 3 init_info
        Output: 3 outputs after fusion
        """

        seqs = [seq1, seq2, seq3]
        trackers = [tracker, tracker2, tracker3]

        # Define outputs
        output1 = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}
        if tracker.params.save_all_boxes:
            output1['all_boxes'] = []
            output1['all_scores'] = []

        output2 = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}
        if tracker2.params.save_all_boxes:
            output2['all_boxes'] = []
            output2['all_scores'] = []

        output3 = {'target_bbox': [], 'time': [], 'max_score': [], 'APCE':[]}
        if tracker3.params.save_all_boxes:
            output3['all_boxes'] = []
            output3['all_scores'] = []

        def _store_outputs(output,tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Read the initial frame of each sequence
        images = [self._read_image(seq.frames[0]) for seq in seqs]
        
        start_time = time.time()
        # Initialize the trackers
        # 存疑：在initialize和track的时候，传入三个原始图片是否合理
        out1 = tracker.initialize(images[0], images[1], images[2], init_info1, init_info2, init_info3)
        out2 = tracker2.initialize(images[1], images[0], images[2], init_info2, init_info1, init_info3)
        out3 = tracker3.initialize(images[2], images[0], images[1], init_info3, init_info1, init_info2)
        
        out1 = {} if out1 is None else out1
        out2 = {} if out2 is None else out2
        out3 = {} if out3 is None else out3

        prev_output1 = OrderedDict(out1)
        # print('init_info1:', type(init_info1))
        init_default1 = {'target_bbox': init_info1.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
        prev_output2 = OrderedDict(out2)
        init_default2 = {'target_bbox': init_info2.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
        prev_output3 = OrderedDict(out3)
        init_default3 = {'target_bbox': init_info3.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
        
        # Store the initial outputs of the trackers
        if tracker.params.save_all_boxes:
            init_default1['all_boxes'] = out1['all_boxes']
            init_default1['all_scores'] = out1['all_scores']
        _store_outputs(output1, out1, init_default1)
        if tracker2.params.save_all_boxes:
            init_default2['all_boxes'] = out2['all_boxes']
            init_default2['all_scores'] = out2['all_scores']
        _store_outputs(output2, out2, init_default2)    
        if tracker3.params.save_all_boxes:
            init_default3['all_boxes'] = out3['all_boxes']
            init_default3['all_scores'] = out3['all_scores']
        _store_outputs(output3, out3, init_default3)

        # Track the sequences
        # print('video length:', type(seq1.frames), len(seq1.frames))
        # print('frame_info:', seq1.frame_info(1), type(seq1.frame_info))

        for frame_num, frame_paths in enumerate(zip(seq1.frames[1:], seq2.frames[1:], seq3.frames[1:]), start=1):
            # print(f"tracking frame_num: {frame_num}")
            images = [self._read_image(frame_path) for frame_path in frame_paths]
            infos = [seq.frame_info(frame_num) for seq in seqs]
            start_time = time.time()
            # print(images[0].shape, infos[0])

            infos[0]['previous_output'] = prev_output1
            infos[1]['previous_output'] = prev_output2
            infos[2]['previous_output'] = prev_output3

            out1, max_score1, responseAPCE1 = tracker.track(1, images[0], images[1], images[2], infos[0], infos[1], infos[2])
            prev_output1 = OrderedDict(out1)
            # _store_outputs(output1, out1, {'time': time.time() - start_time})

            
            out2, max_score2, responseAPCE2 = tracker2.track(2, images[1], images[0], images[2], infos[1], infos[0], infos[2])
            prev_output2 = OrderedDict(out2)
            # _store_outputs(output2, out2, {'time': time.time() - start_time})

            
            out3, max_score3, responseAPCE3 = tracker3.track(3, images[2], images[0], images[1], infos[2], infos[0], infos[1])
            prev_output3 = OrderedDict(out3)
            # 目前的outs记录单独track的输出。后面会基于其表现进行重检测
            outs = [out1, out2, out3]
            # _store_outputs(output3, out3, {'time': time.time() - start_time})
            # print('Can you get here!')
            #These indicators evaluates the tracking status of each tracker
            states = [copy.deepcopy(tracker.state) for tracker in trackers]
            max_scores = [max_score1, max_score2, max_score3]
            apces = [responseAPCE1, responseAPCE2, responseAPCE3]
            # Cross-Drone Target Redetection
            # redet_factor_list = [[5,12], [4,9], [3,5]]
            redet_factor_list = [[9,16], [7,12], [5,8]]
            # redet_factor_list = [[2,4]]
            APEC_threshold = 100
            score_threshold = 0.2
            score_upper = 0.3
            
            # select the tracker with the best tracking status
            max_index = max_scores.index(max(max_scores))
            tmp_max_score = copy.deepcopy(max_scores[max_index])

            # print("max_index:", max_index, "max_score:", tmp_max_score, "from_list:", max_scores)
            tmp_APEC = copy.deepcopy(apces[max_index])
            tmp_image = copy.deepcopy(images[max_index])
            tmp_state = copy.deepcopy(states[max_index])
            tmp_info = copy.deepcopy(infos[max_index])

            for tracker_id in range(3):
                if tracker_id == max_index:
                    continue
                if ((max_scores[tracker_id] < score_threshold and apces[tracker_id] < APEC_threshold) and 
                    (tmp_max_score > score_upper and tmp_APEC > apces[tracker_id])):
                    # 这个for循环内会尝试不同的redetection factor
                #     redet_results = []
                #     # 在遍历不同大小的重检测范围时，每个范围会得到输出结果、最大得分、APCE
                #     for i, factor in enumerate(redet_factor_list):
                #         trackers[tracker_id].state = copy.deepcopy(states[tracker_id])
                #         # out_temp, maxscore_temp, responseAPCE_temp = trackers[tracker_id].three_search_redetect(images[tracker_id], tmp_image,
                #         #                                                                                         tracker_id, copy.deepcopy(tmp_state), factor[0], factor[1],
                #         #                                                                                         infos[tracker_id], tmp_info)
                #         ######2025.4.10加入一种新的重检测方案
                #         out_temp, maxscore_temp, responseAPCE_temp = trackers[tracker_id].feature_matching_redetect(images[tracker_id], tmp_image, tracker_id, copy.deepcopy(tmp_state), factor[0], factor[1], infos[tracker_id], tmp_info)

                #         ######
                #         tmp_dict = {"out":out_temp, "maxscore":maxscore_temp, "responseAPCE":responseAPCE_temp, "droneid":tracker_id, "factor":factor}
                #         redet_results.append(tmp_dict)
                #     # 遍历三个范围下得到的结果，ms代表这三者中最好的结果， label代表最好的结果的index

                #     ms = 0
                #     label = 0
                #     for i, redet_result in enumerate(redet_results):
                #         if redet_result["maxscore"] > ms:
                #             ms = redet_result["maxscore"]
                #             label = i                    
                # ####
                #     # trackers[tracker_id].state = copy.deepcopy(states[tracker_id])
                #     # out_temp, maxscore_temp, responseAPCE_temp = trackers[tracker_id].one_time_redetect(images[tracker_id], tmp_image,
                #     #                                                                                             tracker_id, copy.deepcopy(tmp_state), redet_factor_list,
                #     #                                                                                             infos[tracker_id], tmp_info)   
                #     # tmp_dict = {"out":out_temp, "maxscore":maxscore_temp, "responseAPCE":responseAPCE_temp, "droneid":tracker_id}
                #     # redet_results.append(tmp_dict)             
                #     # label = 0

                # #####
                    
                #     # 如果重检测得到的分数能超越自身跟踪的分数，那么就采用重检测的结果
                #     # trackers的状态记录了当前帧的检测结果，为下一帧提供参照，而outs作为输出
                #     if redet_results[label]['maxscore'] > max_scores[tracker_id]:
                #         # print("used_factor:", redet_factor_list[label])
                #         outs[tracker_id] = redet_results[label]["out"]
                #         max_scores[tracker_id] = redet_results[label]["maxscore"]
                #         # print("result type",type(redet_results[label]["maxscore"]))
                #         apces[tracker_id] = redet_results[label]["responseAPCE"]
                        
                #         trackers[tracker_id].state = outs[tracker_id]["target_bbox"]
                #     else:
                #         # print("remain original score")
                    trackers[tracker_id].state = copy.deepcopy(states[tracker_id])

            # print('Can you finish here to the saving!')

            prev_output1 = OrderedDict(outs[0])
            _store_outputs(output1, outs[0], {'time': time.time() - start_time,  'max_score': max_scores[0].item(),  'APCE':apces[0].item()})
            prev_output2 = OrderedDict(outs[1])
            _store_outputs(output2, outs[1], {'time': time.time() - start_time,  'max_score': max_scores[1].item(),  'APCE':apces[1].item()})
            prev_output3 = OrderedDict(outs[2])
            _store_outputs(output3, outs[2], {'time': time.time() - start_time,  'max_score': max_scores[2].item(),  'APCE':apces[2].item()})
            # print(f"Frame {frame_num}: output1['target_bbox'] length = {len(output1['target_bbox'])}, current_append = {outs[0]}")
            # print(f"Frame {frame_num}: output2['target_bbox'] length = {len(output2['target_bbox'])}, current_append = {outs[1]}")
            # print(f"Frame {frame_num}: output3['target_bbox'] length = {len(output3['target_bbox'])}, current_append = {outs[2]}")



        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output1 and len(output1[key]) <= 1:
                output1.pop(key)
            if key in output2 and len(output2[key]) <= 1:
                output2.pop(key)
            if key in output3 and len(output3[key]) <= 1:
                output3.pop(key)
        
        return output1, output2, output3