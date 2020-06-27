# -----------------------------------------------------------------------------
# THIS FILE IS THE .py VERSION OF THE GIVEN JUPYTER NOTEBOOK BY THE CV3DST TEAM
# -----------------------------------------------------------------------------

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.autonotebook import tqdm

import torch
from torch.utils.data import DataLoader
from tracker.data_track import MOT16Sequences
from tracker.data_obj_detect import MOT16ObjDetect
from tracker.object_detector import FRCNN_FPN
from tracker.tracker import Tracker
from tracker.utils import (plot_sequence, evaluate_mot_accums, get_mot_accum,
                           evaluate_obj_detect, obj_detect_transforms)

import motmetrics as mm


root_dir = "/home/orcun/Desktop/TUM/SecondSemester/CV3-DST/cv3dst_exercise"

sys.path.append(os.path.join(root_dir, 'src'))
mm.lap.default_solver = 'lap'
seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

data_dir = os.path.join(root_dir, 'data/MOT16')
output_dir = os.path.join(root_dir, 'output')


# Configure Object Detector
obj_detect_model_file = os.path.join(root_dir, 'models/faster_rcnn_fpn.model')
obj_detect_nms_thresh = 0.3
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
obj_detect = FRCNN_FPN(num_classes=2, nms_thresh=obj_detect_nms_thresh)
obj_detect_state_dict = torch.load(obj_detect_model_file,
                                   map_location=lambda storage, loc: storage)
obj_detect.load_state_dict(obj_detect_state_dict)
obj_detect.eval()
obj_detect.to(device)


# Dataset
seq_name = 'MOT16-test'
sequences = MOT16Sequences(seq_name, data_dir)


# Thresholds of the tracker
thresholds = {'score_det': 0.5, 'nms_det': 0.3, 'nms_reg': 0.6}

# Configure the tracker
tracker = Tracker(obj_detect, thresholds, device)


# Run tracker
time_total = 0
mot_accums = []
results_seq = {}
for seq in sequences:
    tracker.reset()
    now = time.time()

    print(f"Tracking: {seq}")

    data_loader = DataLoader(seq, batch_size=1, shuffle=False)

    for frame in tqdm(data_loader):
        tracker.step(frame)
    results = tracker.get_results()
    results_seq[str(seq)] = results

    if seq.no_gt:
        print(f"No GT evaluation data available.")
    else:
        mot_accums.append(get_mot_accum(results, seq))

    time_total += time.time() - now

    print(f"Tracks found: {len(results)}")
    print(f"Runtime for {seq}: {time.time() - now:.1f} s.")

    seq.write_results(results, os.path.join(output_dir))

print(f"Runtime for all sequences: {time_total:.1f} s.")
if mot_accums:
    evaluate_mot_accums(mot_accums, [str(s) for s in sequences if not s.no_gt], generate_overall=True)