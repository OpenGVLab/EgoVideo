from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import numpy as np
import argparse
import json

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(output_file, num_clips=30):
  with open(output_file, 'rb') as f:
    preds_list, labels_list, clip_idx, frm_idx = pickle.load(f)

  left_list= []
  right_list= [] 
  left_final_list=[]
  right_final_list=[]

  pred_dict={}
  for i in range(len(preds_list)):
    preds = preds_list[i].numpy()
    labels = labels_list[i].numpy()
    clips = clip_idx[i].cpu().numpy()
    frms = frm_idx[i].cpu().numpy()

    for j in range(len(preds)):
      pred = preds[j]
      label=labels[j]
      clip = clips[j]
      frm = frms[j]
      video_id = str(int(clip)) + '_' + str(int(frm))

      if video_id in pred_dict:
        pred_dict[video_id] += pred
      else:
        pred_dict[video_id] = pred

  for k,v in pred_dict.items():
    pred = v/num_clips

  dumped = json.dumps(pred_dict, cls=NumpyEncoder)
  with open('submission.json', 'a') as f:
      f.write(dumped + '\n') 

if __name__ == "__main__":
    description = 'Evaluation script for egocentric hand movements prediction.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('output_file', type=str,
                   help='output pickle file for predicted future hand positions')
    p.add_argument('num_clips', type=int, help='number of clips for spatial and temporal resampling during testing',default=30)
    main(**vars(p.parse_args()))
