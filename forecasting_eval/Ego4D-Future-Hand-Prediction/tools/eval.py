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

'''
This file is for demonstration purpose only to help you understand the evaluation method. 
You are not able to successfully run this code because we're not providing test.json file. 
'''
def main(output_file, num_clips=30):
  pred_dict={}
  gt_dict={}

  preds_file = 'submission.json'
  f = open(preds_file)
  preds_json = json.load(f)
  f.close()
  for k, v in preds_json.items():
    pred_dict[k] = v

  labels_file = 'test.json'     # 'test.json' is not provided, just for demonstration
  f = open(labels_file)
  labels_json = json.load(f)
  f.close()
  for k, v in labels_json.items():
    gt_dict[k] = v

  left_list= []
  right_list= [] 
  left_final_list=[]
  right_final_list=[]

  for k,v in pred_dict.items():
    pred = [i/num_clips for i in v]
    label = gt_dict[k]

    for k in range(5):
      l_x_pred = pred[k*4]
      l_y_pred = pred[k*4+1]
      r_x_pred = pred[k*4+2]
      r_y_pred = pred[k*4+3]

      l_x_gt = label[k*4]
      l_y_gt = label[k*4+1]
      r_x_gt = label[k*4+2]
      r_y_gt = label[k*4+3]

      if r_x_gt!=0 or r_y_gt!=0:
        dist= np.sqrt((r_y_gt-r_y_pred)**2+(r_x_gt-r_x_pred)**2)
        right_list.append(dist)
      if l_x_gt!=0 or l_y_gt!=0:
        dist= np.sqrt((l_y_gt-l_y_pred)**2+(l_x_gt-l_x_pred)**2)
        left_list.append(dist)

    if r_x_gt!=0 or r_y_gt!=0:
      dist= np.sqrt((r_y_gt-r_y_pred)**2+(r_x_gt-r_x_pred)**2)
      right_final_list.append(dist)
    if l_x_gt!=0 or l_y_gt!=0:
      dist= np.sqrt((l_y_gt-l_y_pred)**2+(l_x_gt-l_x_pred)**2)
      left_final_list.append(dist)  
  

  print('***left hand mean disp error {:.3f}, right hand mean disp error {:.3f}'
        .format(sum(left_list)/len(left_list), sum(right_list)/len(right_list)))
  print('***left hand contact disp error {:.3f}, right hand contact disp error {:.3f}'
        .format(sum(left_final_list)/len(left_final_list), sum(right_final_list)/len(right_final_list)))



if __name__ == "__main__":
    description = 'Evaluation script for egocentric hand movements prediction.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('output_file', type=str,
                   help='output pickle file for predicted future hand positions')
    p.add_argument('num_clips', type=int, help='number of clips for spatial and temporal resampling during testing',default=30)
    main(**vars(p.parse_args()))
