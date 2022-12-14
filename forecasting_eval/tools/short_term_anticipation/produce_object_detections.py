"""This script can be used to extract object detections from the annotated frames"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle
import json

from detectron2.config import get_cfg
from os.path import join
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np

parser = ArgumentParser()
parser.add_argument('path_to_checkpoint', type=Path)
parser.add_argument('path_to_sta_annotations', type=Path)
parser.add_argument('path_to_images', type=Path)
parser.add_argument('path_to_output_json', type=Path)

args = parser.parse_args()

train = json.load(open(args.path_to_sta_annotations / 'fho_sta_train.json'))
val = json.load(open(args.path_to_sta_annotations / 'fho_sta_val.json'))
test = json.load(open(args.path_to_sta_annotations / 'fho_sta_test_unannotated.json'))

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = str(args.path_to_checkpoint)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train['noun_categories'])

predictor = DefaultPredictor(cfg)

detections = {}

for anns in [train, val, test]:
    for ann in tqdm(anns['annotations']):
        uid = ann['uid']
        name = f"{uid}.jpg"
        img_path = args.path_to_images / name
        img = cv2.imread(str(img_path))
        outputs = predictor(img)['instances'].to('cpu')

        dets = []
        for box, score, noun in zip(outputs.pred_boxes.tensor, outputs.scores, outputs.pred_classes):
            box = box.tolist()
            box = [float(x) for x in box]
            score = score.item()
            noun = noun.item()
            dets.append(
                {
                    'box': box,
                    'score': score,
                    'noun_category_id': noun
                }
            )
        detections[uid] = dets
        
json.dump(detections, open(args.path_to_output_json, 'w'))
