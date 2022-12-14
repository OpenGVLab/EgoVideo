"""This script creates a dataset to train an object detector in COCO format, so that it can be used with Detectron2"""

from argparse import ArgumentParser
from pathlib import Path
import json
from datetime import date
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('path_to_annotations', type=Path)
parser.add_argument('path_to_output_json', type=Path)

args = parser.parse_args()

labels = json.load(open(args.path_to_annotations))
noun_categories = labels['noun_categories']

# add +1 to the index to follow the COCO convention
idx_to_obj = {x['id']: x['name'] for x in noun_categories}
obj_to_idx = {v:k for k,v in idx_to_obj.items()}

progressive_image_id = 0
progressive_annotation_id = 0
filenames_to_ids = {}

coco_annotations = {}
info = {
    "description" : "Ego4D Short-Term Object Interaction Anticipation - Next Active Object Dataset",
    "version": "1.0",
    "date_created": str(date.today())
}

categories = [{
    "supercategory": "object",
    "id": k,
    "name": v
} for k,v in idx_to_obj.items()]

images = []
annotations = []

for example in tqdm(labels['annotations']):
    uid = example['uid']
    fname = f"{uid}.jpg"
    video_info = labels['info']['video_metadata'][example['video_id']]

    if fname not in filenames_to_ids:
        filenames_to_ids[fname] = progressive_image_id
        images.append({
            "file_name": fname,
            "width": video_info['frame_width'],
            "height": video_info['frame_height'],
            "id": progressive_image_id
        })
        progressive_image_id += 1

    for obj in example['objects']:
        box = obj['box']
        category_id = obj['noun_category_id']

        width = box[2]-box[0]
        height = box[3]-box[1]
        
        annotations.append({
            "segmentation": [],
            "area": width * height,
            "iscrowd": 0,
            "image_id": filenames_to_ids[fname],
            "bbox": [box[0], box[1], width, height],
            "category_id": category_id,
            "id": progressive_annotation_id
        })
        progressive_annotation_id += 1

coco_annotations = {
    "info": info,
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(args.path_to_output_json, 'w') as f:
    json.dump(coco_annotations, f)