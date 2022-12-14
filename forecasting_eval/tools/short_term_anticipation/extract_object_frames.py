from argparse import ArgumentParser
import json
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import multiprocessing
from tqdm import tqdm
import os
from ego4d.datasets.short_term_anticipation import PyAVVideoReader

parser = ArgumentParser()
parser.add_argument('path_to_annotations', type=Path)
parser.add_argument('path_to_videos', type=Path)
parser.add_argument('path_to_output', type=Path)
parser.add_argument('--fname_format', type=str, default="{video_uid:s}_{frame_number:07d}.jpg")
parser.add_argument('--jobs', default=1, type=int)
parser.add_argument('--clips', action='store_true')

args = parser.parse_args()

args.path_to_output.mkdir(exist_ok=True, parents=True)

images = []

train = json.load(open(args.path_to_annotations / 'fho_sta_train.json'))
val = json.load(open(args.path_to_annotations / 'fho_sta_val.json'))
test = json.load(open(args.path_to_annotations / 'fho_sta_test_unannotated.json'))

names = []
video_ids = []
frame_numbers = []

for ann in [train, val, test]:
    for x in ann['annotations']:
        fname = args.fname_format.format(video_uid=x["video_uid"], frame_number=x["frame"])
        names.append(fname)
        if args.clips:
            video_ids.append(x['clip_uid'])
            frame_numbers.append(x['clip_frame'])
        else:
            video_ids.append(x['video_uid'])
            frame_numbers.append(x['frame'])

#images = sorted(images)

print(f"Found {len(names)} frames to extract")

missing = []
for idx, im in enumerate(names):
    if not os.path.isfile(args.path_to_output / im):
        missing.append(idx)

print(f"Skipping {len(names)-len(missing)} frames already extracted")

names = [names[i] for i in missing]
video_ids = [video_ids[i] for i in missing]
frame_numbers = [frame_numbers[i] for i in missing]

if len(names)==0:
    exit(0)

df = pd.DataFrame({'video': video_ids, 'frame': np.array(frame_numbers).astype(int), 'name': names})

groups = df.groupby('video')

all_video_names = []
all_frames = []
names = []

for g in groups:
    vid = g[0]
    frames = g[1]['frame'].values
    names.extend(g[1]['name'])

    all_video_names.extend([f"{vid}.mp4"]*len(frames))
    all_frames.extend(frames)

df = pd.DataFrame({'video': all_video_names, 'frame': all_frames, 'name': names})

def process_video(argss):
    fname, frames, names = argss
    vr = PyAVVideoReader(fname)

    video_frames = vr[frames]

    for vf, nam in zip(video_frames, names):
        imname = str(args.path_to_output / f"{nam}")
        cv2.imwrite(imname, vf)

params = []
for g in df.groupby('video'):
    vid = g[0]
    fname = str(args.path_to_videos / vid)
    frames = g[1]['frame'].values
    names = g[1]['name'].values

    params.append((fname, frames,names))

pool = multiprocessing.Pool(processes=args.jobs)

for _ in tqdm(pool.imap_unordered(process_video, params), total=len(params)):
    pass


