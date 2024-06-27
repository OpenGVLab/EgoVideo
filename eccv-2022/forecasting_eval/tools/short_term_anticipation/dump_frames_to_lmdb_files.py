from argparse import ArgumentParser
from pathlib import Path
import sys

from typing import DefaultDict
import numpy as np
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import itertools
import json
from ego4d.datasets.short_term_anticipation import PyAVVideoReader, Ego4DHLMDB
from collections import defaultdict

parser = ArgumentParser()

parser.add_argument('path_to_annotations', type=Path)
parser.add_argument('path_to_videos', type=Path)
parser.add_argument('path_to_output_lmdbs', type=Path)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--context_frames', type=int, default=32)
parser.add_argument('--fname_format', type=str, default="{video_id:s}_{frame_number:07d}")
parser.add_argument('--frame_height', type=int, default=320)
parser.add_argument('--video_uid', type=str, default=None)

args = parser.parse_args()


class PyAVSTADataset(Dataset):
    def __init__(self, annotations, path_to_videos, existing_keys, fps=30, max_chunk_size=32, retry=10):
        print("Sampling from {} annotations with a temporal context of {} seconds".format(len(annotations),
                                                                                          args.context_frames / fps))
        existing_frames = defaultdict(list)
        for k in existing_keys:
            video_id, frame_number = k.decode().split("_")
            existing_frames[video_id].append(int(frame_number))

        self.path_to_videos = path_to_videos
        self.retry = retry
        if args.video_uid is not None:
            annotations = [a for a in annotations if a["video_uid"] in args.video_uid]

        frames_per_video = defaultdict(list)

        for ann in annotations:
            video_id = ann["video_uid"]
            last_frame = ann["frame"]
            first_frame = np.max([0, last_frame - args.context_frames + 1])
            frame_numbers = np.arange(first_frame, last_frame + 1)
            frames_per_video[video_id].extend(frame_numbers)

        self.chunks = []

        total_frames = 0

        for k, v in frames_per_video.items():
            frames = np.setdiff1d(np.sort(np.unique(v)), existing_frames[k])

            if (len(frames) > 0):
                ## break at non consecutive frames
                frame_chunks = np.split(frames, np.where(np.diff(frames) != 1)[0] + 1)

                ## add each frame chunk to the list of chunks
                for chunk in frame_chunks:
                    ## if the cunk is too large, break it into smaller chunks
                    if len(chunk) <= max_chunk_size:
                        self.chunks.append((k, chunk))
                        total_frames += len(chunk)  # count the total number of frames
                    else:
                        for chunk in np.array_split(chunk, np.ceil(len(chunk) / max_chunk_size)):
                            self.chunks.append((k, chunk))
                            total_frames += len(chunk)  # count the total number of frames

        total_frames += len(existing_keys)

        avg_bytes = 60000
        total_bytes = total_frames * avg_bytes
        total_gigabytes = total_bytes / 1024 / 1024 / 1024

        print("Sampled {} chunks / {} frames in total".format(len(self.chunks), total_frames))
        print("Skipping {} existing keys".format(len(existing_keys)))
        print("Estimated total size: {:0.2f} GB".format(total_gigabytes))

    def __len__(self):
        return (len(self.chunks))

    def __getitem__(self, index):
        video_id, frame_numbers = self.chunks[index]

        frames = {}

        for i in range(self.retry):
            frame_numbers = np.setdiff1d(frame_numbers, list(frames.keys()))
            vr = PyAVVideoReader(str(self.path_to_videos / (video_id + '.mp4')), height=args.frame_height)
            imgs = vr[frame_numbers]

            added = 0
            for f, img in zip(frame_numbers, imgs):
                if img is not None:
                    frames[f] = img
                    added += 1

            if added == len(frame_numbers) or i == (self.retry - 1):
                keys = [args.fname_format.format(video_id=video_id, frame_number=f) for f in frames.keys()]
                ims = list(frames.values())

                missing_frames = np.setdiff1d(frame_numbers, list(frames.keys()))

                if len(missing_frames) > 0:
                    print(f"WARNING: could not read the following frames from {video_id}:",
                          ", ".join([str(x) for x in missing_frames]))

                return ims, keys


def collate(batch):
    frames = [sample[0] for sample in batch]
    keys = [sample[1] for sample in batch]
    frames = list(itertools.chain.from_iterable(frames))
    keys = list(itertools.chain.from_iterable(keys))

    return frames, keys


train = json.load(open(args.path_to_annotations / 'fho_sta_train.json'))
val = json.load(open(args.path_to_annotations / 'fho_sta_val.json'))
test = json.load(open(args.path_to_annotations / 'fho_sta_test_unannotated.json'))

## Merge all annotations
annotations = []
for j in [train, val, test]:
    annotations += j['annotations']

l = Ego4DHLMDB(args.path_to_output_lmdbs)

## Define the dataset and dataloader
dset = PyAVSTADataset(annotations, args.path_to_videos, existing_keys=l.get_existing_keys())
dloader = DataLoader(dset, batch_size=args.batch_size, collate_fn=collate, num_workers=8)

## Iterate over the dataloader
for (frames, keys) in tqdm(dloader):
    for parent in np.unique([k.split('_')[0] for k in keys]):
        idx = np.where([k.startswith(parent) for k in keys])[0]
        these_keys = [int(keys[i].split('_')[1]) for i in idx]
        these_frames = [frames[i] for i in idx]
        l.put_batch(parent, these_keys, these_frames)

exit(0)
