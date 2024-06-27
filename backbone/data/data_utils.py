
import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import decord
import pandas as pd
import torch
from decord import cpu
import cv2
import io,os
import argparse

try:
    from petrel_client.client import Client
    client = Client()

    # Disable boto logger
    import logging
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('nose').setLevel(logging.WARNING)
except:
    client = None


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

# def get_vr(video_path):
#     video_bytes = client.get(video_path)
#     assert video_bytes is not None, "Get video failed from {}".format(video_path)
#     video_path = video_bytes
#     if isinstance(video_path, bytes):
#         video_path = io.BytesIO(video_bytes)
#     vreader = decord.VideoReader(video_path, ctx=cpu(0))
#     return vreader


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()

def get_videobytesIO(video_path):
    # video_bytes = client.get(video_path, enable_stream=True)
    video_bytes = client.get(video_path)
    assert video_bytes is not None, "Get video failed from {}".format(video_path)
    video_path = video_bytes
    if isinstance(video_path, bytes):
        video_path = io.BytesIO(video_bytes)
    return video_path


def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    if '/mnt/petrelfs' in videoname:
        if fast_rrc:
            video_reader = decord.VideoReader(
                videoname,
                num_threads=num_threads,
                width=rrc_params[0], height=rrc_params[0],
                use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
            )
        elif fast_rcc:
            video_reader = decord.VideoReader(
                videoname,
                num_threads=num_threads,
                width=rcc_params[0], height=rcc_params[0],
                use_rcc=True,
            )
        else:
            video_reader = decord.VideoReader(videoname, num_threads=num_threads)
        #print(video_reader)
        return video_reader
    else:
        video_reader = None
        video_bytes = client.get(videoname)
        assert video_bytes is not None, "Get video failed from {}".format(videoname)
        videoname = video_bytes
        if isinstance(videoname, bytes):
            videoname = io.BytesIO(video_bytes)
            
        if fast_rrc:
            video_reader = decord.VideoReader(
                videoname,
                num_threads=num_threads,
                width=rrc_params[0], height=rrc_params[0],
                use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
            )
        elif fast_rcc:
            video_reader = decord.VideoReader(
                videoname,
                num_threads=num_threads,
                width=rcc_params[0], height=rcc_params[0],
                use_rcc=True,
            )
        else:
            video_reader = decord.VideoReader(videoname, num_threads=num_threads)
        #print(video_reader)
        return video_reader

# def video_loader(root, vid, ext, second, end_second,
#                  chunk_len=300, fps=30, clip_length=32,
#                  threads=1,
#                  fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
#                  fast_rcc=False, rcc_params=(224, ),
#                  jitter=False):
#     # assert fps > 0, 'fps should be greater than 0'
    
#     if chunk_len == -1:
#         vr = get_video_reader(
#             osp.join(root, '{}.{}'.format(vid, ext)),
#             num_threads=threads,
#             fast_rrc=fast_rrc, rrc_params=rrc_params,
#             fast_rcc=fast_rcc, rcc_params=rcc_params,
#         )
#         fps = vr.get_avg_fps() if fps == -1 else fps
        
#         end_second = min(end_second, len(vr) / fps)

#         # calculate frame_ids
#         frame_offset = int(np.round(second * fps))
#         total_duration = max(int((end_second - second) * fps), clip_length)
#         frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

#         # load frames
#         assert max(frame_ids) < len(vr)
#         try:
#             frames = vr.get_batch(frame_ids).asnumpy()
#         except decord.DECORDError as error:
#             print(error)
#             frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    
#         return torch.from_numpy(frames.astype(np.float32))

#     else:
#         chunk_start = int(second) // chunk_len * chunk_len
#         chunk_end = int(end_second) // chunk_len * chunk_len
            
#         # calculate frame_ids
#         frame_ids = get_frame_ids(
#             int(np.round(second * fps)),
#             int(np.round(end_second * fps)),
#             num_segments=clip_length, jitter=jitter
#         )
#         # print(f'Frames: {frame_ids}')
        
#         all_frames = []
#         # allocate absolute frame-ids into the relative ones
#         for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
#             # print(f'Chunk: {chunk}, \t, Rel_frame_ids={rel_frame_ids}')
#             vr = get_video_reader(
#                 # osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
#                 osp.join(root, vid, '{}.{}'.format(str(chunk // chunk_len).zfill(4), ext)),
#                 num_threads=threads,
#                 fast_rrc=fast_rrc, rrc_params=rrc_params,
#                 fast_rcc=fast_rcc, rcc_params=rcc_params,
#             )

#             rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
#             # rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
#             rel_frame_ids = [min(len(vr) - 1, int(frame_id - chunk * fps)) for frame_id in rel_frame_ids]

#             try:
#                 frames = vr.get_batch(rel_frame_ids).asnumpy()
#             except decord.DECORDError as error:
#                 # print(error)
#                 frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
#             except IndexError:
#                 print(root, vid, str(chunk // chunk_len).zfill(4), second, end_second)
#                 print(len(vr), rel_frame_ids)
            
#             all_frames.append(frames)
#             if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
#                 break
            
#         res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
#         assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
#         return res
import time
import func_timeout
from func_timeout import func_set_timeout


def video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=-1, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    # assert fps > 0, 'fps should be greater than 0'
    
    if chunk_len == -1:
    
        if root == '':
            vr = get_video_reader(
                '{}.{}'.format(vid, ext),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )
        else:
            vr = get_video_reader(
                osp.join(root, '{}.{}'.format(vid, ext)),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )
        fps = vr.get_avg_fps() if fps == -1 else fps
        
        end_second = min(end_second, len(vr) / fps)
        if end_second == 0:
            end_second = len(vr) / fps

        # calculate frame_ids
        frame_offset = int(np.round(second * fps))
        total_duration = max(int((end_second - second) * fps), clip_length)

        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

        # load frames
        assert max(frame_ids) < len(vr)
        try:
        
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    
        return torch.from_numpy(frames.astype(np.float32))
    else:
        assert fps > 0, 'fps should be greater than 0'
        
        ## sanity check, for those who have start >= end ##
        end_second = max(end_second, second + 1)

        chunk_start = int(second) // chunk_len * chunk_len
        chunk_end = int(end_second) // chunk_len * chunk_len
        
        # print(f'Vid={vid}, begin_sec={second}, end_sec={end_second}, \t, st_frame={int(np.round(second * fps))}, ed_frame={int(np.round(end_second * fps))}')
        # calculate frame_ids
        frame_ids = get_frame_ids(
            int(np.round(second * fps)),
            int(np.round(end_second * fps)),
            num_segments=clip_length, jitter=jitter
        )
        # print(f'Frames: {frame_ids}')
        
        all_frames = []
        # allocate absolute frame-ids into the relative ones
        for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
            # print(f'Chunk: {chunk}, \t, Rel_frame_ids={rel_frame_ids}')
            vr = get_video_reader(
                # osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
                osp.join(root, vid, '{}.{}'.format(str(chunk // chunk_len).zfill(4), ext)),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )

            rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
            # rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
            rel_frame_ids = [min(len(vr) - 1, int(frame_id - chunk * fps)) for frame_id in rel_frame_ids]

            try:
                frames = vr.get_batch(rel_frame_ids).asnumpy()
            except decord.DECORDError as error:
                # print(error)
                frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
            except IndexError:
                print(root, vid, str(chunk // chunk_len).zfill(4), second, end_second)
                print(len(vr), rel_frame_ids)
            
            all_frames.append(frames)
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
            
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        return res



def video_loader_by_frames(root, vid, frame_ids):
    '''
    args:
        root: root directory of the video
        vid: the unique vid of the video, e.g. hello.mp4 
        frame_ids: the sampled frame indices 
    return:
        frames: torch tensor with shape: [T, H, W, C]
    '''
    vr = get_vr(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0) 

def video_loader_by_timestamp(root, vid, start_timestamp=0, end_timestamp=0, 
            clip_length=4, is_training=False, threads=1, 
            fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
            fast_rcc=False, rcc_params=(224, ),):
    '''
    args:
        root: root directory of the video
        vid: the unique vid of the video, e.g. hello.mp4 
        start_timestamp: the start second of the clip/video
        end_timestamp: the end second of the clip/video
        clip_length: the number of frames to be sampled
        is_training: whether it is training, jitter=True/False for train/test
    return:
        frames: torch tensor with shape: [T, H, W, C]
    '''
    vr = get_video_reader(osp.join(root, vid),
            num_threads=threads,
            fast_rrc=fast_rrc, rrc_params=rrc_params,
            fast_rcc=fast_rcc, rcc_params=rcc_params,
    )
    fps = vr.get_avg_fps()

    start_frame = int(np.round(fps * start_timestamp)) if start_timestamp else 0
    end_frame = int(np.ceil(fps * end_timestamp)) if end_timestamp else len(vr) - 1
    end_frame = min(end_frame, len(vr) - 1)
    
    frame_ids = get_frame_ids(start_frame, end_frame, num_segments=clip_length, jitter=is_training)
    
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid, start_timestamp, end_timestamp, start_frame, end_frame, fps, frame_ids, len(vr) - 1)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    
    return torch.stack(frames, dim=0)



def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    '''
    args:
        start_frame: the beginning frame indice
        end_frame: the end frame indice
        num_segment: number of frames to be sampled
        jitter: True stands for random sampling, False means center sampling
    return:
        seq: a list for the sampled frame indices 
    '''
    assert start_frame <= end_frame
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        
        ### added here to avoid out-of-boundary of frame_id, as np.random.randint ###
        start = min(start, end_frame-1)
        end = min(end, end_frame)

        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2

        seq.append(frame_id)
    return seq

def generate_label_map(dataset, metapath):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        for f in [
            f'{metapath}epic-kitchens-100-annotations/EPIC_100_train.csv',
            f'{metapath}epic-kitchens-100-annotations/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open(f'{metapath}Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open(f'{metapath}action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act

def convert():
    # srcdir =  '/mnt/petrelfs/xujilan/data/ego4d/lavila_generation/ego4d_train.rephraser.no_punkt_top3.pkl'
    srcdir = '/mnt/petrelfs/xujilan/data/ego4d/lavila_generation/ego4d_train.narrator_63690737.return_10.pkl'
    dstdir = srcdir.replace('.pkl', '.csv')
    src = pickle.load(open(srcdir, 'rb'))
    print(len(src))
    vid_list, st_list, ed_list, cap_list = [], [], [], []
    for i in range(len(src)):
        cur = src[i]
        vid_list.append(cur[0])
        st_list.append(cur[1])
        ed_list.append(cur[2])
        cap_list.append(cur[3])

    print(len(vid_list))
    df = pd.DataFrame({
        'video_id': vid_list,
        'start_second': st_list,
        'end_second':ed_list,
        'text': cap_list,
        'dataset': 'ego4d',
    })
    df.to_csv(dstdir)
    print('Done saving')
    
if __name__ == '__main__':
    convert()