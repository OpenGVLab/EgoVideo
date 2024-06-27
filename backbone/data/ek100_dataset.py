
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
from ipdb import set_trace
import cv2
import io,os

from nltk.stem import WordNetLemmatizer
from .data_utils import datetime2sec, get_frame_ids
from .data_utils import video_loader_by_frames, video_loader_by_timestamp, video_loader
from .data_utils import generate_label_map
from petrel_client.client import Client

class EK100Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform=None, is_training=False, tokenizer=None, crop_size=224):
        ### common setups ###
        self.cfg = cfg
        self.root = cfg.root
        self.metadata = cfg.metadata
        self.clip_length = cfg.clip_length
        self.clip_stride = cfg.clip_stride
        self.use_bert = True
        ### maybe customized ###
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        
        self.chunk_len = cfg.video_chunk_len
        self.fps = cfg.fps
        self.threads = cfg.decode_threads
        
        if is_training:
            self.fast_rrc = cfg.fused_decode_crop
            self.rrc_params = (crop_size, (0.5, 1.0))
        else:
            self.fast_rcc = cfg.fused_decode_crop
            self.rcc_params = (crop_size,)

        self.samples = []
        with open(self.metadata) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                pid, vid = row[1:3]
                # start_frame, end_frame = int(row[6]), int(row[7])
                # Deprecated: some videos might have fps mismatch issue
                start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                narration = row[8]
                verb, noun = int(row[10]), int(row[12])
                
                vid_path = '{}.mp4'.format(vid)
                self.samples.append((vid_path, start_timestamp, end_timestamp, narration, verb, noun))
        
        # if self.dataset == 'ek100_mir':
        self.metadata_sentence = pd.read_csv(self.metadata[:self.metadata.index('.csv')] + '_sentence.csv')
        if 'train' in self.metadata:
            self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(self.metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
        elif 'test' in self.metadata:
            self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(self.metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
        else:
            raise ValueError('{} should contain either "train" or "test"!'.format(self.metadata))
        self.relevancy = .1

    def __len__(self):
        return len(self.samples)
    
    def get_raw_item(self, i):
        vid_path, start_timestamp, end_timestamp, narration, verb, noun = self.samples[i]
        # frames = video_loader_by_timestamp(self.root, vid_path, 
        #     start_timestamp=start_frame, end_timestamp=end_frame, 
        #     clip_length=self.clip_length, is_training=self.is_training, 
        #     threads=self.threads, fast_rcc=self.fast_rcc, rcc_params=self.rcc_params
        # )

        if self.is_training:
            frames = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.fps,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)
        else:
            frames = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.fps,
                fast_rcc=self.fast_rcc, rcc_params=self.rcc_params, jitter=self.is_training)
            
        if self.is_training:
            positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[0].tolist()
            if positive_list != []:
                pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                    return frames, self.metadata_sentence.iloc[pos][1], self.relevancy_mat[i][pos]
        else:
            return frames, narration, 1
        
    
    def __getitem__(self, i):
        ### for record info only ###
        vid_path, start_timestamp, end_timestamp, narration, verb, noun = self.samples[i]
        uid = vid_path
        raw_caption = narration

        frames, narration, relevancy = self.get_raw_item(i)
        if self.transform is not None:
            frames = self.transform(frames)
        
        #### this is for ek100_cls ###
        # if self.cfg.dataset == 'ek100_cls':
        #     return frames, '{}:{}'.format(verb, noun)

        #### this is for ek100_mir ###
        if self.tokenizer is not None:
            text = self.tokenizer(narration,max_length=10,truncation=True,
                padding = 'max_length',return_tensors = 'pt')
            #text = self.tokenizer(narration,return_tensors = 'pt')
            caption = text.input_ids.squeeze()
            mask = text.attention_mask
            
        return frames, caption,mask,relevancy       

class EK100Dataset_CLS(torch.utils.data.Dataset):
    def __init__(self, cfg, transform=None, is_training=False, tokenizer=None, crop_size=224):
        ### common setups ###
        self.root = cfg.root
        self.metadata = cfg.metadata
        self.clip_length = cfg.clip_length
        self.clip_stride = cfg.clip_stride
        
        ### maybe customized ###
        self.transform = transform
        self.is_training = is_training
        self.use_bert = True
        self.tokenizer = tokenizer
        
        self.chunk_len = cfg.video_chunk_len
        self.fps = cfg.fps
        self.threads = cfg.decode_threads
        
        if is_training:
            self.fast_rrc = cfg.fused_decode_crop
            self.rrc_params = (crop_size, (0.5, 1.0))
        else:
            self.fast_rcc = cfg.fused_decode_crop
            self.rcc_params = (crop_size,)

        
        self.samples = []
        with open(self.metadata) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                pid, vid = row[1:3]
                # start_frame, end_frame = int(row[6]), int(row[7])
                # Deprecated: some videos might have fps mismatch issue
                start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                narration = row[8]
                verb, noun = int(row[10]), int(row[12])
                
                vid_path = '{}.mp4'.format(vid)
                self.samples.append((vid_path, start_timestamp, end_timestamp, narration, verb, noun))

        self.labels, self.label_mapping = generate_label_map('ek100_cls', cfg.metapath)

    def __len__(self):
        return len(self.samples)
    
    def get_raw_item(self, i):
        # vid_path, start_frame, end_frame, narration, verb, noun = self.samples[i]
        # frames = video_loader_by_timestamp(self.root, vid_path, 
        #     start_timestamp=start_frame, end_timestamp=end_frame, 
        #     clip_length=self.clip_length, is_training=self.is_training, 
        #     threads=self.threads, fast_rcc=self.fast_rcc, rcc_params=self.rcc_params
        # )
        vid_path, start_timestamp, end_timestamp, narration, verb, noun = self.samples[i]

        if self.is_training:
            frames = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.fps,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)
        else:
            frames = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.fps,
                fast_rcc=self.fast_rcc, rcc_params=self.rcc_params, jitter=self.is_training)
            
        return frames, f'{verb}:{noun}', narration
    
    def __getitem__(self, i):
        ### for record info only ###
        vid_path, start_frame, end_frame, narration, verb, noun = self.samples[i]

        frames, label, narration = self.get_raw_item(i)
        raw_caption = narration

        frames = self.transform(frames) if self.transform is not None else None

        if isinstance(label, list):
            # multi-label case
            res_array = np.zeros(len(self.label_mapping))
            for lbl in label:
                res_array[self.label_mapping[lbl]] = 1.
            label = res_array
        else:
            raw_label = label
            label = self.label_mapping[label]

        return frames, label
        


