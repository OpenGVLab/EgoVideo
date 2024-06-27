#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

import itertools

from .build import build_dataset
from collections import defaultdict


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "gt_labels":
            collated_extra_data[key] = torch.tensor(
                np.concatenate(data, axis=0)
            ).float()
        elif key == "boxes" or key == "ori_boxes" or key == "gt_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
                if len(data[i]) > 0
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata" or key == "gt_metadata":
            collated_extra_data[key] = list(itertools.chain(*data))
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data


def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        if cfg.SOLVER.ACCELERATOR != "dp":
            batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        else:
            batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # Create a sampler for multi-process training

    sampler = None
    if not cfg.FBLEARNER:
        # Create a sampler for multi-process training
        if hasattr(dataset, "sampler"):
            sampler = dataset.sampler
        elif cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=get_collate(cfg.DATA.TASK),
    )
    return loader

def get_collate(key):
    if key == "detection":
        return detection_collate
    elif key == "short_term_anticipation":
        return sta_collate
    else:
        return None

def sta_collate(batch):
    """
    Collate function for the short term anticipation task.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    eids, inputs, pred_boxes, verb_labels, ttc_targets, orig_norm_pred_boxes, _extra_data = zip(
        *batch
    )

    eids = default_collate(eids)
    inputs = default_collate(inputs)

    pred_boxes = [torch.from_numpy(b.astype(float)) for b in pred_boxes]
    orig_norm_pred_boxes = [torch.from_numpy(b.astype(float)) for b in orig_norm_pred_boxes]
    verb_labels = [torch.from_numpy(x).long() for x in verb_labels]
    ttc_targets = [torch.from_numpy(x.reshape(-1, 1)).float() for x in ttc_targets]

    extra_data = defaultdict(list)

    for ed in _extra_data:
        for k, v in ed.items():
            extra_data[k].append(v)

    return eids, inputs, pred_boxes, verb_labels, ttc_targets, orig_norm_pred_boxes, extra_data
