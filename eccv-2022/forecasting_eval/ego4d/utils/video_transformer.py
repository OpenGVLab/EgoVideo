#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from pytorchvideo.transforms import (
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsampleRepeated,
)
from torchvision.transforms import (
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
)


"""
    video transform method to normalize, crop, scale, etc.
"""


def random_scale_crop_flip(mode: str, cfg):
    return (
        [
            RandomShortSideScale(
                min_size=cfg.DATA.TRAIN_JITTER_SCALES[0],
                max_size=cfg.DATA.TRAIN_JITTER_SCALES[1],
            ),
            RandomCrop(cfg.DATA.TRAIN_CROP_SIZE),
            RandomHorizontalFlip(p=cfg.DATA.RANDOM_FLIP),
        ]
        if mode == "train"
        else [
            ShortSideScale(cfg.DATA.TRAIN_JITTER_SCALES[0]),
            CenterCrop(cfg.DATA.TRAIN_CROP_SIZE),
        ]
    )


def uniform_temporal_subsample_repeated(cfg):
    return UniformTemporalSubsampleRepeated(
        ((cfg.SLOWFAST.ALPHA, 1) if len(cfg.DATA.INPUT_CHANNEL_NUM) == 2 else (1,))
    )
