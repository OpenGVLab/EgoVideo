#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa

from .sta_models import (
    ShortTermAnticipationResNet,
    ShortTermAnticipationSlowFast,
)  # noqa
from .lta_models import (
    ForecastingEncoderDecoder,
)  # noqa

from .ViT3d import (
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_base_patch16_384,
    vit_large_patch16_224,
    vit_large_patch16_384,
    vit_large_patch16_512,
    vit_huge_patch16_224
)

from .uniformer import(
    uniformer_base_224_ego4d,
    uniformer_base_256_ego4d,
    uniformer_base_320_ego4d
)
