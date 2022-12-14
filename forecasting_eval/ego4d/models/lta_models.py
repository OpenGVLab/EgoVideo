#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi input models."""

import inspect
import random
import heapq
from torch.nn.init import xavier_uniform_
import collections
import torch
from torch.distributions.categorical import Categorical
from functools import reduce
import torch.nn.functional as F
import math
import copy
from einops import rearrange
import torch.nn as nn

from functools import reduce
from operator import mul
from .head_helper import MultiTaskHead, MultiTaskMViTHead
from .video_model_builder import SlowFast, _POOL1, MViT
from .build import MODEL_REGISTRY



@MODEL_REGISTRY.register()
class MultiTaskSlowFast(SlowFast):
    def _construct_network(self, cfg, with_head=False):
        super()._construct_network(cfg, with_head=with_head)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = MultiTaskHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
            ],  # None for AdaptiveAvgPool3d((1, 1, 1))
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )
        self.head_name = "head"
        self.add_module(self.head_name, head)

@MODEL_REGISTRY.register()
class RecognitionSlowFastRepeatLabels(MultiTaskSlowFast):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def forward(self, x, tgts=None):
        # keep only first input
        x = [xi[:, 0] for xi in x]
        x = super().forward(x)

        # duplicate predictions K times
        K = self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        x = [xi.unsqueeze(1).repeat(1, K, 1) for xi in x]
        return x

    def generate(self, x, k=1):
        x = self.forward(x)
        results = []
        for head_x in x:
            preds_dist = Categorical(logits=head_x)
            preds = [preds_dist.sample() for _ in range(k)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)
        return results


@MODEL_REGISTRY.register()
class MultiTaskMViT(MViT):

    def __init__(self, cfg):

        super().__init__(cfg, with_head =False)

        self.head = MultiTaskMViTHead(
            [768],
            cfg.MODEL.NUM_CLASSES,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class ConcatAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_input_clips, D)
        x = x.view(x.shape[0], -1) # (B, num_input_clips * D)
        return x

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES * cfg.FORECASTING.NUM_INPUT_CLIPS

@MODEL_REGISTRY.register()
class MeanAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_input_clips, D)
        x = x.mean(1)
        return x

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)

@MODEL_REGISTRY.register()
class TransformerAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        num_heads = cfg.MODEL.TRANSFORMER_ENCODER_HEADS
        num_layers = cfg.MODEL.TRANSFORMER_ENCODER_LAYERS
        dim_in = cfg.MODEL.MULTI_INPUT_FEATURES
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim_in, num_heads),
            num_layers,
            norm=nn.LayerNorm(dim_in),
        )
        self.pos_encoder = PositionalEncoding(dim_in, dropout=0.2)

    def forward(self, x):
        x = torch.stack(x, dim=1) # (B, num_inputs, D)
        x = x.transpose(0, 1) # (num_inputs, B, D)
        x = self.pos_encoder(x)
        x = self.encoder(x) # (num_inputs, B, D)
        return x[-1] # (B, D) return last timestep's encoding

    @staticmethod
    def out_dim(cfg):
        return cfg.MODEL.MULTI_INPUT_FEATURES 

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class MultiHeadDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        head_classes = [
            reduce((lambda x, y: x + y), cfg.MODEL.NUM_CLASSES)
        ] * self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT
        head_dim_in = MODEL_REGISTRY.get(cfg.FORECASTING.AGGREGATOR).out_dim(cfg)
        self.head = MultiTaskHead(
            dim_in=[head_dim_in],
            num_classes=head_classes,
            pool_size=[None],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
            test_noact=cfg.TEST.NO_ACT,
        )

    def forward(self, x, tgts=None):
        x = x.view(x.shape[0], -1, 1, 1, 1)
        x = torch.stack(self.head([x]), dim=1) # (B, Z, #verbs + #nouns)
        x = torch.split(x, self.cfg.MODEL.NUM_CLASSES, dim=-1) # [(B, Z, #verbs), (B, Z, #nouns)]
        return x

#--------------------------------------------------------------------#

@MODEL_REGISTRY.register()
class ForecastingEncoderDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_clip_backbone()
        self.build_clip_aggregator()
        self.build_decoder()

    # to encode frames into a set of {cfg.FORECASTING.NUM_INPUT_CLIPS} clips
    def build_clip_backbone(self):
        cfg = self.cfg
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        backbone_config = copy.deepcopy(cfg)
        backbone_config.MODEL.NUM_CLASSES = [cfg.MODEL.MULTI_INPUT_FEATURES]
        backbone_config.MODEL.HEAD_ACT = None


        if cfg.MODEL.ARCH == "mvit":
            self.backbone = MViT(backbone_config, with_head=True)
        else:
            self.backbone = SlowFast(backbone_config, with_head=True)
        # replace with:
        # self.backbone = MODEL_REGISTRY.get(cfg.FORECASTING.BACKBONE)(backbone_config, with_head=True)

        if cfg.MODEL.FREEZE_BACKBONE:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Never freeze head.
            for param in self.backbone.head.parameters():
                param.requires_grad = True


    def build_clip_aggregator(self):
        self.clip_aggregator = MODEL_REGISTRY.get(self.cfg.FORECASTING.AGGREGATOR)(self.cfg)

    def build_decoder(self):
        self.decoder = MODEL_REGISTRY.get(self.cfg.FORECASTING.DECODER)(self.cfg)

    # input = [(B, num_inp, 3, T, H, W), (B, num_inp, 3, T', H, W)]
    def encode_clips(self, x):
        # x -> [torch.Size([2, 2, 3, 8, 224, 224]), torch.Size([2, 2, 3, 32, 224, 224])]
        assert isinstance(x, list) and len(x) >= 1

        num_inputs = x[0].shape[1]
        batch = x[0].shape[0]
        features = []
        for i in range(num_inputs):
            pathway_for_input = []
            for pathway in x:
                input_clip = pathway[:, i]
                pathway_for_input.append(input_clip)

            # pathway_for_input -> [torch.Size([2, 3, 8, 224, 224]), torch.Size([2, 3,32, 224, 224])]
            input_feature = self.backbone(pathway_for_input)
            features.append(input_feature)

        return features

    # input = list of clips: [(B, D)] x {cfg.FORECASTING.NUM_INPUT_CLIPS}
    # output = (B, D') tensor after aggregation
    def aggregate_clip_features(self, x):
        return self.clip_aggregator(x)

    # input = (B, D') tensor encoding of full video
    # output = [(B, Z, #verbs), (B, Z, #nouns)] probabilities for each z
    def decode_predictions(self, x, tgts):
        return self.decoder(x, tgts)

    def forward(self, x, tgts=None):
        features = self.encode_clips(x)
        x = self.aggregate_clip_features(features)
        x = self.decode_predictions(x, tgts)
        return x

    def generate(self, x, k=1):
        x = self.forward(x)
        results = []
        for head_x in x:
            if k>1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(k)]
            elif k==1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results
