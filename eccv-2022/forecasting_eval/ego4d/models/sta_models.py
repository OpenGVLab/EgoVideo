#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import numpy as np
import torch
import torch.nn as nn
from detectron2.layers import ROIAlign

from .build import MODEL_REGISTRY
from .video_model_builder import ResNet, SlowFast, _POOL1

class ResNetSTARoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
            self,
            dim_in,
            num_verbs,
            pool_size,
            resolution,
            scale_factor,
            dropout_rate=0.0,
            verb_act_func=(None, "softmax"),
            ttc_act_func=("softplus", "softplus"),
            aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_verbs (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetSTARoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)
        for pathway in range(self.num_pathways):
            temporal_pool = nn.AvgPool3d([pool_size[pathway][0], 1, 1], stride=1)
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
                aligned=aligned,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.verb_projection = nn.Linear(sum(dim_in), num_verbs + 1, bias=True)
        self.ttc_projection = nn.Linear(sum(dim_in), 1, bias=True)

        def get_act(act_func):
            # Softmax for evaluation and testing.
            if act_func == "softmax":
                act = nn.Softmax(dim=1)
            elif act_func == "sigmoid":
                act = nn.Sigmoid()
            elif act_func == "softplus":
                act = nn.Softplus()
            elif act_func == "relu":
                act = nn.ReLU()
            elif act_func == "identity":
                act = None
            elif act_func is None:
                act = None
            else:
                raise NotImplementedError(
                    "{} is not supported as an activation" "function.".format(act_func)
                )
            return act

        self.verb_act = [get_act(x) for x in verb_act_func]
        self.ttc_act = [get_act(x) for x in ttc_act_func]

    def forward(self, inputs, bboxes):
        if bboxes.shape[0]==0: # handle cases in which zero boxes are passed as input
            x_verb = torch.zeros((0, self.verb_projection.out_features))
            x_ttc = torch.zeros((0, self.ttc_projection.out_features))
            return x_verb, x_ttc
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)
        x = x.view(x.shape[0], -1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x_verb = self.verb_projection(x)
        x_ttc = self.ttc_projection(x)

        act_idx = 0 if self.training else 1

        if self.verb_act[act_idx] is not None:
            x_verb = self.verb_act[act_idx](x_verb)

        if self.ttc_act[act_idx] is not None:
            x_ttc = self.ttc_act[act_idx](x_ttc)

        return x_verb, x_ttc


@MODEL_REGISTRY.register()
class ShortTermAnticipationResNet(ResNet):
    def _construct_network(self, cfg):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = ResNetSTARoIHead(
            dim_in=[width_per_group * 32],
            num_verbs=cfg.MODEL.NUM_VERBS,
            pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
            resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
            scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            verb_act_func=(None, cfg.MODEL.HEAD_VERB_ACT),
            ttc_act_func=(cfg.MODEL.HEAD_TTC_ACT,)*2,
            aligned=cfg.DETECTION.ALIGNED,
        )
        self.head_name = "headsta"
        self.add_module(self.head_name, head)


@MODEL_REGISTRY.register()
class ShortTermAnticipationSlowFast(SlowFast):
    def _construct_network(self, cfg, with_head=False):
        super()._construct_network(cfg, with_head=False)

        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        pool_size = _POOL1[cfg.MODEL.ARCH]

        head = ResNetSTARoIHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            num_verbs=cfg.MODEL.NUM_VERBS,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES // cfg.SLOWFAST.ALPHA // pool_size[0][0],
                    1,
                    1,
                ],
                [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
            ],
            resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
            scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            verb_act_func=(None, cfg.MODEL.HEAD_VERB_ACT),
            ttc_act_func=(cfg.MODEL.HEAD_TTC_ACT,)*2,
            aligned=cfg.DETECTION.ALIGNED,
        )
        self.head_name = "headsta"
        self.add_module(self.head_name, head)

    def extract_features(self, x):
        """Performs feature extraction"""
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)

        return x

    def pack_boxes(self, bboxes):
        """Packs images and boxes so that they can be processed in batch"""
        # compute indexes
        idx = torch.from_numpy(np.concatenate([[i]*len(b) for i, b in enumerate(bboxes)]))

        # add indexes as first column of boxes
        bboxes = torch.cat(bboxes, 0)
        bboxes = torch.cat([idx.view(-1,1).to(bboxes.device), bboxes], 1)

        return bboxes

    def postprocess(self,
                    pred_boxes,
                    pred_object_labels,
                    pred_object_scores,
                    pred_verbs,
                    pred_ttcs
                    ):
        """Obtains detections"""

        detections = []
        raw_predictions = []

        for orig_boxes, orig_object_labels, object_scores, verb_scores, ttcs in zip(pred_boxes, pred_object_labels, pred_object_scores, pred_verbs, pred_ttcs):
            if verb_scores.shape[0]>0:
                verb_predictions = verb_scores.argmax(-1)

                dets = {
                    "boxes": orig_boxes,
                    "nouns": orig_object_labels,
                    "verbs": verb_predictions.cpu().numpy(),
                    "ttcs": ttcs.cpu().numpy(),
                    "scores": object_scores
                }
            else:
                dets = {
                    "boxes": np.zeros((0,4)),
                    "nouns": np.array([]),
                    "verbs": np.array([]),
                    "ttcs": np.array([]),
                    "scores": np.array([])
                }

            raw_predictions.append({
                "boxes": orig_boxes,
                "object_labels": orig_object_labels,
                "object_scores": object_scores,
                "verb_scores": verb_scores.cpu().numpy(),
                "ttcs": ttcs.cpu().numpy()
            })

            detections.append(dets)

        return detections, raw_predictions

    def forward(self, videos, bboxes, orig_pred_boxes=None, pred_object_labels=None, pred_object_scores=None):
        """Expects videos to be a batch of input tensors and bboxes
        to be a list associated bounding boxes
        videos : List[short_path, Fast_path] bs*3*t*h*w
        bboxes : List length:bs, each element is a tensor of shape (num_boxes, 4)
        """
        packed_bboxes = self.pack_boxes(bboxes)

        features = self.extract_features(videos)
        ### features[0] bs * ch * t * h * w  e.g. 32 * 2048 * 8 *16 * 16
        ### features[1] bs * ch * t * h * w  e.g. 32 * 256 * 32 *16 * 16
        pred_verbs, pred_ttcs = self.headsta(features, packed_bboxes)

        if self.training:
            return pred_verbs, pred_ttcs
        else:
            assert pred_object_labels is not None
            assert pred_object_scores is not None
            lengths = [len(x) for x in bboxes]  # number of boxes per entry
            pred_verbs = pred_verbs.split(lengths, 0)
            pred_ttcs = pred_ttcs.split(lengths, 0)
            # compute detections and return them
            return self.postprocess(orig_pred_boxes, pred_object_labels, pred_object_scores, pred_verbs, pred_ttcs)


