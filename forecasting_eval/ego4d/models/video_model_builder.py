#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn

from ..utils import weight_init_helper as init_helper
from .batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

import math 
from functools import reduce
import operator
import numpy 

from functools import partial
from torch.nn.init import trunc_normal_

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "mvit": [[]]
}


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg, with_head=True):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = is_detection_enabled(cfg)
        self.num_pathways = 2
        self._construct_network(cfg, with_head=with_head)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg, with_head=True):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[width_per_group * 4, width_per_group * 4 // cfg.SLOWFAST.BETA_INV],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[width_per_group * 8, width_per_group * 8 // cfg.SLOWFAST.BETA_INV],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if not with_head:
            return

        if self.enable_detection:
            head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES[0],
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
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES[0],
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
                ],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.head_name = "head"
        self.add_module(self.head_name, head)

    def forward(self, x, bboxes=None):
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

        if hasattr(self, "head_name"):
            head = getattr(self, self.head_name)
            if self.enable_detection:
                x = head(x, bboxes)
            else:
                x = head(x)
        return x


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg, with_head=True):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = is_detection_enabled(cfg)
        self.num_pathways = 1
        self._construct_network(cfg, with_head=with_head)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg, with_head=True):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if not with_head:
            return

        if self.enable_detection:
            head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES[0],
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES[0],
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        self.head_name = "head"
        self.add_module(self.head_name, head)

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)

        if hasattr(self, "head_name"):
            head = getattr(self, self.head_name)
            if self.enable_detection:
                x = head(x, bboxes)
            else:
                x = head(x)

        return x


def is_detection_enabled(cfg):
    try:
        detection_enabled = (
            cfg.DATA.TASK == "detection" or cfg.DATA.TASK=="short_term_anticipation"
        )
    except Exception:
        # Temporary default config while still using old models without this config option
        detection_enabled = False

    return detection_enabled



@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg, with_head = True):

        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg

        self.num_frames = 16 #self.cfg.DATA.NUM_FRAMES
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size =self.num_frames 

        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        pool_first = cfg.MVIT.POOL_FIRST

        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON

        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL

        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        # num_patches = math.prod(self.patch_dims)
        num_patches = reduce(operator.mul, self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

            
        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, pos_embed_dim, embed_dim)
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[
                    cfg.MVIT.POOL_Q_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]
        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]
        
        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

            
        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None
        input_size = self.patch_dims
        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )

            self.blocks.append(
                MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    pool_first=pool_first,
                    input_size= input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                    rel_pos_spatial=self.rel_pos_spatial,
                    rel_pos_temporal=self.rel_pos_temporal,
                )
            )

            if len(stride_q[i]) > 0:
                input_size = [size // stride for size, stride in zip(input_size, stride_q[i])]

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        if with_head:
                
            self.head = TransformerBasicHead(
                embed_dim,
                num_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        ["pos_embed_spatial", "pos_embed_temporal", "pos_embed_class"]
                    )
                else:
                    names.append(["pos_embed"])
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def forward(self, x):

        # just the slow branch
        if len(x) > 1:
            if x[0].shape[2] == 16:
                x = x[0]
            else:
                downsample = x[1].shape[2] // 16
                assert x[1].shape[2] % 16 == 0
                x = x[1][:, : ,::downsample, : , :]
        else:
            assert x[0].shape[2] == 16

        x = self.patch_embed(x)

        T = self.num_frames // self.patch_stride[0]
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x = x + pos_embed
            else:
                x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        # import pdb; pdb.set_trace()
        thw = [T, H, W]
        for blk in self.blocks:
            x, thw = blk(x, thw)

        
        x = self.norm(x)
        if self.cls_embed_on:
            x = x[:, 0]
        else:
            x = x.mean(1)

        x = self.head(x)
        return x

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        input_size,
        pool_first,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
        # If True, perform pool before projection.
        rel_pos_spatial=False,
        rel_pos_temporal=False,
        cfg=None,
    ):
        super().__init__()
        self.pool_first = pool_first
        self.cfg = cfg
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.k = nn.Linear(dim, dim, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)
        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mask = None
        self.mode = mode

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool3d if mode == "max" else nn.AvgPool3d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = head_dim if mode == "conv" else dim
            self.pool_q = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if self.rel_pos_spatial:
            assert input_size[1] == input_size[2]
            q_size = input_size[1] // stride_q[1] if len(stride_q) > 0 else input_size[1]
            kv_size = input_size[1] // stride_kv[1] if len(stride_kv) > 0 else input_size[1]
            rel_sp_dim = 2 * max(q_size, kv_size) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            trunc_normal_(self.rel_pos_h, std=0.02)
            trunc_normal_(self.rel_pos_w, std=0.02)

        if self.rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(torch.zeros(2 * input_size[0] - 1, dim // num_heads))
            # if not cfg.TRAIN.CHECKPOINT_IN_INIT:
            #     trunc_normal_(self.rel_pos_t, std=0.02)


    def forward(self, x, thw_shape):
        B, N, C = x.shape
        if self.pool_first:
            if self.mode == 'conv_unshared':
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, C // fold_dim).permute(
                0, 2, 1, 3
            )
            q = k = v = x
        else:
            assert self.mode != 'conv_unshared'
            # q = k = v = x
            # q = (
            #     self.q(q)
            #     .reshape(B, N, self.num_heads, C // self.num_heads)
            #     .permute(0, 2, 1, 3)
            # )
            # k = (
            #     self.k(k)
            #     .reshape(B, N, self.num_heads, C // self.num_heads)
            #     .permute(0, 2, 1, 3)
            # )
            # v = (
            #     self.v(v)
            #     .reshape(B, N, self.num_heads, C // self.num_heads)
            #     .permute(0, 2, 1, 3)
            # )
            qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]


        q, q_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        if self.pool_first:
            q_N = (
                numpy.prod(q_shape) + 1
                if self.has_cls_embed
                else numpy.prod(q_shape)
            )
            k_N = (
                numpy.prod(k_shape) + 1
                if self.has_cls_embed
                else numpy.prod(k_shape)
            )
            v_N = (
                numpy.prod(v_shape) + 1
                if self.has_cls_embed
                else numpy.prod(v_shape)
            )

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, C)
            q = (
                self.q(q)
                .reshape(B, q_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, C)
            v = (
                self.v(v)
                .reshape(B, v_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, C)
            k = (
                self.k(k)
                .reshape(B, k_N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )

        N = q.shape[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn, q, self.has_cls_embed, q_shape, k_shape, self.rel_pos_h, self.rel_pos_w
            )
        if self.rel_pos_temporal:
            attn = cal_rel_pos_temporal(
                attn, q, self.has_cls_embed, q_shape, k_shape, self.rel_pos_t,
            )
        # thw = attn.shape[2]
        # attn = attn / thw
        attn = attn.softmax(dim=-1)
        x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape

class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        input_size,
        pool_first,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
        rel_pos_spatial=False,
        rel_pos_temporal=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            input_size=input_size,
            pool_first=pool_first,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )

    def forward(self, x, thw_shape):
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(1, 16, 16),
        stride=(1, 4, 4),
        padding=(1, 7, 7),
        conv_2d=False,
    ):
        super().__init__()
        if conv_2d:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)

def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        logger.info(f"min width {min_width}")
        logger.info(f"width {width} divisor {divisor}")
        logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)

class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # import pdb; pdb.set_trace()

        # Not sure why this error comes with MViT 
        if isinstance(num_classes, list):
            assert len(num_classes) == 1
            num_classes = num_classes[0]
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func is None:
            self.act = nn.Identity()

        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x

def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape

        
def _contract(*args):
    return opt_einsum.contract(*args, backend="torch")

def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        ori_d = rel_pos.shape[0]
        if ori_d == d:
            return rel_pos
        else:
            # Interpolate rel pos.
            new_pos_embed = F.interpolate(
                rel_pos.reshape(1, ori_d, -1).permute(0, 2, 1),
                size=d,
                mode="linear",
            )

            return new_pos_embed.reshape(-1, d).permute(1, 0)

    
def cal_rel_pos_spatial(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w):
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    # Intepolate rel pos if needed.
    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio- torch.arange(k_h)[None, :] * k_h_ratio
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    if q.dim() == 4 and attn.dim() == 4:
        B, n_head, q_N, dim = q.shape

        r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
        # [B, H, q_t, q_h, q_w, dim] -> [q_h, B, H, q_t, q_w, dim] -> [q_h, B*H*q_t*q_w, dim]
        r_q_h = r_q.permute(3, 0, 1, 2, 4, 5).reshape(q_h, B * n_head * q_t * q_w, dim)
        # [B, H, q_t, q_h, q_w, dim] -> [q_w, B, H, q_t, q_h, dim] -> [q_w, B*H*q_t*q_h, dim]
        r_q_w = r_q.permute(4, 0, 1, 2, 3, 5).reshape(q_w, B * n_head * q_t * q_h, dim)

        # [q_h, B*H*q_t*q_w, dim] * [q_h, dim, k_h] = [q_h, B*H*q_t*q_w, k_h] -> [B*H*q_t*q_w, q_h, k_h]
        rel_h = torch.matmul(r_q_h, Rh.permute(0, 2, 1)).transpose(0, 1)
        # [q_w, B*H*q_t*q_h, dim] * [q_w, dim, k_w] = [q_w, B*H*q_t*q_h, k_w] -> [B*H*q_t*q_h, q_w, k_w]
        rel_w = torch.matmul(r_q_w, Rw.permute(0, 2, 1)).transpose(0, 1)

        # [B*H*q_t*q_w, q_h, k_h] -> [B, H, q_t, qh, qw, k_h]
        rel_h = rel_h.view(B, n_head, q_t, q_w, q_h, k_h).permute(0, 1, 2, 4, 3, 5)
        # [B*H*q_t*q_h, q_w, k_w] -> [B, H, q_t, qh, qw, k_w]
        rel_w = rel_w.view(B, n_head, q_t, q_h, q_w, k_w)

        attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
            + rel_h[:, :, :, :, :, None, :, None]
            + rel_w[:, :, :, :, :, None, None, :]
        ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)
    elif q.dim() == 3 and attn.dim() == 3:
        B_n_head, q_N, dim = q.shape
        r_q = q[:, :, sp_idx:].reshape(B_n_head, q_t, q_h, q_w, dim)
        # [BH, q_t, q_h, q_w, dim] -> [q_h, BH, q_t, q_w, dim] -> [q_h, B*H*q_t*q_w, dim]
        r_q_h = r_q.permute(2, 0, 1, 3, 4).reshape(q_h, B_n_head * q_t * q_w, dim)
        # [BH, q_t, q_h, q_w, dim] -> [q_w, BH, q_t, q_h, dim] -> [q_w, B*H*q_t*q_h, dim]
        r_q_w = r_q.permute(3, 0, 1, 2, 4).reshape(q_w, B_n_head * q_t * q_h, dim)
        # [q_h, B*H*q_t*q_w, dim] * [q_h, dim, k_h] = [q_h, B*H*q_t*q_w, k_h] -> [B*H*q_t*q_w, q_h, k_h]
        rel_h = torch.matmul(r_q_h, Rh.permute(0, 2, 1)).transpose(0, 1)
        rel_w = torch.matmul(r_q_w, Rw.permute(0, 2, 1)).transpose(0, 1)
        # [B*H*q_t*q_w, q_h, k_h] -> [BH, q_t, qh, qw, k_h]
        rel_h = rel_h.view(B_n_head, q_t, q_w, q_h, k_h).permute(0, 1, 3, 2, 4)
        rel_w = rel_w.view(B_n_head, q_t, q_h, q_w, k_w)
        attn[:, sp_idx:, sp_idx:] = (
            attn[:, sp_idx:, sp_idx:].view(B_n_head, q_t, q_h, q_w, k_t, k_h, k_w)
            + rel_h[:, :, :, :, None, :, None]
            + rel_w[:, :, :, :, None, None, :]
        ).view(B_n_head, q_t * q_h * q_w, k_t * k_h * k_w)
    else:
        raise NotImplementedError(f"Unsupported q shape {q.shape}")
    return attn
    
def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)
    # Intepolate rel pos if needed.
    rel_pos_t = get_rel_pos(rel_pos_t, dt)

    # Scale up rel pos if shapes for q and k are different.
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = torch.arange(q_t)[:, None] * q_t_ratio- torch.arange(k_t)[None, :] * k_t_ratio
    dist_t += (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    if q.dim() == 4 and attn.dim() == 4:
        B, n_head, q_N, dim = q.shape
        r_q = q[:, :, sp_idx:].reshape(B, n_head, q_t, q_h, q_w, dim)
        # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
        r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w, dim)
        # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
        rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
        # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
        rel = rel.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

        #attn[:, :, 1:, 1:] += attn_t
        if attn.dim() == 4:
            attn[:, :, sp_idx:, sp_idx:] = (
                attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
                + rel[:, :, :, :, :, :, None, None]
            ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)
    elif q.dim() == 3 and attn.dim() == 3:
        B_n_head, q_N, dim = q.shape
        r_q = q[:, :, sp_idx:].reshape(B_n_head, q_t, q_h, q_w, dim)
        # [BH, q_t, q_h, q_w, dim] -> [q_t, BH, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
        r_q = r_q.permute(1, 0, 2, 3, 4).reshape(q_t, B_n_head * q_h * q_w, dim)
        # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
        rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
        # [B*H*q_h*q_w, q_t, k_t] -> [BH, q_t, q_h, q_w, k_t]
        rel = rel.view(B_n_head, q_h, q_w, q_t, k_t).permute(0, 3, 1, 2, 4)
        attn[:, sp_idx:, sp_idx:] = (
                attn[:, sp_idx:, sp_idx:].view(B_n_head, q_t, q_h, q_w, k_t, k_h, k_w)
                + rel[:, :, :, :, :, None, None]
            ).view(B_n_head, q_t * q_h * q_w, k_t * k_h * k_w)
    else:
        raise NotImplementedError(f"Unsupported q shape {q.shape}")

    return attn
