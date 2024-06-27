# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import warnings

from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple

from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 use_checkpoint=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
#                  drop=0., use_checkpoint=False, split_ratio=None):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.use_checkpoint = use_checkpoint
#         self.in_features = in_features
#         self.split_ratio = split_ratio
#         for i in range(split_ratio):
#             fc1 = nn.Linear(in_features, hidden_features // self.split_ratio, bias=True)
#             setattr(self, f"fc1_{i}", fc1)
#         self.act = act_layer()
#         for i in range(split_ratio):
#             fc2 = nn.Linear(hidden_features // self.split_ratio, out_features, bias=False)
#             setattr(self, f"fc2_{i}", fc2)
#
#         self.fc2_bias = nn.Parameter(torch.zeros((out_features)), requires_grad=True)
#
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         def _inner_forward(x, i):
#             fc1 = getattr(self, f"fc1_{i}")
#             x = fc1(x)
#             x = self.act(x)
#             x = self.drop(x)
#             fc2 = getattr(self, f"fc2_{i}")
#             x = fc2(x)
#             x = self.drop(x)
#             return x
#
#         out = 0
#         for i in range(self.split_ratio):
#             if self.use_checkpoint:
#                 out = out + checkpoint.checkpoint(_inner_forward, x, i)
#             else:
#                 out = out + _inner_forward(x, i)
#
#         out += self.fc2_bias
#         return out

    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0,
                 layerscale_init_values=0., final_norm=False, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop, use_checkpoint=use_checkpoint)

        ### ----- layerscale -----
        if layerscale_init_values is not None:
            self.gamma_1 = nn.Parameter(layerscale_init_values * torch.ones((1, 1, dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_init_values * torch.ones((1, 1, dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        ### ----- final norm -----
        if final_norm:
            self.final_norm = nn.LayerNorm(normalized_shape=dim)
        else:
            self.final_norm = None

    def forward(self, x, hw_shape):
        
        def attn_forward(x, hw_shape):
            B, L, C = x.shape
            H, W = hw_shape
            shortcut = x
            x = x.view(B, H, W, C)
    
            # pad feature maps to multiples of window size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            H_pad, W_pad = x.shape[1], x.shape[2]
            
            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # calculate attention mask for SW-MSA
                img_mask = torch.zeros((1, H_pad, W_pad, 1), device=x.device)
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size,
                                  -self.shift_size), slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size,
                                  -self.shift_size), slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
    
                # nW, window_size, window_size, 1
                mask_windows = window_partition(img_mask, self.window_size)
                mask_windows = mask_windows.view(
                    -1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                                  float(-100.0)).masked_fill(
                    attn_mask == 0, float(0.0))
            else:
                shifted_x = x
                attn_mask = None
    
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
    
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
    
            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # B H' W' C
    
            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
                
            if pad_r > 0 or pad_b:
                x = x[:, :H, :W, :].contiguous()
                
            x = x.view(B, H * W, C)
    
            # skip connection (with layerscale)
            if self.gamma_1 is not None:
                x = shortcut + self.drop_path(self.gamma_1 * self.norm1(x))
            else:
                x = shortcut + self.drop_path(self.norm1(x))
            return x
        
        def ffn_forward(x):
            # FFN
            shortcut = x
            x = self.norm2(self.mlp(x))
    
            # skip connection (with layerscale)
            if self.gamma_2 is not None:
                x = shortcut + self.drop_path(self.gamma_2 * x)
            else:
                x = shortcut + self.drop_path(x)
    
            ### ----- final norm -----
            if self.final_norm is not None:
                x = self.final_norm(x)
    
            return x
        
        if self.use_checkpoint:
            x = checkpoint.checkpoint(attn_forward, x, hw_shape)
        else:
            x = attn_forward(x, hw_shape)
            
        # if self.use_checkpoint:
        #     x = checkpoint.checkpoint(ffn_forward, x)
        # else:
        x = ffn_forward(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x, hw_shape):
        """
        x: B, H*W, C
        """
        H, W = hw_shape
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        _, H0, W0, _ = x.shape
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x, (H0, W0)


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 layerscale_init_values=0., final_norm_index=[],
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 layerscale_init_values=layerscale_init_values,
                                 final_norm=True if i in final_norm_index else False,
                                 pretrained_window_size=pretrained_window_size,
                                 use_checkpoint=use_checkpoint)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, hw_shape):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, hw_shape)
            else:
                x = blk(x, hw_shape)
        if self.downsample is not None:
            if self.use_checkpoint:
                x_down, down_hw_shape = checkpoint.checkpoint(self.downsample, x, hw_shape)
            else:
                x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape

        return x, hw_shape, x, hw_shape

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None, use_checkpoint=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # if self.use_checkpoint:
        #     x = checkpoint.checkpoint(self.proj, x)
        # else:
        x = self.proj(x)
        
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x, out_size


@BACKBONES.register_module()
class SwinTransformerV2(BaseModule):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dims (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3,
                 embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 layerscale_init_values=None, final_norm=False,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 out_indices=(0, 1, 2, 3), init_cfg=None,
                 convert_weights=False, pretrained=None, **kwargs):
        self.convert_weights = convert_weights
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'
    
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')
    
        super(SwinTransformerV2, self).__init__(init_cfg=init_cfg)
        
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dims
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = [int(embed_dims * 2 ** i) for i in range(self.num_layers)]

        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims,
            norm_layer=norm_layer if self.patch_norm else None, use_checkpoint=use_checkpoint)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # norm every 6 layers
        if final_norm:
            final_norm_indexs = self.get_final_norm_indexs(depths=depths, interval=6)
            print('final_norm_indexs: ', final_norm_indexs)
        else:
            final_norm_indexs = [[], [], [], []]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               layerscale_init_values=layerscale_init_values,
                               final_norm_index=final_norm_indexs[i_layer],
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)
        
        for i in out_indices:
            layer = nn.LayerNorm(self.num_features[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.ape:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'module' in ckpt:
                state_dict = ckpt['module']
            else:
                state_dict = ckpt
            for k, v in state_dict.items():
                state_dict[k] = v.float()

            # delete relative_position_index since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
            for k in relative_position_index_keys:
                del state_dict[k]
        #
            # interpolate position bias table if needed
            relative_coords_table_keys = [
                k for k in state_dict.keys()
                if 'relative_coords_table' in k
            ]
            for table_key in relative_coords_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                _, h, w, _ = table_current.shape
                table_pretrained = table_pretrained.permute(0, 3, 1, 2)
                table_pretrained_resized = F.interpolate(
                    table_pretrained,
                    size=(h, w),
                    mode='bicubic').permute(0, 2, 3, 1).contiguous()
                state_dict[table_key] = table_pretrained_resized

            # load state_dict
            msg = self.load_state_dict(state_dict, False)
            print(msg)
            
    def get_final_norm_indexs(self, depths, interval=6):
        """
        :param depths: [2,2,42,4]
        :param interval: 6
        :return: [[], [], [1, 7, 13, 19, 25, 31, 37], [1]]
        """
        max_index = sum(depths) // interval
        indexs = [interval * i - 1 for i in range(1, max_index + 1)]

        y = [sum(depths[0:i + 1]) for i, x in enumerate(depths)]  # max index in each stage

        final_norm_indexs = []
        for i in range(4):
            k = []
            for idx in indexs:
                if idx < y[i]:
                    a = idx - y[i - 1] if i > 0 else idx
                    k.append(a)
                else:
                    break
            for i in range(len(k)):
                indexs.pop(0)
            final_norm_indexs.append(k)
        return final_norm_indexs


    def forward(self, x):
        _, _, H, W = x.shape
        x, hw_shape = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outs = []
        for index, layer in enumerate(self.layers):
            x, hw_shape, out, out_hw_shape = layer(x, hw_shape)
            if index in self.out_indices:
                norm_layer = getattr(self, f'norm{index}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[index]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs



if __name__ == '__main__':
    model = SwinTransformerV2(use_checkpoint=True).cuda()
    print(model)
    image = torch.rand(1, 3, 1024, 1024).cuda()
    outs = model(image)
    for out in outs:
        print(out.shape)