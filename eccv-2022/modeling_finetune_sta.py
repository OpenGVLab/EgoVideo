from functools import partial
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from dataclasses import dataclass
from detectron2.layers import ROIAlign


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
        self.ttc_projection = nn.Linear(sum(dim_in),1,bias=True)

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

    def forward(self, inputs, bboxes, box_embed):
        if bboxes.shape[0]==0: # handle cases in which zero boxes are passed as input
            x_verb = torch.zeros((0, self.verb_projection.out_features))
            return x_verb
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
        x_ttc = self.ttc_projection(x+box_embed.type_as(x))
        act_idx = 0 if self.training else 1

        if self.verb_act[act_idx] is not None:
            x_verb = self.verb_act[act_idx](x_verb)
        if self.ttc_act[act_idx] is not None:
            x_ttc = self.ttc_act[act_idx](x_ttc)

        return x_verb,x_ttc


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
                    num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        # B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

@dataclass
class ROIPoolingCfg:
    POOLER_RESOLUTION: int = 7
    POOLER_SCALE: float = 0.0625
    POOLER_SAMPLING_RATIO: int = 0
    POOLER_TYPE: str = 'align3d'
    MEAN_BEFORE_POOLER: bool = True

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)



class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=80,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches  # 8x14x14
        self.grid_size = [img_size//patch_size, img_size//patch_size]  # [14,14]
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.box_pos_embed = get_sinusoid_encoding_table(img_size*img_size, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])

        # self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.norm = norm_layer(embed_dim)  # 这一项是预训练权重中没有的
        self.fc_norm = None
        self.ROIHEAD = ResNetSTARoIHead(
            dim_in=[embed_dim],
            num_verbs=cfg.MODEL.NUM_VERBS,
            pool_size=[
                [
                    all_frames // tubelet_size,
                    1,
                    1,
                ]
            ],
            resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
            scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            verb_act_func=(None, cfg.MODEL.HEAD_VERB_ACT),
            ttc_act_func=(cfg.MODEL.HEAD_TTC_ACT,)*2,
            aligned=cfg.DETECTION.ALIGNED,
        )
        # self.verb_act = [self.get_act(x) for x in (None, cfg.MODEL.HEAD_VERB_ACT)]
        # self.ttc_act = [self.get_act(x) for x in (cfg.MODEL.HEAD_TTC_ACT,)*2]
        # trunc_normal_(self.ROIHEAD.box2ttc.weight,std=.02)
        trunc_normal_(self.ROIHEAD.verb_projection.weight,std=.02)

        # rois setting
        self.head_cfg = ROIPoolingCfg()
        resolution = self.head_cfg.POOLER_RESOLUTION
        self.max_pooler = nn.MaxPool2d((resolution, resolution))

        self.test_ext = (0.1, 0.05)
        self.proposal_per_clip = 100

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, proposals, orig_norm_pred_boxes):
        x = self.patch_embed(x)
        B, width, t, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)

        # B, _, _ = x.size()
        if self.pos_embed is not None:  # 预测阶段插值
            # positional_embedding=[1 8*14*14 768]->[1 8*16*29 768]
            pos_embed = self.pos_embed.reshape(t, -1, width)
            pos_embed = interpolate_pos_embed_online(
                pos_embed, self.grid_size, [h, w], 0).reshape(1, -1, width)
            x = x + pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [b thw=8x14x14 c=768]
        # if self.fc_norm is not None:  # default
        box_embed = self.get_box_pos_embed(proposals)
        # box_embed = None
        x = x.reshape(B, t, h, w, -1).permute(0, 4, 1, 2, 3)  # [b c t h w]
        x_verb,x_ttc = self.ROIHEAD([x], proposals, box_embed)
        return x_verb,x_ttc

    def get_box_pos_embed(self,proposals):
        float_box = proposals[:,1:].clone().to(proposals.device)
        num_boxes,_ = float_box.shape
        # float_box[:,0] = float_box[:,0]*224 + float_box[:,1]
        # float_box[:,1] = float_box[:,2]*224 + float_box[:,3]
        # float_box = float_box[:,0:2]
        float_box = float_box.long().reshape(-1)
        box_embed = self.box_pos_embed[0][float_box].clone().detach()
        box_embed = box_embed.reshape(num_boxes,4,-1)
        box_embed = torch.sum(box_embed,dim=1).to(proposals.device)
        return box_embed
        

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


    def forward(self, x, boxes, orig_norm_pred_boxes, orig_pred_boxes=None, pred_object_labels=None, pred_object_scores=None):
        proposals = self.pack_boxes(boxes)
        orig_norm_pred_boxes = torch.cat(orig_norm_pred_boxes,0).half()
        pred_verbs,pred_ttcs = self.forward_features(x, proposals, orig_norm_pred_boxes)
        if self.training:
            return pred_verbs,pred_ttcs
        else:
            assert pred_object_labels is not None
            assert pred_object_scores is not None
            lengths = [len(x) for x in boxes]  # number of boxes per entry
            pred_verbs = pred_verbs.split(lengths, 0)
            pred_ttcs = pred_ttcs.split(lengths, 0)
            # compute detections and return them
            return self.postprocess(orig_pred_boxes, pred_object_labels, pred_object_scores, pred_verbs, pred_ttcs)



@register_model
def sta_vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def sta_vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def sta_vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def sta_vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


def interpolate_pos_embed_online(
    pos_embed, orig_size: Tuple[int], new_size: Tuple[int], num_extra_tokens: int
):
    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]
    embedding_size = pos_tokens.shape[-1]
    pos_tokens = pos_tokens.reshape(
        -1, orig_size[0], orig_size[1], embedding_size
    ).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=new_size, mode="bicubic", align_corners=False,
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    return new_pos_embed

