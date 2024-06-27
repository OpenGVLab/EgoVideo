from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as cp
from modeling_finetune import Block, PatchEmbed, get_sinusoid_encoding_table


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 verb_num_classes=118,
                 noun_num_classes=582,
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
                 use_mean_pooling=True):
        super().__init__()
        self.verb_num_classes = verb_num_classes
        self.noun_num_classes = noun_num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)

        assert verb_num_classes > 0 or noun_num_classes > 0

        if self.verb_num_classes > 0 and self.noun_num_classes > 0:
            self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
            self.head = nn.Linear(embed_dim, verb_num_classes)
            self.fc_norm2 = norm_layer(embed_dim) if use_mean_pooling else None
            self.head2 = nn.Linear(embed_dim, noun_num_classes)
        elif self.verb_num_classes > 0 and self.noun_num_classes <= 0:
            self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
            self.head = nn.Linear(embed_dim, verb_num_classes)
            self.head2 = nn.Identity()
            self.fc_norm2 = None
        elif self.verb_num_classes <= 0 and self.noun_num_classes > 0:
            self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
            self.head = nn.Linear(embed_dim, noun_num_classes)
            self.head2 = nn.Identity()
            self.fc_norm2 = None
        else:
            raise NotImplementedError

        # self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim,
        #                       verb_num_classes + noun_num_classes) if verb_num_classes + noun_num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        if not isinstance(self.head, nn.Identity):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)
        if not isinstance(self.head2, nn.Identity):
            trunc_normal_(self.head2.weight, std=.02)
            self.head2.weight.data.mul_(init_scale)
            self.head2.bias.data.mul_(init_scale)

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print(x.shape)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            # print(self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach().shape)
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def head_forward(self, x):
        if self.fc_norm is not None and self.fc_norm2 is not None:
            x = x.mean(1)
            x1 = self.fc_norm(x)
            x2 = self.fc_norm2(x)
            x1 = self.head(x1)
            x2 = self.head2(x2)
            x = torch.cat((x1, x2), dim=1)
            return x
        elif self.fc_norm is None and self.fc_norm2 is not None:
            x = x.mean(1)
            x = self.fc_norm2(x)
            x = self.head2(x)
            return x
        elif self.fc_norm is not None and self.fc_norm2 is None:
            x = x.mean(1)
            x = self.fc_norm(x)
            x = self.head(x)
            return x
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head_forward(x)
        return x




@register_model
def vit_tiny_patch16_224_ego4d(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_224_ego4d(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_320_ego4d(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, img_size=320,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224_ego4d(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384_ego4d(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, img_size=384,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_448_ego4d(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, img_size=448,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_huge_patch16_224_ego4d(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
