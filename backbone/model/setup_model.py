#1b:/mnt/petrelfs/share_data/lixinhao/avp_1b_tune_audio.pt\
#base: /mnt/petrelfs/share_data/yujiashuo/model/internvid2/explore_6_3.pth
from .vision_encoder import *
import math
import logging
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from .bert.xbert import *
import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange
import time
from .config import model,TextEncoders,text_enc
from easydict import EasyDict as edict
from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed, interpolate_pos_embed_umtv2
from .bert.tokenization_bert import BertTokenizer
import io
import numpy as np

def interpolate_pos_embed_umtv2(checkpoint_model, model, orig_t_size = 4):
    # interpolate position embedding
    for pos_name in ['vision_encoder.pos_embed', 'vision_encoder.clip_pos_embed']:
        if pos_name in checkpoint_model:

            pos_embed_checkpoint = checkpoint_model[pos_name]
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # we use 8 frames for pretraining
            # new_t_size = args.num_frames * args.num_segments // model.patch_embed.tubelet_size
            new_t_size = 16
            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(orig_t_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (new_t_size))** 0.5)
            print(new_t_size,orig_size,new_size,orig_t_size,embedding_size,num_patches, num_extra_tokens)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            assert 1 < 0
            # class_token and dist_token are kept unchanged
            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size} ({pos_name})")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
                pos_tokens = pos_tokens.view(1, orig_t_size, -1, embedding_size)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, orig_t_size)
                pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=new_t_size, mode='linear')
                pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
                pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model[pos_name] = new_pos_embed
                pos_embed_checkpoint = new_pos_embed

            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size} ({pos_name})")
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model[pos_name] = new_pos_embed
    return checkpoint_model
    if 'pos_embed_spatial' in checkpoint_model or 'pos_embed_temporal' in checkpoint_model:
        raise NotImplementedError

def process_checkpoint_vision(ckpt,model):
    #import pdb
    #pdb.set_trace()
    ckpt = interpolate_pos_embed_umtv2(ckpt, model,4)
    new_ckpt = {}
    model_dict = model.state_dict()
    #for name, param in model.named_parameters():
    #    print(name)
    for k, v in ckpt.items():
        if "vision_encoder" in k:
            if 'vision_encoder.pos_embed' in k or 'vision_encoder.clip_pos_embed' in k:
                print(v.shape)
            new_k = k.replace("vision_encoder.", "")    
            new_ckpt[new_k] = v
    pretrained_dict = {k: v for k, v in new_ckpt.items() if k in model_dict}

    return new_ckpt,pretrained_dict

def process_checkpoint_audio(ckpt,model):
    ckpt_path = '/mnt/petrelfs/share_data/lixinhao/avp_1b_tune_audio.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_ckpt = {}
    model_dict = model.state_dict()

    for k, v in ckpt['module'].items():
        if "audio_encoder" in k:
            new_k = k.replace("audio_encoder.", "")    
            new_ckpt[new_k] = v   
            if new_k == 'pos_embed':
                 s
    pretrained_dict = {k: v for k, v in new_ckpt.items() if k in model_dict}

    return new_ckpt,pretrained_dict

def process_checkpoint_bert(ckpt,model):
    new_ckpt = {}
    model_dict = model.state_dict()

    for k, v in ckpt.items():
        if "text_encoder" in k:
            new_k = k.replace("text_encoder.bert.", "")    
            new_ckpt[new_k] = v    
    pretrained_dict = {k: v for k, v in new_ckpt.items() if k in model_dict}

    return new_ckpt,pretrained_dict


def get_vision_model(num_frames=4):
    from .config import model
    config = edict(model)
    
    model = PretrainVisionTransformer(
        img_size=224, 
        num_frames=num_frames,
        tubelet_size=1,
        patch_size=14, 
        embed_dim=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        use_checkpoint=False,
        checkpoint_num=40,
        use_flash_attn=True,
        use_fused_rmsnorm=True,
        use_fused_mlp=True,
        sep_image_video_pos_embed=False,

    )

    return model

def get_text_model():
    num_frames = 4
    config = edict(model)
    ####text
    text_config = edict(TextEncoders[text_enc])
    text_config_json = '/mnt/petrelfs/peibaoqi/AVION/models/pbq_umt/bert/config.json'

    try:
        text_encoder, loading_info = BertModel.from_pretrained(
            text_config.pretrained,
            config= text_config_json,
            output_loading_info=True, 
            local_files_only=True
        )
    except:
        text_encoder, loading_info = BertModel.from_pretrained(
            text_config.pretrained,
            config= text_config_json,
            output_loading_info=True, 
            local_files_only=False
        )
    
    text_tokenizer = BertTokenizer.from_pretrained(text_config.pretrained, local_files_only=False)

    return text_encoder,text_tokenizer

class EgoVideoModel(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 vision_model: nn.Module,
                 text_model: nn.Module,
                 vision_width: int = None,
                 text_width: int = None,
                 **kwargs
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model


        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        else:
            self.text_projection = None
        self.init_parameters()

    def init_parameters(self):
        if self.image_projection is not None:
            trunc_normal_(self.image_projection, std=self.vision_width ** -0.5)
        if self.text_projection is not None:
            trunc_normal_(self.text_projection, std=self.text_width ** -0.5)

    def encode_visual(self, image):
        return self.encode_image(image)

    def encode_image(self, image):
        #print(image.shape)
        b,c,t,h,w = image.shape
        x = self.visual(image)[1]
        if self.image_projection is not None:
            x = x @ self.image_projection.to(x.dtype)
        return x

    def encode_text(self, text,mask,cast_dtype=None):
        text = text.squeeze(1)
        text_output = self.textual(
            text,
            attention_mask=mask,
            return_dict=True,
            mode="text",
        )

        x = text_output.last_hidden_state
        x = x[:,0]
        if self.text_projection is not None:
            x = x @ self.text_projection.to(x.dtype)
        return x

    def forward(self, image, text,mask):
        #print(image.dtype,text.dtype)
        image_embed = self.encode_image(image)
        #print(image_embed.dtype)
        text_embed = self.encode_text(text,mask)
        return F.normalize(image_embed, dim=-1), F.normalize(text_embed, dim=-1)

def build_model(embed_dim=512,ckpt_path=None,num_frames=4,vision_width=768,text_width=1024):

    vision_model = get_vision_model(num_frames=num_frames)

    text_model,tokenizer = get_text_model()

    model = EgoVideoModel(embed_dim=embed_dim, vision_model=vision_model, text_model=text_model,vision_width=768,text_width=1024)
    
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        new_ckpt = {}
        for k in ckpt:
            new_k = k.replace('module.','')
            new_ckpt[new_k] = ckpt[k]

        msg = model.load_state_dict(new_ckpt,strict=False)
        print(msg)
    return model,tokenizer