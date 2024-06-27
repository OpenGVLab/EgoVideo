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
from petrel_client.client import Client
from .audio.BEATs import BEATs, BEATsConfig
from .bert.tokenization_bert import BertTokenizer
import io
# data2 = client.get('s3://videos_sdd/1b_visual/00000002.npy')
# data2 = io.BytesIO(data2)
conf_path = '~/petreloss.conf'
client = Client(conf_path) # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件
ckpt_path = 'shddnew:s3://features/ego4d_ckpt/ego4d_ckpt'
ckpt = client.get(ckpt_path)
ckpt = io.BytesIO(ckpt)
ckpt = torch.load(ckpt, map_location='cpu')
ckpt = ckpt['module']
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
    num_frames = 4
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

    #_,pretrained_dict = process_checkpoint_vision(ckpt,model)

    
    #message = model.load_state_dict(pretrained_dict)
    #print(f'load vision encoder: {message}')
    # model = model.half()
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
    

    #_,pretrained_dict = process_checkpoint_bert(ckpt,text_encoder)
    #text_model_dict = text_encoder.state_dict()
    #text_model_dict.update(pretrained_dict)
    #message = text_encoder.load_state_dict(text_model_dict)
    #print(print(f'load text encoder: {message}'))
    # for name, param in text_encoder.named_parameters():
    #    print(name,param.shape)
    # for name, p in text_encoder.named_parameters():
    #     if 'encoder' not in name:
    #         # print(f"freeze {name}")
    #         p.requires_grad = False
    #     elif 'encoder' in name:
    #         if '21' in name or '22' in name or '23' in name:
    #             print(f"Unfreeze {name}")
    #         else:
    #             p.requires_grad = False

    text_tokenizer = BertTokenizer.from_pretrained(text_config.pretrained, local_files_only=False)
    # for name, p in text_encoder.named_parameters():
    #     logger.info(f"Freeze {name}")
    #     p.requires_grad = False

    return text_encoder,text_tokenizer

if __name__ == '__main__':

    ckpt_path = '/mnt/petrelfs/share_data/lixinhao/avp_1b_tune_audio.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 4
    config = edict(model)

    model = PretrainVisionTransformer(
        img_size=224, 
        num_frames=4,
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
        checkpoint_num=0,
        use_flash_attn=True,
        use_fused_rmsnorm=True,
        use_fused_mlp=True,
        sep_image_video_pos_embed=False,

    )
    model = model.to('cuda')
    model.half()
    model.eval()

    
    _,pretrained_dict = process_checkpoint_vision(ckpt,model)

    
    message = model.load_state_dict(pretrained_dict)
    print(f'load vision encoder: {message}')

    audio_cfg_path = 'audio/BEATs_iter3_plus_AS2M.pt'
    checkpoint = torch.load(audio_cfg_path, map_location="cpu")
    audio_cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(audio_cfg)
    BEATs_model = BEATs_model.to('cuda')
    BEATs_model.half()
    BEATs_model.eval()
    _,pretrained_dict = process_checkpoint_audio(ckpt,BEATs_model)
    message = BEATs_model.load_state_dict(pretrained_dict)
    print(print(f'load audio encoder: {message}'))

    ####text
    text_config = edict(TextEncoders[text_enc])
    text_config_json = 'bert/config.json'

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
    

    _,pretrained_dict = process_checkpoint_bert(ckpt,text_encoder)
    text_model_dict = text_encoder.state_dict()
    text_model_dict.update(pretrained_dict)
    message = text_encoder.load_state_dict(text_model_dict)
    print(print(f'load text encoder: {message}'))

    vision_encoder = model
    audio_encoder = BEATs_model
    text_encoder = text_encoder.to('cuda').half().eval()
    text_tokenizer = BertTokenizer.from_pretrained(text_config.pretrained, local_files_only=False)
    import pdb
    pdb.set_trace()
    text = ["a ?", "a dog", "a cat"]

    text = text_tokenizer(text, return_tensors="pt")
    text_output = text_encoder(            
    text.input_ids.to('cuda'),
    attention_mask=text.attention_mask.to('cuda'),
    return_dict=True,
    mode="text",
    )

    text_embeds = text_output.last_hidden_state
    pooled_text_embeds = text_embeds[:, 0]

    





