
from .setup_model import *

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
batch_size = 32
max_txt_l = 40

# ========================= model ==========================
text_enc = "bert_large"
model = dict(
    vision_encoder=dict(
        # backbone
        name="pretrain_umt2_6b_patch14_224",
        img_size=224, 
        num_frames=4,
        tubelet_size=1,
        patch_size=14, 
        d_model=3200,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        pretrained="/mnt/petrelfs/share_data/likunchang/model/um_teacher/umt2/vit_6B_2M_CLIP+MAE_300e_pt.pth",
        use_checkpoint=True,
        checkpoint_num=48,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        # clip teacher
        clip_teacher="internvl_clip_6b",
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        # mask
        video_mask_type="random",
        video_mask_ratio=0.8,
        video_double_mask_ratio=0.,
        image_mask_type="attention",
        image_mask_ratio=0.5,
        image_double_mask_ratio=0.,
        sep_image_video_pos_embed=False,
        keep_temporal=False,
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    contra_dim=768,
    av_concat_dim=768,
    temp=0.07,
    find_unused_parameters=True,
    freeze_vision=True,
    freeze_audio=False,
    predictor_dropout=0.5,
    hidden_layer=512,
    predictor_class=50,
)

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.05,
    max_grad_norm=3.,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=1, min_lr_multi=0.01, warmup_epochs=0.2)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=False,  # offload gpu tensors to cpu to save memory.
)

use_half_precision = True
use_bf16 = True

gradient_checkpointing = True # for text encoder
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False
VisionEncoders = dict()
VisionEncoders["beit"] = dict(
    name="beit_base",
    pretrained="microsoft/beit-base-patch16-224-pt22k-ft22k",
    d_model=768,
)
VisionEncoders["beit_large"] = dict(
    name="beit_large",
    pretrained="microsoft/beit-large-patch16-224-pt22k-ft22k",
    d_model=1024,
)

TextEncoders = dict()
TextEncoders["bert"] = dict(
    name="bert_base",
    pretrained="bert-base-uncased",
    config="configs/config_bert.json",
    d_model=768,
    fusion_layer=9,
)
TextEncoders["bert_fusion6"] = dict(
    name="bert_base_fusion6",
    pretrained="bert-base-uncased",
    config="configs/config_bert_fusion6.json",
    d_model=768,
    fusion_layer=6,
)
TextEncoders["bert_large"] = dict(
    name="bert_large",
    pretrained="bert-large-uncased",
    config="configs/config_bert_large.json",
    d_model=1024,
    fusion_layer=19,
)
TextEncoders["med_bert"] = dict(
    name="med_bert_base",
    pretrained="bert-base-uncased",
    config="configs/med_config.json",
    d_model=768,
)
TextEncoders["med_bert_freq2"] = dict(
    name="med_bert_base_freq2",
    pretrained="bert-base-uncased",
    config="configs/med_config_freq2.json",
    d_model=768,
)
TextEncoders["med_bert_freq2_must"] = dict(
    name="med_bert_base_freq2_must",
    pretrained="bert-base-uncased",
    config="configs/med_config_freq2_must.json",
    d_model=768,
)

TextEncoders["med_bert_fusion10"] = dict(
    name="med_bert_base_fusion",
    pretrained="bert-base-uncased",
    config="configs/med_config_fusion.json",
    d_model=768,
    fusion_layer=10
)
TextEncoders["med_bert_fusion9"] = dict(
    name="med_bert_base_fusion",
    pretrained="bert-base-uncased",
    config="configs/med_config_fusion.json",
    d_model=768,
    fusion_layer=9
)
TextEncoders["med_bert_fusion6"] = dict(
    name="med_bert_base_fusion",
    pretrained="bert-base-uncased",
    config="configs/med_config_fusion.json",
    d_model=768,
    fusion_layer=6
)
TextEncoders["med_bert_fusion0"] = dict(
    name="med_bert_base_fusion",
    pretrained="bert-base-uncased",
    config="configs/med_config_fusion.json",
    d_model=768,
    fusion_layer=0
)
TextEncoders["med_bert_fusion3"] = dict(
    name="med_bert_base_fusion",
    pretrained="bert-base-uncased",
    config="configs/med_config_fusion.json",
    d_model=768,
    fusion_layer=3
)
TextEncoders["med_bert_large"] = dict(
    name="med_bert_large",
    pretrained="bert-base-uncased", # not a bug, it just follows BLIP.
    config="configs/med_large_config.json",
    d_model=768
)
