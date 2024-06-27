_base_ = [
    '../_base_/datasets/ego4d_scod_detection.py',
    '../_base_/default_runtime.py'
]
pretrained = 'pretrained/uniformer_large_ls_tl_384.pth'
model = dict(
    type='DINO',
    pretrained=pretrained,
    backbone=dict(
        type='UniFormer',
        embed_dim=[128, 192, 448, 640],
        layers=[5, 10, 24, 7],
        head_dim=64,
        drop_path_rate=0.3,
        use_checkpoint=True,
        checkpoint_num=[5, 10, 24, 7],
        windows=False,
        hybrid=True,
        window_size=14,
        pretrained=pretrained
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 192, 448, 640],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    bbox_head=dict(
        type='DINOHead',
        num_query=900,
        num_classes=1,
        in_channels=2048,  # TODO
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=1000)),
        transformer=dict(
            type='DinoTransformer',
            two_stage_num_proposals=900,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        num_levels=5,
                        embed_dims=256,
                        dropout=0.0),  # 0.1 for DeformDETR
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),  # 0.1 for DeformDETR
                        dict(
                            type='MultiScaleDeformableAttention',
                            num_levels=5,
                            embed_dims=256,
                            dropout=0.0),  # 0.1 for DeformDETR
                    ],
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))  # TODO: Originally 100
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',
         img_scale=[(2000, 800), (2000, 1600)],
         multiscale_mode='range',
         keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2000, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(filter_empty_gt=True, pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
# lr_config = dict(policy='step', step=[11])
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=3,
    save_last=True,
)
evaluation = dict(save_best='auto')
# EMA
# resume_from = None
# custom_hooks = [
#     dict(
#         type='ExpMomentumEMAHook',
#         resume_from=resume_from,
#         momentum=0.0001,
#         priority=49)
# ]
