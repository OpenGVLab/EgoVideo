export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='./workdir/ego4d_hands_uniformer_base' # work dir
DATA_PATH='not use' # it is not to use
MODEL_PATH='MODEL'# the pre-loaded weights for fine-tuning or validation


# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
        --master_port 12320 --nnodes=8 \
        --node_rank=0 --master_addr=$ip_node_0 \
        run_ego4d_verb_cls_pretrain.py \
    --model vit_large_patch16_224_ego4d \
    --nb_noun_classes 0 \
    --data_set ego4d_verb \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 16 \
    --num_sample 1 \
    --warmup_epochs  1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 10 \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --enable_deepspeed