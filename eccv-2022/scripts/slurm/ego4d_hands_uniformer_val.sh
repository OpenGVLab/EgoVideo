export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='./workdir/ego4d_hands_uniformer_base' # work dir
DATA_PATH='not use' # it is not to use
MODEL_PATH='MODEL'# the pre-loaded weights for fine-tuning or validation

JOB_NAME=$1
PARTITION=${PARTITION:-"PARTITION"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-14}
SRUN_ARGS=${SRUN_ARGS:-"--quotatype=reserved"}
PY_ARGS=${@:2}

# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        ${SRUN_ARGS} \
        python -u     run_ego4d_hands.py \
    --model uniformer_base_256_ego4d_finetune \
    --nb_verb_classes 20 \
    --nb_noun_classes 0 \
    --data_set ego4d_hands \
    --data_path ${DATA_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --num_sample 1 \
    --num_segments 8 \
    --finetune ${MODEL_PATH} \
    --warmup_epochs  1 \
    --input_size 320 \
    --short_side_size 320 \
    --save_ckpt_freq 1 \
    --num_frames 16 \
    --layer_decay 1. \
    --opt adamw \
    --no_auto_resume \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 10 \
    --eval \
    --test_num_segment 1 \
    --test_num_crop 1 \
    --test_subset val \
#    --enable_deepspeed \
#    --disable_eval_during_finetuning \
    #    --finetune ${MODEL_PATH} \