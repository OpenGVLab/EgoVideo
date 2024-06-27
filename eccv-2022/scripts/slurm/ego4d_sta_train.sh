export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export PATH=/mnt/lustre/share/gcc/gcc-5.3.0/bin/:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gmp-4.3.2/lib:/mnt/lustre/share/gcc/mpfr-2.4.2/lib:/mnt/lustre/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
export PATH=/mnt/cache/share/cuda-11.3/bin:$PATH
export PATH=/mnt/cache/xingsen/.local/bin:$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/cuda-11.3/lib64:$LD_LIBRARY_PATH
rm -rf /mnt/lustre/xingsen/.cache/torch_extensions
# DATA_PATH does not need to be modified
DATA_PATH='YOUR_PATH/list_kinetics-400'

# The following three parameters need to be replaced with your directory
MODEL_PATH='/mnt/petrelfs/share_data/chenguo/ego_forecasting/pretrained_models/vitl_v_f.pt'
OUTPUT_DIR='/mnt/petrelfs/xingsen/videoMAE_ckp/chenguo_inference_verb_test'
CONFIG_FILE='/mnt/cache/xingsen/videoMAE_kinetics/VideoMAE_ava/short_term_config/VIT3D.yaml'
PARTITION=${PARTITION:-"PARTITION"}
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}
srun -p video \
     --gres=gpu:${GPUS_PER_NODE} \
     --ntasks=${GPUS} \
     --ntasks-per-node=${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --kill-on-bad-exit=1 \
     --quotatype=reserved \
     ${SRUN_ARGS} \
     python -u run_ego4d_sta.py \
      --model sta_vit_large_patch16_224 \
      --data_path ${DATA_PATH} \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 4 \
      --num_sample 1 \
      --input_size 224 \
      --save_ckpt_freq 1 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 0.00025 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 30 \
      --warmup_epochs 0 \
      --data_set "ava" \
      --val_freq 30 \
      --drop_path 0.2 \
      --enable_deepspeed \
      --cfg ${CONFIG_FILE} \
      TRAIN.ENABLE True TEST.ENABLE False ENABLE_LOGGING False \
      ${PY_ARGS}