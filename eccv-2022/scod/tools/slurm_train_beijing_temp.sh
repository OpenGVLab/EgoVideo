#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))

export NCCL_IB_HCA=mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

CONFIG=$1
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}


srun -N 4 \
     --gres=gpu:8 \
     --ntasks-per-node=8 \
     --cpus-per-task=14 \
     --qos=gpugpu \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
