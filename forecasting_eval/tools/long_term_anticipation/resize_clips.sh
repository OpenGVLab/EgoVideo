#!/bin/bash
#SBATCH --job-name=resize_ego4d_clips
#SBATCH --array=0-1720%200
#SBATCH --output=slurm/log_%A_%a.out
#SBATCH --error=slurm/log_%A_%a.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 1
#SBATCH --mem=10GB
#SBATCH --time=1:00:00
#SBATCH --signal=USR1@600
#SBATCH --partition=<slurm partition>
#SBATCH --requeue

source ~/.bashrc
cd /path/to/LTA/repo

EGO4D_CLIP_DIR=data/long_term_anticipation/clips_hq/
TARGET_DIR=data/long_term_anticipation/clips/
mkdir -p ${TARGET_DIR}

readarray -d '' CLIPS < <(find ${EGO4D_CLIP_DIR} -name "*.mp4" -print0)
IFS=$'\n' CLIPS=($(sort <<<"${CLIPS[*]}"))
unset IFS

CLIP=${CLIPS[${SLURM_ARRAY_TASK_ID}]}
FILENAME=`basename $CLIP`

ffmpeg -y -i $CLIP \
	-c:v libx264 \
	-crf 28 \
	-vf "scale=320:320:force_original_aspect_ratio=increase,pad='iw+mod(iw,2)':'ih+mod(ih,2)'" \
	-an ${TARGET_DIR}/${FILENAME}
