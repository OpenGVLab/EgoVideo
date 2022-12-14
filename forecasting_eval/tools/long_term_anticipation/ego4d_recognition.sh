function run(){
    
  NAME=$1
  CONFIG=$2
  shift 2;

  python -m scripts.run_lta \
    --job_name $NAME \
    --working_directory ${WORK_DIR} \
    --cfg $CONFIG \
    ${CLUSTER_ARGS} \
    DATA.PATH_TO_DATA_DIR ${EGO4D_ANNOTS} \
    DATA.PATH_PREFIX ${EGO4D_VIDEOS} \
    CHECKPOINT_LOAD_MODEL_HEAD False \
    MODEL.FREEZE_BACKBONE False \
    DATA.CHECKPOINT_MODULE_FILE_PATH "" \
    CHECKPOINT_FILE_PATH "" \
    $@
}

#-----------------------------------------------------------------------------------------------#

WORK_DIR=$1
mkdir -p ${WORK_DIR}

EGO4D_ANNOTS=$PWD/data/long_term_anticipation/annotations/
EGO4D_VIDEOS=$PWD/data/long_term_anticipation/clips/
CLUSTER_ARGS="--on_cluster NUM_GPUS 8"

# SlowFast
BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/kinetics_slowfast8x8.ckpt
run slowfast \
    configs/Ego4dRecognition/MULTISLOWFAST_8x8_R101.yaml \
    CHECKPOINT_FILE_PATH ${BACKBONE_WTS}

#-----------------------------------------------------------------------------------------------#
#                                    OTHER OPTIONS                                              #
#-----------------------------------------------------------------------------------------------#

# # MViT
# BACKBONE_WTS=$PWD/pretrained_models/long_term_anticipation/kinetics_mvit16x4.ckpt
# run mvit \
#     configs/Ego4dRecognition/MULTIMVIT_16x4.yaml \
#     DATA.CHECKPOINT_MODULE_FILE_PATH ${BACKBONE_WTS}

# # Debug locally using a smaller batch size / fewer GPUs
# CLUSTER_ARGS="NUM_GPUS 2 TRAIN.BATCH_SIZE 8 TEST.BATCH_SIZE 32"