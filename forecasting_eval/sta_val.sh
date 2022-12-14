srun -p video \
     -n1 \
     --gres=gpu:1 \
     --ntasks-per-node 1 \
     --quotatype=reserved \
     python scripts/run_sta.py \
     --cfg configs/Ego4dShortTermAnticipation/VIT3D.yaml \
     TRAIN.ENABLE False \
     TEST.ENABLE True \
     ENABLE_LOGGING False \
     CHECKPOINT_FILE_PATH /mnt/petrelfs/xingsen/videoMAE_ckp/chenguo_inference_verb/checkpoint-10/mp_rank_00_model_states.pt \
     RESULTS_JSON short_term_anticipation/results/which_is_best_box/test_1000_t5.json \
     CHECKPOINT_LOAD_MODEL_HEAD True \
     CHECKPOINT_VERSION "pytorch" \
     TEST.BATCH_SIZE 1 \
     NUM_GPUS 1 \
     EGO4D_STA.OBJ_DETECTIONS short_term_anticipation/data/object_detections.json \
     EGO4D_STA.ANNOTATION_DIR annotations/ \
     EGO4D_STA.RGB_LMDB_DIR /mnt/petrelfs/share_data/chenguo/ego_forecasting/short_term_anticipation/data/lmdb/ \
     EGO4D_STA.TEST_LISTS "['fho_sta_val.json']"