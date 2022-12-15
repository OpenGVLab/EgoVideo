## change this file path to your path
checkpoint_file_path="your/path/of/checkpoint"
results_file_path="your/path/of/result/val_results.json"
object_detections_path="your/path/of/box/file.json"
rgb_lmdb_dir="your/path/of/data/"
annotations_dir="annotations/"
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
     CHECKPOINT_FILE_PATH ${checkpoint_file_path} \
     RESULTS_JSON ${results_file_path} \
     CHECKPOINT_LOAD_MODEL_HEAD True \
     CHECKPOINT_VERSION "pytorch" \
     TEST.BATCH_SIZE 1 \
     NUM_GPUS 1 \
     EGO4D_STA.OBJ_DETECTIONS ${object_detections_path} \
     EGO4D_STA.ANNOTATION_DIR ${annotations_dir} \
     EGO4D_STA.RGB_LMDB_DIR ${rgb_lmdb_dir} \
     EGO4D_STA.TEST_LISTS "['fho_sta_val.json']"