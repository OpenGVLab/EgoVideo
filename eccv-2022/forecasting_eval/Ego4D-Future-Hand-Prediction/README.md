# Ego4D Hand Movement Prediction Baseline

## Installation:
Our method requires the same dependencies as SlowFast. We refer to the official implementation fo [SlowFast](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) for installation details.

## Data Preparation:

**Input**: 60 frames before PRE-1.5s frame (p3). See the definition in paper I-1.1  

**Output**: 5 frames with hand positions on {p3,p2,p1,p,c}; left/right hand position format: x_l, y_l, x_r, y_r

**Note on Ground Truth**: In the dataloader, we choose pad zeros when hand ground truth is not available.

The resulting data should be organized as following:
- datafolder:
  - rootfolder:
    - cropped_videos_ant: folder that contains all recaled input videos with height 256
    - train/val/test/trainval.jsons: json files for each split. (trainval contains all samples from training and validation set)

## Training: 
```shell
 python tools/run_net.py --cfg /path/to/ego4d-hand_ant/configs/Ego4D/I3D_8x8_R50.yaml OUTPUT_DIR /path/to/ego4d-hand_ant/output/
```

## Testing: 
```shell
 python tools/run_net.py --cfg /path/to/ego4d-hand_ant/configs/Ego4D/I3D_8x8_R50.yaml OUTPUT_DIR /path/to/ego4d-hand_ant/output/ TRAIN.ENABLE False
```

## Evaluation: 
```shell
 python tools/eval_functions.py /path/to/ego4d-hand_ant/output/output.pkl 30
```
