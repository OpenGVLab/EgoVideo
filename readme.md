# ego4d-eccv2022-solutions
It is our solutions repository for Ego4D challenges in ECCV2022 workshop.

[Techical report](https://arxiv.org/abs/2211.09529)

[Ego4D Slides (in Chinese)](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/blob/main/assets/VideoIntern_Ego4D.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvideo-ego4d-a-pack-of-champion/state-change-object-detection-on-ego4d)](https://paperswithcode.com/sota/state-change-object-detection-on-ego4d?p=internvideo-ego4d-a-pack-of-champion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvideo-ego4d-a-pack-of-champion/moment-queries-on-ego4d)](https://paperswithcode.com/sota/moment-queries-on-ego4d?p=internvideo-ego4d-a-pack-of-champion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvideo-ego4d-a-pack-of-champion/short-term-object-interaction-anticipation-on)](https://paperswithcode.com/sota/short-term-object-interaction-anticipation-on?p=internvideo-ego4d-a-pack-of-champion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvideo-ego4d-a-pack-of-champion/future-hand-prediction-on-ego4d)](https://paperswithcode.com/sota/future-hand-prediction-on-ego4d?p=internvideo-ego4d-a-pack-of-champion)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/internvideo-ego4d-a-pack-of-champion/natural-language-queries-on-ego4d)](https://paperswithcode.com/sota/natural-language-queries-on-ego4d?p=internvideo-ego4d-a-pack-of-champion)

## 📢News

(2022/12/11) 🚀🚀We release code and checkpoints of pretraining, FHP task and SCOD task.

(2022/12/01) 🚀The VideoMAE features for MQ and NLQ are released.

(2022/11/17) 🔄The repository is created.



## Catalog

- [] Codes for Feature Extractor
- [x] Verb Noun Features (VideoMAE-L) for MQ and NLQ
- [x] Codes for pretraining
- [x] Codes for STA
- [x] Codes for Hands
- [x] Codes and checkpoints for SCOD



## Video Features for MQ and NLQ.
We provide the video features extracted by VideoMAE-L pretrained on verb and noun subset.

|  Feature   | Baidu Netdisk | Zenodo |
|  ----  | ----  | ----  |
| MQ(verb)  | [Download](https://pan.baidu.com/s/1yYRVJmSrUAjrI7EmbUoqPA). code: sxda|[Download](https://zenodo.org/record/7340838) |
| NLQ(verb)  | [Download](https://pan.baidu.com/s/1Q3CHJyV1Onq8skH3xu6XLg). code: teod |[Download](https://zenodo.org/record/7343075)|
| NLQ(noun)  | [Download](https://pan.baidu.com/s/1aspOwXDTMlzpOUkLiIrZFg). code: wrop |[Download](https://zenodo.org/record/7343178) |

You can check more details in our [techical report](https://arxiv.org/abs/2211.09529).


## Pretraining.
Our training strategy is based on the vanilla method and is easy to follow. We use [VideoMAE](https://github.com/MCG-NJU/VideoMAE) codebase for training and validation. Before training, you have to follow it to install the python environment. We split the training annotations filtered by [EgoVLP](https://github.com/showlab/EgoVLP) for rapid development. The second-filtered annotations files are available [here](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/tag/1.0.0). We release the checkpoints in the below table.

|  Method   | Pretrain | Resolution |Subset |Top-1 |Top-5 |Weights |
|  ----  | ----  | ----  |  ----  | ----  | ----  | ----  |
| VideoMAE-L  | K700| 224x224| verb | 52.51 | 86.05 | [Download](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/ego4d_verb_pretrain_vitl_k700.pt) | 
| VideoMAE-L  | K700|224x224 |noun |33.41 | 85.51 | [Download](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/ego4d_noun_pretrain_vitl_k700.pt) | 
| UniFormer-B | K600|320x320 |verb |49.30 | 83.61 | [Download](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/ego4d_verb_uniformer_base_16x320_k600_ep9.pt) | 


### Training
We provide the training script on SLURM mode. If you want to use PyTorch-DDP mode, you can use scripts in `scripts/pytorch_ddp`.

```
bash scripts/slurm/ego4d_verb_slurm_pretrain_vitl_k400.sh
```

In the script, you need to set the approaiate `OUTPUT_DIR` and `MODEL_PATH`.



## STA.
### Training
We use the ViT-Large model to train the STA task.
```bash
sh scripts/slurm/sta_train.sh
```

### Validation
```bash
cd forecasting_eval
sh sta_val.sh
```

## FHP
### Training
We train the FHP task using [Uniformer-B](https://github.com/Sense-X/UniFormer) and the weights pretrained on Ego4D verb subset.
We provide the training script on SLURM mode. If you want to use PyTorch-DDP mode, you can use scripts in `scripts/pytorch_ddp`.

```
bash scripts/slurm/ego4d_hands_uniformer.sh
```

In the script, you need to set the approaiate `OUTPUT_DIR` and `MODEL_PATH`.

### Validation
We also provide the script for validation and testing. You can launch the script below to validate a specific checkpoint's performance.

```
bash scripts/slurm/ego4d_hands_uniformer_val.sh
```

In the script, you need to set the approaiate `OUTPUT_DIR`, `MODEL_PATH`, `--test_subset` and `--test_num_segment`.

## SCOD

Our detection code for SCOD is developed on top of [MMDetection](./scod).

We report the performance on the validation set and release the checkpoint in the below table.

|  Method   | Pretrain | Resolution |AP | AP50 |AP75 | Config | Download | 
|  ----  | ----  | ----  |  ----  | ----  | ----  | ----  | ---- |
| UniFormer-L | [IN-1K](https://drive.google.com/file/d/174rcA6rNzYVG9Ya9ik-NwTGoxW1M79ez/view?usp=sharing) | 800-1600/2000 |24.8 |44.2 | 24.0 | [config](scod/configs/scod/dino_5scale_uniformer-l_8x2_12e_scod_imagenet1k.py) |[ckpt](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/dino_5scale_uniformer-l_8x2_12e_scod_imagenet1k.pth) \| [log](https://github.com/OpenGVLab/ego4d-eccv2022-solutions/releases/download/1.0.0/dino_5scale_uniformer-l_8x2_12e_scod_imagenet1k.txt) | 

To train UniFormer-L + DINO on the SCOD training set with 8 gpus for 12 epochs:

```shell
sh tools/dist_train.sh configs/scod/dino_5scale_uniformer-l_8x2_12e_scod_imagenet1k.py 8
```

To test UniFormer-L + DINO on the SCOD validation set with 8 gpus:

```shell
sh tools/dist_test.sh configs/scod/dino_5scale_uniformer-l_8x2_12e_scod_imagenet1k.py <ckpt-path> 8 --eval bbox
```

It should give:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.248
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.442
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.240
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.075
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.054
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.697
```


# 🎓Citation
If this work is helpful for your research, please consider citing our techical report.
```
@article{chen2022ego4d,
  title={InternVideo-Ego4D: A Pack of Champion Solutions to Ego4D Challenges},
  author={Chen, Guo and Xing, Sen and Chen, Zhe and Wang, Yi and Li, Kunchang and Li, Yizhuo and Liu, Yi and Wang, Jiahao and Zheng, Yin-Dong and Huang, Bingkun and others},
  journal={arXiv preprint arXiv:2211.09529},
  year={2022}
}
```
