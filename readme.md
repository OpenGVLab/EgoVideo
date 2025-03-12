# EgoVideo
Champion Solutions repository for EgoVis challenges in CVPR2024 workshop.

[Techical report](https://arxiv.org/abs/2406.18070)

## 📢News
(2025/03) 🚀Please view the latest work of our EgoVideo Backbone model, and the refined data HOD we used for the champion solutions, in our new repository of ICLR 2025 [here](https://github.com/OpenRobotLab/EgoHOD)

(2022/06/27) 🚀We release the EgoVideo Backbone model and the feature extraction script.

(2024/06/25) 🔄The repository is created.



## Catalog

- [x] Codes for Feature Extractor
- [ ] Codes for Goalsteps and NLQ
- [ ] Codes for EK-100 Action Recognition
- [ ] Codes for EK-100 MIR
- [ ] Codes for STA
- [ ] Codes for LTA
- [ ] Codes for MQ


## The EgoVideo Backbone model and Feature Extraction.
We provide the pretrained EgoVideo Backbone model and feature extraction script.
|  Model   | Google Drive | Ek100 MIR mAP | Ek100 MIR nDCG |
|  :----:  | :----:  | :----: | :----: |
| EgoVideo-4frames  | [Download](https://drive.google.com/file/d/1k6f1eRdcL17IvXtdX_J8WxNbju2Ms3AW/view?usp=sharing)| 45.9 | 38.6 |

You can check more details in the [backbone folder](./backbone).

## Downstream adaptation towards different tasks
We will gradually update the code and models for downstream adaptation on different tasks using our EgoVideo backbone.


# 🎓Citation
If this work is helpful for your research, please consider citing our technical report.
```
@article{pei2024egovideo,
  title={EgoVideo: Exploring Egocentric Foundation Model and Downstream Adaptation},
  author={Pei, Baoqi and Chen, Guo and Xu, Jilan and He, Yuping and Liu, Yicheng and Pan, Kanghua and Huang, Yifei and Wang, Yali and Lu, Tong and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2406.18070 },
  year={2024}
}
```

```
@article{chen2022ego4d,
  title={InternVideo-Ego4D: A Pack of Champion Solutions to Ego4D Challenges},
  author={Chen, Guo and Xing, Sen and Chen, Zhe and Wang, Yi and Li, Kunchang and Li, Yizhuo and Liu, Yi and Wang, Jiahao and Zheng, Yin-Dong and Huang, Bingkun and others},
  journal={arXiv preprint arXiv:2211.09529},
  year={2022}
}
```
Our previous eccv-2022 champion solutions can be found at [eccv-2022](./eccv-2022)
