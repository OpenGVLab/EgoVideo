import itertools
import os

import torch
import torch.utils.data
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import (
    Compose,
    Lambda,
)

from .build import DATASET_REGISTRY
from . import ptv_dataset_helper
from ..utils import logging, video_transformer

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Ego4dRecognition(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" else "random"
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)

        mode_ = 'test_unannotated' if mode=='test' else mode
        data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')
        
        self.dataset = ptv_dataset_helper.clip_recognition_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            video_sampler=sampler,
            decode_audio=False,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
        )
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):
        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(lambda x: x/255.0),
                            Normalize(cfg.DATA.MEAN, cfg.DATA.STD),
                        ]
                        + video_transformer.random_scale_crop_flip(mode, cfg)
                        + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        torch.tensor([x["verb_label"], x["noun_label"]]),
                        str(x["video_name"]) + "_" + str(x["video_index"]),
                        {},
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos


@DATASET_REGISTRY.register()
class Ego4dLongTermAnticipation(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Ego4d ".format(mode)

        sampler = RandomSampler
        if cfg.SOLVER.ACCELERATOR != "dp" and cfg.NUM_GPUS > 1:
            sampler = DistributedSampler

        clip_sampler_type = "uniform" if mode == "test" else "random"
        clip_duration = (
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        ) / self.cfg.DATA.TARGET_FPS
        clip_sampler = make_clip_sampler(clip_sampler_type, clip_duration)

        mode_ = 'test_unannotated' if mode=='test' else mode
        # [!!]
        if mode == 'test' and cfg.TEST.EVAL_VAL:
            mode_ = 'val'
        data_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, f'fho_lta_{mode_}.json')

        self.dataset = ptv_dataset_helper.clip_forecasting_dataset(
            data_path=data_path,
            clip_sampler=clip_sampler,
            num_input_actions=self.cfg.FORECASTING.NUM_INPUT_CLIPS,
            num_future_actions=self.cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT,
            video_sampler=sampler,
            decode_audio=False,
            transform=self._make_transform(mode, cfg),
            video_path_prefix=self.cfg.DATA.PATH_PREFIX,
        )
        self._dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(self.dataset), 2)
        )

    @property
    def sampler(self):
        return self.dataset.video_sampler

    def _make_transform(self, mode: str, cfg):
        class ReduceExpandInputClips:
            def __init__(self, transform):
                self.transform = transform

            def __call__(self, x):
                if x.dim() == 4:
                    x = x.unsqueeze(0)  # Handle num_clips=1

                n, c, t, h, w = x.shape
                x = x.transpose(0, 1)
                x = x.reshape(c, n * t, h, w)
                x = self.transform(x)

                if isinstance(x, list):
                    for i in range(len(x)):
                        c, _, h, w = x[i].shape
                        x[i] = x[i].reshape(c, n, -1, h, w)
                        x[i] = x[i].transpose(1, 0)
                else:
                    c, _, h, w = x.shape
                    x = x.reshape(c, n, t, h, w)
                    x = x.transpose(1, 0)

                return x

        def extract_forecast_labels(x):
            clips = x["forecast_clips"]
            nouns = torch.tensor([y["noun_label"] for y in clips])
            verbs = torch.tensor([y["verb_label"] for y in clips])
            labels = torch.stack([verbs, nouns], dim=-1)
            return labels

        def extract_observed_labels(x):
            clips = x["input_clips"]
            nouns = torch.tensor([y["noun_label"] for y in clips])
            verbs = torch.tensor([y["verb_label"] for y in clips])
            labels = torch.stack([verbs, nouns], dim=-1)
            return labels

        # last visible annotated clip: (clip_uid + action_idx)
        def extract_clip_id(x):
            last_clip = x['input_clips'][-1]
            return f'{last_clip["clip_uid"]}_{last_clip["action_idx"]}'

        def extract_forecast_times(x):
            clips = x["forecast_clips"]
            start_end = [(y["clip_start_sec"], y["clip_end_sec"]) for y in clips]
            return {"label_clip_times": start_end}

        return Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            ReduceExpandInputClips(
                                Compose(
                                    [
                                        Lambda(lambda x: x/255.0),
                                        Normalize(cfg.DATA.MEAN, cfg.DATA.STD)
                                    ]
                                    + video_transformer.random_scale_crop_flip(
                                        mode, cfg
                                    )
                                    + [video_transformer.uniform_temporal_subsample_repeated(cfg)]
                                )
                            ),
                        ]
                    ),
                ),
                Lambda(
                    lambda x: (
                        x["video"],
                        extract_forecast_labels(x),
                        extract_observed_labels(x),
                        extract_clip_id(x),
                        extract_forecast_times(x),
                    )
                ),
            ]
        )

    def __getitem__(self, index):
        value = next(self._dataset_iter)
        return value

    def __len__(self):
        return self.dataset.num_videos

