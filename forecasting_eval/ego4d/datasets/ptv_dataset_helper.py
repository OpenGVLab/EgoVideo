from __future__ import annotations
import gc

import json
import logging
import os
import collections
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr

from pytorchvideo.data.clip_sampling import ClipSampler, ClipInfo
from pytorchvideo.data.video import VideoPathHandler
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler

logger = logging.getLogger(__name__)


class LabeledVideoDataset(torch.utils.data.IterableDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav",
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            decode_audio (bool): If True, also decode audio from video.

            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder
        self.path_handler = VideoPathHandler()

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clips = None
        self._next_clip_start_time = 0.0

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        video_index = next(self._video_sampler_iter)

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            try:
                video_path, info_dict = self._labeled_videos[video_index]
                video = self.path_handler.video_from_path(
                    video_path,
                    decode_audio=self._decode_audio,
                    decoder=self._decoder,
                )
            except Exception as e:
                logger.debug(
                    "Failed to load video with error: {}; trial {}".format(e, i_try)
                )
                continue

            clips = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            if not isinstance(clips, list):
                clips = [clips]

            decoded_clips = []
            video_is_null = False
            for clip_start, clip_end, clip_index, aug_index, is_last_clip in clips:
                clip = video.get_clip(clip_start, clip_end)
                video_is_null = clip is None or clip["video"] is None
                if video_is_null:
                    break
                decoded_clips.append(clip)

            self._next_clip_start_time = clip_end

            if is_last_clip or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                video.close()
                self._next_clip_start_time = 0.0

                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()

                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue


            if len(decoded_clips) == 1:
                frames = decoded_clips[0]["video"]
                audio_samples = decoded_clips[0]["audio"]
            else:
                clip_frames = [
                    uniform_temporal_subsample(x["video"], num_samples=64)
                    for x in decoded_clips
                ]
                frames = torch.stack(clip_frames, dim=0)

                clip_audio = [x["audio"] for x in decoded_clips]
                audio_samples = None
                if None not in clip_audio:
                    audio_samples = torch.stack(clip_audio, dim=0)

            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


def labeled_video_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """
    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset



class UntrimmedClipSampler:
    """
    A wrapper for adapting untrimmed annotated clips from the json_dataset to the
    standard `pytorchvideo.data.ClipSampler` expected format. Specifically, for each
    clip it uses the provided `clip_sampler` to sample between "clip_start_sec" and
    "clip_end_sec" from the json_dataset clip annotation.
    """

    def __init__(self, clip_sampler: ClipSampler) -> None:
        """
        Args:
            clip_sampler (`pytorchvideo.data.ClipSampler`): Strategy used for sampling
                between the untrimmed clip boundary.
        """
        self._trimmed_clip_sampler = clip_sampler

    def __call__(
        self, last_clip_time: float, video_duration: float, clip_info: Dict[str, Any]
    ) -> ClipInfo:
        clip_start_boundary = clip_info["clip_start_sec"]
        clip_end_boundary = clip_info["clip_end_sec"]
        duration = clip_end_boundary - clip_start_boundary

        # Sample between 0 and duration of untrimmed clip, then add back start boundary.
        clip_info = self._trimmed_clip_sampler(last_clip_time, duration, clip_info)
        return ClipInfo(
            clip_info.clip_start_sec + clip_start_boundary,
            clip_info.clip_end_sec + clip_start_boundary,
            clip_info.clip_index,
            clip_info.aug_index,
            clip_info.is_last_clip,
        )


class ForecastingClipSampler:
    def __init__(self, clip_sampler: ClipSampler) -> None:
        self._trimmed_clip_sampler = clip_sampler

    def __call__(
        self, last_clip_time: float, video_duration: float, clip_info: Dict[str, Any]
    ) -> List[ClipInfo]:
        clip_infos = []
        for input_clip in clip_info["input_clips"]:
            clip_start_boundary = input_clip["clip_start_sec"]
            clip_end_boundary = input_clip["clip_end_sec"]
            duration = clip_end_boundary - clip_start_boundary

            # Sample between 0 and duration of untrimmed clip, then add back start boundary.
            clip_info = self._trimmed_clip_sampler(last_clip_time, duration, clip_info)
            clip_infos.append(
                ClipInfo(
                    clip_info.clip_start_sec + clip_start_boundary,
                    clip_info.clip_end_sec + clip_start_boundary,
                    clip_info.clip_index,
                    clip_info.aug_index,
                    clip_info.is_last_clip,
                )
            )
        return clip_infos


def clip_recognition_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
):
    
    assert os.path.exists(data_path), 'Please run data/parse_ego4d_json.py first. Will change this later'

    if g_pathmgr.isfile(data_path):
        try:
            with g_pathmgr.open(data_path, "r") as f:
                annotations = json.load(f)
        except Exception:
            raise FileNotFoundError(f"{data_path} must be json for Ego4D dataset")

        # LabeledVideoDataset requires the data to be list of tuples with format:
        # (video_paths, annotation_dict). For recognition, the annotation_dict contains
        # the verb and noun label, and the annotation boundaries.
        untrimmed_clip_annotations = []
        for entry in annotations:
            untrimmed_clip_annotations.append(
                (
                    os.path.join(video_path_prefix, f'{entry["clip_uid"]}.mp4'),
                    {
                        "clip_start_sec": entry['action_clip_start_sec'],
                        "clip_end_sec": entry['action_clip_end_sec'],
                        "noun_label": entry['noun_label'],
                        "verb_label": entry['verb_label'],
                        "action_idx": entry['action_idx'],
                    },
                )
            )
    else:
        raise FileNotFoundError(f"{data_path} not found.")

    dataset = LabeledVideoDataset(
        untrimmed_clip_annotations,
        UntrimmedClipSampler(clip_sampler),
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset

def clip_forecasting_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    num_input_actions: int,
    num_future_actions: int,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
):
    if g_pathmgr.isfile(data_path):
        try:
            with g_pathmgr.open(data_path, "r") as f:
                entries = json.load(f)['clips']
        except Exception as e:
            raise FileNotFoundError(f"{data_path} must be json for Ego4D dataset. {e}")

        # if entries do not have verb/noun labels (test set) then give dummy ones
        for entry in entries:
            if 'verb_label' not in entry:
                entry.update({'verb_label': -1, 'noun_label': -1})

        # rename keys for pytorchvideo
        for entry in entries:
            entry.update({
                'clip_start_sec': entry.pop('action_clip_start_sec'),
                'clip_end_sec': entry.pop('action_clip_end_sec'),
            })


        # group annotations by clip_uid
        annotations = collections.defaultdict(list)
        for entry in entries:
            annotations[entry['clip_uid']].append(entry)

        # Sort windows by their PNR frame (windows can overlap, but PNR is distinct)
        annotations = {
            clip_uid: sorted(annotations[clip_uid], key=lambda x: x['action_idx'])
            for clip_uid in annotations
        }

        # LabeledVideoDataset requires the data to be list of tuples with format:
        # (video_paths, annotation_dict). For forecasting, annotation_dict contains
        # the input boundaries to be decoded, any observed clip annotations within
        # those boundaries, and a list of num_future_actions clip annotations (including
        # labels and boundaries).
        untrimmed_clip_annotations = []
        for clip_uid, video_clips in annotations.items():
            video_path = os.path.join(video_path_prefix, f'{clip_uid}.mp4')
            if len(video_clips) <= 0:
                continue

            # Extract forecasting annotations from video clips.
            for i in range(
                len(video_clips) - num_future_actions - num_input_actions
            ):
                input_clips = copy.deepcopy(video_clips[i : i + num_input_actions])
                forecast_clips = copy.deepcopy(
                    video_clips[
                        i
                        + num_input_actions : i
                        + num_input_actions
                        + num_future_actions
                    ]
                )
                untrimmed_clip_annotations.append(
                    (
                        video_path,
                        {
                            "input_clips": input_clips,
                            "forecast_clips": forecast_clips,
                        },
                    )
                )
    else:
        raise FileNotFoundError(f"{data_path} not found.")
    
    dataset = LabeledVideoDataset(
        untrimmed_clip_annotations,
        ForecastingClipSampler(clip_sampler),
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset



