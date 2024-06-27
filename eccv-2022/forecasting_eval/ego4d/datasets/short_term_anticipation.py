from hashlib import new
import io
import os
import time
from os.path import join
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms.functional import uniform_temporal_subsample
import decord
import lmdb
import imutils
from tqdm import tqdm
import json
from pathlib import Path
from ego4d.evaluation.sta_metrics import compute_iou
import cv2
import torch
from decord import VideoReader
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale,
)
from ego4d.utils.parser import load_config, parse_args
from ego4d.utils import logging as logging
from ego4d.utils import datasets_utils as utils
from ego4d.utils import transform as transform
from ego4d.datasets import cv2_transform

logger = logging.get_logger(__name__)
decord.bridge.set_bridge("torch")

# trim module
from fractions import Fraction
from typing import Iterable, List
import av
import numpy as np


def pts_difference_per_frame(fps: Fraction, time_base: Fraction) -> int:
    r"""
    utility method to determine the difference between two consecutive video
    frames in pts, based off the fps and time base.
    基于 fps 和时基确定两个连续视频帧之间的差异的实用方法。
    """
    pt = (1 / fps) * (1 / time_base)
    assert pt.denominator == 1, "should be whole number"
    return int(pt)


def frame_index_to_pts(frame: int, start_pt: int, diff_per_frame: int) -> int:
    """
    given a frame number and a starting pt offset, compute the expected pt for the frame.
    Frame is assumed to be an index (0-based)
    """
    return start_pt + frame * diff_per_frame


def pts_to_time_seconds(pts: int, base: Fraction) -> Fraction:
    """
    converts a pt to the time (in seconds)

    returns:
        a Fraction (assuming the base is a Fraction)
    """
    return pts * base


def _get_frames_pts(
        video_pts_set: List[int],
        # pyre-fixme[11]: Annotation `Container` is not defined as a type.
        container: av.container.Container,
        include_audio: bool,
        include_additional_audio_pts: int,
        # pyre-fixme[11]: Annotation `Frame` is not defined as a type.
) -> Iterable[av.frame.Frame]:
    """
    Gets the video/audio frames from a container given:

    Inputs:
        video_pts_set
            the set of video pts to retrieve
        container
            the container to get the pts from
        include_audio
            Determines whether to ignore the audio stream in the first place
        include_additional_audio_pts
            Additional amount of time to include for audio frames.

            pts must be relative to video base
    """
    assert len(container.streams.video) == 1

    min_pts = min(video_pts_set)
    max_pts = max(video_pts_set)
    # pyre-fixme[9]: video_pts_set has type `List[int]`; used as `Set[int]`.
    video_pts_set = set(video_pts_set)  # for O(1) lookup

    video_stream = container.streams.video[0]
    fps: Fraction = video_stream.average_rate
    video_base: Fraction = video_stream.time_base
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    # [start, end) time
    clip_start_sec = pts_to_time_seconds(min_pts, video_base)
    clip_end_sec = pts_to_time_seconds(max_pts, video_base)

    # add some additional time for audio packets
    clip_end_sec += max(
        pts_to_time_seconds(include_additional_audio_pts, video_base), 1 / fps
    )

    # --- setup
    streams_to_decode = {"video": 0}
    if (
            include_audio
            and container.streams.audio is not None
            and len(container.streams.audio) > 0
    ):
        assert len(container.streams.audio) == 1
        streams_to_decode["audio"] = 0
        audio_base: Fraction = container.streams.audio[0].time_base

    # seek to the point we need in the video
    # with some buffer room, just in-case the seek is not precise
    seek_pts = max(0, min_pts - 2 * video_pt_diff)
    container.seek(seek_pts, stream=video_stream)
    if "audio" in streams_to_decode:
        assert len(container.streams.audio) == 1
        audio_stream = container.streams.audio[0]
        # pyre-fixme[61]: `audio_base` may not be initialized here.
        audio_seek_pts = int(seek_pts * video_base / audio_base)
        audio_stream.seek(audio_seek_pts)

    # --- iterate over video

    # used for validation
    previous_video_pts = None
    previous_audio_pts = None

    yielded_frames = 0
    for frame in container.decode(**streams_to_decode):

        if isinstance(frame, av.AudioFrame):
            assert include_audio
            # ensure frames are in order
            assert previous_audio_pts is None or previous_audio_pts < frame.pts
            previous_audio_pts = frame.pts

            # pyre-fixme[61]: `audio_base` may not be initialized here.
            audio_time_sec = pts_to_time_seconds(frame.pts, audio_base)

            # we want all the audio frames in this region
            if audio_time_sec >= clip_start_sec and audio_time_sec < clip_end_sec:
                yield frame
            elif audio_time_sec >= clip_end_sec:
                break

        elif isinstance(frame, av.VideoFrame):
            video_time_sec = pts_to_time_seconds(frame.pts, video_base)
            if video_time_sec >= clip_end_sec:
                break

            # ensure frames are in order
            assert previous_video_pts is None or previous_video_pts < frame.pts

            if frame.pts in video_pts_set:
                # check that the frame is in range
                assert (
                        video_time_sec >= clip_start_sec and video_time_sec < clip_end_sec
                ), f"""
                video frame at time={video_time_sec} (pts={frame.pts})
                out of range for time [{clip_start_sec}, {clip_end_sec}]
                """

                yield frame
                yielded_frames += 1

    if yielded_frames < len(video_pts_set):
        for _ in range(len(video_pts_set) - yielded_frames):
            yield None


def _get_frames(
        video_frames: List[int],
        container: av.container.Container,
        include_audio: bool,
        audio_buffer_frames: int = 0,
) -> Iterable[av.frame.Frame]:
    assert len(container.streams.video) == 1

    video_stream = container.streams.video[0]
    video_start: int = video_stream.start_time
    video_base: Fraction = video_stream.time_base
    fps: Fraction = video_stream.average_rate
    video_pt_diff = pts_difference_per_frame(fps, video_base)

    audio_buffer_pts = (
        frame_index_to_pts(audio_buffer_frames, 0, video_pt_diff)
        if include_audio
        else 0
    )

    time_pts_set = [
        frame_index_to_pts(f, video_start, video_pt_diff) for f in video_frames
    ]
    frames = list(_get_frames_pts(time_pts_set, container, include_audio, audio_buffer_pts))
    assert len(frames) == len(video_frames)
    return frames


class PyAVVideoReader(object):
    def __init__(self, path_to_video, include_audio=False, audio_buffer_frames=0, height=None):
        self.path_to_video = path_to_video
        self.include_audio = include_audio
        self.audio_buffer_frames = audio_buffer_frames
        self.height = height
        self.ceph_client = Client("~/petreloss.conf")

    def __getitem__(self, frame_list):
        if isinstance(frame_list, (int, float)):
            frame_list = [int(frame_list)]
        elif not isinstance(frame_list, (list, tuple)):
            frame_list = [int(f) for f in frame_list]
        else:
            frame_list = list(frame_list)

        # video = self.ceph_client.get(self.path_to_video)
        # if video is None:
        #     new_path_to_video = os.path.join("cluster2:s3://tad_datasets/ego4d/fho_sta_missing_video_full_scale/",
        #                                      os.path.basename(self.path_to_video))
        #     print("change root:", new_path_to_video)
        #     video = self.ceph_client.get(new_path_to_video)
        #     if video is None:
        #         print("not find 2:",new_path_to_video)
        #
        # assert video is not None
        # video = io.BytesIO(video)
        with av.open(self.path_to_video) as input_video:
            frames = _get_frames(frame_list, input_video, include_audio=self.include_audio,
                                 audio_buffer_frames=self.audio_buffer_frames)
            frames = list(frames)
        frames = [f.to_ndarray(format="bgr24") if f is not None else None for f in frames]
        if self.height is not None:
            frames = [imutils.resize(f, height=self.height) if f is not None else None for f in frames]
        return frames


class Ego4DHLMDB():
    def __init__(self, path_to_root: Path, readonly=False, lock=False,
                 frame_template="{video_id:s}_{frame_number:010d}", map_size=1099511627776) -> None:
        self.environments = {}
        self.path_to_root = path_to_root
        if isinstance(self.path_to_root, str):
            self.path_to_root = Path(self.path_to_root)
        self.path_to_root.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly
        self.lock = lock
        self.map_size = map_size
        self.frame_template = frame_template

    def _get_parent(self, parent: str) -> lmdb.Environment:
        return lmdb.open(str(self.path_to_root / parent), map_size=self.map_size, readonly=self.readonly,
                         lock=self.lock)

    def put_batch(self, video_id: str, frames: List[int], data: List[np.ndarray]) -> None:
        with self._get_parent(video_id) as env:
            with env.begin(write=True) as txn:
                for frame, value in zip(frames, data):
                    if value is not None:
                        txn.put(self.frame_template.format(video_id=video_id, frame_number=frame).encode(),
                                cv2.imencode('.jpg', value)[1])

    def put(self, video_id: str, frame: int, data: np.ndarray) -> None:
        if data is not None:
            with self._get_parent(video_id) as env:
                with env.begin(write=True) as txn:
                    txn.put(self.frame_template.format(video_id=video_id, frame_number=frame).encode(),
                            cv2.imencode('.jpg', data)[1])

    def get(self, video_id: str, frame: int) -> np.ndarray:
        with self._get_parent(video_id) as env:
            with env.begin(write=False) as txn:
                data = txn.get(self.frame_template.format(video_id=video_id, frame_number=frame).encode())

                file_bytes = np.asarray(
                    bytearray(io.BytesIO(data).read()), dtype=np.uint8
                )
                return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    def get_batch(self, video_id: str, frames: List[int]) -> List[np.ndarray]:
        out = []
        with self._get_parent(video_id) as env:
            with env.begin() as txn:
                for frame in frames:
                    data = txn.get(self.frame_template.format(video_id=video_id, frame_number=frame).encode())
                    file_bytes = np.asarray(
                        bytearray(io.BytesIO(data).read()), dtype=np.uint8
                    )
                    out.append(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
            return out

    def get_existing_keys(self):
        keys = []
        for parent in self.path_to_root.iterdir():
            with self._get_parent(parent.name) as env:
                with env.begin() as txn:
                    keys += list(txn.cursor().iternext(values=False))
        return keys



class Ego4dShortTermAnticipation(torch.utils.data.Dataset):
    """
    Ego4d Short Term Anticipation Dataset
    """

    def __init__(self, cfg, split):
        # Only support train and val mode.

        self.cfg = cfg              ### 配置文件
        self._split = split         ### train/val/test
        self._sample_rate = cfg.DATA.SAMPLING_RATE  ### 采样频率1，2，4
        self._video_length = cfg.DATA.NUM_FRAMES    ### 采样多少帧
        self._seq_len = self._video_length * self._sample_rate  ### 采样部分的视频总长度
        self._num_classes = cfg.MODEL.NUM_CLASSES   ### 动作种类
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.EGO4D_STA.BGR   ### 默认不适用BGR
        self._fps = cfg.DATA.TARGET_FPS     ### 30
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE          ### 224
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.EGO4D_STA.TRAIN_USE_COLOR_AUGMENTATION   ## default False
            self._pca_jitter_only = cfg.EGO4D_STA.TRAIN_PCA_JITTER_ONLY         ## False
            self._pca_eigval = cfg.EGO4D_STA.TRAIN_PCA_EIGVAL                   ## [0.225, 0.224, 0.229]
            self._pca_eigvec = cfg.EGO4D_STA.TRAIN_PCA_EIGVEC                   ## 
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.EGO4D_STA.TEST_FORCE_FLIP

        if self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == 'lmdb':
            self._hlmdb = Ego4DHLMDB(self.cfg.EGO4D_STA.RGB_LMDB_DIR, readonly=True, lock=False)

        self._obj_detections = json.load(open(cfg.EGO4D_STA.OBJ_DETECTIONS))

        self._load_data(cfg)

    def _load_lists(self, _list):
        def extend_dict(input_dict, output_dict):
            for k, v in input_dict.items():
                output_dict[k] = v
            return output_dict

        res = {
            'videos': {},
            'annotations': []
        }
        for l in _list:
            j = json.load(open(os.path.join(self.cfg.EGO4D_STA.ANNOTATION_DIR, l)))
            res['videos'] = extend_dict(j['info']['video_metadata'], res['videos'])
            res['annotations'] += j['annotations']

        return res

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """

        if self._split == "train":
            self._annotations = self._load_lists(cfg.EGO4D_STA.TRAIN_LISTS)
        elif self._split == "val":
            self._annotations = self._load_lists(cfg.EGO4D_STA.VAL_LISTS)
        else:
            self._annotations = self._load_lists(cfg.EGO4D_STA.TEST_LISTS)

    def __len__(self):
        return len(self._annotations['annotations'])

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]
        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )
            if self.random_horizontal_flip:
                # random flip
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)
            ]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError("Unsupported split mode {}".format(self._split))

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(imgs, self._crop_size, boxes=boxes)

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs, min_size=self._crop_size, max_size=self._crop_size, boxes=boxes
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs, min_size=self._crop_size, max_size=self._crop_size, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError("{} split not supported yet!".format(self._split))

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if self._use_bgr:
            # Convert image format from RGB to BGR.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(boxes, self._crop_size, self._crop_size)

        return imgs, boxes

    def _load_frames_decord(self, video_filename, frame_number, fps):
        assert frame_number > 0

        vr = VideoReader(video_filename, height=320, width=568)

        frames = frame_number - np.arange(
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE,
            step=self.cfg.DATA.SAMPLING_RATE,
        )[::-1]
        frames[frames < 1] = 1

        frames = frames.astype(int)

        video_data = vr.get_batch(frames).permute(3, 0, 1, 2)
        return video_data

    def _load_frames_pyav(self, video_filename, frame_number, fps):
        assert frame_number > 0

        vr = PyAVVideoReader(video_filename, height=320)

        frames = (
                frame_number
                - np.arange(
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE,
            step=self.cfg.DATA.SAMPLING_RATE,
        )[::-1]
        )
        frames[frames < 1] = 1

        frames = frames.astype(int)

        imgs = vr[frames]

        return imgs

    def _load_frames_pytorch_video(self, video_filename, frame_number, fps):
        clip_duration = (
                                self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE - 1
                        ) / fps
        clip_end_sec = frame_number / fps
        clip_start_sec = clip_end_sec - clip_duration

        # truncate if negative timestamp
        clip_start_sec = np.min(clip_start_sec, 0)

        video = EncodedVideo.from_path(video_filename, decode_audio=False)
        video_data = video.get_clip(clip_start_sec, clip_end_sec)["video"]
        video_data = uniform_temporal_subsample(video_data, self.cfg.DATA.NUM_FRAMES)
        # video_data = short_side_scale(video_data, )
        return video_data

    def _retry_load_images_lmdb(self, video_id, frames, retry=10, backend="pytorch"):
        """
        This function is to load images with support of retrying for failed load.

        Args:
            keys (list): paths of images needed to be loaded.
            retry (int, optional): maximum time of loading retrying. Defaults to 10.
            backend (str): `pytorch` or `cv2`.

        Returns:
            imgs (list): list of loaded images.
        """
        for i in range(retry):
            imgs = []
            imgs = self._hlmdb.get_batch(video_id, frames)

            if all(img is not None for img in imgs):
                if backend == "pytorch":
                    imgs = torch.as_tensor(np.stack(imgs))
                return imgs
            else:
                logger.warn("Reading failed. Will retry.")
                time.sleep(1.0)
            if i == retry - 1:
                raise Exception("Failed to load frames from video {}: {}".format(video_id, frames))

    def _sample_frames(self, frame):
        frames = (
                frame
                - np.arange(
            self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE,
            step=self.cfg.DATA.SAMPLING_RATE,
        )[::-1]
        )       ###采样采的是后面几帧
        frames[frames < 0] = 0      ## 小于0的id按0处理

        frames = frames.astype(int)

        return frames

    def _load_annotations(self, idx):
        # get the idx-th annotation
        ann = self._annotations['annotations'][idx]
        uid = ann['uid']

        # get video_id, frame_number, gt_boxes, gt_noun_labels, gt_verb_labels and gt_ttc_targets
        video_id = ann["video_id"]
        frame_number = ann['frame']

        if 'objects' in ann:
            gt_boxes = np.vstack([x['box'] for x in ann['objects']])
            gt_noun_labels = np.array([x['noun_category_id'] for x in ann['objects']])
            gt_verb_labels = np.array([x['verb_category_id'] for x in ann['objects']])
            gt_ttc_targets = np.array([x['time_to_contact'] for x in ann['objects']])
        else:
            gt_boxes = gt_noun_labels = gt_verb_labels = gt_ttc_targets = None

        frame_width, frame_height = self._annotations['videos'][video_id]['frame_width'], \
                                    self._annotations['videos'][video_id]['frame_height']

        fps = self._annotations['videos'][video_id]['fps']

        return uid, video_id, frame_width, frame_height, frame_number, fps, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets

    def _load_detections(self, uid):
        # get the object detections for the current example
        object_detections = self._obj_detections[uid]

        if len(object_detections) > 0:
            pred_boxes = np.vstack([x['box'] for x in object_detections])
            pred_scores = np.array([x['score'] for x in object_detections])
            pred_object_labels = np.array([x['noun_category_id'] for x in object_detections])

            # exclude detections below the theshold
            detected = (
                    pred_scores
                    >= self.cfg.EGO4D_STA.DETECTION_SCORE_THRESH
            )

            pred_boxes = pred_boxes[detected]
            pred_object_labels = pred_object_labels[detected]
            pred_scores = pred_scores[detected]
        else:
            pred_boxes = np.zeros((0, 4))
            pred_scores = pred_object_labels = np.array([])

        return pred_boxes, pred_object_labels, pred_scores

    def _load_frames(self, video_id, frame_number, fps):
        if self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == 'pytorchvideo':
            frames = self._load_frames_pytorch_video(join(self.cfg.EGO4D_STA.VIDEO_DIR, video_id + '.mp4'),
                                                     frame_number, fps)
        elif self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == 'decord':
            frames = self._load_frames_decord(join(self.cfg.EGO4D_STA.VIDEO_DIR, video_id + '.mp4'), frame_number, fps)
        elif self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == 'pyav':
            frames = self._load_frames_pyav(join(self.cfg.EGO4D_STA.VIDEO_DIR, video_id + '.mp4'), frame_number, fps)
        elif self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND == 'lmdb':
            # sample the list of frames in the clip
            # key_list = self._sample_frame_keys(video_id, frame_number)
            frames_list = self._sample_frames(frame_number)
            # # retrieve frames
            frames = self._retry_load_images_lmdb(
                video_id, frames_list, backend="cv2"
            )
        return frames

    def _preprocess_frames_and_boxes(self, frames, boxes):
        if self.cfg.EGO4D_STA.VIDEO_LOAD_BACKEND in ['pytorchvideo', "decord"]:
            video_tensor = frames.permute(1, 0, 2, 3)

            video_tensor, boxes = self._images_and_boxes_preprocessing(
                video_tensor, boxes=boxes
            )

            # T C H W -> C T H W.
            video_tensor = video_tensor.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            video_tensor, boxes = self._images_and_boxes_preprocessing_cv2(
                frames, boxes=boxes
            )
        return video_tensor, boxes

    def getanitem(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            uid: the unique id of the annotation
            imgs: the frames sampled from the video
            pred_boxes: the list of boxes detected in the current frame. These are in the resolution of the input example.
            verb_label: the verb label associated to the current frame
            ttc_target: the ttc target
            extra_data: a dictionary containing extra data fields:
                'orig_pred_boxes': boxes at the original resolution
                'pred_object_scores': associated prediction scores
                'pred_object_labels': associated predicted object labels
                'gt_detections': dictionary containing the ground truth predictions for the current frame
        """
        uid, video_id, frame_width, frame_height, frame_number, fps, gt_boxes, gt_noun_labels, gt_verb_labels, gt_ttc_targets = self._load_annotations(
            idx)
        pred_boxes, pred_object_labels, pred_scores = self._load_detections(uid)
        frames = self._load_frames(video_id, frame_number, fps)

        orig_pred_boxes = pred_boxes.copy()
        nn = np.array([frame_width, frame_height] * 2).reshape(1, -1)
        pred_boxes /= nn
        if gt_boxes is None:  # unlabeled example
            print('unlabeled data pack')
            orig_norm_boxes = pred_boxes.copy()
            video_tensor, pred_boxes = self._preprocess_frames_and_boxes(frames, pred_boxes)
            imgs = utils.pack_pathway_output(self.cfg, video_tensor)
    
            extra_data = {
                'orig_pred_boxes': orig_pred_boxes,
                'pred_object_scores': pred_scores,
                'pred_object_labels': pred_object_labels
            }

            return uid, imgs, pred_boxes, np.array([]), np.array([]), orig_norm_boxes, extra_data
        else:
            orig_gt_boxes = gt_boxes.copy()
            gt_boxes = gt_boxes.astype(np.float64)
            gt_boxes /= nn

            # put all boxes together
            all_boxes = np.vstack([gt_boxes, pred_boxes])
            orig_norm_boxes = all_boxes.copy()

            video_tensor, all_boxes = self._preprocess_frames_and_boxes(frames, all_boxes)

            # separate ground truth from predicted boxes after pre-processing
            gt_boxes = all_boxes[: len(gt_boxes)]
            orig_norm_gt_boxes = orig_norm_boxes[:len(gt_boxes)]
            pred_boxes = all_boxes[len(gt_boxes):]
            orig_norm_pred_boxes = orig_norm_boxes[len(gt_boxes):]
            
            if self._split == 'train' and self.cfg.EGO4D_STA.GT_ONLY:
                pred_boxes = np.concatenate([gt_boxes])
                orig_pred_boxes = np.concatenate([orig_gt_boxes])
                orig_norm_pred_boxes = np.concatenate([orig_norm_gt_boxes])
                pred_object_labels = np.concatenate([gt_noun_labels])
                pred_scores = np.concatenate([np.ones_like(gt_noun_labels)])
            elif self._split == 'train' and self.cfg.EGO4D_STA.PROPOSAL_APPEND_GT:
                pred_boxes = np.concatenate([pred_boxes, gt_boxes])
                orig_pred_boxes = np.concatenate([orig_pred_boxes, orig_gt_boxes])
                orig_norm_pred_boxes = np.concatenate([orig_norm_pred_boxes,orig_norm_gt_boxes])
                pred_object_labels = np.concatenate([pred_object_labels, gt_noun_labels])
                pred_scores = np.concatenate([pred_scores, np.ones_like(gt_noun_labels)])

            # match predicted boxes to ground truth
            # compute IOU values
            ious = compute_iou(pred_boxes, gt_boxes)

            # get the indexes of the largest IOU - these are the matches
            matches = ious.argmax(-1)  # index of the matched gt_box for each pred_box

            # get the largest IOU for each predicted box
            ious = ious.max(-1)

            next_active_labels = (ious >= self.cfg.EGO4D_STA.NAO_IOU_THRESH)

            gt_detections = {
                "boxes": orig_gt_boxes,
                "nouns": gt_noun_labels,
                "verbs": gt_verb_labels,
                "ttcs": gt_ttc_targets,
            }

            imgs = utils.pack_pathway_output(self.cfg, video_tensor)

            # copy the verb labels of the matched boxes
            verb_labels = gt_verb_labels[matches]

            # set verb label to ignore index for not next active objects
            verb_labels[next_active_labels == False] = -100

            # copy ttc targets of the matched boxes
            ttc_targets = gt_ttc_targets[matches]

            # set ttc targets related to non next-active objects to NaN
            # non_idx = np.where(next_active_labels == False)[0]
            # for i in non_idx:
            #     ttc_targets[i] = np.NaN
            # ttc_targets = np.array(ttc_targets, dtype=np.float64)
            try:
                ttc_targets[next_active_labels == False] = np.NaN
            except ValueError:
                ttc_targets = np.array(ttc_targets, dtype=np.float64)
                ttc_targets[next_active_labels == False] = np.NaN

            extra_data = {
                'orig_pred_boxes': orig_pred_boxes,
                'pred_object_scores': pred_scores,
                'pred_object_labels': pred_object_labels,
                'gt_detections': gt_detections
            }

            return (
                uid,
                imgs,
                pred_boxes,
                verb_labels,
                ttc_targets,
                orig_norm_pred_boxes,
                extra_data
            )
    def __getitem__(self,idx):
        data = self.getanitem(idx)
        if np.isnan(data[4]).all() and self._split == 'train':
            return self.getanitem(0)
        else:
            return data
if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    dataset = Ego4dShortTermAnticipation(cfg,'test')
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        print(len(data))