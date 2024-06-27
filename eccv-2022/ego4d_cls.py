import os

import random
import decord
import numpy as np
import torch
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms
import volume_transforms as volume_transforms


class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 keep_tail_frames=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips,), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, total_frames, start_index):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds


def sample_frames_start_end(num_frames, start, end, test_mode=False):
    total_frames = end - start + 1
    sampler = SampleFrames(clip_len=1, frame_interval=1, num_clips=num_frames, test_mode=test_mode)
    frame_idxs = sampler(total_frames=total_frames, start_index=start)
    return frame_idxs


class Ego4dVerbNounClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 crop_size=224, short_side_size=256, new_height=256,
                 new_width=340, keep_aspect_ratio=True, num_segment=1,
                 num_crop=1, test_num_segment=10, test_num_crop=3, args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        annos = []
        with open(self.anno_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                annos.append(line.split(" "))
        self.annos = annos
        self.chunk_sec = 300  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        if (mode == 'train'):
            pass

        elif (mode == 'validation' or mode == "test"):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        else:
            raise NotImplementedError

    def _get_video_path(self, sample):
        video_uid = sample[0]
        video_start_sec = max(float(sample[1]), 0)
        video_end_sec = max(float(sample[2]), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(self.data_path, video_uid, str(chunk_start_id) + ".mp4")
        full_video_end_fp = os.path.join(self.data_path, video_uid, str(chunk_end_id) + ".mp4")
        # print(full_video_start_fp, flush=True)

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec
        return video_fp, video_sec, bound_sec,

    def read_frames_decord_egoclip(self, video_path_1, video_path_2, num_frames, test_mode,
                                   start_sec, end_sec, bound_sec):
        if video_path_1 == video_path_2:
            reader1 = decord.VideoReader(video_path_1)
            reader2 = reader1
            vlen1 = len(reader1)
            vlen2 = vlen1
        else:  # some clips may span two segments.
            reader1 = decord.VideoReader(video_path_1)
            reader2 = decord.VideoReader(video_path_2)
            vlen1 = len(reader1)
            vlen2 = len(reader2)

        # get indexes of sampled frames
        start_f = max(0, int(start_sec * 30))
        end_f = max(0, int(end_sec * 30))
        bound_f = int(bound_sec * 30)
        frame_idxs = sample_frames_start_end(num_frames, start_f, end_f, test_mode=test_mode)

        frames1_idx = []
        frames2_idx = []

        for index in frame_idxs:
            _index = index % (self.chunk_sec * 30)
            if index > bound_f:  # frame from the last video

                _index = min(_index, vlen2)
                frames2_idx.append(_index - 1)
            else:  # frame from the first video
                _index = min(_index, vlen1)
                frames1_idx.append(_index - 1)

        if len(frames1_idx) > 0 and len(frames2_idx) > 0:
            frames1 = reader1.get_batch(frames1_idx).asnumpy()
            frames2 = reader2.get_batch(frames2_idx).asnumpy()
            frames = np.concatenate((frames1, frames2))
        elif len(frames1_idx) <= 0 and len(frames2_idx) > 0:
            frames = reader2.get_batch(frames2_idx).asnumpy()
        elif len(frames2_idx) <= 0 and len(frames1_idx) > 0:
            frames = reader1.get_batch(frames1_idx).asnumpy()
        else:
            raise NotImplementedError

        # if frames.shape[0] < num_frames:
        #     frames = np.concatenate((frames, np.stack([frames[-1] for i in range(num_frames - frames.shape[0])])))
        return frames

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args

            sample = self.annos[index]
            video_fp, video_sec, bound_sec = self._get_video_path(sample)
            verb_label = int(sample[3])
            # verb_label = torch.LongTensor(verb_label)
            if not (os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1])):
                buffer = np.zeros((self.clip_len, self.crop_size, self.crop_size, 3)).astype(np.uint8)
            else:
                buffer = self.read_frames_decord_egoclip(video_fp[0], video_fp[1], self.clip_len,
                                                         test_mode=True if self.mode != "train" else False,
                                                         start_sec=video_sec[0], end_sec=video_sec[1],
                                                         bound_sec=bound_sec)  # T H W C

            assert len(buffer) > 0

            buffer = self._aug_frame(buffer, args)

            return buffer, verb_label, index, {}

        elif self.mode == 'validation' or self.mode == "test":
            args = self.args

            sample = self.annos[index]
            video_fp, video_sec, bound_sec = self._get_video_path(sample)
            verb_label = int(sample[3])
            # verb_label = torch.LongTensor(verb_label)
            if not (os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1])):
                buffer = np.zeros((self.clip_len, self.crop_size, self.crop_size, 3)).astype(np.uint8)
            else:
                buffer = self.read_frames_decord_egoclip(video_fp[0], video_fp[1], self.clip_len,
                                                         test_mode=True if self.mode != "train" else False,
                                                         start_sec=video_sec[0], end_sec=video_sec[1],
                                                         bound_sec=bound_sec)  # T H W C
            assert len(buffer) > 0

            buffer = self.data_transform(buffer)
            return buffer, verb_label, {}

        else:
            raise NotImplementedError

    def _aug_frame(
            self,
            buffer,
            args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.5, 1.0],  # fixme: 0.08->0.5
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = []
            tick = len(vr) / float(self.num_segment)
            all_index = list(np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segment)] +
                                      [int(tick * x) for x in range(self.num_segment)]))
            while len(all_index) < (self.num_segment * self.test_num_segment):
                all_index.append(all_index[-1])
            all_index = list(np.sort(np.array(all_index)))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        average_duration = len(vr) // self.num_segment
        all_index = []
        if average_duration > 0:
            all_index += list(
                np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration,
                                                                                                 size=self.num_segment))
        elif len(vr) > self.num_segment:
            all_index += list(np.sort(np.random.randint(len(vr), size=self.num_segment)))
        else:
            all_index += list(np.zeros((self.num_segment,)))
        all_index = list(np.array(all_index))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        return len(self.annos)


def spatial_sampling(
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


if __name__ == '__main__':
    dataset = Ego4dVideoClsDataset("/mnt/lustre/chenguo/workspace/DataProcessing/ego4d/ego4d_cls_clip.csv",
                                   "/mnt/petrelfs/chenguo/data/ego4d/all_videos_fps30_short320_chunked/",
                                   mode="validation")
    for i in dataset:
        print(i)
        exit()
    print(len(dataset))
