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
from iopath.common.file_io import g_pathmgr
import cv2
import json


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


class Ego4dHandsDataset(Dataset):
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

        path_to_file = self.anno_path
        # assert g_pathmgr.exists(self.anno_path), "{} dir not found".format(
        #     path_to_file
        # )
        self._path_to_ant_videos = []
        self._labels = []
        self._labels_masks = []
        self._spatial_temporal_idx = []
        self._clip_names = []
        f = open(path_to_file)
        # data = json.load(f)
        data = dict(list(json.load(f).items())[5:])
        f.close()
        for clip_id, hand_dicts in data.items():
            for hand_annot in hand_dicts:
                clip_id = hand_annot['clip_id']
                for annot in hand_annot['frames']:
                    pre45_frame = annot['pre_45']['frame']
                    clip_name = str(clip_id) + '_' + str(pre45_frame - 1)
                    # label = []
                    # label_mask = []
                    self._clip_names.append(clip_name)
                    x = os.path.join(self.data_path, 'cropped_clips', clip_name + '.mp4')
                    assert os.path.exists(x)
                    self._path_to_ant_videos.append(
                        os.path.join(self.data_path, 'cropped_clips', clip_name + '.mp4')
                    )
                    # placeholder for the 1x20 hand gt vector (padd zero when GT is not available)
                    # 5 frames have the following order: pre_45, pre_40, pre_15, pre, contact
                    # GT for each frames has the following order: left_x,left_y,right_x,right_y
                    label = [0.0] * 20
                    label_mask = [0.0] * 20
                    # for frame_type, frame_annot in hand_annot.items():
                    for frame_type, frame_annot in annot.items():
                        # if frame_type in ['start_sec', 'end_sec','height', 'width']:
                        if frame_type in ["action_start_sec", "action_end_sec", "action_start_frame",
                                          "action_end_frame", "action_clip_start_sec", "action_clip_end_sec",
                                          "action_clip_start_frame", "action_clip_end_frame"]:
                            continue
                        # frame_gt = frame_annot[1]
                        if len(frame_annot) == 2:
                            continue
                        frame_gt = frame_annot['boxes']
                        if frame_type == 'pre_45':
                            for single_hand in frame_gt:
                                if 'left_hand' in single_hand:
                                    label_mask[0] = 1.0
                                    label_mask[1] = 1.0
                                    label[0] = single_hand['left_hand'][0]
                                    label[1] = single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[2] = 1.0
                                    label_mask[3] = 1.0
                                    label[2] = single_hand['right_hand'][0]
                                    label[3] = single_hand['right_hand'][1]
                        if frame_type == 'pre_30':
                            for single_hand in frame_gt:
                                if 'left_hand' in single_hand:
                                    label_mask[4] = 1.0
                                    label_mask[5] = 1.0
                                    label[4] = single_hand['left_hand'][0]
                                    label[5] = single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[6] = 1.0
                                    label_mask[7] = 1.0
                                    label[6] = single_hand['right_hand'][0]
                                    label[7] = single_hand['right_hand'][1]
                        if frame_type == 'pre_15':
                            for single_hand in frame_gt:
                                if 'left_hand' in single_hand:
                                    label_mask[8] = 1.0
                                    label_mask[9] = 1.0
                                    label[8] = single_hand['left_hand'][0]
                                    label[9] = single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[10] = 1.0
                                    label_mask[11] = 1.0
                                    label[10] = single_hand['right_hand'][0]
                                    label[11] = single_hand['right_hand'][1]
                        if frame_type == 'pre_frame':
                            for single_hand in frame_gt:
                                if 'left_hand' in single_hand:
                                    label_mask[12] = 1.0
                                    label_mask[13] = 1.0
                                    label[12] = single_hand['left_hand'][0]
                                    label[13] = single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[14] = 1.0
                                    label_mask[15] = 1.0
                                    label[14] = single_hand['right_hand'][0]
                                    label[15] = single_hand['right_hand'][1]
                        if frame_type == 'contact_frame':
                            for single_hand in frame_gt:
                                if 'left_hand' in single_hand:
                                    label_mask[16] = 1.0
                                    label_mask[17] = 1.0
                                    label[16] = single_hand['left_hand'][0]
                                    label[17] = single_hand['left_hand'][1]
                                if 'right_hand' in single_hand:
                                    label_mask[18] = 1.0
                                    label_mask[19] = 1.0
                                    label[18] = single_hand['right_hand'][0]
                                    label[19] = single_hand['right_hand'][1]
                    self._labels.append(torch.Tensor(label))
                    self._labels_masks.append(torch.Tensor(label_mask))

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

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args

            sample = self._path_to_ant_videos[index]
            buffer = self.loadvideo_decord(sample)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self._path_to_ant_videos[index]
                    buffer = self.loadvideo_decord(sample)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                label_mask_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self._labels[index]
                    label_mask = self._labels_masks[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    label_mask_list.append(label_mask)
                    index_list.append(index)
                return frame_list, label_list, label_mask_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            return buffer, self._labels[index], self._labels_masks[index], index, {}

        elif self.mode == 'validation' or self.mode == "test":
            sample = self._path_to_ant_videos[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self._path_to_ant_videos[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self._labels[index], self._labels_masks[index], self._clip_names[index]

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
            [0.9, 1.0],  # fixme: 0.08->0.9
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False,
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

        num_frames = len(vr)

        if self.mode == 'validation' or self.mode == "test":
            tick = num_frames / float(self.num_segment)
            all_index = []
            for t_seg in range(self.test_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_num_segment + tick * x)
                    for x in range(self.num_segment)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.array(all_index))
            # all_index = list(np.sort(np.array(all_index)))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        average_duration = num_frames // self.num_segment
        all_index = []
        if average_duration > 0:
            if self.mode == 'validation' or self.mode == "test":
                all_index = list(
                    np.multiply(list(range(self.num_segment)),
                                average_duration) +
                    np.ones(self.num_segment, dtype=int) *
                    (average_duration // 2))
            else:
                all_index = list(
                    np.multiply(list(range(self.num_segment)),
                                average_duration) +
                    np.random.randint(average_duration, size=self.num_segment))
        elif num_frames > self.num_segment:
            if self.mode == 'validation' or self.mode == "test":
                all_index = list(range(self.num_segment))
            else:
                all_index = list(
                    np.sort(
                        np.random.randint(num_frames, size=self.num_segment)))
        else:
            all_index = [0] * (self.num_segment - num_frames) + list(
                range(num_frames))
        all_index = list(np.array(all_index))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        # imgs = []
        # for idx in all_index:
        #     frame_fname = os.path.join(fname,
        #                                self.filename_tmpl.format(idx + 1))
        #     img_bytes = self.client.get(frame_fname)
        #     img_np = np.frombuffer(img_bytes, np.uint8)
        #     img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        #     cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        #     imgs.append(img)
        # buffer = np.array(imgs)
        return buffer

    def __len__(self):
        return len(self._path_to_ant_videos)


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
