import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE

from ego4d_cls import Ego4dVerbNounClsDataset
from ego4d_hands import Ego4dHandsDataset

from forecasting_eval.ego4d.datasets.short_term_anticipation import Ego4dShortTermAnticipation

# pretrain for verb
VERB_TRAINING_ANNO_PATH = "./ego4d_annotations/pretrain/ego4d_verb_cls_train.txt"
VERB_VALIDATION_ANNO_PATH = "./ego4d_annotations/pretrain/ego4d_verb_cls_val.txt"
VIDEO_PATH = "DATA/TO/PATH"

# pretrain for noun
NOUN_TRAINING_ANNO_PATH = "./ego4d_annotations/pretrain/ego4d_noun_cls_train.txt"
NOUN_VALIDATION_ANNO_PATH = "./ego4d_annotations/pretrain/ego4d_noun_cls_val.txt"

PRETRAIN_VIDEO_PATH = "DATA/TO/PATH"

#  ego4d for hands
HANDS_TRAINING_ANNO_PATH = "fho_hands_train.json"
HANDS_VALIDATION_ANNO_PATH = "fho_hands_val.json"
HANDS_TESTING_ANNO_PATH = "fho_hands_test_unannotated.json"
HANDS_VIDEO_PATH = "DATA/TO/PATH"


def build_short_term_dataset(cfg, split):
    return Ego4dShortTermAnticipation(cfg=cfg, split=split)


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == "ego4d_verb":
        if is_train is True:
            mode = 'train'
            anno_path = VERB_TRAINING_ANNO_PATH
        elif test_mode is True:
            mode = 'test'
            anno_path = VERB_VALIDATION_ANNO_PATH
        else:
            mode = 'validation'
            anno_path = VERB_VALIDATION_ANNO_PATH
        dataset = Ego4dVerbNounClsDataset(
            anno_path=anno_path,
            data_path=PRETRAIN_VIDEO_PATH,
            mode=mode,
            clip_len=args.num_frames,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        verb_classes = 118
        noun_classes = 0
    elif args.data_set == "ego4d_noun":
        if is_train is True:
            mode = 'train'
            anno_path = NOUN_TRAINING_ANNO_PATH
        elif test_mode is True:
            mode = 'test'
            anno_path = NOUN_VALIDATION_ANNO_PATH
        else:
            mode = 'validation'
            anno_path = NOUN_VALIDATION_ANNO_PATH
        dataset = Ego4dVerbNounClsDataset(
            anno_path=anno_path,
            data_path=PRETRAIN_VIDEO_PATH,
            mode=mode,
            clip_len=args.num_frames,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        verb_classes = 0
        noun_classes = 582
    elif args.data_set == "ego4d_hands":
        if is_train is True:
            mode = 'train'
            anno_path = HANDS_TRAINING_ANNO_PATH
        elif test_mode is True:
            mode = 'test'
            anno_path = HANDS_TESTING_ANNO_PATH
        else:
            mode = 'validation'
            anno_path = HANDS_VALIDATION_ANNO_PATH
        dataset = Ego4dHandsDataset(
            anno_path=anno_path,
            data_path=HANDS_VIDEO_PATH,
            mode=mode,
            clip_len=args.num_frames,
            num_segment=args.num_segments,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        verb_classes = 20
        noun_classes = 0
    else:
        raise NotImplementedError()

    print(noun_classes, args.nb_noun_classes)
    print(verb_classes, args.nb_verb_classes)
    assert verb_classes == args.nb_verb_classes
    assert noun_classes == args.nb_noun_classes
    # print("Number of the class = %d" % args.nb_classes)
    # print("Now this datasets only support EGO4D!")
    return dataset, verb_classes, noun_classes
