# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
from . import transforms_selsa as TS


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
        random_pad_prob = cfg.INPUT.RANROM_PAD_PROB_TRAIN
        max_pad_size = cfg.INPUT.RANROM_PAD_SIZE_TRAIN
        random_crop_prob = cfg.INPUT.RANROM_CROP_PROB_TRAIN
        random_crop_min_size = cfg.INPUT.RANROM_CROP_MIN_SIZE_TRAIN
        random_crop_max_size = cfg.INPUT.RANROM_CROP_MAX_SIZE_TRAIN
        random_crop_max_ratio = cfg.INPUT.RANROM_CROP_MAX_RATIO_TRAIN
        if not cfg.INPUT.TRANSFORM:
            brightness = 0.0
            contrast = 0.0
            saturation = 0.0
            hue = 0.0
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0
        random_pad_prob = 0.0
        max_pad_size = 0.0
        random_crop_prob = 0.0
        random_crop_min_size = cfg.INPUT.RANROM_CROP_MIN_SIZE_TRAIN
        random_crop_max_size = cfg.INPUT.RANROM_CROP_MAX_SIZE_TRAIN
        random_crop_max_ratio = cfg.INPUT.RANROM_CROP_MAX_RATIO_TRAIN

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    if is_train:
        '''
        transform = T.Compose(
            [
                T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
                T.ChannelShuffle(0.5),
                T.Expand(mean=cfg.INPUT.PIXEL_MEAN, expand_scale=2.0),
                T.RandomSampleCrop(),
                #T.RandomPad(random_pad_prob, max_pad_size),
                #T.RandomCrop(random_crop_prob, random_crop_min_size, random_crop_max_size, random_crop_max_ratio),
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                #T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
        '''
        trans_list = []
        if cfg.INPUT.TRANSFORM:
            trans_list.append(TS.SSDAugmentation(mean=cfg.INPUT.PIXEL_MEAN))
        trans_list.extend([T.Resize(min_size, max_size),
                           T.RandomHorizontalFlip(flip_horizontal_prob),
                           # T.RandomVerticalFlip(flip_vertical_prob),
                           T.ToTensor(),
                           normalize_transform, ])
        transform = T.Compose(trans_list)
    else:
        trans_list = \
            [
                T.Resize(min_size, max_size),
                T.ToTensor(),
            ]
        if cfg.MODEL.VID.METHOD not in ("diffusion"):
            trans_list.append(normalize_transform)
        transform = T.Compose(trans_list)
    return transform
