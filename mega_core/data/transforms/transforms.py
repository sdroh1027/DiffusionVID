# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
import albumentations
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        if target is not None and target.type == 'cur':
            self.size = self.get_size(image.size)
        image = F.resize(image, self.size)
        if target is None:
            return image, target
        type = target.type
        target = target.resize(image.size)
        target.type = type
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.chance = 0.0

    def __call__(self, image, target=None):
        if target is not None and target.type in ['cur', 'global']:
            self.chance = random.random()
        if self.chance < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)

        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            if target is not None:
                target = target.transpose(1)
        return image, target


class ColorJitter(object):
    def __init__(self,
                 brightness=0.166,
                 contrast=0.5,
                 saturation=0.5,
                 hue=0.05,
                 ):
        #self.color_jitter = torchvision.transforms.ColorJitter(
        self.color_jitter = albumentations.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, target=None):
        image = np.array(image).astype(np.uint8)
        image = self.color_jitter(image=image)['image']
        if target is None:
            image = Image.fromarray(image)
        #image = self.color_jitter(image)
        return image, target

class RandomPad(object):
    def __init__(self, prob=0.5, max_size=1.5):
        self.prob = prob
        self.min_size = 0.
        # original SSD implementation expands image upto 4x.
        # therefore, we pad pixels upto 1.5x of original image on each side of the image.
        self.max_size = max_size

    def __call__(self, image, target=None):
        if random.random() < self.prob and target is not None:
            width, height = image.size

            ratio = random.uniform(self.min_size, self.max_size)
            pad_width = random.uniform(0, ratio * width)
            pad_height = random.uniform(0, ratio * height)
            padding = int(min(pad_width, pad_height))
            pad_func = torchvision.transforms.Pad(padding=padding, fill=0)
            pad_image = pad_func(image)

            xmin, ymin, xmax, ymax = target.bbox.split(1, dim=-1)

            padded_xmin = xmin + padding
            padded_ymin = ymin + padding
            padded_xmax = xmax + padding
            padded_ymax = ymax + padding

            padded_box = torch.cat(
                (padded_xmin, padded_ymin, padded_xmax, padded_ymax), dim=-1
            )
            target.bbox = padded_box
            target.padding = padding
            return pad_image, target
        return image, target


class RandomCrop(object):
    def __init__(self, prob=0.5, min_crop_size=0.3, max_crop_size=0.9, max_crop_ratio=2.):
        self.prob = prob
        self.min_size = min_crop_size  # 0~1
        self.max_size = max_crop_size  # 0~1
        self.min_aspect_ratio = 1
        self.max_aspect_ratio = max_crop_ratio  # ratio > 1

    def __call__(self, image, target=None):
        if random.random() < self.prob and target is not None:
            ###### determine crop w, h #####
            #assert target is not None
            if target is not None and len(target) > 0:
                target_bboxes = target.convert("xyxy").bbox
                # select a target box
                ind = random.randrange(0, len(target))
                target_sel = target_bboxes[ind]
                target_sel_w = target_sel[2] - target_sel[0]
                target_sel_h = target_sel[3] - target_sel[1]

                # determine w, h based on iou
                iou_min = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
                iou = 0.
                iter = 0
                while iou < iou_min:
                    iter = iter + 1
                    if iter > 50:
                        #print('random crop failed')
                        return image, target
                    # we do not include minimum w, h limit by img length(0.3)
                    #w = int(random.uniform((target_sel_w * iou_min), min((target_sel_w / iou_min), image.size[0])))
                    #h = int(random.uniform((target_sel_h * iou_min), min((target_sel_h / iou_min), image.size[1])))
                    #h = int(random.uniform(max(target_sel_h * iou_min, self.min_size * image.size[1]), self.max_size * image.size[1]))
                    #w = int(random.uniform(min(h * 0.5, image.size[0]), min(h * 2, image.size[0])))
                    h = int(random.uniform(self.min_size * image.size[1], self.max_size * image.size[1]))
                    w = int(random.uniform(self.min_size * image.size[0], self.max_size * image.size[0]))
                    ratio = w / h
                    if ratio > 2 or ratio < 0.5:
                        continue

                    # select starting point xy
                    middle_gt = (int((target_sel[0] + target_sel[2]) / 2), int((target_sel[1] + target_sel[3]) / 2))
                    start_x = random.randint(max(middle_gt[0] - w + 1, 0), min(middle_gt[0], image.size[0] - w + 1))
                    start_y = random.randint(max(middle_gt[1] - h + 1, 0), min(middle_gt[1], image.size[1] - h + 1))
                    crop_xyxy = (start_x, start_y, start_x + w - 1, start_y + h - 1)
                    iou_list = [compute_iou(target_xyxy, torch.tensor(crop_xyxy)) for target_xyxy in target_bboxes]
                    iou = max(iou_list)

            else:
                # select aspect ratio and size ratio
                aspect_ratio = self.min_aspect_ratio + (self.max_aspect_ratio - self.min_aspect_ratio) * random.random()
                size_long = self.min_size + (self.max_size - self.min_size) * random.random()
                if random.random() < 0.5:
                    # x is longer side
                    w = int(size_long * image.size[0])
                    h = int(min(w / aspect_ratio, image.size[1]))
                else:
                    # y is longer side
                    h = int(size_long * image.size[1])
                    w = int(min(h / aspect_ratio, image.size[0]))
                # select starting point xy
                start_x = torch.randint(0, image.size[0] - w + 1, size=(1,)).item()
                start_y = torch.randint(0, image.size[1] - h + 1, size=(1,)).item()
                crop_xyxy = (start_x, start_y, start_x + w - 1, start_y + h - 1)

            image_cropped = F.crop(image, start_y, start_x, h, w)
            if target is not None and len(target) > 0:
                target_cropped = target.crop(crop_xyxy)
                sel = torch.ones(len(target_cropped), dtype=torch.bool)
                for i in range(len(target_cropped)):
                    middle_gt = (int((target.bbox[i, 0] + target.bbox[i, 2]) / 2), int((target.bbox[i, 1] + target.bbox[i, 3]) / 2))
                    # middle point of gt must be in the cropped image
                    if middle_gt[0] < crop_xyxy[0] or middle_gt[0] > crop_xyxy[2]:
                        sel[i] = False
                    if middle_gt[1] < crop_xyxy[1] or middle_gt[1] > crop_xyxy[3]:
                        sel[i] = False
                target_cropped_new = target_cropped[sel]
                if len(target_cropped_new) == 0:
                    print('error')
                if target.type in ['cur']:
                    target_cropped_new.type = 'cur'
                #image1 = drawbox(image, target)
                #image2 = drawbox(image_cropped, target_cropped_new)
                return image_cropped, target_cropped_new
            return image_cropped, target
        return image, target

def drawbox(image, target):
    # cropping demo
    image_np = np.array(image)
    for box in zip(target.bbox):
        box = box[0].to(torch.int64)
        color = (0, 0, 0)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(image_np, tuple(top_left), tuple(bottom_right), tuple(color), 2)
    return image

def area(box, mode = "xyxy"):
    if mode == "xyxy":
        TO_REMOVE = 1
        area = (box[2] - box[0] + TO_REMOVE) * (box[3] - box[1] + TO_REMOVE)
    elif mode == "xywh":
        area = box[2] * box[3]
    else:
        raise RuntimeError("Should not be here")

    return area

def compute_iou(target_xyxy, crop_xyxy):
    """Compute the intersection over union of two boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """

    area1 = area(target_xyxy, mode="xyxy")
    area2 = area(crop_xyxy, mode="xyxy")

    box1, box2 = target_xyxy, crop_xyxy

    lt = torch.max(box1[:2], box2[:2])  # left_top
    rb = torch.min(box1[2:], box2[2:])  # right_bottom

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[0] * wh[1]  # [N,M]

    iou = inter / (area1 + area2 - inter)
    #iou = inter / area1
    return iou

class ToTensor(object):
    def __call__(self, image, target=None):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        else:
            image = image * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, target
        return image, target


class ChannelShuffle(object):
    def __init__(self, prob=0.5):
        self.ChannelShuffle = albumentations.augmentations.transforms.ChannelShuffle(prob)

    def __call__(self, image, target=None):
        if target is not None:
            #image = np.array(image).astype(np.uint8)
            image = self.ChannelShuffle(image=image)['image']
            #image = Image.fromarray(image)

        return image, target

class Expand(object):
    def __init__(self, mean, expand_scale=2.0):
        self.mean = mean
        self.mean = [round(x) for x in self.mean]
        self.expand_scale = expand_scale

    def __call__(self, image, target):
        if np.random.randint(2) or target is None:
            return image, target
        #image = np.array(image).astype(np.uint8)

        height, width, depth = image.shape
        aspect_ratio = float(height) / float(width)
        ratio = np.random.uniform(1, self.expand_scale)
        left = np.random.uniform(0, width*ratio - width)
        top = np.random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = target.bbox
        labels = target.extra_fields['labels']
        #boxes = boxes.copy()
        boxes[:, :2] = boxes[:, :2] + torch.tensor([int(left), int(top)])
        boxes[:, 2:] = boxes[:, 2:] + torch.tensor([int(left), int(top)])
        target.bbox = boxes

        #image = Image.fromarray(image)

        return image, target


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, crop_pert=0.3, no_iou_limit=False):
        self.crop_pert = crop_pert
        self.no_iou_limit = no_iou_limit
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, target):
        if target is None:
            return image, target
        boxes = target.bbox
        labels = target.extra_fields['labels']
        #image = np.array(image).astype(np.uint8)
        height, width, _ = image.shape
        aspect_ratio = float(height) / float(width)
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if self.no_iou_limit:
                mode = (None, None)
            if mode is None:
                image = Image.fromarray(image)
                return image, target

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = np.random.uniform(self.crop_pert * width, width)
                h = w * aspect_ratio

                # # aspect ratio constraint b/t .5 & 2
                # if h / w < 0.5 or h / w > 2:
                #     continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                #current_boxes = boxes[mask, :].copy()
                current_boxes = boxes[mask, :]

                # take only matching gt labels
                # current_labels = labels[mask]
                current_labels = np.zeros_like(labels, dtype=bool)
                current_labels[mask.numpy()] = True

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                image = Image.fromarray(current_image)
                target.bbox = current_boxes
                target.extra_fields['labels'] = labels[current_labels]

                return image, target


