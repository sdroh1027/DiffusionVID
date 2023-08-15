# from https://raw.githubusercontent.com/amdegroot/ssd.pytorch/master/utils/augmentations.py
import torch
# from torchvision import transforms
import cv2
import numpy as np
import types
import random
from PIL import Image


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


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, type=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels, type)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None, type=None):
        return image.astype(np.float32), boxes, labels

class ConvertToInts(object):
    def __call__(self, image, boxes=None, labels=None, type=None):
        return image.clip(min=0, max=255).astype(np.uint8), boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels



class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None, type=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.saturation = 0.
    def __call__(self, image, boxes=None, labels=None, type=None):
        if type not in ['local'] or True:
            self.do = np.random.randint(2)
            self.saturation = np.random.uniform(self.lower, self.upper)
        if self.do:
            image[:, :, 1] *= self.saturation

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        self.hue = 0.
    def __call__(self, image, boxes=None, labels=None, type=None):
        if type not in ['local'] or True:
            self.do = np.random.randint(2)
            self.hue = np.random.uniform(-self.delta, self.delta)
        if self.do:
            image[:, :, 0] += self.hue
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.perm = 0
    def __call__(self, image, boxes=None, labels=None, type=None):
        if type not in ['local'] or True:
            self.do = np.random.randint(2)
            self.perm = np.random.randint(len(self.perms))
        if self.do:
            swap = self.perms[self.perm]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, type=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        if self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'HSV' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.do = True
        self.alpha = 0.
    # expects float image
    def __call__(self, image, boxes=None, labels=None, type=None):
        if type not in ['local'] or True:
            self.do = np.random.randint(2)
            self.alpha = np.random.uniform(self.lower, self.upper)
        if self.do:
            image *= self.alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
        self.do = False
        self.temp = 0.

    def __call__(self, image, boxes=None, labels=None, type=None):
        if type not in ['local'] or True:
            self.do = np.random.randint(2)
            self.temp = np.random.uniform(-self.delta, self.delta)
        if self.do:
            image += self.temp
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


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
        self.mode = None
        self.rect = None
    def __call__(self, image, boxes=None, labels=None, type=None):
        height, width, _ = image.shape
        aspect_ratio = float(height) / float(width)
        while True:
            # randomly choose a mode
            if type not in ['local'] or True:
                self.mode = random.choice(self.sample_options)
            mode = self.mode
            if self.no_iou_limit:
                mode = (None, None)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image
                if type in ['local'] and False:
                    rect = self.rect
                else:
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
                    if len(overlap) == 0:
                        self.rect = rect
                    else:
                        if overlap.min() < min_iou or max_iou < overlap.max():
                            continue
                        else:
                            self.rect = rect

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                if len(boxes) > 0:
                    # keep overlap with gt box IF center in sampled patch
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    # have any valid boxes? try again if not
                    if not mask.any() and (type not in ['local'] or True):
                        continue

                    # take only matching gt boxes
                    current_boxes = boxes[mask, :].copy()

                    # take only matching gt labels
                    current_labels = labels[mask]
                    #current_labels = np.zeros_like(labels, dtype=bool)
                    #current_labels[mask] = True

                    # should we use the box left and top corner or the crop's
                    current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                      rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, :2] -= rect[:2]

                    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                      rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, 2:] -= rect[:2]
                else:
                    current_boxes = boxes
                    current_labels = labels

                return current_image, current_boxes, current_labels


class Expand(object):
    def __init__(self, mean, expand_scale=2.0, is_RGB=True):
        if is_RGB:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.mean = [round(i) for i in self.mean]
        self.expand_scale = expand_scale
        self.skip = False

    def __call__(self, image, boxes, labels, type):
        if type not in ['local'] or True:
            self.skip = np.random.randint(2)
        if self.skip:
            return image, boxes, labels

        height, width, depth = image.shape
        aspect_ratio = float(height) / float(width)

        if type in ['local'] and False:
            ratio, left, top = self.ratio, self. left, self.top
        else:
            ratio = np.random.uniform(1, self.expand_scale)
            left = np.random.uniform(0, width*ratio - width)
            top = np.random.uniform(0, height*ratio - height)
            self.ratio, self.left, self.top = ratio, left, top

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.clone()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(current='RGB', transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='RGB'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        self.distort_type = None

    def __call__(self, image, boxes, labels, type=None):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels, type)
        if type not in ['local'] or True:
            self.distort_type = np.random.randint(2)
        if self.distort_type:
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels, type)
        return self.rand_light_noise(im, boxes, labels, type)


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123), expand_scale=2, crop_pert=0.3, color=True, no_iou_limit=False):
        self.mean = mean
        self.size = size
        self.expand_scale = expand_scale
        self.crop_pert = crop_pert
        self.color = color
        self.no_iou_limit = no_iou_limit
        if self.color:
            self.augment = Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(self.mean, self.expand_scale, is_RGB=True),
                RandomSampleCrop(self.crop_pert, self.no_iou_limit),
                ConvertToInts(),
            ])
        else:
            self.augment = Compose([
                ConvertFromInts(),
                Expand(self.mean, self.expand_scale, is_RGB=True),
                RandomSampleCrop(self.crop_pert, self.no_iou_limit),
                ConvertToInts(),
            ])

    def __call__(self, img, target):
        img = np.array(img).astype(np.uint8)
        boxes = target.bbox.numpy()
        labels = target.extra_fields['labels']
        img, boxes, labels = self.augment(img, boxes, labels, target.type)
        image = Image.fromarray(img)
        target.bbox = torch.tensor(boxes)
        target.extra_fields['labels'] = labels
        target.size = image.width, image.height
        return image, target

def color_transform(im, target, color_factor):
    if color_factor != 0:
        for _c in range(3):
            random_factor = random.uniform(1.0-color_factor, 1.0+color_factor)
            im[:, :, _c] = np.clip(im[:, :, _c] * random_factor, 0, 255)
    return im