import cv2
import numpy as np
import glob
import os
import tempfile
from collections import OrderedDict
from tqdm import tqdm
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from mega_core.modeling.detector import build_detection_model
from mega_core.utils.checkpoint import DetectronCheckpointer
from mega_core.structures.image_list import to_image_list
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from mega_core.structures.bounding_box import BoxList

from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                 CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)


class Cache(object):

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoProcessor(object):
    def __init__(self, filename, cache_capacity=10):
        if filename is None:
            self._fps = 25
            self._only_output = True
        else:
            self._vcap = cv2.VideoCapture(filename)
            assert cache_capacity > 0
            self._cache = Cache(cache_capacity)
            self._position = 0
            # get basic info
            self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
            self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
            self._fps = self._vcap.get(CAP_PROP_FPS)
            self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
            self._fourcc = self._vcap.get(CAP_PROP_FOURCC)
            self._only_output = False
        self._output_video_name = "visualization.avi"

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: The raw VideoCapture object."""
        return self._vcap

    @property
    def opened(self):
        """bool: Indicate whether the video is opened."""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: Width of video frames."""
        return self._width

    @property
    def height(self):
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self):
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self):
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self):
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "Four character code" of the video."""
        return self._fourcc

    @property
    def position(self):
        """int: Current cursor position, indicating frame decoded."""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        # pos = self._position
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(
                '"frame_id" must be between 0 and {}'.format(self._frame_cnt -
                                                             1))
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited).

        Returns:
            ndarray or None: If the video is fresh, return None, otherwise
                return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_tmpl='{:06d}.jpg',
                   start=0,
                   max_num=0):
        """Convert a video to frame images

        Args:
            frame_dir (str): Output directory to store all the frame images.
            file_start (int): Filenames will start from the specified number.
            filename_tmpl (str): Filename template with the index as the
                placeholder.
            start (int): The starting frame index.
            max_num (int): Maximum number of frames to be written.
        """
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        for i in range(task_num):
            img = self.read()
            if img is None:
                break
            filename = os.path.join(frame_dir,
                                filename_tmpl.format(i + file_start))
            cv2.imwrite(filename, img)

    def frames2videos(self, frames, output_folder):
        if self._only_output:
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            height, width = frames[0].shape[:2]
        else:
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            height = self._height
            width = self._width

        videoWriter = cv2.VideoWriter(os.path.join(output_folder, self._output_video_name), fourcc, self._fps, (width, height))

        for frame_id in range(len(frames)):
            videoWriter.write(frames[frame_id])
        videoWriter.release()

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.get_frame(i)
                for i in range(*index.indices(self.frame_cnt))
            ]
        # support negative indexing
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError('index out of range')
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
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

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class VIDDemo(object):
    CATEGORIES = ['_bg_',  # always index 0
                  'airplane', 'antelope', 'bear', 'bicycle',
                  'bird', 'bus', 'car', 'cattle',
                  'dog', 'domestic_cat', 'elephant', 'fox',
                  'giant_panda', 'hamster', 'horse', 'lion',
                  'lizard', 'monkey', 'motorcycle', 'rabbit',
                  'red_panda', 'sheep', 'snake', 'squirrel',
                  'tiger', 'train', 'turtle', 'watercraft',
                  'whale', 'zebra']

    def __init__(
            self,
            cfg,
            method="base",
            confidence_threshold=0.7,
            output_folder="demo/visulaization"
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        self.method = method
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # for video output
        self.vprocessor = VideoProcessor(None)

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transforms_list = [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
        ]
        if "diffusion" not in cfg.MODEL.VID.METHOD:
            transforms_list.append(normalize_transform)
        transform = T.Compose(transforms_list)
        return transform

    def build_pil_transform(self):
        """
        Creates a basic transformation that was used in generalized_rnn_{}._forward_test()
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]] * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x)

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transforms_list = [
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
        ]
        if "diffusion" not in cfg.MODEL.VID.METHOD:
            transforms_list.append(normalize_transform)
        transform = T.Compose(transforms_list)
        return transform

    def perform_transform(self, original_image):
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        return image_list

    def run_on_image_folder(self, image_folder, suffix='.JPEG', track_refs=False):
        image_names = glob.glob(image_folder + '/*' + suffix)
        image_names = sorted(image_names)
        image_set_index = [i.split(suffix)[0] for i in image_names]
        start_id = int(image_set_index[0].split("/")[-1])

        img_dir = "%s" + suffix
        frame_seg_len = len(image_names)
        pattern = image_folder + "/%06d"

        images_with_boxes = []
        # preparation for visualization
        self.model.demo = True
        if self.method != 'diffusion':
            self.model.roi_heads.box.feature_extractor.demo = True
        self.model.features_all = []
        self.model.proposals_all = []
        self.affines = []
        self.contributors = []
        self.l2_norms = []
        self.l2_norms_key = []
        self.proposals_global = []
        self.proposals_global_last = []
        self.target_labels = []
        self.class_probs = []
        self.cur_feat = []
        self.enhanced_feat = []
        self.cur_feat_nms_idx = []
        pil_transform = self.build_pil_transform()
        shuffled_indices = []
        for idx in tqdm(range(frame_seg_len)):
            original_image = cv2.imread(image_names[idx])
            frame_id = int(image_set_index[idx].split("/")[-1])
            img_cur = self.perform_transform(original_image)
            if self.method == "base":
                image_with_boxes, _ = self.run_on_image(original_image, img_cur)
                images_with_boxes.append(image_with_boxes)
            elif self.method in ("dff", "fgfa", "rdn", "mega", "dafa", "diffusion"):
                infos = {}
                infos["cur"] = img_cur
                infos["frame_category"] = 0 if idx == 0 else 1
                infos["seg_len"] = frame_seg_len
                infos["pattern"] = pattern
                infos["img_dir"] = img_dir
                infos["transforms"] = pil_transform

                infos["frame_id"] = frame_id
                infos["start_id"] = 0  # dir id of the first frame of video
                infos["end_id"] = frame_seg_len - 1  # dir id of the last frame of video
                ref_id_final = min(frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET, frame_seg_len - 1)
                infos["last_queue_id"] = ref_id_final

                if self.method == "dff":
                    infos["is_key_frame"] = True if frame_id % 10 == 0 else False
                elif self.method in ("fgfa", "rdn"):
                    img_refs = []
                    if self.method == "fgfa":
                        max_offset = self.cfg.MODEL.VID.FGFA.MAX_OFFSET
                    else:
                        max_offset = self.cfg.MODEL.VID.RDN.MAX_OFFSET
                    ref_id = min(frame_seg_len - 1, frame_id + max_offset)
                    ref_filename = pattern % ref_id
                    img_ref = cv2.imread(img_dir % ref_filename)
                    img_ref = self.perform_transform(img_ref)
                    img_refs.append(img_ref)

                    infos["ref"] = img_refs
                    targets = None
                elif self.method in ["mega", "dafa", "diffusion"]:
                    img_refs_l = []
                    '''
                    # reading other images of the queue (not necessary to be the last one, but last one here)
                    ref_id = min(frame_seg_len - 1, frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET)
                    ref_filename = pattern % ref_id
                    img_ref = cv2.imread(img_dir % ref_filename)
                    img_ref = self.perform_transform(img_ref)
                    img_refs_l.append(img_ref)
                    '''
                    # only supports ImageNet VID
                    # read prev & future frames (max size)
                    ref_id_final = min(frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET, frame_seg_len - 1)
                    if infos["frame_category"] == 0:
                        size_local = self.cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL - self.cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION - 1
                        assert size_local == self.cfg.MODEL.VID.MEGA.MAX_OFFSET
                        ref_id_start = max(ref_id_final - self.cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL + 1, 0)
                    else:
                        filename_prev = image_set_index[idx - 1]
                        frame_id_prev = int(filename_prev.split("/")[-1])
                        frame_diff = frame_id - frame_id_prev
                        num_ref = min(frame_diff, self.cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL)
                        ref_id_start = max(ref_id_final - num_ref + 1, start_id)

                    for id in range(ref_id_start, ref_id_final + 1):
                        ref_filename = pattern % id
                        img_ref = cv2.imread(img_dir % ref_filename)
                        img_ref = self.perform_transform(img_ref)
                        img_refs_l.append(img_ref)

                    img_refs_g = []
                    if self.cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        size = self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE if infos["frame_category"] == 0 else 0
                        shuffled_index = np.arange(frame_seg_len)
                        if self.cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        for id in range(size):
                            sid = shuffled_index[(frame_id + self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) % frame_seg_len]
                            shuffled_indices.append(sid)
                            filename = pattern % sid
                            img = cv2.imread(img_dir % filename)
                            img = self.perform_transform(img)
                            img_refs_g.append(img)

                    infos["ref_l"] = img_refs_l
                    infos["ref_g"] = img_refs_g
                    infos['frame_id_g'] = shuffled_indices

                    # get anotation (GT) information
                    anno_filename = image_names[frame_id].replace('/Data/', '/Annotations/').replace('.JPEG', '.xml')
                    if os.path.exists(anno_filename):
                        tree = ET.parse(anno_filename).getroot()
                        anno = self._preprocess_annotation(tree)
                        height, width = anno["im_info"]
                        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
                        target.add_field("labels", anno["labels"])
                        img_size = infos["cur"].image_sizes[0]
                        target = target.resize((img_size[1], img_size[0])).to(self.device)
                        targets = [target]
                    else:
                        targets = []
                else:
                    pass

                image_with_boxes, predictions = self.run_on_image(original_image, infos, targets)
                if self.method in ["mega", "dafa"]:
                    self.affines.append(self.model.roi_heads.box.feature_extractor.affine)
                    self.contributors.append(self.model.roi_heads.box.feature_extractor.contributor)  # which ref feature in memory contributes most
                    self.l2_norms.append(self.model.roi_heads.box.feature_extractor.l2_norm)
                    self.l2_norms_key.append(self.model.roi_heads.box.feature_extractor.l2_norm_key)
                    self.proposals_global.append(self.model.proposals_global_mem)  # best contributor's objectness score
                    self.proposals_global_last.append(self.model.proposals_global_mem_last)
                    self.target_labels.append(self.model.roi_heads.box.target_labels)
                    self.class_probs.append(self.model.roi_heads.box.class_prob)
                    self.cur_feat.append(self.model.roi_heads.box.cur_feat)
                    self.enhanced_feat.append(self.model.roi_heads.box.enhanced_feat)
                    self.cur_feat_nms_idx.append(self.model.roi_heads.box.cur_feat_nms_idx)
                images_with_boxes.append(image_with_boxes)
                num_predictions = min(2, len(predictions))
                score_sort, idx_sort = predictions.extra_fields['scores'].topk(k=num_predictions, largest=True)
                label_sort = predictions.extra_fields['labels'][idx_sort]
                if self.method in ["dafa"]:
                    feat_idx_nms_sorted = self.cur_feat_nms_idx[-1][idx_sort]
                    cur_feat_sorted = self.cur_feat[-1][feat_idx_nms_sorted]
                    contributors_each_feat = self.contributors[idx][feat_idx_nms_sorted]
                    contributors_fid = []
                    for ref_fid in contributors_each_feat:
                        contributors_fid.append(self.proposals_global[idx].extra_fields['frame_id'][ref_fid])
                    if track_refs:
                        # track top5 ref boxes per objects in frames
                        for i in range(num_predictions):
                            fids = contributors_fid[i]
                            mem_ids = contributors_each_feat[i]
                            proposals_contrib = self.proposals_global_last[idx][mem_ids]
                            for j in range(len(fids)):
                                # draw ref bbox in their images
                                assert proposals_contrib.extra_fields['frame_id'][j] == fids[j]
                                ref_image = cv2.imread(image_names[fids[j]])
                                # reshape prediction (a BoxList) into the original image size
                                height, width = original_image.shape[:-1]
                                proposal_contrib = proposals_contrib[j:j + 1].resize((width, height))
                                ref_image_with_box = ref_image.copy()
                                ref_image_with_box = self.overlay_boxes(ref_image_with_box, proposal_contrib)
                                name = os.path.join(self.output_folder,
                                             "%06d" % idx + "_obj" + str(i) + "_ref" + str(j) + "_fid" + str(int(fids[j])) + ".jpg")
                                cv2.imwrite(name, ref_image_with_box)
            else:
                raise NotImplementedError("method {} is not implemented.".format(self.method))

        from visualizer import plot_histogram, contrib_L2_plots, plot_TSNE
        if False:
            mem = self.model.roi_heads.box.feature_extractor.global_cache[0]['feats']
            mem_selfatten = self.model.roi_heads.box.feature_extractor.update_lm(mem, 1)
            class_logits, box_regression = self.model.roi_heads.box.predictor(mem_selfatten)
            _, class_labels = class_logits.max(dim=-1)
            class_labels = class_labels.cpu().numpy()

            plot_TSNE(self.model.features_all, self.model.proposals_all, mem, class_labels)


        #contrib_L2_plots(enhancement_scores, self.l2_norms, 'Classification contribution')

        #contrib_mean = [x.mean(0).mean(0) for x in self.contributions]
        #contrib_L2_plots(contrib_mean, self.l2_norms, 'Attention contribution')
        #contrib_L2_plots(contrib_mean, self.l2_norms_key, 'Attention contribution')

        # plot_histogram(contrib_mean, self.l2_norms)

        return images_with_boxes

    def run_on_video(self, video_path):
        if not os.path.isfile(video_path):
            raise FileNotFoundError('file "{}" does not exist'.format(video_path))
        self.vprocessor = VideoProcessor(video_path)
        tmpdir = tempfile.mkdtemp()
        self.vprocessor.cvt2frames(tmpdir)
        results = self.run_on_image_folder(tmpdir, suffix='.jpg')

        return results

    def run_on_image(self, image, infos=None, targets=None):
        """
        Arguments:
            image
            infos
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image, infos, targets)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)
        if False:
            # print region proposal boxes
            rpn_result = self.rpn_proposals[:75]
            rpn_result.bbox = rpn_result.bbox.cpu()
            rpn_result.extra_fields['scores'] = rpn_result.extra_fields['objectness'].cpu()
            rpn_result.extra_fields['labels'] = torch.zeros(len(rpn_result), dtype=torch.int64)
            result = self.overlay_boxes(result, rpn_result)
            result = self.overlay_class_names(result, rpn_result)

        return result, predictions

    def compute_prediction(self, original_image, infos, targets):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # compute predictions
        #with torch.cuda.amp.autocast():
            # output is float16 because linear layers autocast to float16.
        with torch.no_grad():
            predictions = self.model(infos, targets)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=False)   # ascending order in order that higher score boxes drawn later
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        try:
            labels = predictions.get_field("labels")
            colors = self.compute_colors_for_labels(labels).tolist()
        except:
            colors = [[255, 0, 0] for i in range(len(predictions.bbox))] #bgr
        boxes = predictions.bbox

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2
            )

        return image

    def draw_text(self, img, text,
                  pos=(0, 0),
                  font=cv2.FONT_HERSHEY_PLAIN,
                  font_scale=3,
                  text_color=(0, 255, 0),
                  font_thickness=2,
                  text_color_bg=(0, 0, 0)
                  ):

        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(img, text, (x, int(y + text_h + int(font_scale) - 1)), font, font_scale, text_color, font_thickness)

        return text_size

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels")
        labels_text = [self.CATEGORIES[i] for i in labels.tolist()]
        boxes = predictions.bbox
        box_colors = self.compute_colors_for_labels(labels).tolist()

        template = "{}: {:.2f}"
        for box, score, label, box_color in zip(boxes, scores, labels_text, box_colors):
            x, y = box[:2]
            s = template.format(label, score)
            self.draw_text(
                 image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, tuple(box_color)
            )

        return image

    def generate_images(self, visualization_results):
        for frame_id in range(len(visualization_results)):
            cv2.imwrite(os.path.join(self.output_folder, "%06d.jpg" % frame_id), visualization_results[frame_id])

    def generate_video(self, visualization_results):
        self.vprocessor.frames2videos(visualization_results, self.output_folder)

    def _preprocess_annotation(self, target):
        classes_map = ['__background__',  # always index 0
                       'n02691156', 'n02419796', 'n02131653', 'n02834778',
                       'n01503061', 'n02924116', 'n02958343', 'n02402425',
                       'n02084071', 'n02121808', 'n02503517', 'n02118333',
                       'n02510455', 'n02342885', 'n02374451', 'n02129165',
                       'n01674464', 'n02484322', 'n03790512', 'n02324045',
                       'n02509815', 'n02411705', 'n01726692', 'n02355227',
                       'n02129604', 'n04468005', 'n01662784', 'n04530566',
                       'n02062744', 'n02391049']
        self.classes_to_ind = dict(zip(classes_map, range(len(classes_map))))
        boxes = []
        gt_classes = []

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        objs = target.findall("object")
        for obj in objs:
            if not obj.find("name").text in self.classes_to_ind:
                continue

            bbox =obj.find("bndbox")
            box = [
                np.maximum(float(bbox.find("xmin").text), 0),
                np.maximum(float(bbox.find("ymin").text), 0),
                np.minimum(float(bbox.find("xmax").text), im_info[1] - 1),
                np.minimum(float(bbox.find("ymax").text), im_info[0] - 1)
            ]
            boxes.append(box)
            gt_classes.append(self.classes_to_ind[obj.find("name").text.lower().strip()])

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res
