from PIL import Image
import sys
import numpy as np
import torch.utils.data
import os
import pickle
import scipy

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from .YouTubeObjects.ap_VOCategories import ap_VOCategories
from .YouTubeObjects.initVideoObjectOptions import initVideoObjectOptions
from .YouTubeObjects.ap_VOGetTubes import ap_VOGetTubes
from .YouTubeObjects.ap_VOGetVideos import ap_VOGetVideos
from mega_core.config import cfg
from mega_core.structures.bounding_box import BoxList
from mega_core.utils.comm import is_main_process

class YOTDataset(torch.utils.data.Dataset):
    classes_yot = ['__background__',  # always index 0
                'aeroplane',  'bird',  'boat',  'car',  'cat',
               'cow',  'dog',  'horse',  'motorbike',  'train']

    classes_vid = ['__background__',  # always index 0
               'airplane', 'antelope', 'bear', 'bicycle',
               'bird', 'bus', 'car', 'cattle',
               'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion',
               'lizard', 'monkey', 'motorcycle', 'rabbit',
               'red_panda', 'sheep', 'snake', 'squirrel',
               'tiger', 'train', 'turtle', 'watercraft',
               'whale', 'zebra']

    classes_map = ['__background__',  # always index 0
                    'n02691156', 'n02419796', 'n02131653', 'n02834778',
                    'n01503061', 'n02924116', 'n02958343', 'n02402425',
                    'n02084071', 'n02121808', 'n02503517', 'n02118333',
                    'n02510455', 'n02342885', 'n02374451', 'n02129165',
                    'n01674464', 'n02484322', 'n03790512', 'n02324045',
                    'n02509815', 'n02411705', 'n01726692', 'n02355227',
                    'n02129604', 'n04468005', 'n01662784', 'n04530566',
                    'n02062744', 'n02391049']

    yot_to_vid_idx = [0,
                    1, 5, 28, 7, 10,
                    8, 9, 15, 19, 26]

    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        #super(VIDMEGADataset, self).__init__(image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=is_train)
        self.det_vid = image_set.split("_")[0]
        self.image_set = image_set
        self.transforms = transforms

        self.data_dir = data_dir
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.img_index = img_index

        self.is_train = is_train

        self._img_dir = os.path.join(self.img_dir, "%s.jpg")
        self._anno_path = os.path.join(self.anno_path, "%s.mat")

        self.classes_to_ind = dict(zip(self.classes_map, range(len(self.classes_map))))
        self.categories = dict(zip(range(len(self.classes_vid)), self.classes_vid))

        '''
        # read YOT v1.0
        GT = not self.is_train
        params = initVideoObjectOptions(minlen=-1, GT=GT)
        params.is_train = self.is_train
        shots_all = []
        tubes_all = []
        for cls in self.classes_yot:
            if 'background' in cls:
                continue
            class_dir = ap_VOCategories(cls)
            tubes_selection_file = class_dir + '/selected_tubes_VID.list'

            # The following call returns the tubes automatically selected by our
            # approach
            #shots, tubes = ap_VOGetTubes(cls, params, tubes_selection_file)
            #shots_all = shots_all + shots
            #tubes_all = tubes_all + tubes
        '''

        # read YOT v2.2
        self.image_set_index, self.pattern = [], []
        self.annos = []
        self.start_id, self.end_id, self.shot_num, self.shot_part_num, self.img_size = [], [], [], [], []
        # self.start_id : first dir index of each shot-part
        # self.end_id : last dir index of each shot-part
        if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            self.shuffled_index = {}

        num_gt = 0
        num_box = 0  # 4306(train) + 2667(test)
        if self.is_train:
            matfile = 'bb_gtTraining_'
        else:
            matfile = 'bb_gtTest_'
        for i, cls in enumerate(self.classes_yot):
            if 'background' in cls:
                continue
            gt = scipy.io.loadmat(self._anno_path % os.path.join(cls, matfile + cls))[matfile[:-1]][0]
            num_gt += len(gt)
            name_len = len(cls)
            image_set_index_cls = [os.path.join(cls, data[0][0][name_len:]) for data in gt]
            pattern = [os.path.dirname(dirr) + "/%08d" for dirr in image_set_index_cls]
            self.image_set_index = self.image_set_index + image_set_index_cls
            self.pattern = self.pattern + pattern
            boxes = [data[1] for data in gt]
            for b in boxes:
                num_box += len(b)
            annos = [{'boxes': torch.tensor(b.astype(np.float32)),
                      'labels': torch.tensor([self.yot_to_vid_idx[i] for _ in range(len(b))])}
                     for b in boxes]
            self.annos += annos

            # range of shot partitions
            img_idx = scipy.io.loadmat(os.path.join(self.img_index, 'ranges_'+ cls +'.mat'))['ranges']
            start_id, end_id, shot_num, shot_part_num, img_size = [], [], [], [], []
            shuffled_index = {}
            for frame in image_set_index_cls:
                frame_num = int(os.path.basename(frame))
                img_size.append(Image.open(self._img_dir % frame).convert("RGB").size)
                # find shot_part_num
                idx1 = np.where(frame_num >= img_idx[0])[0].max()
                idx2 = np.where(frame_num <= img_idx[1])[0].min()
                assert idx1 == idx2
                shot_part_num.append(idx1)
                # read and save dir_ids and shot_num
                start_id.append(img_idx[0][idx1])
                end_id.append(img_idx[1][idx1])
                shot_num.append(img_idx[2][idx1])
                # save shuffled_index
                if str(idx1) in shuffled_index.keys():
                    continue
                else:
                    if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        shuffled_indexes = np.arange(img_idx[0][idx1], img_idx[1][idx1] + 1)
                        if cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_indexes)
                        shuffled_index[str(idx1)] = shuffled_indexes
            self.start_id += start_id
            self.end_id += end_id
            self.shot_num += shot_num
            self.shot_part_num += shot_part_num
            self.img_size += img_size
            self.shuffled_index[cls] = shuffled_index

        # make start_index (not start_id)
        if not self.is_train:
            self.start_index = []
            assert len(self.shot_part_num) == len(self.image_set_index) == len(self.start_id) == len(self.end_id)
            for i, image_index in enumerate(self.image_set_index):
                if self.shot_part_num[i] != self.shot_part_num[i - 1]:
                    self.start_index.append(i)

        return

    def __len__(self):
        return len(self.image_set_index)

    def get_img_info(self, idx):
        im_info = self.img_size[idx]
        return {"height": im_info[1], "width": im_info[0]}

    def __getitem__(self, idx):
        if self.is_train:
            return self._get_train(idx)
        else:
            return self._get_test(idx)

    def get_groundtruth(self, idx):
        anno = self.annos[idx]
        hw = self.get_img_info(idx)
        height, width = hw['height'], hw['width']
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])

        return target

    def _get_train(self, idx):
        raise NotImplementedError

    def _get_test(self, idx):
        raise NotImplementedError

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return YOTDataset.classes_vid[class_id]


    def load_annos(self, cache_file):
        annos = []
        for idx in range(len(self)):
            if idx % 10000 == 0:
                print("Had processed {} images".format(idx))

            filename = self.image_set_index[idx]

            tree = ET.parse(self._anno_path % filename).getroot()  # get label info (im_info + objects info)
            anno = self._preprocess_annotation(tree)
            annos.append(anno)
        print("Had processed {} images".format(len(self)))

        if is_main_process():
            with open(cache_file, "wb") as fid:
                pickle.dump(annos, fid)
            print("Saving {}'s annotation information into {}".format(self.det_vid, cache_file))

        return annos

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        # im_info is tuple [image_y, image_x]

        objs = target.findall("object")  # get objects
        for obj in objs:
            if not obj.find("name").text in self.classes_to_ind:
                continue

            bbox = obj.find("bndbox") # get bounding box [xmin, ymin, xmax, ymax]
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