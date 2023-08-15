from PIL import Image
import numpy as np
import os

from mega_core.config import cfg
from .yot import YOTDataset

class YOTMEGADataset(YOTDataset):
    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        super(YOTMEGADataset, self).__init__(image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=is_train)

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # if a video dataset
        img_refs_l = []
        img_refs_m = []
        img_refs_g = []
        if hasattr(self, "pattern"):
            # local frames
            if cfg.MODEL.VID.MEGA.LOCAL.ENABLE:
                offsets = np.random.choice(cfg.MODEL.VID.MEGA.MAX_OFFSET - cfg.MODEL.VID.MEGA.MIN_OFFSET + 1,
                                           cfg.MODEL.VID.MEGA.REF_NUM_LOCAL, replace=False) + cfg.MODEL.VID.MEGA.MIN_OFFSET
                for i in range(len(offsets)):
                    ref_id = min(max(self.frame_seg_id[idx] + offsets[i], 0), self.frame_seg_len[idx] - 1)
                    ref_filename = self.pattern[idx] % ref_id
                    img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                    img_refs_l.append(img_ref)

            # memory frames
            if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
                ref_id_center = max(self.frame_seg_id[idx] - cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL, 0)
                offsets = np.random.choice(cfg.MODEL.VID.MEGA.MAX_OFFSET - cfg.MODEL.VID.MEGA.MIN_OFFSET + 1,
                                           cfg.MODEL.VID.MEGA.REF_NUM_MEM, replace=False) + cfg.MODEL.VID.MEGA.MIN_OFFSET
                for i in range(len(offsets)):
                    ref_id = min(max(ref_id_center + offsets[i], 0), self.frame_seg_len[idx] - 1)
                    ref_filename = self.pattern[idx] % ref_id
                    img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                    img_refs_m.append(img_ref)

            # global frames
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                try:
                    ref_ids = np.random.choice(self.frame_seg_len[idx], cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL, replace=False)
                except:
                    ref_ids = np.random.choice(self.frame_seg_len[idx], 5, replace=False)
                targets_g = []
                for ref_id in ref_ids:
                    ref_filename = self.pattern[idx] % ref_id
                    img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                    img_refs_g.append(img_ref)
                    # get gt of global img
                    target_g = self.get_groundtruth_from_filename(ref_filename)
                    targets_g.append(target_g.clip_to_image(remove_empty=True))
        else:
            if cfg.MODEL.VID.MEGA.LOCAL.ENABLE:
                for i in range(cfg.MODEL.VID.MEGA.REF_NUM_LOCAL):
                    img_refs_l.append(img.copy())
            if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
                for i in range(cfg.MODEL.VID.MEGA.REF_NUM_MEM):
                    img_refs_m.append(img.copy())
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                #for i in range(cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL):
                for i in range(1): # no redundant features
                    img_refs_g.append(img.copy())

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)
        if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            targets_g = targets_g if hasattr(self, "pattern") else [target.copy_with_fields([]) for i in range(cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL)]
        else:
            targets_g = []
        target.type = 'cur'  # mark as current frame information
        for i in range(len(img_refs_g)):
            targets_g[i].type = 'global'

        # select comparative frames
        img_news = []
        targets_new = []
        labels = []
        if False:
            label_t = np.random.choice(target.extra_fields['labels'])
            labels.append(label_t)
            # positive image
            while True:
                new_filename = np.random.choice(self.image_set_index)
                img_new = Image.open(self._img_dir % new_filename).convert("RGB")
                target_new = self.get_groundtruth_from_filename(new_filename)
                label_p = np.random.choice(target_new.extra_fields['labels'])
                if label_p == label_t:
                    labels.append(label_p)
                    img_news.append(img_new)
                    targets_new.append(target_new.clip_to_image(remove_empty=True))
                    break
            # negative image
            while True:
                new_filename = np.random.choice(self.image_set_index)
                img_new = Image.open(self._img_dir % new_filename).convert("RGB")
                target_new = self.get_groundtruth_from_filename(new_filename)
                label_n = np.random.choice(target_new.extra_fields['labels'])
                if label_n != label_t:
                    labels.append(label_n)
                    img_news.append(img_new)
                    targets_new.append(target_new.clip_to_image(remove_empty=True))
                    break

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_m)):
                img_refs_m[i], _ = self.transforms(img_refs_m[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)  # targets_g[i])
            for i in range(len(img_news)):
                img_news[i], _ = self.transforms(img_news[i], None)  # targets_new[i])

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l
        images["ref_m"] = img_refs_m
        images["ref_g"] = img_refs_g
        images["ref_c"] = [img_news, targets_new, labels]

        return images, [[target], targets_g], idx

    def _get_test(self, idx):
        # idx: index of dataset loader
        filename = self.image_set_index[idx]
        shuffled_cur = False
        init_and_no_update_g = True
        frame_seg_len_idx = self.end_id[idx] - self.start_id[idx] + 1
        if shuffled_cur:
            # shuffled test
            shuffled_index_cur = self.shuffled_index[str(self.start_id[idx])]
            idx_new_local = shuffled_index_cur[idx - self.start_id[idx]]
            filename_new = self.pattern[idx] % idx_new_local
            img = Image.open(self._img_dir % filename_new).convert("RGB")
            idx_new = self.start_id[idx] + idx_new_local
        else:
            img = Image.open(self._img_dir % filename).convert("RGB")

        # give the current frame a category. 0 if meet test frame of a clip firstly
        # frame category for test clip with sparse frame.
        # frame_id: dir_id
        frame_id = int(filename.split("/")[-1])
        #if frame_id != self.start_id[idx]:
        if idx == 0:
            # new video clip
            frame_category = 0
        if self.shot_part_num[idx] != self.shot_part_num[idx - 1]:
            # new video clip
            frame_category = 0
        else:
            # old video clip
            frame_category = 1
            filename_prev = self.image_set_index[idx-1]
            frame_id_prev = int(filename_prev.split("/")[-1])
            frame_diff = frame_id - frame_id_prev

        if shuffled_cur:
            img_refs_l = [img]
            ref_id_final = frame_id
        else:
            img_refs_l = []
            # read prev & future frames (max size)
            ref_id_final = min(frame_id + cfg.MODEL.VID.MEGA.MAX_OFFSET, self.end_id[idx])
            if frame_category == 0:
                size_local = cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL - cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION - 1
                assert size_local == cfg.MODEL.VID.MEGA.MAX_OFFSET
                ref_id_start = max(ref_id_final - cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL + 1, self.start_id[idx])
            else:
                num_ref = min(frame_diff, cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL)
                ref_id_start = max(ref_id_final - num_ref + 1, self.start_id[idx])

            for id in range(ref_id_start, ref_id_final + 1):
                ref_filename = self.pattern[idx] % id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs_l.append(img_ref)

        img_refs_g = []
        size_g = 0 if init_and_no_update_g else 1
        if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            cls = os.path.dirname(self.pattern[idx])
            size = cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_category == 0 else size_g
            shuffled_index = self.shuffled_index[cls][str(self.shot_part_num[idx])]
            for id in range(size):
                filename = self.pattern[idx] % shuffled_index[
                    (idx - self.start_id[idx] + cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1)
                    % frame_seg_len_idx]
                img_temp = Image.open(self._img_dir % filename).convert("RGB")
                img_refs_g.append(img_temp)

        if shuffled_cur:
            idx = idx_new

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)
        target.type = 'cur'
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l
        images["ref_g"] = img_refs_g
        images["frame_category"] = frame_category
        # for yot dataset, global_id == dir_id
        # for vid dataset, global_id != dir_id
        images["frame_id"] = frame_id  # dir_id of current frame
        images["start_id"] = self.start_id[idx]
        images["end_id"] = self.end_id[idx]
        images["seg_len"] = frame_seg_len_idx
        images["last_queue_id"] = ref_id_final
        images["pattern"] = self.pattern[idx]
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms

        return images, target, idx
