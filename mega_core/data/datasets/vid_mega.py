from PIL import Image
import sys
import numpy as np

from .vid import VIDDataset
from mega_core.config import cfg
import torch
import cv2

class VIDMEGADataset(VIDDataset):
    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        super(VIDMEGADataset, self).__init__(image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=is_train)
        self.stop_update_after_init_g_test = cfg.MODEL.VID.MEGA.GLOBAL.STOP_UPDATE_AFTER_INIT_TEST
        self.shuffled_cur_test = cfg.MODEL.VID.MEGA.SHUFFLED_CUR_TEST
        self.infer_batch = cfg.INPUT.INFER_BATCH
        if not self.is_train:
            self.start_index = []
            self.start_id = []
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE or self.shuffled_cur_test:
                self.shuffled_index = {}
            for id, image_index in enumerate(self.image_set_index):
                frame_id = int(image_index.split("/")[-1])
                if frame_id == 0:
                    self.start_index.append(id)
                    if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE or self.shuffled_cur_test:
                        shuffled_index = np.arange(self.frame_seg_len[id])
                        if cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        self.shuffled_index[str(id)] = shuffled_index

                    self.start_id.append(id)
                else:
                    self.start_id.append(self.start_index[-1])

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")
        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        # if a video dataset
        img_refs_l = []
        img_refs_m = []
        img_refs_g = []
        targets_l = []
        targets_m = []
        targets_g = []
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
                    target_l = self.get_groundtruth_from_filename(ref_filename)
                    targets_l.append(target_l.clip_to_image(remove_empty=True))

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
                    target_m = self.get_groundtruth_from_filename(ref_filename)
                    targets_m.append(target_m.clip_to_image(remove_empty=True))

            # global frames
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                try:
                    ref_ids = np.random.choice(self.frame_seg_len[idx], cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL, replace=False)
                except:
                    ref_ids = np.random.choice(self.frame_seg_len[idx], 5, replace=False)
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
                    targets_l.append(target.copy_with_fields(['labels']))
            if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
                for i in range(cfg.MODEL.VID.MEGA.REF_NUM_MEM):
                    img_refs_m.append(img.copy())
                    targets_m.append(target.copy_with_fields(['labels']))
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                #for i in range(cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL):
                for i in range(1): # no redundant features
                    img_refs_g.append(img.copy())
                    targets_g.append(target.copy_with_fields(['labels']))

        target.type = 'cur'  # mark as current frame information
        for i in range(len(img_refs_l)):
            targets_l[i].type = 'local'
        for i in range(len(img_refs_m)):
            targets_m[i].type = 'local'
        for i in range(len(img_refs_g)):
            targets_g[i].type = 'global'

        if False:
            img_before = np.array(img.copy()) # RGB format
            proposal_before = target.copy_with_fields(['labels'])
            height, width = img_before.shape[:-1]
            proposal_before_ref = proposal_before.resize((width, height))
            img_before = overlay_boxes(img_before, proposal_before_ref)

            img_before_ref = []
            for i in range(len(img_refs_g)):
                img_before_ref.append(np.array(img_refs_g[i].copy()))  # RGB format
                proposal_before_ref = targets_g[i].copy_with_fields(['labels'])
                height_ref, width_ref = img_before_ref[i].shape[:-1]
                proposal_before_ref_resized = proposal_before_ref.resize((width_ref, height_ref))
                img_before_ref[i] = overlay_boxes(img_before_ref[i], proposal_before_ref_resized)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], targets_l[i] = self.transforms(img_refs_l[i], targets_l[i])
            for i in range(len(img_refs_m)):
                img_refs_m[i], targets_m[i] = self.transforms(img_refs_m[i], targets_m[i])
            for i in range(len(img_refs_g)):
                img_refs_g[i], targets_g[i] = self.transforms(img_refs_g[i], targets_g[i])  # targets_g[i])

        if False:
            img_new = img.clone().permute(1, 2, 0).cpu().numpy()
            means = self.transforms.transforms[-1].mean
            for i in range(3):
                img_new[:,:,i] = img_new[:,:,i] + means[i]
            img_new = np.ascontiguousarray(img_new, dtype=np.uint8)
            proposal = target.copy_with_fields(['labels']) # target
            height1, width1 = img_new.shape[:-1]
            proposal_resized = proposal.resize((width1, height1))
            img_new = overlay_boxes(img_new, proposal_resized)

            img_new_ref = []
            for i in range(len(img_refs_g)):
                img_new_ref.append(img_refs_g[i].clone().permute(1, 2, 0).cpu().numpy())  # RGB format
                for j in range(3):
                    img_new_ref[i][:, :, j] = img_new_ref[i][:, :, j] + means[j]
                img_new_ref[i] = np.ascontiguousarray(img_new_ref[i], dtype=np.uint8)
                proposal_ref = targets_g[i].copy_with_fields(['labels'])
                height_ref1, width_ref1 = img_new_ref[i].shape[:-1]
                proposal_ref_resized = proposal_ref.resize((width_ref1, height_ref1))
                img_new_ref[i] = overlay_boxes(img_new_ref[i], proposal_ref_resized)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l
        images["ref_m"] = img_refs_m
        images["ref_g"] = img_refs_g

        return images, [[target], targets_g, targets_l], idx

    def _get_test(self, idx):
        filename = self.image_set_index[idx]
        if self.shuffled_cur_test:
            # shuffled test
            shuffled_index_cur = self.shuffled_index[str(self.start_id[idx])]
            idx_new_local = shuffled_index_cur[idx - self.start_id[idx]]
            filename_new = self.pattern[idx] % idx_new_local
            img = Image.open(self._img_dir % filename_new).convert("RGB")
            idx_new = self.start_id[idx] + idx_new_local
        else:
            img = Image.open(self._img_dir % filename).convert("RGB")

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(filename.split("/")[-1])
        frame_category = 0
        if frame_id != 0:
            frame_category = 1
            filename_prev = self.image_set_index[idx-1]
            frame_id_prev = int(filename_prev.split("/")[-1])
            frame_diff = frame_id - frame_id_prev

        if self.shuffled_cur_test:
            img_refs_l = [img]
            ref_id_final = frame_id
        else:
            img_refs_l = []
            '''
            # reading other images of the queue (not necessary to be the last one, but last one here)
            ref_id = min(self.frame_seg_len[idx] - 1, frame_id + cfg.MODEL.VID.MEGA.MAX_OFFSET)
            ref_filename = self.pattern[idx] % ref_id
            img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
            img_refs_l.append(img_ref)
            '''
            # read prev & future frames (max size)
            ref_id_final = min(frame_id + cfg.MODEL.VID.MEGA.MAX_OFFSET, self.frame_seg_len[idx] - 1)
            if frame_category == 0:
                size_local = cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL - cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION - 1
                assert size_local == cfg.MODEL.VID.MEGA.MAX_OFFSET
                ref_id_start = max(ref_id_final - cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL + 1, 0)
            else:
                num_ref = min(frame_diff, cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL)
                ref_id_start = max(ref_id_final - num_ref + 1, 0)

            for id in range(ref_id_start, ref_id_final + 1):
                ref_filename = self.pattern[idx] % id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs_l.append(img_ref)

        img_refs_g = []
        size_g = 0 if self.stop_update_after_init_g_test else 1
        if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            size = cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == 0 else size_g
            shuffled_index = self.shuffled_index[str(self.start_id[idx])]
            for id in range(size):
                filename = self.pattern[idx] % shuffled_index[
                    (idx - self.start_id[idx] + cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) % self.frame_seg_len[idx]]
                img_temp = Image.open(self._img_dir % filename).convert("RGB")
                img_refs_g.append(img_temp)

        if self.shuffled_cur_test:
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
        images["frame_id"] = frame_id
        images["start_id"] = 0
        images["end_id"] = self.frame_seg_len[idx] - 1  # minus 1 cause VID directory ids starts from 0.
        images["seg_len"] = self.frame_seg_len[idx]
        images["last_queue_id"] = ref_id_final
        images["pattern"] = self.pattern[idx]
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms

        return images, target, [idx + i for i in range(self.infer_batch)]  # idx

def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    try:
        labels = predictions.get_field("labels")
        colors = compute_colors_for_labels(labels).tolist()
    except:
        colors = [[255, 0, 0] for i in range(len(predictions.bbox))]  # bgr
    boxes = predictions.bbox

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 2
        )

    return image

PIXEL_MEAN = [123.675, 116.280, 103.530]  # RGB
PIXEL_STD = [58.395, 57.120, 57.375]

def view_image_with_boxes(img, boxes):
    # visualization when debugging
    # over_th = results[0].extra_fields['scores'] > 0.1
    # results[0] = results[0][over_th]
    img_new = img.tensors[0].permute(1, 2, 0).cpu().numpy()
    for i in range(3):
        img_new[:, :, i] = img_new[:, :, i] * PIXEL_STD[i]
        img_new[:, :, i] = img_new[:, :, i] + PIXEL_MEAN[i]
    img_new = np.ascontiguousarray(img_new, dtype=np.uint8)
    # height1, width1 = img_new.shape[:-1]
    # proposal_resized = results[0].resize((width1, height1))
    img_new_box = overlay_boxes(img_new, boxes[0][:10])

    return img_new_box

