# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import time
import random

from PIL import Image
from collections import deque, namedtuple
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import numpy as np

from ..backbone import build_backbone
from ..roi_heads.box_head.box_head import DynamicHead
from ..roi_heads.box_head.loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from ..roi_heads.box_head.loss import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.layers import batched_nms

from mega_core.structures.bounding_box import BoxList
from mega_core.structures.image_list import to_image_list
from mega_core.structures.boxlist_ops import cat_boxlist
from mega_core.data.datasets.vid_mega import view_image_with_boxes
from mega_core.layers import fps

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    import math
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#from detectron2.config import CfgNode as CN
from yacs.config import CfgNode as CN

def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6
    cfg.MODEL.DiffusionDet.NUM_HEADS_LOCAL = 0

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.BIAS_LR_FACTOR = 1.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0001

    '''
    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000],
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
    '''

    # for DiffusionDet
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS = 256
    cfg.MODEL.FPN.NORM = ''
    cfg.MODEL.FPN.FUSE_TYPE = 'sum'
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]

    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

    cfg.MODEL.RESNETS.NORM = "FrozenBN"
    cfg.MODEL.RESNETS.DEPTH = 101
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False

    # not used params
    cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = (False, False, False, False)
    cfg.MODEL.RESNETS.DEFORM_MODULATED = False
    cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1

    cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

    cfg.INPUT.TO_BGR255 = False  # because torchvision model use rgb image
    cfg.INPUT.INFER_BATCH = 1

    '''
    # this is reference detectron2 cfg, which is used by DiffusionDet
    # you can compare cfg with this when you have error building model
    from detectron2.config import get_cfg
    cfg_temp = get_cfg()
    '''

class DiffusionDet(nn.Module):
    """
    Main class for Sparse R-CNN.
    """

    def __init__(self, cfg):
        super(DiffusionDet, self).__init__()
        self.device = cfg.MODEL.DEVICE

        self.global_enable = cfg.MODEL.VID.MEGA.GLOBAL.ENABLE

        self.mem_management_metric = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_METRIC
        
        self.base_num = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N
        self.advanced_num = int(self.base_num * cfg.MODEL.VID.MEGA.RATIO)

        self.all_frame_interval = cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL
        self.key_frame_location = cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION

        self.local_box_enable = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ENABLE
        self.mem_management_size_test = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_SIZE_TEST

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES  # cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS

        self.infer_batch = cfg.INPUT.INFER_BATCH

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0).to(torch.float32)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build DynamicHead
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        self.num_heads_local = self.head.num_heads_local
        self.top_k = self.head.top_k

        # Loss parameters:
        class_weight = cfg.MODEL.DiffusionDet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionDet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionDet.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionDet.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads + self.num_heads_local - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1) / 255.
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1) / 255.
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, images, targets=None):
        """
        Arguments:
            #images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            images["cur"] = to_image_list(images["cur"])
            images["ref_l"] = [to_image_list(image) for image in images["ref_l"]]
            images["ref_m"] = [to_image_list(image) for image in images["ref_m"]]
            images["ref_g"] = [to_image_list(image) for image in images["ref_g"]]

            return self._forward_train(images["cur"], images["ref_l"], images["ref_m"], images["ref_g"], targets)
        else:
            images["cur"] = to_image_list(images["cur"])
            images["ref_l"] = [to_image_list(image) for image in images["ref_l"]]
            images["ref_g"] = [to_image_list(image) for image in images["ref_g"]]

            infos = images.copy()
            infos.pop("cur")
            return self._forward_test(images["cur"], infos, targets)

    def _forward_train(self, img_cur, imgs_l, imgs_m, imgs_g, targets):
        targets, targets_g, targets_l = targets
        targets = targets + targets_l + targets_g

        num_imgs = 1 + len(imgs_l) + len(imgs_g)
        imgs_all = torch.cat([img_cur.tensors, *[img.tensors for img in imgs_l], *[img.tensors for img in imgs_g]], dim=0)
        features_dict = self.backbone(imgs_all)
        features = list()
        for p in self.in_features:
            feature = features_dict[p]  # [p][0:1]
            features.append(feature)

        # DiffusionDet training generates noisy bboxes from GT bboxes
        h, w = img_cur.image_sizes[0]
        images_whwh = torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
        targets, x_boxes, noises, t = self.prepare_targets(targets)
        for tar in targets:
            h1, w1 = img_cur.tensors.size()[-2:]
            tar["image_size_xyxy_pred"] = torch.tensor([w1, h1, w1, h1], dtype=torch.float32, device=self.device)
        t = t.squeeze(-1)
        x_boxes = x_boxes * images_whwh[None, None, :]

        outputs_class, outputs_coord = self.head(features, x_boxes, t, init_features=None, box_extract=0)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.deep_supervision:
            output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                     for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

        # new_img = view_image_with_boxes(img_cur, outputs_coord[-1])

        targets = targets[:(1 + len(imgs_l))] if self.local_box_enable else targets
        loss_dict = self.criterion(output, targets)  # [targets[0]])  # for fast training
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def _forward_test(self, imgs, infos, targets=None):
        """
        forward for the test phase.
        :param imgs:
        :param infos:
        :param targets:
        :return:
        """
        if targets is not None and not self.demo:
            raise ValueError("In testing mode, targets should be None")

        # initialization
        if infos["frame_category"] == 0:  # a new video
            self.local_img_queue = []
            self.head.proposal_feats_global = [None, None]
            self.head.proposal_feats_local = [None, None]
            self.feats = deque(maxlen=self.all_frame_interval)
            self.classes_300 = deque(maxlen=self.all_frame_interval)
            self.proposals_init_300 = deque(maxlen=self.all_frame_interval)
            self.proposals_300 = deque(maxlen=self.all_frame_interval)
            self.proposals = deque(maxlen=self.all_frame_interval)
            self.proposals_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_300 = deque(maxlen=self.all_frame_interval)

        # get frame info
        self.frame_id = infos["frame_id"]  # dir id of the current frame
        self.start_id = infos["start_id"]  # dir id of the first frame of video
        self.end_id = infos["end_id"]  # dir id of the last frame of video
        self.seg_len = infos["seg_len"]  # total video length
        self.last_queue_id = infos["last_queue_id"]  # dir id of the last frame of queue

        if self.frame_id % self.infer_batch != 0:
            self.local_img_queue += infos["ref_l"]
            return []
        else:
            infos["ref_l"] = self.local_img_queue + infos["ref_l"]
            self.local_img_queue = []

        # 1. extract features
        if infos["ref_l"] or infos["ref_g"]:
            local_imgs = [img.tensors for img in infos["ref_l"]]
            global_imgs = [img.tensors for img in infos["ref_g"]]
            total_imgs_tensor = torch.cat(local_imgs + global_imgs)
            total_imgs_tensor = self.normalizer(total_imgs_tensor)
            batch_split = self.infer_batch
            total_imgs_splits = total_imgs_tensor.split(batch_split)
            total_feats_split = []
            for imgs_split in total_imgs_splits:
                feats = self.backbone(imgs_split)
                total_feats_split.append(feats)
            total_feats, feats_l, feats_g = {}, {}, {}
            len_l, len_g = len(local_imgs), (global_imgs)
            for p in self.in_features:
                total_feats[p] = torch.cat([feats[p] for feats in total_feats_split], dim=0)
                feats_l[p] = total_feats[p][:len_l]
                feats_g[p] = total_feats[p][len_l:]

            classes_list_all, boxes_list_all, proposals_list_all, proposals_list_k1, proposals_list_k2, \
                boxes_init_list_all = [], [], [], [], [], []
            for bi, feats in enumerate(total_feats_split):
                f = []
                for p in self.in_features:
                    f.append(feats[p])

                B = len(f[0])
                h, w = imgs.image_sizes[0]
                images_whwh = torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
                images_whwh_B = images_whwh.unsqueeze(0).expand(B,-1)

                shape_g = (B, self.num_proposals, 4)
                box_init = torch.randn(shape_g, device=self.device)
                time_g = 999
                time_cond_g = torch.full((B,), time_g, device=self.device, dtype=torch.long)

                proposal_all, proposal_k1, proposal_k2 = self.model_predictions(f, images_whwh_B, box_init, time_cond_g,
                                                                              None, clip_x_start=True, box_extract=bi+1)
                boxes_init_list_all.append(box_init)
                classes_list_all.append(proposal_all[0])
                boxes_list_all.append(proposal_all[1])
                proposals_list_all.append(proposal_all[2])
                proposals_list_k1.append(proposal_k1)
                proposals_list_k2.append(proposal_k2)

            # concat all extracted data
            boxes_init_t = torch.cat(boxes_init_list_all, dim=0).view(-1, self.num_proposals, 4)
            classes_t = torch.cat(classes_list_all, dim=0).view(-1, self.num_proposals, 30)
            boxes_t = torch.cat(boxes_list_all, dim=0).view(-1, self.num_proposals, 4)
            proposals_t = torch.cat(proposals_list_all, dim=1).view(-1, self.num_proposals, self.hidden_dim)
            proposals_t1 = torch.cat(proposals_list_k1, dim=0).view(-1, self.top_k[0], self.hidden_dim)
            proposals_t2 = torch.cat(proposals_list_k2, dim=0).view(-1, self.top_k[1], self.hidden_dim)

            # prepare data for local queue and global mem
            boxes_init_all = boxes_init_t[:len_l]
            classes_all = classes_t[:len_l]
            boxes_all = boxes_t[:len_l]
            proposals_all = proposals_t[:len_l]
            proposals_l1, proposals_g1 = proposals_t1[:len_l], proposals_t1[len_l:]
            proposals_l2, proposals_g2 = proposals_t2[:len_l], proposals_t2[len_l:]

        # 2. update global mem
        if infos["ref_g"]:
            global_feat_new, idx = update_erase_memory(
                feats_new=proposals_g1.view(-1, self.hidden_dim),
                feats_mem=self.head.proposal_feats_global[0],
                target_size=self.mem_management_size_test)
            global_feat_dis_new, idx2 = update_erase_memory(
                feats_new=proposals_g2.view(-1, self.hidden_dim),
                feats_mem=self.head.proposal_feats_global[1],
                target_size=150)
            self.head.proposal_feats_global = [global_feat_new, global_feat_dis_new]

        # 3. update local mem
        if infos["frame_category"] == 0:  # a new video
            frame_diff = self.frame_id - self.start_id
            fill_idx = [0] * (self.key_frame_location - frame_diff) + list(range(len(local_imgs))) \
                       + [len(local_imgs)-1] * (self.all_frame_interval - ((self.key_frame_location - frame_diff) + len(local_imgs)))
        elif infos["frame_category"] == 1:
            fill_idx = range(len(local_imgs))
        # fill sampled local features queue
        for i in fill_idx:
            frame_feat = []
            for p in self.in_features:
                frame_feat.append(feats_l[p][i].unsqueeze(0))
            self.feats.append(frame_feat)
            self.classes_300.append(classes_all[i].unsqueeze(0))
            self.proposals_init_300.append(boxes_init_all[i].unsqueeze(0))
            self.proposals_300.append(boxes_all[i].unsqueeze(0))
            self.proposals_feat_300.append(proposals_all[i])
            if self.local_box_enable:
                self.proposals_feat.append(proposals_l1[i])
                self.proposals_feat_dis.append(proposals_l2[i])

        if self.local_box_enable:
            self.head.proposal_feats_local = [torch.cat(list(self.proposals_feat), dim=0), torch.cat(list(self.proposals_feat_dis), dim=0)]

        # get preprocessed current batch feature & queries
        batch = min(self.infer_batch, self.end_id - self.frame_id + 1) # 1  # len(self.feats)
        range_start = self.key_frame_location
        range_end = self.key_frame_location + batch
        feats_cur = []
        for j in range(len(self.in_features)):
            feats_cur.append(torch.cat([self.feats[i][j] for i in range(range_start, range_end)]))
        self.head.proposals_feat_cur = [[torch.cat([self.classes_300[i] for i in range(range_start, range_end)], dim=0),
                                        torch.cat([self.proposals_300[i] for i in range(range_start, range_end)], dim=0),
                                        torch.cat([self.proposals_feat_300[i] for i in range(range_start, range_end)], dim=0).unsqueeze(0)]]
        #feats_cur = self.feats[self.key_frame_location]

        # diffusion preparation
        images_whwh = list()
        for i in range(batch):
            h, w = imgs.image_sizes[0]  # assume image size are same
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # random init. of predict boxes
        img = torch.randn(shape, device=self.device)
        #if infos["frame_category"] == 1:
            #img = torch.cat([self.last_output["pred_boxes"], img], dim=1)[:, :self.num_proposals, :]

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        clip_denoised = True
        do_postprocess = False

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(feats_cur, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][:, :self.num_proposals], outputs_coord[-1][:, :self.num_proposals]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(input=score_per_image, dim=-1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx, dim=-1)

                pred_noise = [pred_noise[i, keep_idx[i]] for i in range(batch)]
                x_start = [x_start[i, keep_idx[i]] for i in range(batch)]
                if self.head.use_topk:
                    img = [img[self.head.topk_idx_bool].view(*box_per_image.size())[i, keep_idx[i]] for i in range(batch)]
                else:
                    img = [img[i, keep_idx[i]] for i in range(batch)]
            if time_next < 0:
                img = x_start
                continue

            # DDIM
            alpha = self.alphas_cumprod[time].to('cpu').to(torch.float64)
            alpha_next = self.alphas_cumprod[time_next].to('cpu').to(torch.float64)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            alpha_next = self.alphas_cumprod[time_next]

            for i in range(batch):
                noise = torch.randn_like(img[i])

                img[i] = x_start[i] * alpha_next.sqrt() + \
                         c.to(torch.float32).to('cuda') * pred_noise[i] + \
                         sigma.to(torch.float32).to('cuda') * noise

                if self.box_renewal:  # filter
                    # replenish with randn boxes
                    img[i] = torch.cat((img[i], torch.randn(self.num_proposals - num_remain[i], 4, device=img[i].device)), dim=0)
            img = torch.stack(img, dim=0)

            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_batch, scores_batch, labels_batch = self.inference(outputs_class[-1],
                                                                            outputs_coord[-1],
                                                                            imgs.image_sizes * batch)
                ensemble_score.append(scores_batch)
                ensemble_label.append(labels_batch)
                ensemble_coord.append(box_pred_batch)

        # NMS
        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_batch = torch.cat(ensemble_coord, dim=1)
            scores_batch = torch.cat(ensemble_score, dim=1)
            labels_batch = torch.cat(ensemble_label, dim=1)

            results = []
            image_size = imgs.image_sizes[0]
            for i, (scores_per_image, box_pred_per_image, labels_per_image) in enumerate(zip(
                    scores_batch, box_pred_batch, labels_batch)):
                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                boxlist = BoxList(box_pred_per_image, image_size[::-1], mode="xyxy")
                boxlist.add_field("scores", scores_per_image)
                boxlist = boxlist.clip_to_image(remove_empty=False)
                if (labels_per_image == 0).sum():
                    raise NotImplementedError('Not supported model')
                boxlist.add_field("labels", labels_per_image)
                results.append(boxlist)
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            self.last_output = {'pred_logits': outputs_class[-1][keep_idx], 'pred_boxes': outputs_coord[-1][keep_idx]}
            results = self.inference(box_cls, box_pred, imgs.image_sizes * batch)

        # new_img = view_image_with_boxes(imgs, results)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, imgs.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False, box_extract=0):
        # test phase input output coordinates change
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        if box_extract:
            return self.head(backbone_feats, x_boxes, t, None, box_extract)
        else:
            outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        k = 0  # self.key_frame_location

        x_start = outputs_coord[-1]  # (stages, batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[k, None, :]  # images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        if self.head.use_topk:
            pred_noise = self.predict_noise_from_start(x[self.head.topk_idx_bool].view(*x_start.size()), t, x_start)
        else:
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord


    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample ( a(t) * signal + (1-a(t)) + noise, a is cosine func )
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            w, h = targets_per_image.size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.get_field('labels').long()
            gt_boxes = targets_per_image.bbox / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.bbox.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.area().to(self.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        num_proposals = box_cls.size(1)
        results = []
        box_pred_batch, scores_batch, labels_batch = [], [], []
        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            # logits 0~30 -> predict labels 1~31
            labels = torch.arange(1, self.num_classes + 1, device=self.device). \
                unsqueeze(0).repeat(num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    box_pred_batch.append(box_pred_per_image)
                    scores_batch.append(scores_per_image)
                    labels_batch.append(labels_per_image)
                    continue

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                if False:
                    # use torchvision 'Boxes' class
                    result.pred_boxes = Boxes(box_pred_per_image)
                    result.scores = scores_per_image
                    result.pred_classes = labels_per_image
                    results.append(result)
                else:
                    # use BoxList
                    boxlist = BoxList(box_pred_per_image, image_size[::-1], mode="xyxy")
                    boxlist.add_field("scores", scores_per_image)
                    boxlist = boxlist.clip_to_image(remove_empty=False)
                    if (labels_per_image == 0).sum():
                        raise NotImplementedError('Not supported model')
                    boxlist.add_field("labels", labels_per_image)
                    results.append(boxlist)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_batch = torch.stack(box_pred_batch, dim=0)
                scores_batch = torch.stack(scores_batch, dim=0)
                labels_batch = torch.stack(labels_batch, dim=0)
                return box_pred_batch, scores_batch, labels_batch
        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms and False:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

def update_erase_memory(feats_new=None, feats_mem=None, rois_new=None, rois_mem=None, target_size=None, mem_management_type="greedy"):
    #  feats_mem: n obj features
    #  feats_new: k obj features
    #  returns target_size updated feats
    assert target_size is not None

    merged_feat_list = [feats_mem, feats_new]
    merged_feat_list = [f for f in merged_feat_list if f is not None]
    merged_feat = torch.cat(merged_feat_list, dim=0)
    if len(merged_feat) <= target_size:
        return merged_feat, torch.arange(len(merged_feat), device=merged_feat.device)

    if mem_management_type == "greedy":
        idx_to_be_remained = select_farthest_k_greedy_cuda(merged_feat=merged_feat, k=target_size)
    elif mem_management_type == "random":
        idx_to_be_remained = np.random.choice(len(merged_feat), target_size, replace=False)
    else:
        raise NotImplementedError

    result_feat = merged_feat[idx_to_be_remained]

    if rois_new is not None:
        merged_rois_list = torch.cat([rois_mem, rois_new], dim=0)
        result_rois = merged_rois_list[idx_to_be_remained]
        return result_feat, result_rois
    else:
        return result_feat, idx_to_be_remained

def select_farthest_k_greedy_cuda(merged_feat: torch.Tensor, k: int) -> torch.Tensor:
    """
    Uses iterative furthest point sampling to select a set of npoint features that have the largest
    minimum distance
    :param merged_feat: (N, 1024) where N > k
    :param distance: (N, N)
    :param k: int, number of features in the sampled set
    :return:
         output: (B, k) tensor containing the set
    """
    assert merged_feat.is_contiguous()
    distance = torch.cdist(merged_feat, merged_feat, p=2.0)  # l2 distance n * n

    #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    #cos_sim = cos(merged_feat.repeat_interleave(len(merged_feat), dim=0), merged_feat.repeat(len(merged_feat), 1))
    ##distance = 1 - cos_sim.view(len(merged_feat), len(merged_feat))
    #distance = cos_sim.arccos().view(len(merged_feat), len(merged_feat))  # Angular distance

    # dot_product = torch.bmm(merged_feat[None, :, :], merged_feat[None, :, :].transpose(1, 2))[0] / 32.
    # distance = 1 / (dot_product + 1e-10)

    B = 1
    N = merged_feat.size()[0]
    output = torch.cuda.IntTensor(B, k)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

    fps(B, N, k, distance, temp, output)
    return output[0].type(torch.LongTensor)