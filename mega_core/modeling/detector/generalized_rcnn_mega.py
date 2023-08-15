# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import time

from PIL import Image
from collections import deque

import torch
from torch import nn

from mega_core.structures.image_list import to_image_list
from mega_core.structures.boxlist_ops import cat_boxlist

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

import torch.autograd.profiler as profiler

class GeneralizedRCNNMEGA(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNNMEGA, self).__init__()
        self.device = cfg.MODEL.DEVICE

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.demo = self.roi_heads.box.feature_extractor.demo

        self.memory_enable = cfg.MODEL.VID.MEGA.MEMORY.ENABLE
        self.global_enable = cfg.MODEL.VID.MEGA.GLOBAL.ENABLE
        self.global_box_enable = cfg.MODEL.VID.MEGA.GLOBAL.BOX_ATTEND
        self.global_pixel_enable = cfg.MODEL.VID.MEGA.GLOBAL.PIXEL_ATTEND
        self.global_pixel_mem_train = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_SIZE_PIXEL_TRAIN
        self.global_pixel_mem_test = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_SIZE_PIXEL_TEST

        self.mem_management_metric = self.roi_heads.box.feature_extractor.mem_management_metric

        self.base_num = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N
        self.advanced_num = int(self.base_num * cfg.MODEL.VID.MEGA.RATIO)

        self.all_frame_interval = cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL
        self.key_frame_location = cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION
        self.num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.local_box_enable = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ENABLE and cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.STAGE > 0
        self.local_pixel_enable = cfg.MODEL.VID.MEGA.LOCAL.PIXEL_ATTEND

    def local_frame_selector(self, sel_future=5, sel_prev=5):
        interval = self.all_frame_interval
        keyframe = self.key_frame_location
        min_val = -keyframe
        max_val = interval - keyframe - 1
        idx_future = [2 ** i for i in range(sel_future)]
        idx_prev = [-(2 ** i) for i in range((sel_prev - 1), -1, -1)]
        idx_all = idx_prev + [0] + idx_future
        idx_all = [max(min(x, max_val), min_val) for x in idx_all]
        idx_all = list(set(idx_all))
        idx_all.sort()
        idx_all = torch.tensor(idx_all) + keyframe

        return idx_all

    def local_pixel_level_attention(self, feats_cur, sparse=0.1):
            idx_idx = self.idx_all >= self.key_frame_location - self.frame_id
            real_idx = self.idx_all[idx_idx]
            feats_ref_selected = torch.cat([self.feats[i] for i in real_idx])
            feats_cur_enhanced = self.update_lm_pixel_with_transpose(feats=feats_cur,
                                                                     feats_ref=feats_ref_selected,
                                                                     ps=self.ps,
                                                                     sparse=sparse)
            return feats_cur_enhanced

    def update_lm_pixel_with_transpose(self, feats=None, feats_ref=None, ps=None, i=0, sparse=1., read_mem=True):
        # pixel level attention
        if feats_ref is None:
            # self attention, usually for encoding
            feats_ref = feats
        if feats.dim() == 4:
            # feats size must be [1, 1024, h ,w]
            assert len(feats) == 1
            if ps is not None:
                feats = feats + ps
            b, dim, h, w = feats.size()
            feats = feats[0].view([dim, -1]).transpose(-1, -2)
        if feats_ref.dim() == 4:
            # feats size must be [n, 1024, h ,w]
            if ps is not None:
                feats_ref = feats_ref + ps
            feats_ref = feats_ref.reshape(-1, dim, h * w).permute(0, 2, 1)

            # read frame features sparsely
            assert sparse > 0
            if sparse < 1:
                feats_refs = []
                for i in range(len(feats_ref)):
                    indices = torch.randperm(len(feats_ref[i]))[:round(len(feats_ref[i]) * sparse)]
                    feats_refs.append(feats_ref[i][indices])
                feats_ref = torch.cat(feats_refs)
        else:
            assert feats_ref.dim() == 2

        # read external mem
        if read_mem:
            if self.pixel_external_mem is not None:
                indices = torch.randperm(len(self.pixel_external_mem))[:2000]
                feats_ref = torch.cat([feats_ref, self.pixel_external_mem[indices]])
            if self.global_cache_pixel is not None:
                indices = torch.randperm(len(self.global_cache_pixel))[:2000]
                feats_ref = torch.cat([feats_ref, self.global_cache_pixel[indices]])

        feats_new_pixel, atten_weights = self.roi_heads.box.feature_extractor.update_lm_pixel(feats=feats, feats_ref=feats_ref, i=i)
        feats_new = feats_new_pixel.transpose(-1, -2).view(1, dim, h, w)

        if False:
            # update external mem with attention weights
            val, ind = atten_weights[0].sum(dim=0).topk(int(0.2*len(feats_ref)))
            self.pixel_external_mem = feats_ref[ind]

        return feats_new

    def select_pixel_ref(self, feats, feats_enhanced=None, proposals=None, mode='random', update_mem=None):
        # flatten and sample pixel-level features
        # feat : shape must be [1, 1024, h, w] or [n, 1024]
        if feats.dim() == 4:
            n, dim, h, w = feats.size()
            assert n == 1
            feats_reshaped = feats.reshape([1024, -1]).T
        elif feats.dim() == 2:
            assert mode == 'none'
            feats_reshaped = feats
        if mode == 'pooler':
            # pixels after roi-align (7x7 bi-linear pooler)
            pixels_ref = self.roi_heads.box.feature_extractor.pooler((feats,), proposals)
            pixels_ref = pixels_ref.view([len(pixels_ref), dim, -1]).transpose(-1, -2).reshape(
                [-1, 1024])  # -> [n, h*w, 1024] -> [n*h*w, 1024]
        elif mode == 'box':
            # pixels in the rpn boxes
            over_threshold_idx = proposals[0].extra_fields['scores'] > 0.9
            proposals1 = [proposals[0][over_threshold_idx]]
            over_threshold_idx2 = proposals[0].extra_fields['scores'] > 0.5
            proposals2 = [proposals[0][over_threshold_idx2]]
            pixels_index = self.roi_heads.box.feature_extractor.get_pixels_index(proposals1[0].bbox, (h, w), nsample=100)
            pixels_ref = feats_reshaped.index_select(0, pixels_index)  # new copy
            pixels_index2 = self.roi_heads.box.feature_extractor.get_pixels_index(proposals2[0].bbox, (h, w), nsample=100)
            pixels_ref2 = feats_reshaped.index_select(0, pixels_index2)  # new copy
            self.roi_heads.box.feature_extractor.pixels_last_high = pixels_ref2
        elif mode == 'random':
            # randomly select pixels
            indices = torch.randperm(len(feats_reshaped))[:250]
            pixels_ref = feats_reshaped[indices]
        elif mode == 'score':
            # pixel level ranking with objectness score
            objectness, rpn_box_regression = self.rpn.head((feats_enhanced,))
            objectness = torch.sigmoid(objectness[0])
            max_objectness = objectness[0].max(dim=0)[0]
            topk_val, pixels_index = max_objectness.flatten().topk(150)
            pixels_ref = feats_reshaped.index_select(0, pixels_index)
        elif mode == 'none':
            pixels_ref = feats_reshaped
        else:
            NotImplementedError('not supported pixel select mode')

        if update_mem is not None:
            l2_norm = feats.pow(2).sum(dim=1).sqrt() / 32.
            l2_norm = l2_norm.flatten()
            keep_irrelevant = (torch.softmax(l2_norm, dim=0) > 1 / len(l2_norm))
            pixels_ref_distinct = feats_reshaped[keep_irrelevant]
            indices = torch.randperm(len(pixels_ref_distinct))[:100]
            pixels_ref_distinct = pixels_ref_distinct[indices].clone().detach()
        if update_mem == 'local':
            self.roi_heads.box.feature_extractor.pixels_irr = pixels_ref_distinct
            if not self.training:
                if self.pixel_external_mem is None:
                    self.pixel_external_mem = pixels_ref
                else:
                    self.pixel_external_mem = torch.cat([self.pixel_external_mem, pixels_ref])
                    num_mem = 24000
                    if len(self.pixel_external_mem) > num_mem:
                        indices = torch.randperm(len(self.pixel_external_mem))[:num_mem]
                        self.pixel_external_mem = self.pixel_external_mem[indices]
        elif update_mem == 'global':
            self.roi_heads.box.feature_extractor.pixels_irr_g = pixels_ref_distinct
            if not self.training:
                feats_g_pixel_new, _ = self.roi_heads.box.feature_extractor.update_erase_memory(
                    feats_new=pixels_ref,
                    feats_mem=self.global_cache_pixel,
                    target_size=self.global_pixel_mem_test)
                self.global_cache_pixel = feats_g_pixel_new
        else:
            NotImplementedError('not supported update_mem type')

        return pixels_ref

    def update_pixel_memory(self, mem_pix=None, feats_new=None):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor

        # update pixel-level memory
        feats_pixel = feats_new.view([1024, -1]).transpose(-1, -2)  # [2268, 1024]
        mem_pix_new, _ = self.roi_heads.box.feature_extractor.update_erase_memory(feats_new=feats_pixel,
                                                                                  feats_mem=mem_pix,
                                                                                  target_size=500)
        #merged_feat = torch.cat([mem_pix, feats_g_pixel_view], dim=0)
        #idx_to_be_remained = self.roi_heads.box.feature_extractor.select_farthest_k_greedy(merged_feat=merged_feat, k=1000)
        #mem_pix = merged_feat[idx_to_be_remained]

        return mem_pix_new

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
        # 1. build memory
        proposals_m_list = []
        if imgs_m:
            concat_imgs_m = torch.cat([img.tensors for img in imgs_m], dim=0)
            concat_feats_m = self.backbone(concat_imgs_m)[0]
            feats_m_list = torch.chunk(concat_feats_m, len(imgs_m), dim=0)

            for i in range(len(imgs_m)):
                proposals_ref = self.rpn(imgs_m[i], (feats_m_list[i], ), version="ref")
                proposals_m_list.append(proposals_ref[0])
        else:
            feats_m_list = []

        # 3. build global frames
        proposals_g_list = []
        pixels_ga_list = []
        pixels_g_list = []
        self.global_cache_pixel = None  # global pixel mem
        self.pixel_external_mem = None  # local pixel mem
        self.roi_heads.box.feature_extractor.pixels_irr = None
        if imgs_g:
            concat_imgs_g = torch.cat([img.tensors for img in imgs_g], dim=0)
            concat_feats_g = self.backbone(concat_imgs_g)[0]
            feats_g_list = torch.chunk(concat_feats_g, len(imgs_g), dim=0)

            if self.global_pixel_enable:
                # ps encoding
                b, d, h, w = feats_g_list[0].shape
                self.device = feats_g_list[0].device
                self.ps = self.roi_heads.box.feature_extractor.cal_positional_embedding_pixel(d, h, w).to(self.device)

                # mining pixel-level features
                for i in range(len(imgs_g)):
                    # just use all pixels
                    pixels_ref = feats_g_list[i][0].reshape([1024, -1]).T
                    pixels_ga_list.append(pixels_ref)
                    # randomly select pixels
                    indices = torch.randperm(len(pixels_ref))[:500]
                    pixels_g_list.append(pixels_ref[indices])

                feats_ga_pixel = torch.cat(pixels_ga_list, dim=0)
                feats_g_pixel = torch.cat(pixels_g_list, dim=0)
                if True:
                    # update global pixel mem
                    self.global_cache_pixel = feats_g_pixel
                    # update pixels_irr mem
                    self.select_pixel_ref(feats_ga_pixel, mode='none', update_mem='global')
                else:
                    feats_g_pixel_new, _ = self.roi_heads.box.feature_extractor.update_erase_memory(
                        feats_new=None,
                        feats_mem=feats_ga_pixel,
                        target_size=self.global_pixel_mem_train)
                    self.global_cache_pixel = feats_g_pixel_new

            if self.global_box_enable:
                feats_g_list_new = []
                for i in range(len(feats_g_list)):
                    feats = feats_g_list[i]
                    if self.global_pixel_enable:
                        # enhance global feats
                        feats = self.update_lm_pixel_with_transpose(feats=feats, feats_ref=None,  # self
                                                                    ps=self.ps, sparse=0.25, read_mem=True)
                        feats_g_list_new.append(feats)
                    # with torch.no_grad():
                    proposals_ref = self.rpn(imgs_g[i], (feats, ), version="ref")
                    proposals_g_list.append(proposals_ref[0])
                if self.global_pixel_enable:
                    feats_g_list = feats_g_list_new

        else:
            feats_g_list = []

        # 2. build local frames
        concat_imgs_l = torch.cat([img_cur.tensors, *[img.tensors for img in imgs_l]], dim=0)
        concat_feats_l, concat_feats_l_origin = self.backbone(concat_imgs_l)

        if self.local_pixel_enable and False:
            feats_l_list = []
            pixels_l_list = []
            concat_feats_l_split = concat_feats_l.split(1)
            for i in range(len(concat_feats_l_split)):
                pixels_ref = concat_feats_l_split[i][0].reshape([1024, -1]).T
                indices = torch.randperm(len(pixels_ref))[:500]
                pixels_l_list.append(pixels_ref[indices])
            feats_l_pixel = torch.cat(pixels_l_list, dim=0)

            if not self.local_box_enable:
                imgs_l = []
            for i in range(1 + len(imgs_l)):
                feats_l_list.append(self.update_lm_pixel_with_transpose(feats=concat_feats_l_split[i],
                                                                        feats_ref=feats_l_pixel))
        else:
            # use all feats
            num_imgs = 1 + len(imgs_l)
            feats_l_list = torch.chunk(concat_feats_l, num_imgs, dim=0)

            if not self.local_box_enable:
                imgs_l = []
                if self.local_pixel_enable:
                    pixels_ref_all = concat_feats_l.permute(0, 2, 3, 1).reshape(-1, d)
                    self.select_pixel_ref(feats=pixels_ref_all, mode='none', update_mem='local')
                    # merge global & local distinct pixels for better training
                    self.roi_heads.box.feature_extractor.pixels_irr = \
                        torch.cat([self.roi_heads.box.feature_extractor.pixels_irr, self.roi_heads.box.feature_extractor.pixels_irr_g])
                    #pixel enhancement with local pixels
                    feats_cur_enhanced = self.update_lm_pixel_with_transpose(feats=feats_l_list[0], feats_ref=concat_feats_l,
                                                                             ps=self.ps, sparse=0.25, read_mem=True)
                    if False:
                        # embeding & de-embedding
                        feats_cur_enhanced = self.backbone.body.new_conv2(feats_cur_enhanced) + concat_feats_l_origin[0]
                    feats_l_list = [feats_cur_enhanced, ]

        proposals, proposal_losses = self.rpn(img_cur, (feats_l_list[0],), targets, version="key")

        proposals_l_list = []
        proposals_cur = self.rpn(img_cur, (feats_l_list[0], ), version="ref")
        proposals_l_list.append(proposals_cur[0])
        for i in range(len(imgs_l)):
            proposals_ref = self.rpn(imgs_l[i], (feats_l_list[i + 1], ), version="ref")
            proposals_l_list.append(proposals_ref[0])

        feats_list = [feats_l_list, feats_m_list, feats_g_list]
        proposals_list = [proposals, proposals_l_list, proposals_m_list, proposals_g_list]
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats_list,
                                                        proposals_list,
                                                        targets)
        else:
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def _forward_test(self, imgs, infos, targets=None):
        """
        forward for the test phase.
        :param imgs:
        :param infos:
        :param targets:
        :return:
        """
        def update_feature(img=None, feats=None, proposals=None, proposals_feat=None):
            assert (img is not None) or (feats is not None)
            #with profiler.record_function("(CUR) BACKBONE"):
            if feats is None:
                feats = self.backbone(img)[0]
                # note here it is `imgs`! for we only need its shape, it would not cause error, but is not explicit.
                if self.global_pixel_enable:
                    feats_new = self.update_lm_pixel_with_transpose(feats)
                    feats = feats_new
            if proposals is None:
                proposals, _ = self.rpn(imgs, (feats,), None)
            if proposals_feat is None:
                proposals_feat = self.roi_heads.box.feature_extractor(feats, [proposals[0]], pre_calculate=True)

            self.feats.append(feats)
            self.proposals_300.append(proposals)
            self.proposals.append(proposals[0][:self.base_num])
            self.proposals_dis.append(proposals[0][:self.advanced_num])
            self.proposals_feat_300.append(proposals_feat)
            self.proposals_feat.append(proposals_feat[:self.base_num])
            self.proposals_feat_dis.append(proposals_feat[:self.advanced_num])

        if targets is not None and not self.demo:
            raise ValueError("In testing mode, targets should be None")

        if infos["frame_category"] == 0:  # a new video
            if self.global_enable:
                if self.global_box_enable:
                    self.roi_heads.box.feature_extractor.init_global()
                    self.proposals_global_mem = None
                    self.proposals_global_mem_last = None
                    self.roi_heads.box.feature_extractor.feat_support = None
                    self.roi_heads.box.feature_extractor.alpha = [None, None]
                if self.global_pixel_enable or self.local_pixel_enable:
                    # global_pixel_memories
                    self.global_cache_pixel = None
                    # local_pixel_memories
                    self.pixel_external_mem = None
                    self.roi_heads.box.feature_extractor.pixels_irr = None
                    self.roi_heads.box.feature_extractor.pixels_last_high = None

        # get frame info
        self.frame_id = infos["frame_id"]  # dir id of the current frame
        self.start_id = infos["start_id"]  # dir id of the first frame of video
        self.end_id = infos["end_id"]  # dir id of the last frame of video
        self.seg_len = infos["seg_len"]  # total video length
        self.last_queue_id = infos["last_queue_id"]  # dir id of the last frame of queue

        # 1. update global
        if infos["ref_g"]:
            # extract features from global images
            feats_g_list = []
            feats_g_enhanced_list = []
            global_imgs = torch.cat([img.tensors for img in infos["ref_g"]])
            global_imgs_splits = global_imgs.split(32)
            for imgs_split in global_imgs_splits:
                feats_g_split = self.backbone(imgs_split)[0]
                feats_g_list_splits = feats_g_split.chunk(len(feats_g_split))
                feats_g_list.extend(feats_g_list_splits)
            # sample global pixel features
            if self.global_pixel_enable:
                for i in range(len(feats_g_list)):
                    feats_g = feats_g_list[i]
                    # this is now self-attention encoding
                    #feats_enhanced = self.update_lm_pixel_with_transpose(feats=feats_g)
                    #feats_g_enhanced_list.append(feats_enhanced)
                    # global pixel-level mem update
                    self.select_pixel_ref(feats=feats_g, mode='random', update_mem='global')  # feats or feats_enhanced

            if self.global_box_enable:
                b, d, h, w = feats_g_list[0].shape
                self.device = feats_g_list[0].device
                self.ps = self.roi_heads.box.feature_extractor.cal_positional_embedding_pixel(d, h, w).to(self.device)
                for i, global_img in enumerate(infos["ref_g"]):
                    # object level memory update
                    if self.global_pixel_enable:
                        feats_enhanced = self.update_lm_pixel_with_transpose(feats=feats_g_list[i], feats_ref=None,
                                                                             ps=self.ps, sparse=0.25, read_mem=True)
                        feats_g_enhanced_list.append(feats_enhanced)
                        proposals = self.rpn(global_img, (feats_g_enhanced_list[i],), version="ref")
                        proposals_feat = self.roi_heads.box.feature_extractor(feats_g_enhanced_list[i], proposals, pre_calculate=True)
                        #proposals_feat = self.roi_heads.box.feature_extractor(feats_g_list[i], proposals, pre_calculate=True)
                    else:
                        proposals = self.rpn(global_img, (feats_g_list[i],), version="ref")
                        if self.demo:
                            proposals[0].extra_fields['frame_id'] = torch.full([75], infos['frame_id_g'][i], device=self.device)
                        proposals_feat = self.roi_heads.box.feature_extractor(feats_g_list[i], proposals, pre_calculate=True)
                    if self.mem_management_metric == 'distance' or self.mem_management_metric == 'mamba':
                        neg_feats, neg_proposals = self.roi_heads.box.feature_extractor.filter_irr_feats(proposals_feat, proposals)
                        if False:
                            neg_feats, idx = self.roi_heads.box.feature_extractor.update_erase_memory(
                                feats_new=neg_feats,
                                feats_mem=self.roi_heads.box.feature_extractor.feat_support,
                                target_size=25)
                        self.roi_heads.box.feature_extractor.feat_support = neg_feats
                        global_feat_new, idx = self.roi_heads.box.feature_extractor.update_erase_memory(
                                            feats_new=proposals_feat,
                                            feats_mem=self.roi_heads.box.feature_extractor.global_cache[0]['feats'],
                                            target_size=self.roi_heads.box.feature_extractor.mem_management_size_test)
                        self.roi_heads.box.feature_extractor.replace_global(global_feat_new, i=0)
                        if len(self.roi_heads.box.feature_extractor.global_cache) >= 2:
                            global_feat_new2, idx2 = self.roi_heads.box.feature_extractor.update_erase_memory(
                                feats_new=proposals_feat[:25],
                                feats_mem=self.roi_heads.box.feature_extractor.global_cache[1]['feats'],
                                target_size=150)
                            self.roi_heads.box.feature_extractor.replace_global(global_feat_new2, i=1)
                    elif self.mem_management_metric == 'queue':  # queue-type memory
                        self.roi_heads.box.feature_extractor.update_global(proposals_feat, i=0)
                        if len(self.roi_heads.box.feature_extractor.global_cache) >= 2:
                            self.roi_heads.box.feature_extractor.update_global(proposals_feat[:25], i=1)
                    else:
                        raise NotImplementedError

                    if self.demo and self.mem_management_metric != 'queue':
                        # DEMO: track proposals of memory features
                        box_merged = cat_boxlist([f for f in [proposals[0], self.proposals_global_mem] if f is not None])
                        self.proposals_global_mem = box_merged[idx]
                    if False:
                        # calculate score correction multiplier with min cosine distance
                        result_feat = self.roi_heads.box.feature_extractor.global_cache[0]['feats']
                        N = result_feat.size(0)
                        matrix = torch.cdist(result_feat, result_feat, p=2.0)
                        #feats_batch = result_feat.unsqueeze(dim=0).expand(N, N, 1024)
                        #matrix = 1 - torch.cosine_similarity(result_feat.unsqueeze(1), feats_batch, dim=-1)
                        min_distance = matrix.topk(2, dim=-1, largest=False)[0][:, 1]
                        mean_distance = matrix.topk(10, dim=-1, largest=False)[0][:, 1].mean(dim=-1)
                        mean_of_min = min_distance.mean()
                        self.roi_heads.box.feature_extractor.alpha[0] = min_distance / mean_of_min
                if self.demo and self.mem_management_metric != 'queue':
                    self.proposals_global_mem_last = cat_boxlist([f for f in [self.proposals_global_mem, neg_proposals] if f is not None])

        if infos["ref_l"]:
            # get future frames from ref_l images
            local_imgs = torch.cat([img.tensors for img in infos["ref_l"]])
            local_imgs_splits = local_imgs.split(32)
            feats_l_list = []
            feats_l_list_origin = []
            for imgs_split in local_imgs_splits:
                feats_l_split, feats_l_split_origin = self.backbone(imgs_split)
                feats_l_list_splits = feats_l_split.chunk(len(feats_l_split))
                feats_l_list.extend(feats_l_list_splits)
                feats_l_list_origin_splits = feats_l_split_origin.chunk(len(feats_l_split_origin))
                feats_l_list_origin.extend(feats_l_list_origin_splits)

        if infos["frame_category"] == 0:  # a new video
            self.feats = deque(maxlen=self.all_frame_interval)
            self.feats_origin = deque(maxlen=self.all_frame_interval)
            self.proposals_300 = deque(maxlen=self.all_frame_interval)
            self.proposals = deque(maxlen=self.all_frame_interval)
            self.proposals_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_dis = deque(maxlen=self.all_frame_interval)
            self.proposals_feat_300 = deque(maxlen=self.all_frame_interval)

            self.roi_heads.box.feature_extractor.init_memory()

            #create box collector when seq_nms is True
            if self.roi_heads.box.post_processor.seq_nms:
                self.roi_heads.box.post_processor.all_boxes = [[] for _ in range(self.num_classes)]

            feats_cur, feats_cur_origin = self.backbone(imgs.tensors)
            if self.demo:
                # DEMO: track current features
                self.roi_heads.box.feats_cur = feats_cur

            frame_diff = self.frame_id - self.start_id
            if not self.local_box_enable and self.local_pixel_enable:
                if len(feats_l_list) > 0:
                    # initialization of local pixel-level attention
                    # initialize ps-encoding fit to image size
                    b, d, h, w = feats_l_list[0].shape
                    self.device = feats_l_list[0].device
                    self.ps = self.roi_heads.box.feature_extractor.cal_positional_embedding_pixel(d, h, w).to(self.device)
                    # initialize local frame indexing
                    self.idx_all = self.local_frame_selector(sel_future=5, sel_prev=5)

                for i in range(self.key_frame_location - frame_diff):
                    self.feats.append(feats_cur)
                    self.feats_origin.append(feats_cur_origin)
                for img, feats_l, feats_l_origin in zip(infos["ref_l"], feats_l_list, feats_l_list_origin):
                    self.feats.append(feats_l)
                    self.feats_origin.append(feats_l_origin)
                for i in range(self.all_frame_interval - len(self.feats)):
                    self.feats.append(feats_l_list[-1])
                    self.feats_origin.append(feats_l_list_origin[-1])
            else:
                proposals_cur, _ = self.rpn(imgs, (feats_cur,), None)
                proposals_feat_cur = self.roi_heads.box.feature_extractor(feats_cur, [proposals_cur[0]], pre_calculate=True)
                # when not first frame, get previous frames from_ref_l
                for i in range(self.key_frame_location - frame_diff):
                    update_feature(None, feats_cur, proposals_cur, proposals_feat_cur)
                # fill sampled local features
                for img, feats_l in zip(infos["ref_l"], feats_l_list):
                    proposals_l, _ = self.rpn(img, (feats_l,), None)
                    proposals_feat_l = self.roi_heads.box.feature_extractor(feats_l, [proposals_l[0]], pre_calculate=True)
                    update_feature(None, feats_l, proposals_l, proposals_feat_l)
                # copy last local feature
                for i in range(self.all_frame_interval - len(self.feats)):
                    update_feature(None, feats_l, proposals_l, proposals_feat_l)

        elif infos["frame_category"] == 1:
            for img, feats_l, feats_l_origin in zip(infos["ref_l"], feats_l_list, feats_l_list_origin):
                if not self.local_box_enable and self.local_pixel_enable:
                    self.feats.append(feats_l)
                    self.feats_origin.append(feats_l_origin)
                else:
                    proposals_l, _ = self.rpn(img, (feats_l,), None)
                    proposals_feat_l = self.roi_heads.box.feature_extractor(feats_l, [proposals_l[0]], pre_calculate=True)
                    update_feature(None, feats_l, proposals_l, proposals_feat_l)

        if not self.local_box_enable and self.local_pixel_enable:  # local pixel-level enhance of feats_cur
            feats_cur = self.feats[self.key_frame_location]
            feats_cur_origin = self.feats_origin[self.key_frame_location]
            feats_cur_enhanced = self.local_pixel_level_attention(feats_cur, sparse=0.1)
            if self.backbone.body.new_conv2 is not None:
                feats_cur_enhanced = self.backbone.body.new_conv2(feats_cur_enhanced) + feats_cur_origin
            proposals, _ = self.rpn(imgs, (feats_cur_enhanced,), None)
            feats = self.roi_heads.box.feature_extractor(feats_cur_enhanced, [proposals[0]], pre_calculate=True)
            #feats = self.roi_heads.box.feature_extractor(feats_cur, [proposals[0]], pre_calculate=True)
            proposals_ref = None
            proposals_ref_dis = None
            proposals_feat_ref = None
            proposals_feat_ref_dis = None

        else:
            feats = self.proposals_feat_300[self.key_frame_location]
            proposals = self.proposals_300[self.key_frame_location]

            proposals_ref = cat_boxlist(list(self.proposals))
            proposals_ref_dis = cat_boxlist(list(self.proposals_dis))
            proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)
            proposals_feat_ref_dis = torch.cat(list(self.proposals_feat_dis), dim=0)

        proposals_list = [proposals, proposals_ref, proposals_ref_dis, proposals_feat_ref, proposals_feat_ref_dis]

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats, proposals_list, targets)
            # local pixel-level mem update
            if not self.local_box_enable and self.local_pixel_enable:
                new_pixels_ref = self.select_pixel_ref(feats_cur_enhanced, proposals=result, mode='box', update_mem='local')
            # box-level mem update
            if self.global_box_enable and False:
                # method 1. detected boxes. use final detection score (self.roi_heads.box.cur_feat_nms_idx)
                #x_sel = x[self.roi_heads.box.cur_feat_nms_idx]
                # method 2. high objectness score
                x_sel = x[proposals[0].extra_fields['objectness'] > 0.5]
                x_sel2 = x[proposals[0].extra_fields['objectness'] > 0.9]
                pos_boxes = [proposals[0][proposals[0].extra_fields['objectness'] > 0.5]]
                if self.roi_heads.box.feature_extractor.global_cache[1]['feats'] is None:
                    self.roi_heads.box.feature_extractor.global_cache[0]['feats'] = x_sel[:75]
                    self.roi_heads.box.feature_extractor.global_cache[1]['feats'] = x_sel2[:25]
                else:
                    merged1 = torch.cat([self.roi_heads.box.feature_extractor.global_cache[0]['feats'], x_sel[:75]])
                    merged2 = torch.cat([self.roi_heads.box.feature_extractor.global_cache[1]['feats'], x_sel2[:25]])
                    if len(merged1) > 24000:
                        indices = torch.randperm(len(merged1))[:24000]
                        self.roi_heads.box.feature_extractor.global_cache[0]['feats'] = merged1[indices]
                    else:
                        self.roi_heads.box.feature_extractor.global_cache[0]['feats'] = merged1
                    if len(merged2) > 8000:
                        indices = torch.randperm(len(merged2))[:8000]
                        self.roi_heads.box.feature_extractor.global_cache[1]['feats'] = merged2[indices]
                    else:
                        self.roi_heads.box.feature_extractor.global_cache[1]['feats'] = merged2

                neg_feats, neg_proposals = self.roi_heads.box.feature_extractor.filter_irr_feats(x, None, negative=False)
                self.roi_heads.box.feature_extractor.feat_support = neg_feats
        else:
            result = proposals

        if self.demo:
            self.features_all.append(proposals_feat_ref)
            self.proposals_all.append(proposals_ref)
            #labels, regression_targets = self.roi_heads.box.loss_evaluator.prepare_targets(proposals[0], targets_g[i])

        return result
