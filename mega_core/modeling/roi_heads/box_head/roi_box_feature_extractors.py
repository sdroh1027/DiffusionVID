# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
from collections import deque

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mega_core.modeling import registry
from mega_core.modeling.backbone import resnet
from mega_core.modeling.poolers import Pooler
from mega_core.modeling.make_layers import group_norm
from mega_core.modeling.make_layers import make_fc, Conv2d

from mega_core.structures.boxlist_ops import cat_boxlist

import numpy as np
#import time

from mega_core.layers import fps
import torch.autograd.profiler as profiler

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("ResNetConv52MLPFeatureExtractor")
class ResNetConv52MLPFeatureExtractor(nn.Module):
    """
    Heads for Faster R-CNN MSRA version for classification
    """

    def __init__(self, cfg, in_channels):
        super(ResNetConv52MLPFeatureExtractor, self).__init__()

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=1,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION,
        )

        in_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 2 ** (stage.index - 1)
        if cfg.MODEL.VID.ROI_BOX_HEAD.REDUCE_CHANNEL:
            new_conv = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)
            nn.init.kaiming_uniform_(new_conv.weight, a=1)
            nn.init.constant_(new_conv.bias, 0)
            output_channel = 256
        else:
            new_conv = None
            output_channel = in_channels

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.head = head
        self.conv = new_conv
        self.pooler = pooler

        input_size = output_channel * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)

        self.out_channels = representation_size

    def forward(self, x, proposals):
        if self.conv is not None:
            x = self.head(x[0])
            x = (F.relu(self.conv(x)),)
        else:
            x = (self.head(x[0]),)
        x = self.pooler(x, proposals)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class AttentionExtractor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(AttentionExtractor, self).__init__()

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        return embedding

    @staticmethod
    def extract_position_matrix(bbox, ref_bbox):
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

        return position_matrix

    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding,
                                    feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                    index=0):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param ref_feat: [num_nongt_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, num_nongt_rois]
        position_feat_1 = F.relu(self.Wgs[index](position_embedding))
        # aff_weight, [num_rois, group, num_nongt_rois, 1]
        aff_weight = position_feat_1.permute(2, 1, 3, 0)
        # aff_weight, [num_rois, group, num_nongt_rois]
        aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

    def cal_position_embedding(self, rois1, rois2):
        # [num_rois, num_nongt_rois, 4]
        position_matrix = self.extract_position_matrix(rois1, rois2)
        # [num_rois, num_nongt_rois, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.permute(2, 0, 1)
        # [1, 64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.unsqueeze(0)

        return position_embedding

    def cal_positional_embedding_pixel(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe[None, :]

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("RDNFeatureExtractor")
class RDNFeatureExtractor(AttentionExtractor):
    """
    Heads for Faster R-CNN MSRA version for classification
    """

    def __init__(self, cfg, in_channels):
        super(RDNFeatureExtractor, self).__init__(cfg, in_channels)

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=1,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION,
        )

        in_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 2 ** (stage.index - 1)
        if cfg.MODEL.VID.ROI_BOX_HEAD.REDUCE_CHANNEL:
            new_conv = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)
            nn.init.kaiming_uniform_(new_conv.weight, a=1)
            nn.init.constant_(new_conv.bias, 0)
            output_channel = 256
        else:
            new_conv = None
            output_channel = in_channels

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.head = head
        self.conv = new_conv
        self.pooler = pooler

        input_size = output_channel * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN

        if cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ENABLE:
            self.embed_dim = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.EMBED_DIM
            self.groups = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.GROUP
            self.feat_dim = representation_size

            self.base_stage = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.STAGE
            self.advanced_stage = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ADVANCED_STAGE

            self.base_num = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N
            self.advanced_num = int(self.base_num * cfg.MODEL.VID.RDN.RATIO)

            fcs, Wgs, Wqs, Wks, Wvs = [], [], [], [], []

            for i in range(self.base_stage + self.advanced_stage + 1):
                r_size = input_size if i == 0 else representation_size

                if i == self.base_stage and self.advanced_stage == 0:
                    break

                if i != self.base_stage + self.advanced_stage:
                    fcs.append(make_fc(r_size, representation_size, use_gn))
                Wgs.append(Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
                Wqs.append(make_fc(self.feat_dim, self.feat_dim))
                Wks.append(make_fc(self.feat_dim, self.feat_dim))
                Wvs.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0,
                                  groups=self.groups))
                for l in [Wgs[i], Wvs[i]]:
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
            self.fcs = nn.ModuleList(fcs)
            self.Wgs = nn.ModuleList(Wgs)
            self.Wqs = nn.ModuleList(Wqs)
            self.Wks = nn.ModuleList(Wks)
            self.Wvs = nn.ModuleList(Wvs)

        self.out_channels = representation_size

    def forward(self, x, proposals, pre_calculate=False):
        if pre_calculate:
            return self._forward_ref(x, proposals)

        if self.training:
            return self._forward_train(x, proposals)
        else:
            return self._forward_test(x, proposals)

    def _forward_train(self, x, proposals):
        num_refs = len(x) - 1
        x = self.head(torch.cat(x, dim=0))
        if self.conv is not None:
            x = F.relu(self.conv(x))
        x, x_refs = torch.split(x, [1, num_refs], dim=0)

        proposals, proposals_cur, proposals_refs = proposals[0][0], proposals[1], proposals[2:]

        x, x_cur = torch.split(self.pooler((x,), [cat_boxlist([proposals, proposals_cur], ignore_field=True), ]),
                               [len(proposals), len(proposals_cur)], dim=0)
        x, x_cur = x.flatten(start_dim=1), x_cur.flatten(start_dim=1)

        if proposals_refs:
            x_refs = self.pooler((x_refs,), proposals_refs)
            x_refs = x_refs.flatten(start_dim=1)
            x_refs = torch.cat([x_cur, x_refs], dim=0)
        else:
            x_refs = x_cur

        rois_cur = proposals.bbox
        rois_ref = cat_boxlist([proposals_cur, *proposals_refs]).bbox
        position_embedding = self.cal_position_embedding(rois_cur, rois_ref)

        x_refs = F.relu(self.fcs[0](x_refs))

        for i in range(self.base_stage):
            x = F.relu(self.fcs[i](x))
            attention = self.attention_module_multi_head(x, x_refs, position_embedding,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=i)
            x = x + attention

        if self.advanced_stage > 0:
            x_refs_adv = torch.cat([x[:self.advanced_num] for x in torch.split(x_refs, self.base_num, dim=0)], dim=0)
            rois_ref_adv = torch.cat([x[:self.advanced_num] for x in torch.split(rois_ref, self.base_num, dim=0)],
                                     dim=0)
            position_embedding_adv = torch.cat(
                [x[..., :self.advanced_num] for x in torch.split(position_embedding, self.base_num, dim=-1)], dim=-1)

            position_embedding = self.cal_position_embedding(rois_ref_adv, rois_ref)

            for i in range(self.advanced_stage):
                attention = self.attention_module_multi_head(x_refs_adv, x_refs, position_embedding,
                                                             feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                             index=i + self.base_stage)
                x_refs_adv = x_refs_adv + attention
                x_refs_adv = F.relu(self.fcs[i + self.base_stage](x_refs_adv))

            attention = self.attention_module_multi_head(x, x_refs_adv, position_embedding_adv,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=self.base_stage + self.advanced_stage)
            x = x + attention

        return x

    def _forward_ref(self, x, proposals):
        if self.conv is not None:
            x = self.head(x)
            x = (F.relu(self.conv(x)),)
        else:
            x = (self.head(x),)
        x = self.pooler(x, proposals)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fcs[0](x))

        return x

    def _forward_test(self, x, proposals):
        proposals, proposals_ref, x_refs = proposals

        rois_cur = cat_boxlist(proposals).bbox
        rois_ref = proposals_ref.bbox

        if self.conv is not None:
            x = self.head(x)
            x = (F.relu(self.conv(x)),)
        else:
            x = (self.head(x),)
        x = self.pooler(x, proposals)
        x = x.flatten(start_dim=1)

        position_embedding = self.cal_position_embedding(rois_cur, rois_ref)

        for i in range(self.base_stage):
            x = F.relu(self.fcs[i](x))
            attention = self.attention_module_multi_head(x, x_refs, position_embedding,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=i)
            x = x + attention

        if self.advanced_stage > 0:
            x_refs_adv = torch.cat([x[:self.advanced_num] for x in torch.split(x_refs, self.base_num, dim=0)], dim=0)
            rois_ref_adv = torch.cat([x[:self.advanced_num] for x in torch.split(rois_ref, self.base_num, dim=0)],
                                     dim=0)
            position_embedding_adv = torch.cat(
                [x[..., :self.advanced_num] for x in torch.split(position_embedding, self.base_num, dim=-1)], dim=-1)

            position_embedding = self.cal_position_embedding(rois_ref_adv, rois_ref)

            for i in range(self.advanced_stage):
                attention = self.attention_module_multi_head(x_refs_adv, x_refs, position_embedding,
                                                             feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                             index=i + self.base_stage)
                x_refs_adv = x_refs_adv + attention
                x_refs_adv = F.relu(self.fcs[i + self.base_stage](x_refs_adv))

            attention = self.attention_module_multi_head(x, x_refs_adv, position_embedding_adv,
                                                         feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                         index=self.base_stage + self.advanced_stage)
            x = x + attention

        return x


def select_farthest_k(feat_select, feat_target, k, reciprocal=False):
    distance = torch.cdist(feat_select, feat_target, p=2.0)  # l2 distance n+k * n
    if reciprocal:
        distance_reciprocal = 1 / distance
        #distance_reciprocal = distance_reciprocal.fill_diagonal_(0.)
        metric = distance_reciprocal.sum(-1)
        not_inf = (metric < 10000).sum()
        assert not_inf > 0
        k = min(k, not_inf)
        _, idx_to_be_remained = metric.topk(k, dim=-1, largest=False, sorted=False)  # 1/r 의 경우 False
    else:
        metric = distance.sum(-1)
        _, idx_to_be_remained = metric.topk(k, dim=-1, largest=True, sorted=False)  # 1/r 의 경우 False
    return idx_to_be_remained


def select_farthest_k_no_fill_zero(feat_select, feat_target, k):
    distance = torch.cdist(feat_select, feat_target, p=2.0)  # l2 distance n+k * n
    distance_reciprocal = 1 / distance
    # distance_reciprocal = distance_reciprocal.fill_diagonal_(0.)
    metric = distance_reciprocal.sum(-1)
    not_inf = (metric < 10000).sum()
    # assert not_inf > 0
    if not_inf == 0:
        return None
    k_new = min(k, not_inf)
    # feature vector 간 metric top k개 선택
    # _, topk_metric = metric.topk(k_new, dim=-1, largest=True, sorted=False)  # 1/r 의 경우 False
    _, topk_metric = metric.topk(k_new, dim=-1, largest=False, sorted=False)  # 1/r 의 경우 False
    idx_to_be_remained = topk_metric
    return idx_to_be_remained


def select_farthest_k_sequential(merged_feat, k):
    distance = torch.cdist(merged_feat, merged_feat, p=2.0)  # l2 distance n * n
    distance_reciprocal = 1 / distance
    #distance_reciprocal = distance_reciprocal.fill_diagonal_(0.)
    D = distance_reciprocal
    perm = torch.zeros(k, dtype=torch.int64, device=merged_feat.device) # np.zeros(N, dtype=np.int64)
    perm[0] = 0
    ds = D[0, :]
    for i in range(1, k):
        idx = torch.argmin(ds)
        perm[i] = idx
        ds = ds + D[idx, :]
    return perm  # torch.tensor(perm)# perm # .sort()[0] #(perm, lambdas)

@torch.jit.script
def getGreedyPerm(D, N, start):
    # type: (torch.Tensor, int, int) -> torch.Tensor # List[torch.Tensor]
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    Parameters
    ----------
    D : Tensor (N, N)
        An NxN distance matrix for points
    N : int
        target sample size
    Return
    ------
    perm : Tensor (N)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = torch.zeros(N, dtype=torch.int64, device=D.device) # np.zeros(N, dtype=np.int64)
    #perm = torch.zeros(N, dtype=torch.int64, device=torch.device('cpu'))
    perm[0] = start
    #perm = [start]
    #lambdas = np.zeros(N)
    ds = D[start, :]
    for i in range(1, N):
        idx = torch.argmax(ds)
        # idx = np.argmin(ds)
        perm[i] = idx
        #perm.append(idx)
        # lambdas[i] = ds[idx]
        ds = torch.min(ds, D[idx, :]) # np.minimum(ds, D[idx, :])
        # ds = np.maximum(ds, D[idx, :])
    return perm  # torch.tensor(perm)# perm # .sort()[0] #(perm, lambdas)

@torch.jit.script
def getGreedyPerm2(D, N, start):
    # type: (torch.Tensor, int, int) -> torch.Tensor
    """
    A Naive O(N^2) algorithm to do lowest redundancy sampling
    Parameters
    ----------
    D : Tensor (N, N)
        An NxN distance matrix for points
    N : int
        target sample size
    Return
    ------
    perm : Tensor (N)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    perm = torch.zeros(N, dtype=torch.int64, device=D.device) # np.zeros(N, dtype=np.int64)
    perm[0] = start
    #lambdas = np.zeros(N)
    ds = D[start, :]
    for i in range(1, N):
        idx = torch.argmin(ds)
        perm[i] = idx
        ds = ds + D[idx]
    return perm


@torch.jit.script
def select_farthest_k_greedy(merged_feat, k):
    # type: (torch.Tensor, int) -> torch.Tensor # List[torch.Tensor]
    distance = torch.cdist(merged_feat, merged_feat, p=2.0)  # l2 distance n * n
    # distance_reciprocal = 1 / distance
    # start = int(torch.randint(len(merged_feat), [1]))
    #distance = distance.cpu()
    idx_to_be_remained = getGreedyPerm(distance, k, 0)  # fixed initialization is more stable
    #idx_to_be_remained = idx_to_be_remained.to(merged_feat.device)
    return idx_to_be_remained #new_merged_feat  # , metric_mean

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
    '''
    # test triangle inequality
    for i in range(100):
        a, b, c = torch.randint(N, [3])
        if distance[a, b] + distance[b, c] < distance[a, c]:
            print(1)
    '''
    B = 1
    N = merged_feat.size()[0]
    output = torch.cuda.IntTensor(B, k)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

    fps(B, N, k, distance, temp, output)
    return output[0].type(torch.LongTensor)

@torch.jit.script
def select_farthest_k_greedy2(merged_feat, k):
    # type: (torch.Tensor, int) -> torch.Tensor
    distance = torch.cdist(merged_feat, merged_feat, p=2.0)  # l2 distance n * n
    distance_reciprocal = 1 / distance
    idx_to_be_remained = getGreedyPerm2(distance_reciprocal, k, 0)

    return idx_to_be_remained


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("MEGAFeatureExtractor")
class MEGAFeatureExtractor(AttentionExtractor):
    def __init__(self, cfg, in_channels):
        super(MEGAFeatureExtractor, self).__init__(cfg, in_channels)
        #torch.manual_seed(4)
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=cfg.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=cfg.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=cfg.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=cfg.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=1,
            res2_out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=cfg.MODEL.RESNETS.RES5_DILATION,
        )

        in_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * 2 ** (stage.index - 1)
        if cfg.MODEL.BACKBONE.CONV_BODY == "R-101-C5":
            head = torch.nn.Identity()
            in_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

        if cfg.MODEL.VID.ROI_BOX_HEAD.REDUCE_CHANNEL:
            reduce_dim = 1024
            new_conv = nn.Conv2d(in_channels, reduce_dim, kernel_size=1, stride=1)
            nn.init.kaiming_uniform_(new_conv.weight, a=1)
            nn.init.constant_(new_conv.bias, 0)
            output_channel = reduce_dim
        else:
            new_conv = None
            output_channel = in_channels

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.head = head
        self.conv = new_conv
        self.pooler = pooler

        input_size = output_channel * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN

        self.all_frame_interval = cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL

        if cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ENABLE:
            self.embed_dim = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.EMBED_DIM
            self.groups = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.GROUP
            self.feat_dim = representation_size

            self.stage = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.STAGE

            self.base_num = cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N
            self.advanced_num = int(self.base_num * cfg.MODEL.VID.MEGA.RATIO)

            fcs, Wgs, Wqs, Wks, Wvs, us = [], [], [], [], [], []

            for i in range(self.stage):
                r_size = input_size if i == 0 else representation_size
                fcs.append(make_fc(r_size, representation_size, use_gn))
                Wgs.append(Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
                Wqs.append(make_fc(self.feat_dim, self.feat_dim))
                Wks.append(make_fc(self.feat_dim, self.feat_dim))
                Wvs.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0,
                                  groups=self.groups))
                us.append(nn.Parameter(torch.Tensor(self.groups, 1, self.embed_dim)))
                for l in [Wgs[i], Wvs[i]]:
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                for weight in [us[i]]:
                    torch.nn.init.normal_(weight, std=0.01)

                self.l_fcs = nn.ModuleList(fcs)
                self.l_Wgs = nn.ModuleList(Wgs)
                self.l_Wqs = nn.ModuleList(Wqs)
                self.l_Wks = nn.ModuleList(Wks)
                self.l_Wvs = nn.ModuleList(Wvs)
                self.l_us = nn.ParameterList(us)

            if self.stage == 0:
                self.l_fcs = nn.ModuleList([make_fc(input_size, representation_size, use_gn)])

        # Long Range Memory
        self.memory_enable = cfg.MODEL.VID.MEGA.MEMORY.ENABLE
        self.memory_size = cfg.MODEL.VID.MEGA.MEMORY.SIZE

        # Global Box Aggregation Stage
        self.global_box_enable = cfg.MODEL.VID.MEGA.GLOBAL.BOX_ATTEND
        self.global_size = cfg.MODEL.VID.MEGA.GLOBAL.SIZE
        self.global_res_stage = cfg.MODEL.VID.MEGA.GLOBAL.RES_STAGE
        self.vanilla_MHA_box = cfg.MODEL.VID.MEGA.MHA
        if self.global_box_enable:
            if self.vanilla_MHA_box:
                att = []
                for i in range(self.global_res_stage):
                    att.append(torch.nn.MultiheadAttention(self.feat_dim, self.groups, dropout=0.0, bias=True))
                self.g_MHAttention = nn.ModuleList(att)
            else:
                Wqs, Wks, Wvs, us = [], [], [], []
                for i in range(max(self.global_res_stage, 2)): # support old version
                    Wqs.append(make_fc(self.feat_dim, self.feat_dim))
                    Wks.append(make_fc(self.feat_dim, self.feat_dim))
                    Wvs.append(Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0,
                                      groups=self.groups))
                    us.append(nn.Parameter(torch.Tensor(self.groups, 1, self.embed_dim)))
                    for l in [Wvs[i]]:
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)
                    for weight in [us[i]]:
                        torch.nn.init.normal_(weight, std=0.01)
                self.g_Wqs = nn.ModuleList(Wqs)
                self.g_Wks = nn.ModuleList(Wks)
                self.g_Wvs = nn.ModuleList(Wvs)
                self.g_us = nn.ParameterList(us)

            fcs = []
            for i in range(self.global_res_stage - 1):
                fcs.append(make_fc(representation_size, representation_size, use_gn))
            self.g_fcs = nn.ModuleList(fcs)

        # Pixel Aggregation Stage
        self.global_pixel_enable = cfg.MODEL.VID.MEGA.GLOBAL.PIXEL_ATTEND
        self.local_pixel_enable = cfg.MODEL.VID.MEGA.LOCAL.PIXEL_ATTEND
        self.feat_dim_p = representation_size
        self.groups_p = 8
        self.embed_dim_p = int(self.feat_dim_p / self.groups_p)
        self.global_pixel_stage = cfg.MODEL.VID.MEGA.GLOBAL.PIXEL_STAGE
        self.global_pixel_mem_train = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_SIZE_PIXEL_TRAIN
        self.global_pixel_mem_test = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_SIZE_PIXEL_TEST
        self.vanilla_MHA_pix = cfg.MODEL.VID.MEGA.MHA
        if self.global_pixel_enable:
            if self.vanilla_MHA_pix:
                self.gp_MHAttention = torch.nn.MultiheadAttention(self.feat_dim_p, self.groups_p, dropout=0.0, bias=True)
                self.gp_fcs = make_fc(self.feat_dim_p, self.feat_dim_p)
            else:
                fcs, Wqs, Wgs, Wks, Wvs, us = [], [], [], [], [], []

                for i in range(self.global_pixel_stage):
                    #fcs.append(make_fc(representation_size, representation_size, use_gn))
                    Wqs.append(make_fc(self.feat_dim_p, self.feat_dim_p))
                    #Wgs.append(make_fc(4, self.groups_p))  # pixel positional encoding
                    Wks.append(make_fc(self.feat_dim_p, self.feat_dim_p))
                    Wvs.append(Conv2d(self.feat_dim_p * self.groups_p, self.feat_dim_p, kernel_size=1, stride=1, padding=0,
                                      groups=self.groups_p))
                    us.append(nn.Parameter(torch.Tensor(self.groups_p, 1, self.embed_dim_p)))
                    for l in [Wvs[i]]:
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)
                    for weight in [us[i]]:
                        torch.nn.init.normal_(weight, std=0.01)

                self.gp_Wqs = nn.ModuleList(Wqs)
                #self.gp_Wgs = nn.ModuleList(Wgs)
                self.gp_Wks = nn.ModuleList(Wks)
                self.gp_Wvs = nn.ModuleList(Wvs)
                self.gp_us = nn.ParameterList(us)
                #self.gp_fcs = nn.ModuleList(fcs)

        self.out_channels = representation_size

        self.demo = False
        self.local_box_enable = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ENABLE and cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.STAGE > 0
        self.mem_management_metric = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_METRIC
        self.mem_management_type = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_TYPE
        self.mem_management_size_test = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_SIZE_TEST
        self.mem_management_size_train = cfg.MODEL.VID.MEGA.MEMORY_MANAGEMENT_SIZE_TRAIN
        self.method = cfg.MODEL.VID.METHOD
        # self.distance_linear = make_fc(r_size, representation_size, use_gn)  # move to other vector space
        # self.distance_multiplier = nn.Parameter(torch.randn(representation_size), requires_grad=True)  # multiply
        #self.confience1 = make_fc(self.feat_dim, 1)
        #self.confience2 = make_fc(self.feat_dim, 1)
        #self.tau = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding,
                                    feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                    index=0, ver="local"):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param relative_pe: [1, demb_dim, num_rois, num_nongt_rois]
        :param non_gt_index:
        :param fc_dim: same as group
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        if ver in ("local", "memory"):
            Wgs, Wqs, Wks, Wvs, us = self.l_Wgs, self.l_Wqs, self.l_Wks, self.l_Wvs, self.l_us
        elif ver == "global_p":
            Wqs, Wks, Wvs, us = self.gp_Wqs, self.gp_Wks, self.gp_Wvs, self.gp_us
        else:
            assert position_embedding is None
            Wqs, Wks, Wvs, us = self.g_Wqs, self.g_Wks, self.g_Wvs, self.g_us

        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, nongt_dim]
        if position_embedding is not None:
            position_feat_1 = F.relu(Wgs[index](position_embedding))
            # aff_weight, [num_rois, group, num_nongt_rois, 1]
            aff_weight = position_feat_1.permute(2, 1, 3, 0)
            # aff_weight, [num_rois, group, num_nongt_rois]
            aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        #alpha = self.alpha[index]
        #alpha = alpha.unsqueeze(dim=-1) if alpha is not None else 1.
        k_data = Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff_a, [group, num_rois, num_nongt_rois]
        aff_a = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))

        # aff_c, [group, 1, num_nongt_rois]
        aff_c = torch.bmm(us[index], k_data_batch.transpose(1, 2))

        # aff = aff_a + aff_b + aff_c + aff_d
        aff = aff_a + aff_c

        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        if position_embedding is not None:
            weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        else:
            weighted_aff = aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])
        if index == 0 and ver in ("global") and self.demo:
            self.affine = weighted_aff.mean(dim=1).max(dim=0)[1]
            # find which ref feature contributes most
            #heads_score, heads_idx = aff_softmax.max(dim=-1)
            #all_score, all_idx = heads_score.max(dim=-1)
            #self.contributor = heads_idx.gather(dim=1, index=all_idx[:, None]).squeeze(-1)
            score, self.contributor = aff_softmax.max(dim=1)[0].topk(k=3, dim=-1)
            self.l2_norm = ref_feat.pow(2).sum(dim=1).sqrt() / 32.
            #self.l2_norm_key = k_data.pow(2).sum(dim=1).sqrt() / 32.
            self.l2_norm_key = (k_data_batch.pow(2).sum(dim=-1).sqrt() / 8.).mean(dim=0)
        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)
        '''
        Wv_new = make_fc(self.feat_dim, self.feat_dim).cuda()
        Wv_new.weight = torch.nn.Parameter(Wvs[index].weight.squeeze(dim=-1).squeeze(dim=-1))
        vv = Wv_new(v_data)
        vvv = vv.reshape(-1, group, int(dim_group[0])).permute(1, 0, 2)
        a = aff_softmax.unsqueeze(dim=-1) * vvv.unsqueeze(dim=0)
        output2 = a.sum(dim=-2).reshape(-1, 1024)
        '''
        return output

    def attention_module_multi_head2(self, roi_feat, ref_feat, position_embedding,
                                    feat_dim=1024, dim=(1024, 1024, 1024), group=16,
                                    index=0, ver="local"):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param relative_pe: [1, demb_dim, num_rois, num_nongt_rois]
        :param non_gt_index:
        :param fc_dim: same as group
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        if ver in ("local", "memory"):
            Wgs, Wqs, Wks, Wvs, us = self.l_Wgs, self.l_Wqs, self.l_Wks, self.l_Wvs, self.l_us
        elif ver == "global_p":
            Wqs, Wks, Wvs = self.gp_Wqs, self.gp_Wks, self.gp_Wvs
        else:
            assert position_embedding is None
            Wqs, Wks, Wvs, us = self.g_Wqs, self.g_Wks, self.g_Wvs, self.g_us

        #dim = (64, 64, 64)
        #group = 1
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, nongt_dim]
        if position_embedding is not None:
            position_feat_1 = F.relu(Wgs[index](position_embedding))
            # aff_weight, [num_rois, group, num_nongt_rois, 1]
            aff_weight = position_feat_1.permute(2, 1, 3, 0)
            # aff_weight, [num_rois, group, num_nongt_rois]
            aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = Wqs[index](roi_feat).unsqueeze(dim=0)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = Wks[index](ref_feat).unsqueeze(dim=0)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff_a, [group, num_rois, num_nongt_rois]
        aff_a = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        # l2 = torch.cdist(q_data_batch, k_data_batch, p=2)
        key_abs = torch.norm(k_data_batch, dim=-1).unsqueeze(dim=1)
        cur_abs = torch.norm(q_data_batch, dim=-1).unsqueeze(dim=2)
        abss = key_abs * cur_abs * self.tau
        # cur_normpow = q_data_batch.pow(2).sum(dim=-1).unsqueeze(dim=2)
        # key_normpow = k_data_batch.pow(2).sum(dim=-1).unsqueeze(dim=1)

        # aff_c, [group, 1, num_nongt_rois]
        #aff_c = torch.bmm(us[index], k_data_batch.transpose(1, 2))

        # aff = aff_a + aff_b + aff_c + aff_d
        # aff = aff_a + aff_c
        aff = aff_a / abss  #+ aff_c   # cosine affines
        #aff = 2 * aff_a - + 2 * aff_c - 2 * key_normpow  # L2 distance affines
        #aff, idx_topk = torch.topk(input=aff, k=50, dim=-1, largest=True)

        #aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        aff_scale = aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        if position_embedding is not None:
            weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        else:
            weighted_aff = aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)
        #aff_softmax = torch.zeros_like(weighted_aff).scatter_(2, idx_topk, aff_softmax).permute(1, 0, 2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

    def forward(self, x, proposals, pre_calculate=False):
        if pre_calculate:
            return self._forward_ref(x, proposals)

        if self.training:
            return self._forward_train(x, proposals)
        else:
            return self._forward_test(x, proposals)

    def init_memory(self):
        self.mem_queue_list = []
        self.mem = []
        for i in range(self.stage):
            queue = {"rois": deque(maxlen=self.memory_size),
                     "feats": deque(maxlen=self.memory_size)}
            self.mem_queue_list.append(queue)
            self.mem.append(dict())

    def init_global(self):
        self.global_queue_list = []
        self.global_cache = []
        for i in range(max(self.global_res_stage, 2)):
            queue = {"feats": deque(maxlen=self.global_size)}
            self.global_queue_list.append(queue)
            self.global_cache.append(dict())
            self.global_cache[i]['feats'] = None

    def update_global(self, feats, i=0):
        self.global_queue_list[i]["feats"].append(feats)
        self.global_cache[i]["feats"] = torch.cat(list(self.global_queue_list[i]["feats"]), dim=0)

    def replace_global(self, feats, i=0):
        #self.global_queue_list[i]["feats"].append(feats)
        self.global_cache[i]["feats"] = feats

    def update_erase_memory(self, feats_new=None, feats_mem=None, rois_new=None, rois_mem=None, target_size=None):
        #  feats_mem: n obj features
        #  feats_new: k obj features
        #  returns target_size updated feats
        assert target_size is not None

        merged_feat_list = [feats_mem, feats_new]
        merged_feat_list = [f for f in merged_feat_list if f is not None]
        merged_feat = torch.cat(merged_feat_list, dim=0)
        if len(merged_feat) <= target_size:
            return merged_feat, torch.arange(len(merged_feat), device=merged_feat.device)

        if self.mem_management_type == "sequential":
            idx_to_be_remained = select_farthest_k_sequential(merged_feat=merged_feat, k=target_size)
        elif self.mem_management_type == "once":
            idx_to_be_remained = select_farthest_k(merged_feat, feats_mem, k=target_size, reciprocal=False)
        elif self.mem_management_type == "twice":
            select_new = select_farthest_k_no_fill_zero(feats_new, feats_mem, k=int(len(feats_new) / 2))
            if select_new is None:
                return feats_mem, None
            feats_new_selected = feats_new[select_new]
            merged_feat = torch.cat([feats_mem, feats_new_selected], dim=0)
            idx_to_be_remained = select_farthest_k_no_fill_zero(merged_feat, feats_mem, k=target_size)
        elif self.mem_management_type == "greedy":
            idx_to_be_remained = select_farthest_k_greedy_cuda(merged_feat=merged_feat, k=target_size)
        elif self.mem_management_type == "greedy2":
            idx_to_be_remained = select_farthest_k_greedy2(merged_feat=merged_feat, k=target_size)
        elif self.mem_management_type == "random" or self.mem_management_metric == "mamba":
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
        # 현재는 기준(roi_feat)을 g_cache로 했지만 x_key 나 다른걸 생각해봐도 될듯
        # distance 기준도 affine으로 했지만 달라질 수 있음 (euclidean?)
        # gaussian 분포로 투영한 다음 cdist 사용???

    def update_memory(self, i, cache):
        number_to_push = self.base_num if i == 0 else self.advanced_num

        rois = cache["rois_ref"][:number_to_push]
        feats = cache["feats_ref"][:number_to_push]

        self.mem_queue_list[i]["rois"].append(rois)
        self.mem_queue_list[i]["feats"].append(feats)

        rois_list = list(self.mem_queue_list[i]["rois"])
        feats_list = list(self.mem_queue_list[i]["feats"])

        if self.local_box_enable and self.mem[i] and False:
            num_feats_max = self.memory_size * number_to_push
            if len(self.mem[i]["feats"]) >= num_feats_max:
                new_feats, new_rois = self.update_erase_memory(feats_new=feats,
                                                               feats_mem=self.mem[i]["feats"],
                                                               rois_new=rois,
                                                               rois_mem=self.mem[i]["rois"],
                                                               target_size=num_feats_max
                                                               )
                self.mem[i] = {"rois": new_rois,
                               "feats": new_feats}
            else:
                self.mem[i] = {"rois": torch.cat(rois_list, dim=0),
                               "feats": torch.cat(feats_list, dim=0)}
        else:
            self.mem[i] = {"rois": torch.cat(rois_list, dim=0),
                           "feats": torch.cat(feats_list, dim=0)}

    def update_lm(self, feats, i=0):
        feats_ref = self.global_cache[0]["feats"]
        if feats_ref is None:
            feats_ref = feats[:self.base_num]
        #feats_ref = F.relu(feats_ref) # for old models
        if len(feats_ref) > 2000:
            idx = torch.randperm(len(feats_ref))[:2000]
            feats_ref = feats_ref[idx]
        # if self.global_res_stage == 1 and i == 1: # DAFA do not requires pre-attention stage
        if self.global_res_stage == 1:  # MEGA requires pre-attention stage
            if self.method == "dafa":
                if i == 0:
                    return feats
                elif i == 1:
                    i = 0
                    if not self.training:
                        feats_ref = torch.cat([feats_ref, self.feat_support], dim=0)
            if self.vanilla_MHA_box:
                feats = feats[:, None, :]
                feats_ref = feats_ref[:, None, :]
                attention, weights = self.g_MHAttention[i](feats, feats_ref, feats_ref)
                feats = feats + attention
                feats = feats.squeeze(1)
            else:
                attention = self.attention_module_multi_head(feats, feats_ref, None,
                                                             feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                             index=i, ver="global")  # i for MEGA, 0 for l3g1
                feats = feats + attention

            return feats
        elif self.global_res_stage == 2 and i == 1:
            if self.training:
                feats_query = torch.cat([feats, feats_ref])
            else:
                if self.global_cache[1]["feats"] is None:
                    feats_ref_advanced = feats[:self.advanced_num]
                else:
                    feats_ref_advanced = self.global_cache[-1]["feats"]
                    feats_ref_advanced = torch.cat([feats_ref_advanced, self.feat_support], dim=0)
                if len(feats_ref_advanced) > 400:
                    idx2 = torch.randperm(len(feats_ref_advanced))[:400]
                    feats_ref_advanced = feats_ref_advanced[idx2]
                feats_query = torch.cat([feats, feats_ref_advanced])
            if not self.vanilla_MHA_box:
                attention1 = self.attention_module_multi_head(feats_query, feats_ref, None,
                                                             feat_dim=1024, group=self.groups, dim=(1024, 1024, 1024),
                                                             index=0, ver="global")
            else:
                feats_query = feats_query[:, None, :]
                feats_ref = feats_ref[:, None, :]
                attention1, weights = self.g_MHAttention[0](feats_query, feats_ref, feats_ref)

            result = feats_query + attention1  # * self.confience1(feats_query)
            # if not last stage, do fc layer + relu
            result = F.relu(self.g_fcs[0](result))

            feats_query2 = result[:len(feats)]
            feats_ref2 = result[len(feats):]
            if not self.vanilla_MHA_box:
                attention2 = self.attention_module_multi_head(feats_query2, feats_ref2, None,
                                                             feat_dim=1024, group=self.groups, dim=(1024, 1024, 1024),
                                                             index=1, ver="global")
                result = feats_query2 + attention2  # * self.confience2(feats_query2)
            else:
                attention2, weights = self.g_MHAttention[1](feats_query2, feats_ref2, feats_ref2)
                result = feats_query2 + attention2
                result = result.squeeze(1)

            return result
        else:
            return feats

    def update_lm_pixel(self, feats, i=0, feats_ref=None):
        if feats_ref is None:
            feats_ref = feats
        if self.pixels_irr is not None:
            if self.training:
                feats_ref = torch.cat([feats_ref, self.pixels_irr], dim=0)
            else:
                feats_ref = torch.cat([feats_ref, self.pixels_irr, self.pixels_last_high], dim=0)
        if not self.vanilla_MHA_pix:
            attention = self.attention_module_multi_head(feats, feats_ref, position_embedding=None,
                                                         feat_dim=1024, group=self.groups_p, dim=(self.feat_dim_p, self.feat_dim_p, self.feat_dim_p),
                                                         index=i, ver="global_p")
            feats_pixel = feats + attention
            weights = None
        else:
            feats = feats[:, None, :]
            feats_ref = feats_ref[:, None, :]
            attention, weights = self.gp_MHAttention(feats, feats_ref, feats_ref)
            feats_pixel = feats + attention
            feats_pixel = feats_pixel.squeeze(1)

        return feats_pixel, weights

    def generate_feats(self, x, proposals, proposals_key=None, ver="local"):
        x = self.head(torch.cat(x, dim=0))  # 다른 레포 기준으로 5 진행
        if self.conv is not None:
            x = F.relu(self.conv(x))

        if proposals_key is not None:
            assert ver == "local"
            rois_key = proposals_key[0].bbox
            x_key = self.pooler((x[0:1, ...],), proposals_key)
            x_key = x_key.flatten(start_dim=1)

        if proposals:
            x = self.pooler((x,), proposals)
            x = x.flatten(start_dim=1)

        rois = cat_boxlist(proposals).bbox

        if ver == "local":
            x_key = F.relu(self.l_fcs[0](x_key))
        if not (ver == "local" and self.stage == 0):
            x = F.relu(self.l_fcs[0](x))

        if self.global_cache and self.global_box_enable:
            if ver == "local":
                x_key = self.update_lm(x_key)
            if not (ver == "local" and self.stage == 0):
                x = self.update_lm(x)

        # distillation
        if ver in ("local", "memory"):
            x_dis = torch.cat([x[:self.advanced_num] for x in torch.split(x, self.base_num, dim=0)], dim=0)
            rois_dis = torch.cat([x[:self.advanced_num] for x in torch.split(rois, self.base_num, dim=0)], dim=0)

        if ver == "memory":
            self.memory_cache.append({"rois_cur": rois_dis,
                                      "rois_ref": rois,
                                      "feats_cur": x_dis,
                                      "feats_ref": x})
            for _ in range(self.stage - 1):
                self.memory_cache.append({"rois_cur": rois_dis,
                                          "rois_ref": rois_dis})
        elif ver == "local":
            if self.stage == 0 or self.stage == 1:
                self.local_cache.append({"rois_cur": rois_key,
                                         "rois_ref": rois,
                                         "feats_cur": x_key,
                                         "feats_ref": x})
            else:
                self.local_cache.append({"rois_cur": torch.cat([rois_key, rois_dis], dim=0),
                                         "rois_ref": rois,
                                         "feats_cur": torch.cat([x_key, x_dis], dim=0),
                                         "feats_ref": x})
                for _ in range(self.stage - 2):
                    self.local_cache.append({"rois_cur": torch.cat([rois_key, rois_dis], dim=0),
                                             "rois_ref": rois_dis})
                self.local_cache.append({"rois_cur": rois_key,
                                         "rois_ref": rois_dis})
        elif ver == "global":
            if self.global_box_enable and self.mem_management_metric == "distance":
                # with torch.no_grad():
                num_boxes = [len(b) for b in proposals]
                xgs = torch.split(x, num_boxes)
                x_temp = xgs[0]
                for i in range(1, len(xgs)):
                    x_temp, _ = self.update_erase_memory(feats_new=xgs[i],
                                                         feats_mem=x_temp,
                                                         target_size=self.mem_management_size_train)
                self.global_cache.append({"feats": x_temp})
            else:
                self.global_cache.append({"feats": x})

    def generate_feats_test(self, x, proposals):
        proposals, proposals_ref, proposals_ref_dis, x_ref, x_ref_dis = proposals

        if self.global_box_enable: #and self.stage != 0:
            if self.stage == 0:
                x_ref = None
                x_ref_dis = None
            else:
                x = self.update_lm(x)
                x_ref = self.update_lm(x_ref)
                x_ref_dis = self.update_lm(x_ref_dis)

        if self.stage == 0:
            self.local_cache.append({"feats_cur": x})
        else:
            rois_key = proposals[0].bbox
            rois = proposals_ref.bbox
            rois_dis = proposals_ref_dis.bbox
            if self.stage == 1:
                self.local_cache.append({"rois_cur": torch.cat([rois_key], dim=0),
                                         "rois_ref": rois,
                                         "feats_cur": torch.cat([x], dim=0),
                                         "feats_ref": x_ref})
            else:
                self.local_cache.append({"rois_cur": torch.cat([rois_key, rois_dis], dim=0),
                                         "rois_ref": rois,
                                         "feats_cur": torch.cat([x, x_ref_dis], dim=0),
                                         "feats_ref": x_ref})
                for _ in range(self.stage - 2):
                    self.local_cache.append({"rois_cur": torch.cat([rois_key, rois_dis], dim=0),
                                             "rois_ref": rois_dis})
                self.local_cache.append({"rois_cur": rois_key,
                                         "rois_ref": rois_dis})

    def _forward_train_single(self, i, cache, memory=None, ver="memory"):
        rois_cur = cache.pop("rois_cur")
        rois_ref = cache.pop("rois_ref")
        feats_cur = cache.pop("feats_cur")
        feats_ref = cache.pop("feats_ref")

        if memory is not None:
            rois_ref = torch.cat([rois_ref, memory["rois"]], dim=0)
            feats_ref = torch.cat([feats_ref, memory["feats"]], dim=0)

        if ver == "memory":
            self.mem.append({"rois": rois_ref, "feats": feats_ref})
            if i == self.stage - 1:
                return

        if rois_cur is not None:
            position_embedding = self.cal_position_embedding(rois_cur, rois_ref)
        else:
            position_embedding = None

        attention = self.attention_module_multi_head(feats_cur, feats_ref, position_embedding,
                                                     feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                     index=i, ver=ver)
        feats_cur = feats_cur + attention

        if i != self.stage - 1:
            feats_cur = F.relu(self.l_fcs[i + 1](feats_cur))

        return feats_cur

    def _forward_test_single(self, i, cache, memory):
        rois_cur = cache.pop("rois_cur")
        rois_ref = cache.pop("rois_ref")
        feats_cur = cache.pop("feats_cur")
        feats_ref = cache.pop("feats_ref")

        if memory is not None:
            if memory["rois"] is not None:
                rois_ref = torch.cat([rois_ref, memory["rois"]], dim=0)
            else:
                rois_ref = None
            feats_ref = torch.cat([feats_ref, memory["feats"]], dim=0)

        if rois_cur is not None and rois_ref is not None:
            position_embedding = self.cal_position_embedding(rois_cur, rois_ref)
        else:
            position_embedding = None

        attention = self.attention_module_multi_head(feats_cur, feats_ref, position_embedding,
                                                     feat_dim=1024, group=16, dim=(1024, 1024, 1024),
                                                     index=i)
        feats_cur = feats_cur + attention

        if i != self.stage - 1:
            feats_cur = F.relu(self.l_fcs[i + 1](feats_cur))

        return feats_cur

    def _forward_train(self, x, proposals):
        proposals, proposals_l, proposals_m, proposals_g = proposals
        x_l, x_m, x_g = x

        self.global_cache = []
        self.memory_cache = []
        self.local_cache = []
        if proposals_g and self.global_box_enable and self.global_res_stage > 0:
            self.generate_feats(x_g, proposals_g, ver="global")

        if proposals_m:
            with torch.no_grad():
                self.generate_feats(x_m, proposals_m, ver="memory")

        self.generate_feats(x_l, proposals_l, proposals, ver="local")

        # 1. generate long range memory
        with torch.no_grad():
            if self.memory_cache:
                self.mem = []
                for i in range(self.stage):
                    feats = self._forward_train_single(i, self.memory_cache[i], None, ver="memory")

                    if i == self.stage - 1:
                        break

                    self.memory_cache[i + 1]["feats_cur"] = feats
                    self.memory_cache[i + 1]["feats_ref"] = feats
            else:
                self.mem = None

        # 2. update current feats
        for i in range(self.stage):
            if self.mem is not None:
                memory = self.mem[i]
            else:
                memory = None

            feats = self._forward_train_single(i, self.local_cache[i], memory, ver="local")

            if i == self.stage - 1:
                x = feats
            elif i == self.stage - 2:
                self.local_cache[i + 1]["feats_cur"] = feats[:len(proposals[0])]
                self.local_cache[i + 1]["feats_ref"] = feats[len(proposals[0]):]
            else:
                self.local_cache[i + 1]["feats_cur"] = feats
                self.local_cache[i + 1]["feats_ref"] = feats[len(proposals[0]):]

        if self.stage == 0:
            x = self.local_cache[0]["feats_cur"]

        if self.global_box_enable:
            #for i in range(self.global_res_stage):
            #    x = self.update_lm(x, i + 1)
            x = self.update_lm(x, 1)

        return x

    def _forward_ref(self, x, proposals):
        if self.conv is not None:
            x = self.head(x)
            x = (F.relu(self.conv(x)),)
        else:
            x = (self.head(x),)
        x = self.pooler(x, proposals)
        x = x.flatten(start_dim=1)

        x = F.relu(self.l_fcs[0](x))

        return x

    def _forward_test(self, x, proposals):
        # proposals, proposals_ref, x_refs = proposals
        if False:
            with profiler.record_function("(CUR_300) OBJ_FEAT EXTRACT"):
                if self.conv is not None:
                    x = self.head(x)
                    x = (F.relu(self.conv(x)),)
                else:
                    x = (self.head(x),)
                x = self.pooler(x, proposals[0])
                x = x.flatten(start_dim=1)
                x = F.relu(self.l_fcs[0](x))
                x_original = x.clone()

        self.local_cache = []

        self.generate_feats_test(x, proposals)

        for i in range(self.stage):
            memory = self.mem[i] if self.mem[i] else None

            if self.memory_enable:
                self.update_memory(i, self.local_cache[i])

            feat_cur = self._forward_test_single(i, self.local_cache[i], memory)

            if i == self.stage - 1:
                x = feat_cur
            elif i == self.stage - 2:
                self.local_cache[i + 1]["feats_cur"] = feat_cur[:len(proposals[0][0])]
                self.local_cache[i + 1]["feats_ref"] = feat_cur[len(proposals[0][0]):]
            else:
                self.local_cache[i + 1]["feats_cur"] = feat_cur
                self.local_cache[i + 1]["feats_ref"] = feat_cur[len(proposals[0][0]):]

        if self.stage == 0:
            x = self.local_cache[0]["feats_cur"]

        if self.global_box_enable:
            # for i in range(self.global_res_stage):
            #    x = self.update_lm(x, i + 1)
            x = self.update_lm(x, 1)

        return x

    def get_pixels_index(self, bboxes, wh, nsample=0):
        if len(bboxes) == 0:
            return torch.tensor([], device=bboxes.device)
        # https://discuss.pytorch.org/t/how-to-collect-coordinates-within-some-bounding-boxes-efficiently-by-pytorch/6347/7
        # Bounding boxes and points.
        bboxes = bboxes * 0.0625  # (x1,y1,x2,y2)
        x = torch.arange(wh[0], device=bboxes.device) + 0.5
        y = torch.arange(wh[1], device=bboxes.device) + 0.5
        grid_x, grid_y = torch.meshgrid(x, y)
        points = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1).reshape(-1, 2)

        # Keep a reference to the original strided `points` tensor.
        old_points = points

        # Permutate all points for every single bounding box.
        points = points.unsqueeze(1)
        points = points.repeat(1, len(bboxes), 1)

        # Create the conditions necessary to determine if a point is within a bounding box.
        # x >= left, x <= right, y >= top, y <= bottom
        c1 = points[:, :, 0] <= bboxes[:, 2]
        c2 = points[:, :, 0] >= bboxes[:, 0]
        c3 = points[:, :, 1] <= bboxes[:, 3]
        c4 = points[:, :, 1] >= bboxes[:, 1]

        # Add all of the conditions together. If all conditions are met, sum is 4.
        # Afterwards, get all point indices that meet the condition (a.k.a. all non-zero mask-summed values)
        mask = c1 & c2 & c3 & c4
        if nsample <= 0:
            # select pixels in the union of sampled bboxes.
            pixel_indexes = torch.nonzero(mask.sum(dim=-1)).squeeze()
        else:
            # select 'nsample' pixels for each bbox.
            ls = []
            for i in range(len(bboxes)):
                a = torch.nonzero(mask[:, i]).squeeze(dim=-1)
                if nsample < mask[:, i].sum():
                    idx = torch.randperm(len(a), device=a.device)[:nsample]
                    b = a[idx]
                else:
                    b = a
                ls.append(b)
            pixel_indexes = torch.cat(ls)

        return pixel_indexes  # old_points.index_select(dim=0, index=mask)

    def filter_irr_feats(self, feat, proposals=None, negative=True):
        # filter current feats with high norm
        l2_norm = feat.pow(2).sum(dim=1).sqrt() / 32.0
        keep_norm = (torch.nn.functional.softmax(l2_norm, dim=0) > 1.0 / len(feat))
        idx_keep = torch.where(keep_norm)[0]
        rand = torch.randperm(len(idx_keep))[:100]
        idx_keep = idx_keep[rand]
        proposals_keep = None
        # filter current feats with high objectness score
        if proposals and negative:
            if 'scores' in proposals[0].extra_fields.keys():
                score = cat_boxlist(proposals).get_field('scores')
            elif 'objectness' in proposals[0].extra_fields.keys():
                score = cat_boxlist(proposals).get_field('objectness')
            keep_score = score < 0.5
            # idx_keep = torch.unique(torch.cat([idx_distinct, idx_highscore]))
            idx_keep = torch.where(keep_norm * keep_score)[0]
            rand = torch.randperm(len(idx_keep))[:100]
            idx_keep = idx_keep[rand]
            proposals_keep = proposals[0][idx_keep]
        return feat[idx_keep], proposals_keep

    @staticmethod
    def extract_position_matrix_pixel(width, height):
        coords_h = torch.arange(height)
        coords_w = torch.arange(width)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        result = ((relative_coords / (width - 1)).abs() + 1.).log()
        return result

@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPN2MLPFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNXconv1fcFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_BOX_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs, ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
