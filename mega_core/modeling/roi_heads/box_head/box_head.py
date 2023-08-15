# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import copy

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from mega_core.modeling.make_layers import make_fc

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from timm.models.layers import Mlp

import math
_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


class ROIAttentionBoxHead(ROIBoxHead):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIAttentionBoxHead, self).__init__(cfg, in_channels)
        self.datasets_test = cfg.DATASETS.TEST

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals[0] = self.loss_evaluator.subsample(proposals[0], targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if not self.training and 'YouTube_Objects' in self.datasets_test:
            no_yot = list({i for i in range(31)} - {0, 1, 5, 28, 7, 10, 8, 9, 15, 19, 26})
            class_logits[:, no_yot] = - 99.

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals[0])
            self.cur_feat_nms_idx = self.post_processor.keep_feat_idx
            if hasattr(self.feature_extractor, 'demo'):
                if self.feature_extractor.demo:
                    # demo features
                    if len(targets[0]) > 0:
                        target_labels, regression_targets = self.loss_evaluator.prepare_targets(proposals[0], targets)
                        self.target_labels = target_labels[0]
                    else:
                        self.target_labels = torch.zeros(len(proposals[0][0]), dtype=torch.int64)
                    self.class_prob = torch.softmax(class_logits, dim=-1)
                    self.cur_feat = features
                    self.enhanced_feat = x

            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals[0],
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    if cfg.MODEL.VID.ENABLE:
        if cfg.MODEL.VID.METHOD in ("rdn", "mega", "dafa"):
            return ROIAttentionBoxHead(cfg, in_channels)

    return ROIBoxHead(cfg, in_channels)

# Diffusion DynamicHead
class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        d_model = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.DiffusionDet.NHEADS
        dropout = cfg.MODEL.DiffusionDet.DROPOUT
        activation = cfg.MODEL.DiffusionDet.ACTIVATION
        num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.num_heads_local = cfg.MODEL.DiffusionDet.NUM_HEADS_LOCAL
        self.adaptive_norm = True
        self.head_series_cond = nn.ModuleList([RCNNHead_cond(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation, adaptive_norm=self.adaptive_norm)
                                                for i in range(self.num_heads_local)])
        self.return_intermediate = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION

        self.local_interval = cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL
        self.key_frame = cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION
        self.local_enable = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ENABLE
        self.local_attention = []
        if self.local_enable:
            self.local_stage = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.STAGE
            for i in range(self.local_stage):
                att = []
                att.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout))
                att.append(nn.Dropout(dropout))
                att.append(nn.LayerNorm(d_model))
                self.local_attention.append(nn.ModuleList(att))
            self.local_attention = nn.ModuleList(self.local_attention)

        self.global_enable = cfg.MODEL.VID.MEGA.GLOBAL.ENABLE
        self.global_stage = cfg.MODEL.VID.MEGA.GLOBAL.RES_STAGE
        self.global_attention = []
        if self.global_enable:
            if self.global_stage > 0:
                for i in range(self.global_stage):
                    att = []
                    att.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout))
                    if not self.adaptive_norm:
                        att.append(nn.Dropout(dropout))
                        att.append(nn.LayerNorm(d_model))
                        #if i < len(self.global_stage) - 1:
                        mlp_hidden_dim = int(d_model * 4)
                        att.append(Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=0.))
                        #att.append(make_fc(d_model, d_model))
                        att.append(nn.LayerNorm(d_model))
                    self.global_attention.append(nn.ModuleList(att))
                self.global_attention = nn.ModuleList(self.global_attention)

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = cfg.MODEL.DiffusionDet.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

        self.infer_batch = cfg.INPUT.INFER_BATCH
        self.top_k = [75, 25]
        self.top_k = [min(x, cfg.MODEL.DiffusionDet.NUM_PROPOSALS) for x in self.top_k]
        self.sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, t, init_features, box_extract=0):
        # assert t shape (batch_size)
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        dim = features[0].size(1)
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]
        key_frame = 0  # self.key_frame

        if self.training or box_extract > 0 or self.sampling_timesteps > 1:
            if init_features is not None:
                init_features = init_features[None].repeat(1, bs, 1)
                proposal_features = init_features.clone()
            else:
                proposal_features = None

            # self frame feature generation task
            for head_idx, rcnn_head in enumerate(self.head_series):
                class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler, time)
                if self.return_intermediate and self.training:
                    inter_class_logits.append(class_logits) #[key_frame].unsqueeze(0))  # fast train
                    inter_pred_bboxes.append(pred_bboxes) #[key_frame].unsqueeze(0))  # fast train
                bboxes = pred_bboxes.detach()
        elif box_extract == 0:
            # skip 0~2 stage
            class_logits, bboxes, proposal_features = self.proposals_feat_cur.pop()

        if self.training or box_extract > 0:
            # select top k local box features per frame
            class_logits_max, _ = torch.max(input=class_logits, dim=-1, keepdim=False)
            topk_val, topk_idx = class_logits_max.topk(k=self.top_k[0], dim=-1)
            topk_idx_bool = torch.zeros_like(class_logits_max, dtype=torch.bool)
            topk_idx_bool.scatter_(1, topk_idx, 1)
            topk_idx_bool2 = torch.zeros_like(class_logits_max, dtype=torch.bool)
            topk_idx_bool2.scatter_(1, topk_idx[:, :self.top_k[1]], 1)
            # for i in range(len(topk_idx_bool)):
            #    topk_idx_bool[i, topk_idx[i]] = 1
            if box_extract > 0:
                proposal_feat_frame = proposal_features.view([-1, num_boxes, dim])
                return [class_logits, bboxes, proposal_features], \
                        proposal_feat_frame[topk_idx_bool], proposal_feat_frame[topk_idx_bool2]  # , bboxes[topk_idx_bool].view(bboxes.size(0), k, 4).detach()

        # local attention task
        if self.local_enable or self.global_enable:
            #features_cur = [p[key_frame].unsqueeze(0) for p in features]
            proposal_feat_frame = proposal_features.view([-1, num_boxes, dim])
            if self.training:
                local_interval = 3 if self.local_enable else 1
                if self.local_enable:
                    local_proposals = proposal_feat_frame[:local_interval]
                    local_topk_idx_bool = topk_idx_bool[:local_interval]
                    local_kv_ = local_proposals[local_topk_idx_bool].unsqueeze(1)  # all: proposal_features.permute(1, 0, 2)
                if self.global_enable:
                    global_proposals = proposal_feat_frame[local_interval:]
                    global_topk_idx_bool = topk_idx_bool[local_interval:]
                    global_kv1_ = global_proposals[global_topk_idx_bool].unsqueeze(1)
                    #global_topk_idx_bool2 = topk_idx_bool2[local_interval:]
                    #global_kv2_ = global_proposals[global_topk_idx_bool2].unsqueeze(1)
                    global_kv_ = [global_kv1_, global_kv1_]
            else:
                local_interval = bs
                local_kv_ = [f.unsqueeze(1) if self.local_enable else None for f in self.proposal_feats_local]
                global_kv_ = [f.unsqueeze(1) if self.global_enable else None for f in self.proposal_feats_global]

            if self.training and self.local_enable:
                query_ = local_proposals.view(local_interval * num_boxes, 1, dim)  # [local_interval * num_boxes, 1, 256]
                bboxes2 = bboxes[:local_interval]
                class_logits2 = class_logits[:local_interval]
                features = [f[:local_interval] for f in features]
                time = time[:local_interval]
            elif True:
                # enhance cur batch features
                query_ = proposal_features.permute(1, 0, 2)
                bboxes2 = bboxes
                class_logits2 = class_logits
            else:
                # enhance cur single features
                query_ = proposal_feat_frame[key_frame].unsqueeze(1)
                bboxes2 = bboxes[key_frame].unsqueeze(0)  # bboxes[topk_idx_bool].view(pred_bboxes.size(0), k, 4).detach()
                features = features_cur
                time = time[key_frame].unsqueeze(0)

            # local box-level attention
            for i in range(len(self.local_attention)):
                local_attn, dropout, layer_norm = self.local_attention[i]
                attn_ = local_attn(query=query_, key=local_kv_[i], value=local_kv_[i])[0]
                attn_ = layer_norm(attn_)

            # global box-level attention
            if len(self.global_attention) >= 2:
                query_ = torch.cat([query_, global_kv_[1]], dim=0)
            for i in range(len(self.global_attention)):
                if self.adaptive_norm:
                    global_attn = self.global_attention[i][0]
                    attn_ = global_attn(query=query_, key=global_kv_[i], value=global_kv_[i])[0]
                else:
                    global_attn, dropout, layer_norm, mlp, layer_norm2 = self.global_attention[i]
                    attn_ = global_attn(query=query_, key=global_kv_[i], value=global_kv_[i])[0]
                    query_ = query_ + dropout(attn_) # Mlp of timm
                    query_ = layer_norm(query_)
                    query_ = query_ + mlp(query_)
                    query_ = layer_norm2(query_)
                if len(self.global_attention) >= 2 and i == 0:
                    query_, global_kv_[1] = query_[:-len(global_kv_[1])], query_[-len(global_kv_[1]):]

            if self.adaptive_norm:
                B = local_interval if self.local_enable else bs
                attn_ = attn_.reshape(B, num_boxes, dim)
                if self.training:
                    # classifier free guidance
                    # Probability of class embeddings being the null embedding
                    self.p_uncond = 0.1
                    probs = torch.rand(B)
                    null = torch.where(probs < self.p_uncond, 1, 0).to(torch.bool).to(attn_.device)
                    mask = torch.ones_like(attn_)
                    mask[null] = 0.
                    attn_ = attn_ * mask
                attn_ = attn_.reshape(-1, dim)

            for head_idx, rcnn_head_local in enumerate(self.head_series_cond):
                self.use_topk = False
                if self.use_topk:  # attention score
                    if head_idx == 0:
                        k_level = [150, 75, 75]
                        low_k = 50
                    if k_level[head_idx] != k_level[head_idx - 1]:
                        scores, _ = class_logits2.max(dim=-1)
                        #scores = self.last_att_score.view(-1, k_level[head_idx - 1])
                        topk_val, topk_idx = scores.topk(k=k_level[head_idx], dim=-1)
                        lowest_k_val, lowest_k_idx = scores.topk(k=low_k, dim=-1, largest=False)
                        topk_idx_bool = torch.zeros([local_interval, num_boxes], dtype=torch.bool, device=topk_idx.device)
                        topk_idx_bool.scatter_(1, topk_idx, 1)
                        topk_idx_bool.scatter_(1, lowest_k_idx, 1)
                        self.topk_idx_bool = topk_idx_bool
                        # insert random query
                        #topk_idx_bool = topk_idx_bool + torch.where(torch.rand(topk_idx_bool.size(0), topk_idx_bool.size(1),
                        #                                                       device=topk_idx_bool.device) >= 0.2, 0, 1).to(torch.bool)
                        query_ = query_.view(-1, num_boxes, dim)[topk_idx_bool].unsqueeze(1)
                        bboxes2 = bboxes2[topk_idx_bool].view(-1, k_level[head_idx] + low_k, 4)
                        attn_ = attn_.view(-1, num_boxes, dim)[topk_idx_bool]
                proposal_features2 = query_.permute(1, 0, 2)
                class_logits2, pred_bboxes2, proposal_features2 = rcnn_head_local(features, bboxes2, proposal_features2,
                                                                                   self.box_pooler, time, attn_)
                if self.return_intermediate:
                    inter_class_logits.append(class_logits2)
                    inter_pred_bboxes.append(pred_bboxes2)
                bboxes2 = pred_bboxes2.detach()
                query_ = proposal_features2.permute(1, 0, 2)

        if self.return_intermediate and self.training:
            if self.local_enable:
                inter_class_logits = [r[:local_interval] for r in inter_class_logits]
                inter_pred_bboxes = [r[:local_interval] for r in inter_pred_bboxes]
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)
        elif self.local_enable or self.global_enable:
            return class_logits2[None], pred_bboxes2[None]
        else:
            # for baseline inference
            return class_logits[None], bboxes[None]

# Diffusion RCNNHead
class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = cfg.MODEL.DiffusionDet.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.DiffusionDet.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        assert (pred_boxes[:, 2:] >= pred_boxes[:, :2]).all()

        return pred_boxes


class RCNNHead_cond(RCNNHead):
    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0), adaptive_norm=True):
        super(RCNNHead_cond, self).__init__(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation,
                 scale_clamp, bbox_weights)

        # conditioning
        self.adaptive_norm = adaptive_norm
        if self.adaptive_norm:
            self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model))
            self.c_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, features, bboxes, pro_features, pooler, time_emb, cond=None):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        if self.adaptive_norm:
            shift = self.c_mlp(cond)
            scale = self.block_time_mlp(time_emb)
            scale = torch.repeat_interleave(scale, nr_boxes, dim=0)
            fc_feature = fc_feature * (scale + 1) + shift
        else:
            scale_shift = self.block_time_mlp(time_emb)
            scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
            scale, shift = scale_shift.chunk(2, dim=1)
            fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DiffusionDet.DIM_DYNAMIC # 64
        self.num_dynamic = cfg.MODEL.DiffusionDet.NUM_DYNAMIC # 2
        self.num_params = self.hidden_dim * self.dim_dynamic # 256 * 64
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params) # 2 * 256 * 64

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2) # (N * nr_boxes, 49, 256)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2) # (N * nrboxes , 1, 2*256*64)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic) # (N*nr_boxes, 256, 64)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim) # (N*nr_boxes, 64, 256)

        features = torch.bmm(features, param1)  # [N * nr_boxes, 49, 64]  FC 256->64
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)  # [N * nr_boxes, 49, 256]  FC 64->256
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)  # [N * nr_boxes, 64*256]
        features = self.out_layer(features)  # [N * nr_boxes, 256]
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
class sparse_attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Build heads.
        d_model = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.DiffusionDet.NHEADS
        dropout = cfg.MODEL.DiffusionDet.DROPOUT
        activation = cfg.MODEL.DiffusionDet.ACTIVATION

        att = []
        #att.append(MSDeformAttn(d_model, n_levels=3, n_heads=8, n_points=4))
        att.append(nn.MultiheadAttention(d_model, nhead, dropout=dropout))
        att.append(nn.Dropout(dropout))
        att.append(nn.LayerNorm(d_model))
        mlp_hidden_dim = int(d_model * 4)
        att.append(Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=nn.ReLU, drop=0.))
        att.append(nn.LayerNorm(d_model))
        self.local_attn = nn.ModuleList(att)

    def make_random_attention(self, in_spatial_length, seq_length, random_attention_probability=0.1):
        return torch.where(
            torch.rand((in_spatial_length * seq_length, in_spatial_length * seq_length)) >= random_attention_probability,
            1, 0)

    def make_position_attention(self, in_spatial_length, seq_length):
        one_frame = torch.eye(in_spatial_length, in_spatial_length)

        return one_frame.repeat(seq_length, seq_length)

    def make_frame_attention(self, in_spatial_length, seq_length):
        init_attention = torch.zeros((in_spatial_length * seq_length, in_spatial_length * seq_length))
        for i in range(seq_length):
            init_attention[i * in_spatial_length: (i + 1) * in_spatial_length, i * in_spatial_length: (i + 1) * in_spatial_length] = 1
        return init_attention

    def make_positional_attention(self, in_spatial_length, seq_length, random_attention_probability):
        return self.make_random_attention(in_spatial_length, seq_length, random_attention_probability) + \
                self.make_position_attention(in_spatial_length, seq_length) + \
                self.make_frame_attention(in_spatial_length, seq_length)

    def forward(self, features):
        N = features[0].size(0)
        n_levels = len(features)
        spatial_shapes = [features[i].size()[2:] for i in range(len(features))]  # [(h,w), (h,w), ...]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=features[0].device)

        for i in range(2, n_levels):
            cross_att, dropout, norm, mlp, norm2 = self.local_attn
            mask = self.make_positional_attention(in_spatial_length=features[i].size(2) * features[i].size(3),
                                                  seq_length=len(features[i]),
                                                  random_attention_probability=0.1).to('cuda')
            src = features[i].permute(0, 2, 3, 1).reshape(-1, 1, dim)
            src2 = cross_att(src, src, src, attn_mask=mask)[0]
            src = src + dropout(src2)
            src = norm(src)

            src = src + mlp(src)
            src = norm2(src)

            src = src.reshape(N, spatial_shapes[i, 0], spatial_shapes[i, 1], dim).permute(0, 3, 1, 2)
            features[i] = src

        return features