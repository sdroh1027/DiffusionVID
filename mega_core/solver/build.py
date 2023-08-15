# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import itertools

from .lr_scheduler import WarmupMultiStepLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from detectron2.solver.build import maybe_add_gradient_clipping

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "backbone" in key and "fpn_" not in key:
            lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
        if "backbone" in key and "fpn_" in key: # and 'swin' in cfg.MODEL.BACKBONE.NAME:
            lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
        if "bias" in key:
            lr = lr * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    if cfg.SOLVER.OPTIMIZER_TYPE == "adamw":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=cfg.SOLVER.WEIGHT_DECAY, amsgrad=False
        )
    elif cfg.SOLVER.OPTIMIZER_TYPE == "sgd":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
        )
    else:
        raise NotImplementedError(f"no optimizer type {cfg.SOLVER.OPTIMIZER_TYPE}")
    if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
        optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_SCHEDULER_TYPE == "cosine":
        return CosineLRScheduler(
            optimizer,
            t_initial=cfg.SOLVER.MAX_ITER,
            lr_min=cfg.SOLVER.BASE_LR * (1/10 ** 2),
            warmup_lr_init=cfg.SOLVER.BASE_LR * (1/10 ** 3),
            warmup_t=cfg.SOLVER.WARMUP_ITERS,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
