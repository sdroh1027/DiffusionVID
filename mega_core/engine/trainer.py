# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from mega_core.data import make_data_loader
from mega_core.utils.comm import get_world_size, synchronize
from mega_core.utils.metric_logger import MetricLogger
from mega_core.engine.inference import inference

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    tensorboard_writer
):
    logger = logging.getLogger("mega_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"] * cfg.SOLVER.ACCUMULATION_STEPS
    model.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    optimizer.zero_grad()

    for iter, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iter + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iter = iter + 1  # iter for lataloader
        iteration = iter // cfg.SOLVER.ACCUMULATION_STEPS + 1  # this is real iteration
        arguments["iteration"] = iteration - 1

        if not cfg.MODEL.VID.ENABLE:
            images = images.to(device)
        else:
            method = cfg.MODEL.VID.METHOD
            if method in ("base", ):
                images = images.to(device)
            elif method in ("rdn", "mega", "dafa", "diffusion", "fgfa", "dff"):
                images["cur"] = images["cur"].to(device)
                for key in ("ref", "ref_l", "ref_m", "ref_g"):
                    if key in images.keys():
                        images[key] = [img.to(device) for img in images[key]]
            else:
                raise ValueError("method {} not supported yet.".format(method))
        if method in ("mega", "dafa", "diffusion"):
            targets_c, targets_g, targets_l = targets[0]
            targets_c = [target.to(device) for target in targets_c]
            targets_g = [tg.to(device) for tg in targets_g]
            targets_l = [tl.to(device) for tl in targets_l]
            targets = [targets_c, targets_g, targets_l]
        else:
            targets = [target.to(device) for target in targets]

        if method in ("mega", "dafa", "diffusion"):
            num_boxes_targets = [len(target.bbox) for target in targets_g]
            idxs = [-1] + [i for i, x in enumerate(num_boxes_targets) if x > 0]
            total_reuse_count = min(cfg.SOLVER.BATCH_REUSE_STEPS, len(idxs))
            if len(targets_g) <= 1:
                total_reuse_count = 1
        else:
            idxs = [-1]
            total_reuse_count = 1
        for i in range(total_reuse_count):
            idx = idxs[i]
            if idx != -1 and method in ("mega", "dafa", "diffusion"):
                # swap a current(target) image and an global ref image
                # 1. randomly select a global ref img
                images["cur"], images["ref_g"][idx].tensors = images["ref_g"][idx].tensors[0], images["cur"][None,:]
                targets[0][0], targets[1][idx] = targets_g[idx], targets_c[0]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values()) / (cfg.SOLVER.ACCUMULATION_STEPS * total_reuse_count)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()

        # update weights with accumulated gradients
        if iter % cfg.SOLVER.ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            if cfg.SOLVER.LR_SCHEDULER_TYPE == "cosine":
                scheduler.step_update(iter // cfg.SOLVER.ACCUMULATION_STEPS)
            else:
                scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iter)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (iter % (20 * cfg.SOLVER.ACCUMULATION_STEPS) == 0 or iter == max_iter) and torch.cuda.current_device() == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if tensorboard_writer is not None:
                for key, val in meters.meters.items():
                    if 'loss' in key.lower():
                        tensorboard_writer.add_scalar('Train/' + key,
                                                      val.global_avg, iteration)
                        tensorboard_writer.add_scalar('Train_Avg20/' + key,
                                                      val.avg, iteration)
                tensorboard_writer.add_scalar('Train/RunningLearningRate',
                                              optimizer.param_groups[-1]['lr'], iteration)
                                              #scheduler.get_last_lr()[0], iteration)
        if iter % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iter == max_iter:
            checkpointer.save("model_final", **arguments)
        if data_loader_val is not None and test_period > 0 and (iter % test_period == 0 or iter == max_iter):
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            val_result = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                cfg,
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                # make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=False)[0],
                data_loader_val[0],
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('Val/mAP', val_result[0]['map'], iteration)
            synchronize()
            model.train()
            '''
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            '''
        if iter == max_iter:
            break
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
