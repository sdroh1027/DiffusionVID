# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from mega_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

from seq_nms import seq_nms
from mega_core.structures.boxlist_ops import boxlist_nms
from mega_core.structures.boxlist_ops import cat_boxlist

import torch.autograd.profiler as profiler

def compute_on_dataset(model, data_loader, device, bbox_aug, method, timer=None, do_seq_nms=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                if method in ("base", ):
                    images = images.to(device)
                elif method in ("rdn", "mega", "dafa", "diffusion", "fgfa", "dff"):
                    images["cur"] = images["cur"].to(device)
                    for key in ("ref", "ref_l", "ref_m", "ref_g"):
                        if key in images.keys():
                            images[key] = [img.to(device) for img in images[key]]
                else:
                    raise ValueError("method {} not supported yet.".format(method))
                '''
                with profiler.profile(record_shapes=True) as prof:
                    with profiler.record_function("model_inference"):
                        output = model(images)
                print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))
                '''
                output = model(images)
                '''
                while i < 10:
                    prof.export_chrome_trace("trace_" + str(i) + ".json")
                '''
                if do_seq_nms:
                    ###  Codes are revised vesion of FGFA github
                    ###  https://github.com/msracver/Flow-Guided-Feature-Aggregation
                    if images["frame_id"] == images["seg_len"]-1:
                        all_boxes = model.roi_heads.box.post_processor.all_boxes
                        num_classes = model.num_classes
                        thresh_nms = model.roi_heads.box.post_processor.score_thresh
                        video = [all_boxes[j][:] for j in range(1, num_classes)]
                        dets_all = seq_nms(video)
                        for cls_ind, dets_cls in enumerate(dets_all):
                            for frame_ind, dets in enumerate(dets_cls):
                                # boxlist_nms works with one class Boxlist
                                # original nms() returns keeped index. MEGA's boxlist_nms() returns keeped boxlists
                                keep, keep_box_idx = boxlist_nms(boxlist=dets, nms_thresh=thresh_nms) # nms(dets)
                                #all_boxes[cls_ind + 1][frame_ind] = dets[keep, :]
                                all_boxes[cls_ind + 1][frame_ind] = keep  # call by ref
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            if not do_seq_nms:
                output = [o.to(cpu_device) for o in output]
        if do_seq_nms:
            if images["frame_id"] == 0:
                image_ids_list = [image_ids[0]]
            else:
                image_ids_list.append(image_ids[0])
            if images["frame_id"] == images["seg_len"]-1:
                # output contains list of Boxlists for each frames
                boxes_frame_wise = [cat_boxlist([boxes_one_cls[j] for boxes_one_cls in all_boxes[1:]])
                                    for j in range(images["seg_len"])]
                output = boxes_frame_wise
                #  if you use seq_nms, results_dict is updated collectively when a video ends.
                results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids_list, output)}
                )
        else:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids[0], output)}
            )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("mega_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        motion_specific=False,
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("mega_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, cfg.MODEL.VID.METHOD, inference_timer, cfg.TEST.SEQ_NMS)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder and cfg.TEST.SEQ_NMS:
        torch.save(predictions, os.path.join(output_folder, "predictions_seq_nms.pth"))
    elif output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        motion_specific=motion_specific,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def inference_no_model(
        data_loader,
        iou_types=("bbox",),
        motion_specific=False,
        box_only=False,
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    dataset = data_loader.dataset

    predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
    print("prediction loaded.")

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        motion_specific=motion_specific,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
