import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
import torch.nn as nn

import json


def isnan(x):
    return torch.isnan(x).int().sum() != 0


def isinf(x):
    return torch.isinf(x).int().sum() != 0


def train_class_batch(model, samples, target, masks):
    outputs = model(samples)
    outputs = outputs * masks
    avg = masks.sum()
    loss_fun = nn.SmoothL1Loss(reduction="sum", beta=5.0)

    # Compute the loss.
    loss = loss_fun(outputs, target)
    loss = loss / avg
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, targets_mask, _, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)

        targets_mask = targets_mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        assert not isnan(samples)
        assert not isnan(targets_mask)
        assert not isnan(targets)
        assert not isinf(samples)
        assert not isinf(targets_mask)
        assert not isinf(targets)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, targets_mask)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, targets_mask)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # if mixup_fn is None:
        #     # class_acc = (output.max(-1)[-1] == targets).float().mean()
        #     acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        # else:
        #     # class_acc = None
        #     acc1 = None
        #     acc5 = None
        acc1 = None
        acc5 = None
        metric_logger.update(loss=loss_value)
        # metric_logger.update(class_acc=class_acc)
        metric_logger.update(acc1=acc1)
        metric_logger.update(acc5=acc5)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            # log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(acc1=acc1, head="loss")
            log_writer.update(acc5=acc5, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def convert_anno_2_results(anno_file):
    with open(anno_file, "r") as f:
        a = json.load(f)
    clips = a['clips']

    gts = dict()

    for clip in clips:
        clip_id = clip["clip_id"]
        # print(clip)
        for frame in clip["frames"]:

            pre45_frame = frame['pre_45']['frame']
            clip_name = str(clip_id) + '_' + str(pre45_frame - 1)
            label = [0] * 20
            for frame_type, frame_annot in frame.items():
                # if frame_type in ['start_sec', 'end_sec','height', 'width']:
                if frame_type in ["action_start_sec", "action_end_sec", "action_start_frame",
                                  "action_end_frame", "action_clip_start_sec", "action_clip_end_sec",
                                  "action_clip_start_frame", "action_clip_end_frame"]:
                    continue
                # frame_gt = frame_annot[1]
                if len(frame_annot) == 2:
                    continue
                frame_gt = frame_annot['boxes']
                if frame_type == 'pre_45':
                    for single_hand in frame_gt:
                        if 'left_hand' in single_hand:
                            label[0] = single_hand['left_hand'][0]
                            label[1] = single_hand['left_hand'][1]
                        if 'right_hand' in single_hand:
                            label[2] = single_hand['right_hand'][0]
                            label[3] = single_hand['right_hand'][1]
                if frame_type == 'pre_30':
                    for single_hand in frame_gt:
                        if 'left_hand' in single_hand:
                            label[4] = single_hand['left_hand'][0]
                            label[5] = single_hand['left_hand'][1]
                        if 'right_hand' in single_hand:
                            label[6] = single_hand['right_hand'][0]
                            label[7] = single_hand['right_hand'][1]
                if frame_type == 'pre_15':
                    for single_hand in frame_gt:
                        if 'left_hand' in single_hand:
                            label[8] = single_hand['left_hand'][0]
                            label[9] = single_hand['left_hand'][1]
                        if 'right_hand' in single_hand:
                            label[10] = single_hand['right_hand'][0]
                            label[11] = single_hand['right_hand'][1]
                if frame_type == 'pre_frame':
                    for single_hand in frame_gt:
                        if 'left_hand' in single_hand:
                            label[12] = single_hand['left_hand'][0]
                            label[13] = single_hand['left_hand'][1]
                        if 'right_hand' in single_hand:
                            label[14] = single_hand['right_hand'][0]
                            label[15] = single_hand['right_hand'][1]
                if frame_type == 'contact_frame':
                    for single_hand in frame_gt:
                        if 'left_hand' in single_hand:
                            label[16] = single_hand['left_hand'][0]
                            label[17] = single_hand['left_hand'][1]
                        if 'right_hand' in single_hand:
                            label[18] = single_hand['right_hand'][0]
                            label[19] = single_hand['right_hand'][1]
            gts[clip_name] = label
    return gts


def eval_hands(pred_dict, anno_file, num_clips=30):
    # pred_dict = {}
    # gt_dict = {}
    #
    # preds_file = output_file
    # # preds_file = 'submission.json'
    # f = open(preds_file)
    # preds_json = json.load(f)
    # f.close()
    # for k, v in preds_json.items():
    #     pred_dict[k] = v

    # labels_file = '/mnt/petrelfs/chenguo/workspace/ego4d/forecasting/fhp/annotations/fho_hands_val.json'  # 'test.json' is not provided, just for demonstration
    # f = open(labels_file)
    # labels_json = json.load(f)
    # f.close()
    # for k, v in labels_json.items():
    #     gt_dict[k] = v

    gt_dict = convert_anno_2_results(anno_file)

    left_list = []
    right_list = []
    left_final_list = []
    right_final_list = []

    for k, v in pred_dict.items():
        pred = [i / num_clips for i in v]
        label = gt_dict[k]
        # print(pred)

        for k in range(5):
            l_x_pred = pred[k * 4]
            l_y_pred = pred[k * 4 + 1]
            r_x_pred = pred[k * 4 + 2]
            r_y_pred = pred[k * 4 + 3]

            l_x_gt = label[k * 4]
            l_y_gt = label[k * 4 + 1]
            r_x_gt = label[k * 4 + 2]
            r_y_gt = label[k * 4 + 3]

            if r_x_gt != 0 or r_y_gt != 0:
                dist = np.sqrt((r_y_gt - r_y_pred) ** 2 + (r_x_gt - r_x_pred) ** 2)
                right_list.append(dist)
            if l_x_gt != 0 or l_y_gt != 0:
                dist = np.sqrt((l_y_gt - l_y_pred) ** 2 + (l_x_gt - l_x_pred) ** 2)
                left_list.append(dist)

        if r_x_gt != 0 or r_y_gt != 0:
            dist = np.sqrt((r_y_gt - r_y_pred) ** 2 + (r_x_gt - r_x_pred) ** 2)
            right_final_list.append(dist)
        if l_x_gt != 0 or l_y_gt != 0:
            dist = np.sqrt((l_y_gt - l_y_pred) ** 2 + (l_x_gt - l_x_pred) ** 2)
            left_final_list.append(dist)

    print('***left hand mean disp error {:.3f}, right hand mean disp error {:.3f}'
          .format(sum(left_list) / len(left_list), sum(right_list) / len(right_list)))
    print('***left hand contact disp error {:.3f}, right hand contact disp error {:.3f}'
          .format(sum(left_final_list) / len(left_final_list), sum(right_final_list) / len(right_final_list)))


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = nn.SmoothL1Loss(reduction="mean", beta=5.0)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    anno_file = data_loader.dataset.anno_path
    # switch to evaluation mode
    model.eval()
    final_result = dict()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        target_mask = batch[2]
        clip_names = batch[3]
        # clip_names = batch[3]
        # print(clip_names)
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        # compute output

        with torch.cuda.amp.autocast():
            outputs = model(videos)
            outputs = outputs * target_mask
            avg = target_mask.sum()

            loss_fun = nn.SmoothL1Loss(reduction="sum", beta=5.0)

            # Compute the loss.
            loss = loss_fun(outputs, target)
            loss = loss / avg

        for i in range(outputs.size(0)):
            final_result[clip_names[i]] = outputs[i].cpu().numpy()

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())

    eval_hands(final_result, num_clips=1, anno_file=anno_file)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@torch.no_grad()
def final_test(data_loader, model, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    final_result = dict()

    mode = data_loader.dataset.mode
    anno_file = data_loader.dataset.anno_path
    header = f'Test:[{mode}]'

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        target_mask = batch[2]
        clip_names = batch[3]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)
        loss = None
        # compute output
        with torch.cuda.amp.autocast():
            test_output_ls = []
            for i in range(args.test_num_segment):
                inputs = videos[:, :, i * args.num_segments:(i + 1) * args.num_segments]
                outputs = model(inputs)
                test_output_ls.append(outputs)
            outputs = torch.stack(test_output_ls).mean(dim=0)
            if "test" not in mode:
                outputs = outputs * target_mask
                avg = target_mask.sum()
                loss_fun = nn.SmoothL1Loss(reduction="sum", beta=5.0)
                # Compute the loss.
                loss = loss_fun(outputs, target)
                loss = loss / avg
        for i in range(outputs.size(0)):
            final_result[clip_names[i]] = outputs[i].cpu().numpy()

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        if loss is not None:
            metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if loss is not None:
        print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    if "test" not in mode:
        eval_hands(final_result, num_clips=1, anno_file=anno_file)

    with open(os.path.join(args.output_dir, f"submission_{mode}.json"), "w") as f:
        json.dump(final_result, f, cls=NumpyEncoder)


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
