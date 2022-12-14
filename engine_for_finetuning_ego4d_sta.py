import os
from unittest import result
import numpy as np
import math
import sys
import time
import datetime
import logging
from typing import Iterable, Optional
import torch
import torch.nn as nn
import json
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
import forecasting_eval.ego4d.models.losses as losses


def train_class_batch(model, samples, boxes, verb_labels, ttc_targets, orig_norm_pred_boxes, losslist, cfg = None):
    pred_verb,pred_ttc = model(samples,boxes,orig_norm_pred_boxes)
    verb_labels = torch.cat(verb_labels, 0)
    ttc_targets = torch.cat(ttc_targets, 0)

    valid_verbs = verb_labels!=-100
    verb_labels = verb_labels[valid_verbs]
    pred_verb = pred_verb[valid_verbs]
    verb_loss = losslist[0](pred_verb, verb_labels)
    valid_ttcs = ~torch.isnan(ttc_targets)
    pred_ttc = pred_ttc[valid_ttcs]
    ttc_targets = ttc_targets[valid_ttcs]
    ttc_loss = losslist[1](
        pred_ttc, ttc_targets
    )
    lossw = cfg.MODEL.STA_LOSS_WEIGHTS #损失权重
    # print(verb_loss.item(),ttc_loss.item())
    loss = lossw[1] * ttc_loss + lossw[0] * verb_loss
    return loss, pred_verb

def train_class_batch_verb(model, samples, verb_labels):
    pred_verb = model(samples)
    # pred_verb = F.softmax(pred_verb,dim=1)
    verb_labels = torch.cat(verb_labels, 0)
    # import pdb
    # pdb.set_trace()
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(pred_verb, verb_labels)
    return loss, None


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, cfg=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    verb_loss_fun = losses.get_loss_func(cfg.MODEL.VERB_LOSS_FUNC)(reduction="mean")#动词损失函数
    ttc_loss_fun = losses.get_loss_func(cfg.MODEL.TTC_LOSS_FUNC)(reduction="mean")##ttc损失函数
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    count = 0
    for data_iter_step, (uid,samples, boxes, verb_labels,ttc_targets,orig_norm_pred_boxes,_) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples[0]
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
        boxes = [box.to(device=device) for box in boxes]  # boxlist
        orig_norm_pred_boxes = [box.to(device=device) for box in orig_norm_pred_boxes]
        verb_labels = [verb_label.to(device=device) for verb_label in verb_labels]
        # for i in range(len(ttc_targets)):
        #     valid_ttc = ~torch.isnan(ttc_targets[i])
        #     ttc_targets[i] = ttc_targets[i][valid_ttc][0].reshape(-1,1)
        ttc_targets = [ttc_target.to(device=device) for ttc_target in ttc_targets]
        
        # targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, _ = train_class_batch(
                model, samples, boxes, verb_labels, ttc_targets, orig_norm_pred_boxes,[verb_loss_fun,ttc_loss_fun], cfg=cfg)
            #loss, _ = train_class_batch_verb(model,samples,verb_labels)
        else:
            with torch.cuda.amp.autocast():
                loss, _ = train_class_batch(
                    model, samples, boxes, verb_labels, ttc_targets, orig_norm_pred_boxes, [verb_loss_fun,ttc_loss_fun], cfg=cfg)
                # loss, _ = train_class_batch_verb(model,samples,verb_labels)

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
        #     class_acc = (output.max(-1)[-1] == targets).float().mean()
        # else:
        #     class_acc = None
        metric_logger.update(loss=loss_value)
        # metric_logger.update(class_acc=class_acc)
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

def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
