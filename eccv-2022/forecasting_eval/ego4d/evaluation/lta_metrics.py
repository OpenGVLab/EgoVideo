#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import numpy as np
import torch

import editdistance

from ..utils import distributed as du
from ..utils import logging

from sklearn.metrics import average_precision_score

logger = logging.get_logger(__name__)


def distributed_topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k. Average reduces the result with all other
    distributed processes.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    errors = topk_errors(preds, labels, ks)
    return errors


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"

    # Find the top max_k predictions for each sample
    maxk = max(ks)
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, maxk, dim=1, largest=True, sorted=True
    )

    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].reshape(-1).float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]

def edit_distance(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
        dists.append(dist)
    return np.mean(dists)

def distributed_edit_distance(preds, labels):
    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    return edit_distance(preds, labels)

def AUED(preds, labels):
    N, Z, K = preds.shape
    preds = preds.numpy()  # (N, Z, K)
    labels = labels.squeeze(-1).numpy()  # (N, Z)
    ED = np.vstack(
        [edit_distance(preds[:, :z], labels[:, :z]) for z in range(1, Z + 1)]
    )
    AUED = np.trapz(y=ED, axis=0) / (Z - 1)

    output = {"AUED": AUED}
    output.update({f"ED_{z}": ED[z] for z in range(Z)})
    return output

def distributed_AUED(preds, labels):
    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
    labels = torch.cat(du.all_gather_unaligned(labels), dim=0)
    return AUED(preds, labels)
