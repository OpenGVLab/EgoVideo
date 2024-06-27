#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Distributed helpers."""

import functools
import logging
import os
import pickle

import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def all_gather_unaligned(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data = pickle.loads(buffer)
        if isinstance(data, torch.Tensor):
            data = data.cpu()

        data_list.append(data)

    return data_list


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    Returns:
        (group): pytorch dist group.
    """
    return dist.group.WORLD


def init_distributed_groups(cfg):
    """
    Initialize distributed sub groups for each machine.
    """
    if cfg.NUM_GPUS == 1:
        return

    num_gpus_per_machine = cfg.NUM_GPUS
    num_machines = dist.get_world_size() // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == get_node_rank():
            global _LOCAL_PROCESS_GROUP
            _LOCAL_PROCESS_GROUP = pg


def _serialize_to_tensor(data, group):
    """
    Seriialize the tensor to ByteTensor. Note that only `gloo` and `nccl`
        backend is supported.
    Args:
        data (data): data to be serialized.
        group (group): pytorch dist group.
    Returns:
        tensor (ByteTensor): tensor that serialized.
    """

    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    if isinstance(data, torch.Tensor):
        data = data.to(device)

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Padding all the tensors from different GPUs to the largest ones.
    Args:
        tensor (tensor): tensor to pad.
        group (group): pytorch dist group.
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def is_master_proc():
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


def get_world_size():
    """
    Get the size of the world.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Get the rank of the current process.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_node_rank() -> int:
    """
    Get the rank of the current process.
    """
    return get_rank() // get_local_size()


def get_local_size() -> int:
    """
    Returns:
        The number of local gpus, which is required to be equivalent to the local
        number of processes.
    """
    return torch.cuda.device_count()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    return int(os.environ.get("LOCAL_RANK", 0))
