#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""


import logging
import os
import sys

from . import distributed as du


def setup_logging(output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    _logger = logging.getLogger("lightning")
    _logger.handlers = []

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if output_dir is not None:
        filename = os.path.join(output_dir, f"stdout_{du.get_rank()}.log")
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)
