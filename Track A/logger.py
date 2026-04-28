#!/usr/bin/env python
# -*-coding:utf-8 -*-

import logging
import sys

LOG_FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def set_root_log_level(log_level: str):
    logger = logging.getLogger()
    level: int | str = log_level.upper()
    try:
        level = int(log_level)
    except ValueError:
        pass
    try:
        logger.setLevel(level)  # type: ignore
    except Exception:
        logger.warning(
            f"Failed to set logging level to {log_level}, using default 'NOTSET'"
        )
        logger.setLevel(logging.NOTSET)


def init_logger(log_file: str | None = None, name: str | None = None, level: str = "NOTSET"):
    """
    Setup logging.

    Args:
        log_file: A file name to save file logs to.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
    """
    set_root_log_level(level)
    logger = logging.getLogger(name)

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    logger.handlers.clear()
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)

    if log_file is not None:
        # build file handler
        file_handler = logging.FileHandler(log_file, "a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(LOG_FORMATTER)
        # update logger
        logger = logging.getLogger()
        logger.addHandler(file_handler)

    return logger