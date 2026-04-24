# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import logging
import os
from contextlib import contextmanager
from types import MethodType
from typing import Optional

from .platforms import Platform


def _parse_log_level(level) -> int:
    """Parse log level from string or int.

    Accepts:
      - Standard names: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
      - Numeric strings: '10', '20'
      - Integers: 10, 20
    """
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        level = level.strip().upper()
        # Try numeric first
        try:
            return int(level)
        except ValueError:
            pass
        # Try standard name
        numeric = getattr(logging, level, None)
        if isinstance(numeric, int):
            return numeric
    return logging.INFO


log_level = _parse_log_level(os.getenv('LOG_LEVEL', 'INFO'))


# Avoid circular reference
def _is_local_master():
    local_rank = Platform.get_local_rank()
    return local_rank in {-1, 0}


init_loggers = {}

# old format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_format = logging.Formatter('[%(asctime)s][%(levelname)s:%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

info_set = set()
warning_set = set()


def _rank_info(self, msg, *args, ranks='master', **kwargs):
    """Log at INFO level with rank control.

    Args:
        ranks: 'master' (default) - only local rank 0;
               'all' - every rank (prefixed with rank id).
    """
    if ranks == 'all':
        rank = Platform.get_local_rank()
        tagged = f'[rank{rank}] {msg}'
        if _is_local_master():
            self._original_info(tagged, *args, **kwargs)
        else:
            with logger_context(self, logging.INFO):
                self._original_info(tagged, *args, **kwargs)
    else:
        self._original_info(msg, *args, **kwargs)


def _rank_warning(self, msg, *args, ranks='master', **kwargs):
    """Log at WARNING level with rank control.

    Args:
        ranks: 'master' (default) - only local rank 0;
               'all' - every rank (prefixed with rank id).
    """
    if ranks == 'all':
        rank = Platform.get_local_rank()
        tagged = f'[rank{rank}] {msg}'
        if _is_local_master():
            self._original_warning(tagged, *args, **kwargs)
        else:
            with logger_context(self, logging.WARNING):
                self._original_warning(tagged, *args, **kwargs)
    else:
        self._original_warning(msg, *args, **kwargs)


def info_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.info(msg)


def warning_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.warning(msg)


def info_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in info_set:
        return
    info_set.add(hash_id)
    self.info(msg)


def warning_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in warning_set:
        return
    warning_set.add(hash_id)
    self.warning(msg)


def get_logger(log_file: Optional[str] = None,
               log_level: Optional[int] = None,
               file_mode: str = 'w',
               only_local_master: bool = True) -> logging.Logger:
    """ Get logging logger

    Args:
        log_file: Log filename, if specified, file handler will be added to
            logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file, if filename is
            specified (if filemode is unspecified, it defaults to 'w').
        only_local_master: Output log only when it's local master, default True.
    """
    if log_level is None:
        log_level = _parse_log_level(os.getenv('LOG_LEVEL', 'INFO'))
    elif isinstance(log_level, str):
        log_level = _parse_log_level(log_level)
    logger_name = __name__.split('.')[0]
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    if logger_name in init_loggers:
        add_file_handler_if_needed(logger, log_file, file_mode, log_level)
        return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    is_worker0 = _is_local_master() or not only_local_master

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(logger_format)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_worker0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    init_loggers[logger_name] = True

    # Preserve originals before overriding, for rank-aware wrappers
    logger._original_info = logger.info
    logger._original_warning = logger.warning
    logger.info = MethodType(_rank_info, logger)
    logger.warning = MethodType(_rank_warning, logger)
    logger.info_once = MethodType(info_once, logger)
    logger.warning_once = MethodType(warning_once, logger)
    logger.info_if = MethodType(info_if, logger)
    logger.warning_if = MethodType(warning_if, logger)
    return logger


logger = get_logger(log_level=log_level)


@contextmanager
def logger_context(logger, log_leval):
    origin_log_level = logger.level
    logger.setLevel(log_leval)
    try:
        yield
    finally:
        logger.setLevel(origin_log_level)


def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if importlib.util.find_spec('torch') is not None:
        is_worker0 = int(os.getenv('LOCAL_RANK', -1)) in {-1, 0}
    else:
        is_worker0 = True

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(logger_format)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
