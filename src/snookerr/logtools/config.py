# logging_config.py
# Copyright (C) 2026  Murdock Grewar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import enum
import logging
import os
import pathlib
import sys
import time

from ..version import version

_module_name = "snookerr"


class LogLevel(enum.Enum):
    """Enum of log levels."""

    ERROR = 10
    WARN = 20
    INFO = 30
    DEBUG = 40


def make_logger(  # noqa: C901
    path: pathlib.Path | None,
    *,
    name: str = _module_name,
    stderr_loglevel: LogLevel = LogLevel.WARN,
    create_dir: bool = False,
    allow_append: bool = False,
    ) -> logging.Logger:
    """Initialise logging configuration for the current process."""
    # Process argument 'stderr_loglevel'
    match stderr_loglevel:
        case LogLevel.ERROR:
            stderr_loglevel_ = logging.ERROR
        case LogLevel.WARN:
            stderr_loglevel_ = logging.WARN
        case LogLevel.INFO:
            stderr_loglevel_ = logging.INFO
        case LogLevel.DEBUG:
            stderr_loglevel_ = logging.DEBUG
        case _:
            raise ValueError()
    del stderr_loglevel

    if False:
        # Disable all the other loggers, like from matplotlib.
        # We need to import problematic packages here first, so that their loggers are picked up and disabled here
        import matplotlib.pyplot  # noqa: F401, ICN001, PLC0415
        try:
            from .. import image_view  # noqa: F401, PLC0415
        except:  # noqa: E722
            pass
        for name, logger in logging.root.manager.loggerDict.items():  # noqa: PERF102
            logger.disabled = True  # ty:ignore[invalid-assignment]

    ##### Logging setup #####
    logger = logging.getLogger(name)
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format='[[{levelname: <8}] [{name: <10}] {asctime:s}] {message:s}',
    #     # datefmt='%Y-%d-%m %H:%M:%S %z',
    #     datefmt='%H:%M:%S',
    #     style='{',
    # )

    def filter(record: logging.LogRecord):
        # record.name_ = record.name
        # This is not compatible with tqdm's redirection, apparently... sigh.
        return record

    formatter = logging.Formatter(
        fmt='[[{levelname: <8}] [{name: <8}] {asctime:s}] {message:s}',
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Adjust the stderr handler to the specified log level
    # logging.root.handlers[0].setLevel(stderr_loglevel_)
    # logging.root.handlers[0].setFormatter(formatter)
    # logging.root.handlers[0].addFilter(filter)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(stderr_loglevel_)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(filter)
    logger.addHandler(stderr_handler)

    # Set up logfiles
    if path is not None:
        # formatter = logging.root.handlers[0].formatter
        if create_dir:
            os.makedirs(path, exist_ok=allow_append)
        else:
            if not os.path.isdir(path):
                msg = f"The path '{path}' does not exist. Please create it first, or use the 'create_dir' option."
                raise FileNotFoundError(msg)
        log_info_path: pathlib.Path = pathlib.Path(path.joinpath('log_info.txt'))
        log_debug_path: pathlib.Path = pathlib.Path(path.joinpath('log_debug.txt'))
        if not allow_append:
            if os.path.isfile(log_info_path):
                msg = f"File already exists: {log_info_path}"
                raise ValueError(msg)
            if os.path.isfile(log_debug_path):
                msg = f"File already exists: {log_debug_path}"
                raise ValueError(msg)
        fh_info  = logging.FileHandler(log_info_path, mode='a')
        fh_info.setLevel(logging.INFO)
        fh_info.setFormatter(formatter)
        fh_info.addFilter(filter)
        fh_debug = logging.FileHandler(log_debug_path, mode='a')
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        fh_debug.addFilter(filter)
        # logging.getLogger().addHandler(fh_info)
        # logging.getLogger().addHandler(fh_debug)
        logger.addHandler(fh_info)
        logger.addHandler(fh_debug)

    # Log the fact that logging has been initiated.
    logger.info(time.strftime("Log start datetime: %Y-%d-%m %H:%M:%S %z"))
    logger.info(f"Current `{_module_name:s}` module version: {version:s}")
    logger.info(f"Current working directory: '{os.getcwd()}'")
    logger.info("Script called with the following commandline program and arguments:")
    logger.info(f"{sys.argv}")
    logger.info("")
    #########################

    return logger
