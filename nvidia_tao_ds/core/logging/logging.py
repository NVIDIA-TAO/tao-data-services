# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logger class for data-services."""

import atexit
from datetime import datetime
import json
import logging as _logging
import os

from nvidia_tao_core.microservices.handlers.cloud_handlers.utils import status_callback

from torch import distributed as torch_distributed
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn


# Mapping of string log levels to logging constants
LOG_LEVEL_MAPPING = {
    'DEBUG': _logging.DEBUG,
    'INFO': _logging.INFO,
    'WARNING': _logging.WARNING,
    'WARN': _logging.WARNING,
    'ERROR': _logging.ERROR,
    'CRITICAL': _logging.CRITICAL,
    'FATAL': _logging.CRITICAL,
}


def get_logging_level():
    """Get logging level from environment variable.

    Reads TAO_LOGGING_LEVEL environment variable and returns the corresponding
    logging level. Defaults to INFO if not set or invalid.

    Returns:
        int: Python logging level constant (e.g., logging.INFO)
    """
    level_str = os.getenv('TAO_LOGGING_LEVEL', 'INFO').upper()
    level = LOG_LEVEL_MAPPING.get(level_str, _logging.INFO)
    return level


class MessageFormatter(_logging.Formatter):
    """Formatter that supports colored logs."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        _logging.DEBUG: grey + fmt + reset,
        _logging.INFO: grey + fmt + reset,
        _logging.WARNING: yellow + fmt + reset,
        _logging.ERROR: red + fmt + reset,
        _logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        """Format the log message."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = _logging.Formatter(log_fmt)
        return formatter.format(record)


# Get logging level from environment variable
_log_level = get_logging_level()

logger = _logging.getLogger('TAO Data-service')
logger.setLevel(_log_level)
ch = _logging.StreamHandler()
ch.setLevel(_log_level)
ch.setFormatter(MessageFormatter())
logger.addHandler(ch)
logging = logger


class StatusLoggerHandler(_logging.Handler):
    """Handler that forwards standard logging to StatusLogger.

    This handler bridges the standard Python logging system with the
    TAO StatusLogger, allowing log messages to be written to both
    console and status log files simultaneously.

    The handler fetches the current status logger on each emit, so it
    automatically uses the most recently set StatusLogger instance.
    """

    def __init__(self):
        """Initialize the StatusLoggerHandler."""
        super().__init__()
        self._get_status_logger = None
        self._Verbosity = None
        self._Status = None

    def emit(self, record):
        """Forward log records to the status logger.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        try:
            # Lazy import to avoid circular dependencies
            if self._get_status_logger is None:
                # Import from this module since Verbosity and Status are defined here
                self._get_status_logger = get_status_logger
                self._Verbosity = Verbosity
                self._Status = Status

            # Get the current status logger (may change if set_status_logger is called again)
            status_logger = self._get_status_logger()

            # Map Python logging levels to StatusLogger verbosity
            level_map = {
                _logging.DEBUG: self._Verbosity.DEBUG,
                _logging.INFO: self._Verbosity.INFO,
                _logging.WARNING: self._Verbosity.WARNING,
                _logging.ERROR: self._Verbosity.ERROR,
                _logging.CRITICAL: self._Verbosity.CRITICAL
            }

            verbosity_level = level_map.get(record.levelno, self._Verbosity.INFO)

            # Determine status based on log level
            if record.levelno >= _logging.ERROR:
                status_level = self._Status.FAILURE
            else:
                status_level = self._Status.RUNNING

            # Format the message
            message = self.format(record)

            # Write to status logger
            status_logger.write(
                data={},
                status_level=status_level,
                verbosity_level=verbosity_level,
                message=message
            )
        except Exception:
            self.handleError(record)


def enable_dual_logging():
    """Enable logging to both console and status logger.

    This function adds a StatusLoggerHandler to the TAO Data-service logger,
    which forwards all log messages to the status logger in addition to
    the standard console output.

    Note: This is automatically called by set_status_logger(), so you typically
    don't need to call it manually. It's safe to call multiple times - if the
    handler is already added, this function silently returns.

    Example:
        >>> from nvidia_tao_ds.core.logging.logging import StatusLogger, set_status_logger, logging
        >>>
        >>> # Setup status logger (automatically enables dual logging)
        >>> status_logger = StatusLogger(filename="status.json")
        >>> set_status_logger(status_logger)
        >>>
        >>> # Now all logging calls automatically go through both systems
        >>> logging.info("This goes to both console and status file!")
    """
    # Check if handler is already added (safe for multiple calls)
    for handler in logger.handlers:
        if isinstance(handler, StatusLoggerHandler):
            # Handler already exists, silently return
            return

    # Add the handler only if it doesn't exist
    status_handler = StatusLoggerHandler()
    status_handler.setLevel(get_logging_level())
    logger.addHandler(status_handler)


class Verbosity():
    """Verbosity levels."""

    DISABLE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# Defining a log level to name dictionary.
log_level_to_name = {
    Verbosity.DISABLE: "DISABLE",
    Verbosity.DEBUG: 'DEBUG',
    Verbosity.INFO: 'INFO',
    Verbosity.WARNING: 'WARNING',
    Verbosity.ERROR: 'ERROR',
    Verbosity.CRITICAL: 'CRITICAL'
}


class Status():
    """Status levels."""

    SUCCESS = 0
    FAILURE = 1
    STARTED = 2
    RUNNING = 3
    SKIPPED = 4


status_level_to_name = {
    Status.SUCCESS: 'SUCCESS',
    Status.FAILURE: 'FAILURE',
    Status.STARTED: 'STARTED',
    Status.RUNNING: 'RUNNING',
    Status.SKIPPED: 'SKIPPED'
}


class BaseLogger(object):
    """File logger class."""

    def __init__(self, verbosity=Verbosity.INFO):
        """Base logger class.

        Args:
            verbsority (int): Logging level

        """
        self.verbosity = verbosity
        self.categorical = {}
        self.graphical = {}
        self.kpi = {}

    @property
    def date(self):
        """Get date from the status.

        Returns:
            Formatted string containing mm/dd/yyyy.
        """
        date_time = datetime.now()
        date_object = date_time.date()
        return "{}/{}/{}".format(
            date_object.month,
            date_object.day,
            date_object.year
        )

    @property
    def time(self):
        """Get date from the status.

        Returns:
            Formatted string with time in hh:mm:ss
        """
        date_time = datetime.now()
        time_object = date_time.time()
        return "{}:{}:{}".format(
            time_object.hour,
            time_object.minute,
            time_object.second
        )

    @property
    def categorical(self):
        """Property getter for categorical data to be logged."""
        return self._categorical

    @categorical.setter
    def categorical(self, value: dict):
        """Set categorical data to be logged."""
        self._categorical = value

    @property
    def graphical(self):
        """Property getter for graphical data to be logged."""
        return self._graphical

    @graphical.setter
    def graphical(self, value: dict):
        """Set graphical data to be logged."""
        self._graphical = value

    @property
    def kpi(self):
        """Set KPI data."""
        return self._kpi

    @kpi.setter
    def kpi(self, value: dict):
        """Set KPI data."""
        self._kpi = value

    @rank_zero_only
    def flush(self):
        """Flush the logger."""
        pass

    def format_data(self, data: dict):
        """Format the data."""
        if not isinstance(data, dict):
            raise TypeError(f"Data must be a dictionary and not type {type(data)}.")
        data_string = json.dumps(data)
        return data_string

    @rank_zero_only
    def log(self, level, string):
        """Log the data string.

        This method is implemented only for rank 0 process in a multiGPU
        session.

        Args:
            level (int): Log level requested.
            string (string): Message to be written.
        """
        if level >= self.verbosity:
            logging.log(level, string)

    @rank_zero_only
    def write(self, data=None,
              status_level=Status.RUNNING,
              verbosity_level=Verbosity.INFO,
              message=None):
        """Write data out to the log file.

        Args:
            data (dict): Dictionary of data to be written out.
            status_level (nvidia_tao_pytorch.core.loggers.api_logging.Status): Current status of the
                process being logged. DEFAULT=Status.RUNNING
            verbosity level (nvidia_tao_pytorch.core.loggers.api_logging.Vebosity): Setting
                logging level of the Status logger. Default=Verbosity.INFO
        """
        if self.verbosity > Verbosity.DISABLE:
            if not data:
                data = {}
            # Define generic data.
            data["date"] = self.date
            data["time"] = self.time
            data["status"] = status_level_to_name.get(status_level, "RUNNING")
            data["verbosity"] = log_level_to_name.get(verbosity_level, "INFO")

            if message:
                data["message"] = message

            if self.categorical:
                data["categorical"] = self.categorical

            if self.graphical:
                data["graphical"] = self.graphical

            if self.kpi:
                data["kpi"] = self.kpi

            data_string = self.format_data(data)
            self.log(verbosity_level, data_string)
            self.flush()
            status_callback(data_string)


class StatusLogger(BaseLogger):
    """Simple logger to save the status file."""

    def __init__(self, filename=None,
                 verbosity=Verbosity.INFO,
                 append=True):
        """Logger to write out the status.

        Args:
            filename (str): Path to the log file.
            verbosity (str): Logging level. Default=INFO
            append (bool): Flag to open the log file in
                append mode or write mode. Default=True
        """
        super().__init__(verbosity=verbosity)
        self.log_path = os.path.realpath(filename)
        if os.path.exists(self.log_path):
            rank_zero_warn(
                f"Log file already exists at {self.log_path}"
            )
        # Open the file only if rank == 0.
        distributed = torch_distributed.is_initialized() and torch_distributed.is_available()
        global_rank_0 = (not distributed) or (distributed and torch_distributed.get_rank() == 0)
        if global_rank_0:
            self.l_file = open(self.log_path, "a" if append else "w")
            atexit.register(self.l_file.close)

    @rank_zero_only
    def log(self, level, string):
        """Log the data string.

        This method is implemented only for rank 0 process in a multiGPU
        session.

        Args:
            level (int): Log level requested.
            string (string): Message to be written.
        """
        if level >= self.verbosity:
            self.l_file.write(string + "\n")

    @rank_zero_only
    def flush(self):
        """Flush contents of the log file."""
        self.l_file.flush()

    @staticmethod
    def format_data(data):
        """Format the dictionary data.

        Args:
            data(dict): Dictionary data to be formatted to a json string.

        Returns
            data_string (str): json formatted string from a dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Data must be a dictionary and not type {type(data)}.")
        data_string = json.dumps(data)
        return data_string


# Define the logger here so it's static.
_STATUS_LOGGER = BaseLogger()


def set_status_logger(status_logger):
    """Set the status logger.

    This function also automatically enables dual logging, which forwards
    all standard Python logging calls to the status logger in addition to
    console output.

    Args:
        status_logger: An instance of the logger class.
    """
    global _STATUS_LOGGER  # pylint: disable=W0603
    _STATUS_LOGGER = status_logger

    # Automatically enable dual logging when status logger is set
    try:
        enable_dual_logging()  # No pylint needed here - function is in same module
    except Exception:
        # Silently fail if dual logging cannot be enabled (e.g., in non-standard environments)
        pass


def get_status_logger():
    """Get the status logger."""
    global _STATUS_LOGGER  # pylint: disable=W0602,W0603
    return _STATUS_LOGGER
