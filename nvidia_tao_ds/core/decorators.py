# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common decorators used in TAO Toolkit."""

from functools import wraps
import os

# Import Hydra exception classes for config error handling
try:
    from hydra.errors import ConfigCompositionException, MissingConfigException
    from omegaconf.errors import ConfigKeyError, MissingMandatoryValue, UnsupportedInterpolationType
except ImportError:
    # Fallback for older versions or if imports fail
    ConfigCompositionException = Exception
    MissingConfigException = Exception
    ConfigKeyError = Exception
    MissingMandatoryValue = Exception
    UnsupportedInterpolationType = Exception

# Import validation error classes
try:
    from marshmallow.exceptions import ValidationError as MarshmallowValidationError
except ImportError:
    MarshmallowValidationError = Exception

import nvidia_tao_ds.core.logging.logging as status_logging
from nvidia_tao_ds.core.mlops.wandb import alert


def monitor_status(name='Data-services', mode='analyze'):
    """Status monitoring decorator."""
    def inner(runner):
        @wraps(runner)
        def _func(cfg, **kwargs):
            # set up status logger
            os.makedirs(cfg.results_dir, exist_ok=True)

            status_file = os.path.join(cfg.results_dir, "status.json")
            status_logging.set_status_logger(
                status_logging.StatusLogger(
                    filename=status_file,
                    verbosity=1,
                    append=True
                )
            )
            s_logger = status_logging.get_status_logger()
            try:
                s_logger.write(
                    status_level=status_logging.Status.STARTED,
                    message=f"Starting {name} {mode}."
                )
                alert(
                    title=f'{mode.capitalize()} started',
                    text=f'{mode.capitalize()} {name} has started',
                    level=0
                )
                runner(cfg, **kwargs)
                s_logger.write(
                    status_level=status_logging.Status.RUNNING,
                    message=f"{mode.capitalize()} finished successfully."
                )
                if os.getenv("CLOUD_BASED") == "True":
                    s_logger.write(
                        status_level=status_logging.Status.RUNNING,
                        message="Job artifacts in results dir are being uploaded to the cloud"
                    )
            except (KeyboardInterrupt, SystemError):
                s_logger.write(
                    message=f"User/System interruption: {mode.capitalize()} was interrupted",
                    verbosity_level=status_logging.Verbosity.INFO,
                    status_level=status_logging.Status.FAILURE
                )
            except (
                ConfigCompositionException,
                MissingConfigException,
                ConfigKeyError,
                MissingMandatoryValue,
                UnsupportedInterpolationType,
            ) as e:
                s_logger.write(
                    message=f"Configuration error: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e
            except NotImplementedError as e:
                s_logger.write(
                    message=f"Feature not implemented: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e
            except (ValueError, TypeError) as e:
                s_logger.write(
                    message=f"Parameter validation error: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e
            except (FileNotFoundError, PermissionError, OSError, IOError) as e:
                s_logger.write(
                    message=f"File system error: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e
            except MarshmallowValidationError as e:
                s_logger.write(
                    message=f"Schema validation error: {str(e)}",
                    status_level=status_logging.Status.FAILURE
                )
                raise e

        return _func
    return inner
