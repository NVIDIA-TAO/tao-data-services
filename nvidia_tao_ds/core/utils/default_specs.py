# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This script is responsible for generating default experiment.yaml files from dataclasses."""

from __future__ import annotations

import os
from os import makedirs, listdir
from os.path import abspath, dirname, exists, join

from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass

from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.logging.logging import logging

# Usage example:
# ==============
"""
python default_specs \
    results_dir=/results/annotations/ \
    module_name=annotations
"""


# Get the config root from nvidia_tao_ds
import nvidia_tao_ds
CONFIG_ROOT = join(dirname(abspath(nvidia_tao_ds.__file__)), "config")

# Some config dir names don't match the on-disk DS module dir name.
# Map: config dir name -> DS module dir name.
CONFIG_TO_DS_MODULE_ALIASES = {"analytics": "data_analytics"}


def get_supported_modules():
    """
    Get list of supported modules from config directory that are also implemented in nvidia_tao_ds.

    This function checks both:
    1. Modules defined in nvidia_tao_ds/config/
    2. Modules actually implemented in nvidia_tao_ds (annotations, augmentation, auto_label, data_analytics, image)

    Returns:
        List[str]: List of module names that have both config definitions and implementations
    """
    if not exists(CONFIG_ROOT):
        logging.warning(f"Config root not found at {CONFIG_ROOT}")
        return []

    # Get all config modules from nvidia_tao_ds/config/
    config_modules = [
        item for item in listdir(CONFIG_ROOT)
        if item not in ["utils", "__pycache__", "common"] and os.path.isdir(join(CONFIG_ROOT, item))
    ]

    # Get TAO DS path from current file location without importing
    # __file__ is in nvidia_tao_ds/core/utils/default_specs.py
    # Go up 3 levels to get to nvidia_tao_ds/
    nvidia_tao_ds_dir = dirname(dirname(dirname(abspath(__file__))))

    # Get all implemented modules from tao-dataservices (top-level directories with entrypoints)
    ds_modules = set()
    if exists(nvidia_tao_ds_dir):
        for item in listdir(nvidia_tao_ds_dir):
            item_path = join(nvidia_tao_ds_dir, item)
            if os.path.isdir(item_path) and item not in ["__pycache__", "core", "api", "backbone", "config", "dataclass_to_rst"]:
                # Check if it has an entrypoint (indicates it's a network module)
                entrypoint_path = join(item_path, "entrypoint")
                if exists(entrypoint_path):
                    ds_modules.add(item)

    # Return only modules that exist in both places, honoring known name aliases.
    supported = [
        module for module in config_modules
        if module in ds_modules or CONFIG_TO_DS_MODULE_ALIASES.get(module) in ds_modules
    ]

    if not supported:
        logging.warning(
            f"No matching modules found between config ({len(config_modules)} modules) "
            f"and dataservices implementation ({len(ds_modules)} modules)"
        )

    return sorted(supported)


def import_module_from_path(module_name):
    """
    Import a module from its full path.

    Args:
        module_name (str): Full module path (e.g., 'nvidia_tao_ds.config.annotations.default_config')

    Returns:
        module: The imported module
    """
    import importlib
    return importlib.import_module(module_name)


def dataclass_to_yaml(dataclass_obj, yaml_file_path):
    """
    Convert a dataclass object to a YAML file using omegaconf.

    Args:
        dataclass_obj (object): The dataclass object to convert.
        yaml_file_path (str): The path to the output YAML file.

    Returns:
        None
    """
    if not hasattr(dataclass_obj, "__dataclass_fields__"):
        raise ValueError("Provided object is not a dataclass instance.")

    # Convert dataclass to OmegaConf structured object
    conf = OmegaConf.structured(dataclass_obj)

    # Save as YAML
    output_dir = dirname(yaml_file_path)
    if output_dir and not exists(output_dir):
        makedirs(output_dir, exist_ok=True)
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write(OmegaConf.to_yaml(conf))
        logging.info(f"Generated default spec: {yaml_file_path}")


@dataclass
class DefaultConfig:
    """This is a structured config for generating default specs."""

    # Minimalistic experiment manager.
    results_dir: str = MISSING
    module_name: str = MISSING


spec_path = dirname(abspath(__file__))


@hydra_runner(config_path=spec_path, config_name="default_specs", schema=DefaultConfig)
def main(cfg: DefaultConfig) -> None:
    """Script to generate default experiment YAML from dataclasses.

    Args:
        cfg (OmegaConf.DictConf): Hydra parsed config object.
    """
    logging.info(f"Generating default spec for module: {cfg.module_name}")

    # Validate module name
    supported_modules = get_supported_modules()
    if cfg.module_name not in supported_modules:
        error_msg = (f"Module '{cfg.module_name}' is not supported.\n"
                     f"Supported modules: {', '.join(supported_modules)}")
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Create results directory if it doesn't exist
    if not exists(cfg.results_dir):
        makedirs(cfg.results_dir, exist_ok=True)
        logging.info(f"Created results directory: {cfg.results_dir}")

    # Set output file path
    output_filename = "experiment.yaml"
    output_path = join(cfg.results_dir, output_filename)
    if exists(output_path):
        logging.warning(f"Output file already exists and will be overwritten: {output_path}")

    # Import the module and get the ExperimentConfig dataclass
    module_path = f"nvidia_tao_ds.config.{cfg.module_name}.default_config"
    try:
        imported_module = import_module_from_path(module_path)
        if not hasattr(imported_module, 'ExperimentConfig'):
            raise AttributeError(f"Module '{module_path}' does not have 'ExperimentConfig' dataclass")

        # Generate YAML from dataclass
        dataclass_to_yaml(imported_module.ExperimentConfig, output_path)

        # Success logging
        logging.info(f"Default specification file for {cfg.module_name} generated at '{output_path}'")

    except ImportError as e:
        error_msg = f"Failed to import module '{module_path}': {str(e)}"
        logging.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Failed to generate spec for {cfg.module_name}: {str(e)}"
        logging.error(error_msg)
        raise


if __name__ == "__main__":
    main()
