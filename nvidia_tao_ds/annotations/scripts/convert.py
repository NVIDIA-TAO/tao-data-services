# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entrypoint script to run TAO data format conversion."""

import os
import sys

# Cap OPENBLAS threads before numpy is imported — OpenBLAS allocates thread metadata at library
# load time, so this must be set before any downstream import triggers numpy initialization.
# Caps to min(user_value, available_cpus, 64) to handle both >128-core systems (e.g. GH200)
# and machines where fewer CPUs are available than the requested thread count.
_n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
os.environ["OPENBLAS_NUM_THREADS"] = str(min(int(os.environ.get("OPENBLAS_NUM_THREADS", 64)), _n_cpus, 64))

from nvidia_tao_core.config.annotations.default_config import ExperimentConfig  # noqa: E402
from nvidia_tao_ds.annotations.conversion.mapping import CONVERSION_MAPPING  # noqa: E402
from nvidia_tao_ds.core.decorators import monitor_status  # noqa: E402
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner  # noqa: E402


@hydra_runner(
    config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../experiment_specs"),
    config_name="annotations", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for format conversion."""
    os.makedirs(cfg.results_dir, exist_ok=True)
    run_conversion(cfg=cfg)


@monitor_status(mode='Annotation conversion')
def run_conversion(cfg: ExperimentConfig):
    """TAO annotation convert wrapper."""
    try:
        if cfg.get("kitti", {}).get("mapping", None) == "":
            cfg["kitti"]["mapping"] = None
        input_format = cfg.data.input_format.lower()
        output_format = cfg.data.output_format.lower()
        if input_format not in CONVERSION_MAPPING or output_format not in CONVERSION_MAPPING[input_format]:
            print(f"Unsupported conversion mapping: {input_format} -> {output_format}")
        else:
            CONVERSION_MAPPING[input_format][output_format](cfg, verbose=cfg.verbose)
    except KeyboardInterrupt as e:
        print(f"Interrupting data conversion with error: {e}")
        sys.exit()
    except RuntimeError as e:
        print(f"Data conversion run failed with error: {e}")
        sys.exit()


if __name__ == '__main__':
    main()
