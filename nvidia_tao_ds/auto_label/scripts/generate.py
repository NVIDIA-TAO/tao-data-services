# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate annotations using existing AI models in TAO Toolkit"""

import os

from nvidia_tao_ds.config.auto_label.default_config import ExperimentConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_ds.core.decorators import monitor_status
from nvidia_tao_ds.core.logging.logging import enable_dual_logging

from nvidia_tao_ds.auto_label.grounding_dino.inference import run_grounding_inference
from nvidia_tao_ds.auto_label.image_grounding.inference import run_image_grounding_inference
from nvidia_tao_ds.auto_label.image_referring_expression.inference import run_image_referring_expression_inference
from nvidia_tao_ds.auto_label.mal.inference import run_mal_inference
from nvidia_tao_ds.auto_label.video_reasoning_annotation.inference import run_video_reasoning_annotation_inference


@monitor_status(mode='Auto-label')
def run_experiment(cfg, results_dir=None):
    """Start the inference."""
    enable_dual_logging()
    os.makedirs(results_dir, exist_ok=True)
    if cfg.autolabel_type == "grounding_dino":
        run_grounding_inference(cfg, results_dir)
    elif cfg.autolabel_type == "mal":
        run_mal_inference(cfg, results_dir)
    elif cfg.autolabel_type == "video_reasoning_annotation":
        run_video_reasoning_annotation_inference(cfg, results_dir)
    elif cfg.autolabel_type == "image_grounding":
        run_image_grounding_inference(cfg, results_dir)
    elif cfg.autolabel_type == "image_referring_expression":
        run_image_referring_expression_inference(cfg, results_dir)
    else:
        raise NotImplementedError(f"{cfg.autolabel_type}")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="generate", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process."""
    run_experiment(cfg,
                   results_dir=cfg.results_dir)


if __name__ == "__main__":
    main()
