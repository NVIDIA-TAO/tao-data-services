# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
from omegaconf import OmegaConf

from nvidia_tao_core.config.auto_label.default_config import ExperimentConfig
from nvidia_tao_ds.auto_label.grounding_dino.inference import run_grounding_inference


TEST_IMAGE_ROOT = "/media/scratch_metropolis2/tao_ci/data_services/data/Auto_Label/gdino/image_20/"
TEST_JSONL_ROOT = "/media/scratch_metropolis2/tao_ci/data_services/data/Auto_Label/gdino/noun_chunks.jsonl"
MODEL_PATH = "/media/scratch_metropolis2/tao_ci/data_services/models/swint.pth"


@pytest.fixture(scope='session')
def results_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('ds_annot')


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("visualize", [True, False])
@pytest.mark.parametrize("iteration", [[{'conf_threshold': 0.5, 'nms_threshold': 0.7}],
                                       [{'conf_threshold': 0.5, 'nms_threshold': 0.7},
                                        {'conf_threshold': 0.3, 'nms_threshold': 0.7}]])
def test_grounding_dino_closed(results_dir, batch_size, iteration, visualize):
    """Test function to validate closed-set autolabeling of GDINO."""

    # override config
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.autolabel_type = "grounding_dino"
    cfg.results_dir = results_dir

    cfg.batch_size = batch_size
    cfg.num_workers = 0  # For small shm size
    cfg.grounding_dino.visualize = visualize
    cfg.grounding_dino.model.backbone = "swin_tiny_224_1k"
    cfg.grounding_dino.dataset.image_dir = TEST_IMAGE_ROOT
    cfg.grounding_dino.dataset.class_names = ['person', 'clothes']
    cfg.grounding_dino.checkpoint = MODEL_PATH
    cfg.grounding_dino.iteration_scheduler = iteration
    os.environ['TAO_VISIBLE_DEVICES'] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TAO_VISIBLE_DEVICES"]
    run_grounding_inference(cfg, results_dir)

    # Check if the final annotation files were dumped correctly.
    output_json = os.path.join(results_dir, "final_annotation.jsonl")
    assert os.path.exists(output_json), f"Output jsonl file {output_json} doesn't exist"
    labelmap_json = os.path.join(results_dir, "labelmap.json")
    assert os.path.exists(labelmap_json), f"Labelmap json file {labelmap_json} doesn't exist {os.listdir(results_dir)}"

    # Check if visualization was stored correctly
    if visualize:
        visualize_dir = os.path.join(results_dir, "auto_label0", "images_annotated")
        vis_len = len(os.listdir(visualize_dir))
        assert vis_len, f"Visualization not dumped correctly {visualize_dir}"

    # Check the annotation format
    with open(output_json, "r") as f:
        metas = [json.loads(l) for l in f]

    total = 0
    for meta in metas:
        assert meta["file_name"], "file_name not found"
        assert meta["width"], "width not found"
        assert meta["height"], "height not found"
        assert meta["detection"], "detection not found"
        assert isinstance(meta['detection'], dict), "incorrect format of detection"
        
        for inst in meta["detection"]["instances"]:
            total += len(inst)
    
    assert total > 0, "No detection found after merging"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("visualize", [True, False])
@pytest.mark.parametrize("iteration", [[{'conf_threshold': 0.5, 'nms_threshold': 0.7}],
                                       [{'conf_threshold': 0.5, 'nms_threshold': 0.7},
                                        {'conf_threshold': 0.3, 'nms_threshold': 0.7}]])
def test_grounding_dino_grounding(results_dir, batch_size, iteration, visualize):
    """Test function to validate grounding autolabeling of GDINO."""

    # override config
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.autolabel_type = "grounding_dino"
    cfg.results_dir = results_dir

    cfg.batch_size = batch_size
    cfg.num_workers = 0  # For small shm size
    cfg.grounding_dino.visualize = visualize
    cfg.grounding_dino.model.backbone = "swin_tiny_224_1k"
    cfg.grounding_dino.dataset.image_dir = TEST_IMAGE_ROOT
    cfg.grounding_dino.dataset.noun_chunk_path = TEST_JSONL_ROOT
    cfg.grounding_dino.checkpoint = MODEL_PATH
    cfg.grounding_dino.iteration_scheduler = iteration
    os.environ['TAO_VISIBLE_DEVICES'] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TAO_VISIBLE_DEVICES"]
    run_grounding_inference(cfg, results_dir)

    # Check if the final annotation files were dumped correctly.
    output_json = os.path.join(results_dir, "final_annotation.jsonl")
    assert os.path.exists(output_json), f"Output jsonl file {output_json} doesn't exist"

    # Check if visualization was stored correctly
    if visualize:
        visualize_dir = os.path.join(results_dir, "auto_label0", "images_annotated")
        vis_len = len(os.listdir(visualize_dir))
        assert vis_len, f"Visualization not dumped correctly {visualize_dir}"

    # Check the annotation format
    with open(output_json, "r") as f:
        metas = [json.loads(l) for l in f]

    total = 0
    for meta in metas:
        assert meta["file_name"], "file_name not found"
        assert meta["width"], "width not found"
        assert meta["height"], "height not found"
        assert "grounding" in meta, f"grounding not found {meta.keys()}"
        assert isinstance(meta['grounding'], dict), "incorrect format of grounding"
        
        for inst in meta["grounding"]["regions"]:
            total += len(inst)
    
    assert total > 0, "No grounding found after merging"
