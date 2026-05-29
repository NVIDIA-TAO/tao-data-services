# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default config file."""

import warnings
from typing import List, Optional, Dict
from dataclasses import dataclass

from nvidia_tao_core.config.grounding_dino.dataset import GDINOAugmentationConfig
from nvidia_tao_core.config.grounding_dino.model import GDINOModelConfig
from nvidia_tao_core.config.grounding_dino.train import GDINOTrainExpConfig
from nvidia_tao_core.config.mal.default_config import (
    MALTrainExpConfig,
    MALEvalExpConfig,
    MALInferenceExpConfig,
    MALDatasetConfig,
    MALModelConfig,
)
from nvidia_tao_ds.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    FLOAT_FIELD,
    BOOL_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class MALConfig:
    """MAL config."""

    dataset: MALDatasetConfig = DATACLASS_FIELD(
        MALDatasetConfig(),
        description="Configuration parameters for MAL dataset"
    )
    train: MALTrainExpConfig = DATACLASS_FIELD(
        MALTrainExpConfig(),
        description="Configuration parameters for MAL train"
    )
    model: MALModelConfig = DATACLASS_FIELD(
        MALModelConfig(),
        description="Configuration parameters for MAL model"
    )
    inference: MALInferenceExpConfig = DATACLASS_FIELD(
        MALInferenceExpConfig(),
        description="Configuration parameters for MAL inference"
    )
    evaluate: MALEvalExpConfig = DATACLASS_FIELD(
        MALEvalExpConfig(),
        description="Configuration parameters for MAL evaluation"
    )
    checkpoint: Optional[str] = STR_FIELD(
        None,
        default_value="",
        description="MAL model checkpoint path",
    )
    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Result directory",
    )


@dataclass
class GDINOConfig:
    """Grounding DINO config."""

    @dataclass
    class GDINODataConfig:
        """DINO dataset config used for auto-labeling."""

        image_dir: Optional[str] = STR_FIELD(
            None,
            default_value="",
            description="Image root directory",
        )
        noun_chunk_path: Optional[str] = STR_FIELD(
            value=None,
            default_value=""
        )
        class_names: Optional[List[str]] = LIST_FIELD(
            arrList=[],
            description="List of classes to run auto-labeling"
        )
        augmentation: GDINOAugmentationConfig = DATACLASS_FIELD(
            GDINOAugmentationConfig(),
            description="Configuration parameters for Grounding DINO augmenation"
        )

    train: GDINOTrainExpConfig = DATACLASS_FIELD(
        GDINOTrainExpConfig(),
        description="Configuration parameters for Grounding DINO train"
    )
    model: GDINOModelConfig = DATACLASS_FIELD(
        GDINOModelConfig(),
        description="Configuration parameters for Grounding DINO model"
    )
    dataset: GDINODataConfig = DATACLASS_FIELD(
        GDINODataConfig(),
        description="Configuration parameters for Grounding DINO dataset"
    )

    checkpoint: Optional[str] = STR_FIELD(
        None,
        default_value="",
        description="Grounding model checkpoint path",
    )

    results_dir: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Result directory",
    )

    iteration_scheduler: List[Dict[str, float]] = LIST_FIELD(
        arrList=[{"conf_threshold": 0.5, "nms_threshold": 0.0}],
        default_values=[{"conf_threshold": 0.5, "nms_threshold": 0.0}],
        description="""The list of iteration schedule. Default is one iteration with confidence threshold of 0.5.
                    Next iteration eliminates classes/noun chunks that have been already detected."""
    )
    visualize: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Flag to enable visualization of bounding boxes."
    )


@dataclass
class VideoReasoningAnnotationGeminiConfig:
    """Gemini API configuration for video reasoning annotation pipeline."""

    api_key: str = STR_FIELD(
        value="",
        default_value="",
        description="Google Gemini API key (or set GOOGLE_API_KEY env var)",
    )
    model: str = STR_FIELD(
        value="gemini-3.1-flash-lite-preview",
        default_value="gemini-3.1-flash-lite-preview",
        description="Gemini model name",
    )
    media_resolution: str = STR_FIELD(
        value="MEDIA_RESOLUTION_LOW",
        default_value="MEDIA_RESOLUTION_LOW",
        description="Media resolution for video input",
    )
    temperature: float = FLOAT_FIELD(
        value=0.3,
        default_value=0.3,
        description="Sampling temperature",
    )
    max_output_tokens: int = INT_FIELD(
        value=8192,
        default_value=8192,
        description="Maximum output tokens",
    )
    timeout: int = INT_FIELD(
        value=120,
        default_value=120,
        description="Request timeout in seconds",
    )


@dataclass
class VideoReasoningAnnotationOpenAIConfig:
    """OpenAI-compatible endpoint configuration for video reasoning annotation pipeline."""

    api_key: str = STR_FIELD(
        value="",
        default_value="",
        description="API key for OpenAI-compatible endpoint",
    )
    base_url: str = STR_FIELD(
        value="",
        default_value="",
        description="Base URL for OpenAI-compatible endpoint",
    )
    model_name: str = STR_FIELD(
        value="",
        default_value="",
        description="Model name for OpenAI-compatible endpoint",
    )
    temperature: float = FLOAT_FIELD(
        value=0.7,
        default_value=0.7,
        description="Sampling temperature",
    )
    max_tokens: int = INT_FIELD(
        value=4096,
        default_value=4096,
        description="Maximum output tokens",
    )
    timeout: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Request timeout in seconds",
    )


@dataclass
class VideoReasoningAnnotationLLMConfig:
    """LLM backend selection and configuration for video reasoning annotation pipeline."""

    backend: str = STR_FIELD(
        value="gemini",
        default_value="gemini",
        description="LLM backend to use",
        valid_options="gemini,openai",
    )
    gemini: VideoReasoningAnnotationGeminiConfig = DATACLASS_FIELD(
        VideoReasoningAnnotationGeminiConfig(),
        description="Gemini API configuration",
    )
    openai: VideoReasoningAnnotationOpenAIConfig = DATACLASS_FIELD(
        VideoReasoningAnnotationOpenAIConfig(),
        description="OpenAI-compatible endpoint configuration",
    )


@dataclass
class VideoReasoningAnnotationWorkflowConfig:
    """Pipeline execution parameters for video reasoning annotation."""

    steps: List[str] = LIST_FIELD(
        arrList=["0", "1a", "1b", "1c", "2", "3", "4"],
        default_values=["0", "1a", "1b", "1c", "2", "3", "4"],
        description="Pipeline steps to execute",
    )
    mode: str = STR_FIELD(
        value="auto",
        default_value="auto",
        description="Pipeline mode: auto (VLM classifies), anomaly, or normal",
        valid_options="auto,anomaly,normal",
    )
    max_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        description="Maximum concurrent workers for video processing",
    )
    max_video_length_sec: int = INT_FIELD(
        value=300,
        default_value=300,
        description="Maximum video length in seconds",
    )
    chunk_duration_options: List[int] = LIST_FIELD(
        arrList=[5, 10, 15, 20, 30],
        default_values=[5, 10, 15, 20, 30],
        description="Chunk duration options in seconds",
    )
    max_chunks: int = INT_FIELD(
        value=10,
        default_value=10,
        description="Maximum number of chunks per video",
    )
    highlight_before_sec: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="Seconds to include before anomaly timestamp in highlight clip",
    )
    highlight_after_sec: float = FLOAT_FIELD(
        value=3.0,
        default_value=3.0,
        description="Seconds to include after anomaly timestamp in highlight clip",
    )
    long_video_threshold_sec: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Duration threshold (seconds) above which videos are sampled as frames",
    )
    long_video_sample_fps: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Frame sampling rate for long videos",
    )
    long_video_max_frames: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Maximum frames to sample from long videos",
    )
    qa_types: List[str] = LIST_FIELD(
        arrList=[
            "mcq", "bcq", "open_qa",
            "causal_linkage", "temporal_localization", "temporal_event_desc",
            "scene_description", "event_summary",
        ],
        default_values=[
            "mcq", "bcq", "open_qa",
            "causal_linkage", "temporal_localization", "temporal_event_desc",
            "scene_description", "event_summary",
        ],
        description="QA types to generate",
    )


@dataclass
class VideoReasoningAnnotationDataConfig:
    """Input data specification for video reasoning annotation pipeline.

    At least one of ``video_root`` or ``input_jsonl_files`` must be provided.
    Both may be used together — the resulting video lists are merged.

    When using ``input_jsonl_files``, each JSONL file should contain one JSON
    object per line with at least a ``"video_path"`` (or ``"video"``) field::

        {"video_path": "/absolute/path/to/video.mp4"}

    Additional fields are allowed. If ``filter_field`` is set, only entries
    where that boolean field is truthy are included.
    """

    video_root: str = STR_FIELD(
        value="",
        default_value="",
        description="Root directory containing input videos (walked recursively). "
                    "At least one of video_root or input_jsonl_files must be provided; both may be used together.",
    )
    input_jsonl_files: List[str] = LIST_FIELD(
        arrList=[],
        default_values=[],
        description="Optional list of JSONL files listing video paths. "
                    "Each line must have a 'video_path' (or 'video') field. "
                    "Can be used instead of or in addition to video_root.",
    )
    filter_field: Optional[str] = STR_FIELD(
        value=None,
        default_value="",
        description="Optional boolean field name to filter entries in input JSONL files",
    )


@dataclass
class VideoReasoningAnnotationConfig:
    """Video reasoning annotation pipeline configuration."""

    vlm: VideoReasoningAnnotationLLMConfig = DATACLASS_FIELD(
        VideoReasoningAnnotationLLMConfig(),
        description="VLM (vision-language model) configuration for video steps",
    )
    llm: VideoReasoningAnnotationLLMConfig = DATACLASS_FIELD(
        VideoReasoningAnnotationLLMConfig(),
        description="LLM (text-only) configuration for text steps",
    )
    workflow: VideoReasoningAnnotationWorkflowConfig = DATACLASS_FIELD(
        VideoReasoningAnnotationWorkflowConfig(),
        description="Pipeline workflow parameters",
    )
    data: VideoReasoningAnnotationDataConfig = DATACLASS_FIELD(
        VideoReasoningAnnotationDataConfig(),
        description="Input data configuration",
    )
    license: str = STR_FIELD(
        value="",
        default_value="",
        description=(
            "License string written to metadata.license in the "
            "tao-vl-reason-v1.0 output envelope (e.g. 'CC-BY-4.0'). "
            "Empty string by default."
        ),
    )
    description_extra: str = STR_FIELD(
        value="",
        default_value="",
        description=(
            "Extra text appended to the per-task description in the "
            "tao-vl-reason-v1.0 output metadata. Useful for naming "
            "the dataset, source, or other context."
        ),
    )
    prompts_module: str = STR_FIELD(
        value="",
        default_value="",
        description="Optional Python module path for custom prompt templates",
    )


# =============================================================================
# Image Grounding / Referring Data Engine Configs (image_grounding, image_referring_expression)
# =============================================================================


# Alias for readability in image_grounding / image_referring_expression configs. Reuses the same
# {backend, gemini, openai} layout as video reasoning annotation for consistency.
LLMBackendConfig = VideoReasoningAnnotationLLMConfig


@dataclass
class ImageGDWorkflowConfig:
    """Pipeline execution parameters for image_grounding pipeline."""

    steps: List[str] = LIST_FIELD(
        arrList=["0", "1"],
        default_values=["0", "1"],
        description="Pipeline steps to execute (0=expression extraction, 1=phrase grounding)",
    )
    max_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        description="Maximum concurrent workers for per-sample API calls",
    )
    force_reprocess: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Ignore cached outputs and reprocess from scratch",
    )
    long_video_threshold_sec: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Unused by image pipelines; retained for LLMClient compatibility",
    )
    long_video_sample_fps: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Unused by image pipelines; retained for LLMClient compatibility",
    )
    long_video_max_frames: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Unused by image pipelines; retained for LLMClient compatibility",
    )


@dataclass
class ImageGDDataConfig:
    """Input data specification for image_grounding pipeline.

    Input JSONL should have one JSON object per line with at least
    ``image_path`` and ``caption`` fields. ``width``, ``height``, and
    ``image_id`` are optional and auto-filled when missing.
    """

    input_jsonl: str = STR_FIELD(
        value="",
        default_value="",
        description="Path to input JSONL with image_path and caption fields",
    )
    image_root: str = STR_FIELD(
        value="",
        default_value="",
        description="Optional prefix for resolving relative image_path values",
    )


@dataclass
class ImageGDConfig:
    """Image grounding-data-engine configuration."""

    vlm: LLMBackendConfig = DATACLASS_FIELD(
        LLMBackendConfig(),
        description="VLM backend configuration for image-based inference",
    )
    workflow: ImageGDWorkflowConfig = DATACLASS_FIELD(
        ImageGDWorkflowConfig(),
        description="Pipeline workflow parameters",
    )
    data: ImageGDDataConfig = DATACLASS_FIELD(
        ImageGDDataConfig(),
        description="Input data configuration",
    )


@dataclass
class ImageREWorkflowConfig:
    """Pipeline execution parameters for image_referring_expression pipeline."""

    steps: List[str] = LIST_FIELD(
        arrList=["0", "1", "2", "3"],
        default_values=["0", "1", "2", "3"],
        description=(
            "Pipeline steps to execute "
            "(0=region_expr, 1=image_caption, 2=grounding_expr, 3=double_check)"
        ),
    )
    max_workers: int = INT_FIELD(
        value=4,
        default_value=4,
        valid_min=1,
        description="Maximum concurrent workers for per-image API calls within each step",
    )
    force_reprocess: bool = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Ignore cached outputs and reprocess from scratch",
    )
    output_format: str = STR_FIELD(
        value="jsonl",
        default_value="jsonl",
        description=(
            "Output format: 'jsonl' (unified schema only), 'legacy' "
            "(2d-data-engine .txt.stepN files only), or 'both'"
        ),
        valid_options="jsonl,legacy,both",
    )
    long_video_threshold_sec: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Unused by image pipelines; retained for LLMClient compatibility",
    )
    long_video_sample_fps: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="Unused by image pipelines; retained for LLMClient compatibility",
    )
    long_video_max_frames: int = INT_FIELD(
        value=60,
        default_value=60,
        description="Unused by image pipelines; retained for LLMClient compatibility",
    )


@dataclass
class ImageREDataConfig:
    """Input data specification for image_referring_expression pipeline."""

    image_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Directory containing input images (.jpg/.png)",
    )
    kitti_label_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Directory containing KITTI-format bounding box labels",
    )
    input_annotations_jsonl: str = STR_FIELD(
        value="",
        default_value="",
        description=(
            "Optional unified annotations.jsonl to seed the pipeline "
            "(for resuming or running on pre-computed regions)"
        ),
    )


@dataclass
class ImageREConfig:
    """Image referring-data-engine configuration."""

    vlm: LLMBackendConfig = DATACLASS_FIELD(
        LLMBackendConfig(),
        description="VLM backend configuration for image-based inference",
    )
    workflow: ImageREWorkflowConfig = DATACLASS_FIELD(
        ImageREWorkflowConfig(),
        description="Pipeline workflow parameters",
    )
    data: ImageREDataConfig = DATACLASS_FIELD(
        ImageREDataConfig(),
        description="Input data configuration",
    )


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        default_value=[0],
        description="Indices of GPUs to use"
    )
    num_gpus: int = INT_FIELD(value=1,
                              default_value=1,
                              description="Number of GPUs to use")
    batch_size: int = INT_FIELD(value=4,
                                default_value=4,
                                valid_min=1,
                                description="Batch size")
    num_workers: int = INT_FIELD(value=8,
                                 default_value=8,
                                 valid_min=1,
                                 description="Number of workers for dataloader")

    autolabel_type: str = STR_FIELD(
        value="mal",
        default_value="mal",
        description="Type of auto-labeling to run",
        valid_options=(
            "mal,grounding_dino,video_reasoning_annotation,"
            "image_grounding,image_referring_expression,vllm_captioning"
        )
    )

    mal: MALConfig = DATACLASS_FIELD(
        MALConfig(),
        description="Configuration parameters for MAL"
    )
    grounding_dino: GDINOConfig = DATACLASS_FIELD(
        GDINOConfig(),
        description="Configuration parameters for Grounding DINO"
    )
    video_reasoning_annotation: VideoReasoningAnnotationConfig = DATACLASS_FIELD(
        VideoReasoningAnnotationConfig(),
        description="Configuration parameters for video reasoning annotation pipeline"
    )
    image_grounding: ImageGDConfig = DATACLASS_FIELD(
        ImageGDConfig(),
        description="Configuration parameters for image grounding data engine"
    )
    image_referring_expression: ImageREConfig = DATACLASS_FIELD(
        ImageREConfig(),
        description="Configuration parameters for image referring data engine"
    )
    results_dir: str = STR_FIELD(
        value="",
        default_value="",
        description="Result directory",
    )

    def __post_init__(self):
        """assertion check."""
        valid_types = [
            "mal", "grounding_dino", "video_reasoning_annotation",
            "image_grounding", "image_referring_expression", "vllm_captioning"
        ]
        assert self.autolabel_type in valid_types, \
            f"Invalid option encountered. {self.autolabel_type}"
        if self.autolabel_type == "vllm_captioning":
            tensor_parallel_size = self.vllm_captioning.model.llm.tensor_parallel_size
            assert self.num_gpus == tensor_parallel_size, (
                f"num_gpus ({self.num_gpus}) must match vllm_captioning.model.llm.tensor_parallel_size "
                f"({tensor_parallel_size})."
            )
            warnings.warn(
                "batch_size is not used for vllm_captioning; vLLM manages batching internally.",
                UserWarning,
            )
