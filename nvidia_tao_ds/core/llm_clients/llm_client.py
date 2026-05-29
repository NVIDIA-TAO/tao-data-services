# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM client abstraction and factory."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM and VLM clients.

    Subclasses must implement ``generate_text`` and ``generate_with_video``
    to provide text-only and video+text generation capabilities respectively.
    """

    @abstractmethod
    def generate_text(self, prompt, temperature=None, max_tokens=None):
        """Generate text from a text-only prompt.

        Args:
            prompt (str): The input prompt text.
            temperature (float | None): Sampling temperature override.
                Uses the client default when None.
            max_tokens (int | None): Maximum output token count override.
                Uses the client default when None.

        Returns:
            str: The generated text response.
        """

    @abstractmethod
    def generate_with_video(self, video_path, prompt, temperature=None):
        """Generate text conditioned on a video and a text prompt.

        Args:
            video_path (str): Path to the input video file.
            prompt (str): The text prompt accompanying the video.
            temperature (float | None): Sampling temperature override.
                Uses the client default when None.

        Returns:
            str: The generated text response.
        """

    @abstractmethod
    def generate_with_image(self, image_path, prompt, temperature=None):
        """Generate text conditioned on a single image and a text prompt.

        Args:
            image_path (str): Path to the input image file.
            prompt (str): The text prompt accompanying the image.
            temperature (float | None): Sampling temperature override.
                Uses the client default when None.

        Returns:
            str: The generated text response.
        """


def create_client(llm_config, workflow_cfg=None):
    """Instantiate an LLMClient from a backend configuration.

    Selects GeminiClient or OpenAICompatibleClient based on the ``backend``
    field. When *workflow_cfg* is provided, long-video parameters are
    forwarded to the backend config.

    Args:
        llm_config (object | dict): Configuration object (or dict) with a
            ``backend`` field (``"gemini"`` or ``"openai"``) and a
            corresponding sub-config (``gemini`` / ``openai``).
        workflow_cfg (object | None): Optional workflow config supplying
            ``long_video_threshold_sec``, ``long_video_sample_fps``, and
            ``long_video_max_frames``.

    Returns:
        LLMClient: A concrete client instance ready for generation calls.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    from nvidia_tao_ds.core.llm_clients.gemini_client import GeminiClient
    from nvidia_tao_ds.core.llm_clients.openai_compatible_client import OpenAICompatibleClient

    backend = llm_config.backend if hasattr(llm_config, "backend") else llm_config.get("backend", "gemini")
    if backend == "gemini":
        gemini_cfg = llm_config.gemini if hasattr(llm_config, "gemini") else llm_config.get("gemini", {})
        _inject_workflow_params(gemini_cfg, workflow_cfg)
        return GeminiClient(gemini_cfg)
    if backend == "openai":
        openai_cfg = llm_config.openai if hasattr(llm_config, "openai") else llm_config.get("openai", {})
        _inject_workflow_params(openai_cfg, workflow_cfg)
        return OpenAICompatibleClient(openai_cfg)
    raise ValueError(f"Unknown LLM backend: {backend}")


def _inject_workflow_params(backend_cfg, workflow_cfg):
    """Copy long-video params from workflow config into backend config."""
    if workflow_cfg is None:
        return
    for key in ("long_video_threshold_sec", "long_video_sample_fps", "long_video_max_frames"):
        if not hasattr(backend_cfg, key) and hasattr(workflow_cfg, key):
            try:
                setattr(backend_cfg, key, getattr(workflow_cfg, key))
            except (AttributeError, TypeError):
                pass
