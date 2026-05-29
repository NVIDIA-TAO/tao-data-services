# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible LLM client implementation."""

import base64
import os
import re
import time

from openai import OpenAI

from nvidia_tao_ds.core.llm_clients.llm_client import LLMClient
from nvidia_tao_ds.core.logging.logging import logging as logger
from nvidia_tao_ds.core.utils.video_utils import (
    get_video_length_sec,
    sample_frames,
)

_IMAGE_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


def _image_mime_type(image_path):
    """Infer image MIME type from file extension; default to image/jpeg."""
    ext = os.path.splitext(image_path)[1].lower()
    return _IMAGE_MIME_BY_EXT.get(ext, "image/jpeg")


class OpenAICompatibleClient(LLMClient):
    """Client for OpenAI-compatible endpoints (vLLM, NVIDIA NIM, etc.)."""

    def __init__(self, cfg):
        """Initialize the OpenAI-compatible client.

        Args:
            cfg (object): OpenAIConfig-like object with attributes:
                ``api_key``, ``base_url``, ``model_name``, ``temperature``,
                ``max_tokens``, ``timeout``, and optionally
                ``long_video_threshold_sec``, ``long_video_sample_fps``,
                ``long_video_max_frames``.
        """
        self.client = OpenAI(
            api_key=cfg.api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=cfg.base_url,
            timeout=cfg.timeout,
        )
        self.model_name = cfg.model_name
        self.default_temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.long_video_threshold_sec = getattr(cfg, "long_video_threshold_sec", 60)
        self.long_video_sample_fps = getattr(cfg, "long_video_sample_fps", 0.5)
        self.long_video_max_frames = getattr(cfg, "long_video_max_frames", 60)

    def generate_text(self, prompt, temperature=None, max_tokens=None):
        """Generate text from a text-only prompt via an OpenAI-compatible endpoint.

        Args:
            prompt (str): The input prompt text.
            temperature (float | None): Sampling temperature override.
            max_tokens (int | None): Maximum output token count override.

        Returns:
            str: Generated text with thinking tags stripped.
        """
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        return self._call_with_retry(
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=tokens,
        )

    def generate_with_video(self, video_path, prompt, temperature=None):
        """Generate text conditioned on a video and prompt via an OpenAI-compatible endpoint.

        Long videos are frame-sampled and sent as images; short videos are
        sent as inline base64-encoded MP4.

        Args:
            video_path (str): Path to the input video file (MP4).
            prompt (str): The text prompt accompanying the video.
            temperature (float | None): Sampling temperature override.

        Returns:
            str: Generated text with thinking tags stripped.
        """
        temp = temperature if temperature is not None else self.default_temperature
        duration = get_video_length_sec(video_path)

        # Long videos: sample frames
        if duration is not None and duration > self.long_video_threshold_sec:
            frames = sample_frames(
                video_path, duration,
                self.long_video_sample_fps, self.long_video_max_frames,
            )
            if frames:
                content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                    for b64 in frames
                ]
                content.append({"type": "text", "text": prompt})
                return self._call_with_retry(
                    messages=[{"role": "user", "content": content}],
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )

        # Short videos: send full video as base64
        with open(video_path, "rb") as f:
            b64_video = base64.b64encode(f.read()).decode("utf-8")

        content = [
            {
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{b64_video}"},
            },
            {"type": "text", "text": prompt},
        ]
        return self._call_with_retry(
            messages=[{"role": "user", "content": content}],
            temperature=temp,
            max_tokens=self.max_tokens,
        )

    def generate_with_image(self, image_path, prompt, temperature=None):
        """Generate text conditioned on a single image and a text prompt.

        The image is sent as a base64-encoded data URL via the OpenAI
        ``image_url`` content block, with a MIME type inferred from the
        file extension.

        Args:
            image_path (str): Path to the input image file.
            prompt (str): The text prompt accompanying the image.
            temperature (float | None): Sampling temperature override.

        Returns:
            str: Generated text with thinking tags stripped.
        """
        temp = temperature if temperature is not None else self.default_temperature
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        mime = _image_mime_type(image_path)

        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
            {"type": "text", "text": prompt},
        ]
        return self._call_with_retry(
            messages=[{"role": "user", "content": content}],
            temperature=temp,
            max_tokens=self.max_tokens,
        )

    def _call_with_retry(self, messages, temperature, max_tokens,
                         max_retries=3, retry_delay=5):
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM")
                content = re.sub(
                    r"<think>.*?</think>", "", content, flags=re.DOTALL
                ).strip()
                if not content:
                    raise ValueError("Response empty after stripping thinking tags")
                return content
            except Exception as e:
                last_error = e
                logger.warning(
                    "OpenAI API error (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise last_error
