# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Google Gemini LLM client implementation."""

import base64
import os
import re
import time

from google import genai
from google.genai import types

from nvidia_tao_ds.core.llm_clients.llm_client import LLMClient
from nvidia_tao_ds.core.logging.logging import logging as logger
from nvidia_tao_ds.core.utils.video_utils import (
    get_video_length_sec,
    sample_frames,
    video_mime_type,
)

INLINE_VIDEO_MAX_BYTES = 20 * 1024 * 1024  # 20 MB

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


class GeminiClient(LLMClient):
    """Google Gemini via the google-genai SDK (v1alpha for media_resolution)."""

    def __init__(self, cfg):
        """Initialize the Gemini client.

        Args:
            cfg (object): GeminiConfig-like object with attributes:
                ``api_key``, ``model``, ``temperature``,
                ``max_output_tokens``, ``timeout``, and optionally
                ``media_resolution``, ``long_video_threshold_sec``,
                ``long_video_sample_fps``, ``long_video_max_frames``.
        """
        api_key = cfg.api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GOOGLE_API_KEY env var "
                "or provide api_key in config."
            )
        self.client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"},
        )
        self.model = cfg.model
        self.media_resolution = getattr(cfg, "media_resolution", None)
        self.default_temperature = cfg.temperature
        self.max_output_tokens = cfg.max_output_tokens
        self.timeout = cfg.timeout
        self.long_video_threshold_sec = getattr(cfg, "long_video_threshold_sec", 60)
        self.long_video_sample_fps = getattr(cfg, "long_video_sample_fps", 0.5)
        self.long_video_max_frames = getattr(cfg, "long_video_max_frames", 60)

    def generate_text(self, prompt, temperature=None, max_tokens=None):
        """Generate text from a text-only prompt via the Gemini API.

        Args:
            prompt (str): The input prompt text.
            temperature (float | None): Sampling temperature override.
            max_tokens (int | None): Maximum output token count override.

        Returns:
            str: Generated text with thinking tags stripped.
        """
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.max_output_tokens

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=tokens,
        )
        response = self._call_with_retry(
            contents=[types.Content(parts=[types.Part(text=prompt)])],
            config=config,
        )
        return self._extract_text(response)

    def _make_video_config(self, temperature=None):
        """Build GenerateContentConfig with media_resolution for video calls."""
        temp = temperature if temperature is not None else self.default_temperature
        kwargs = {"temperature": temp, "max_output_tokens": self.max_output_tokens}
        if self.media_resolution:
            kwargs["media_resolution"] = self.media_resolution
        return types.GenerateContentConfig(**kwargs)

    def generate_with_video(self, video_path, prompt, temperature=None):
        """Generate text conditioned on a video and prompt via Gemini.

        Automatically selects the upload strategy based on video length
        and file size: frame sampling for long videos, the Files API for
        large files, or inline bytes for short/small videos.

        Args:
            video_path (str): Path to the input video file (MP4).
            prompt (str): The text prompt accompanying the video.
            temperature (float | None): Sampling temperature override.

        Returns:
            str: Generated text with thinking tags stripped.
        """
        config = self._make_video_config(temperature)

        duration = get_video_length_sec(video_path)
        file_size = os.path.getsize(video_path)

        # Long videos: sample frames and send as images
        if duration is not None and duration > self.long_video_threshold_sec:
            logger.info(
                "Long video (%.0fs), sampling frames: %s",
                duration, os.path.basename(video_path),
            )
            frames = sample_frames(
                video_path, duration,
                self.long_video_sample_fps, self.long_video_max_frames,
            )
            if frames:
                parts = [
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=base64.b64decode(b64),
                        ),
                    )
                    for b64 in frames
                ]
                parts.append(types.Part(text=prompt))
                response = self._call_with_retry(
                    contents=[types.Content(parts=parts)],
                    config=config,
                )
                return self._extract_text(response)

        # Large files: upload via Files API
        if file_size > INLINE_VIDEO_MAX_BYTES:
            return self._generate_with_uploaded_video(
                video_path, prompt, config
            )

        # Small/short videos: inline
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        parts = [
            types.Part(inline_data=types.Blob(mime_type=video_mime_type(video_path), data=video_bytes)),
            types.Part(text=prompt),
        ]
        response = self._call_with_retry(
            contents=[types.Content(parts=parts)],
            config=config,
        )
        return self._extract_text(response)

    def generate_with_image(self, image_path, prompt, temperature=None):
        """Generate text conditioned on a single image and a text prompt via Gemini.

        The image is sent inline as a base64-encoded blob with a MIME type
        inferred from the file extension.

        Args:
            image_path (str): Path to the input image file.
            prompt (str): The text prompt accompanying the image.
            temperature (float | None): Sampling temperature override.

        Returns:
            str: Generated text with thinking tags stripped.
        """
        temp = temperature if temperature is not None else self.default_temperature
        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=self.max_output_tokens,
        )

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        parts = [
            types.Part(
                inline_data=types.Blob(
                    mime_type=_image_mime_type(image_path),
                    data=image_bytes,
                ),
            ),
            types.Part(text=prompt),
        ]
        response = self._call_with_retry(
            contents=[types.Content(parts=parts)],
            config=config,
        )
        return self._extract_text(response)

    def _generate_with_uploaded_video(self, video_path, prompt, config):
        """Upload video via Files API, poll until ready, then generate."""
        uploaded = self.client.files.upload(file=video_path)
        logger.info("Uploaded %s, waiting for processing...", video_path)

        while uploaded.state.name == "PROCESSING":
            time.sleep(2)
            uploaded = self.client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(
                f"Video upload failed for {video_path}: state={uploaded.state.name}"
            )

        parts = [
            types.Part(file_data=types.FileData(file_uri=uploaded.uri, mime_type=video_mime_type(video_path))),
            types.Part(text=prompt),
        ]
        response = self._call_with_retry(
            contents=[types.Content(parts=parts)],
            config=config,
        )
        return self._extract_text(response)

    def _call_with_retry(self, contents, config, max_retries=3, retry_delay=5):
        last_error = None
        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Gemini API error (attempt %d/%d): %s",
                    attempt + 1, max_retries, e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise last_error

    @staticmethod
    def _extract_text(response):
        text = response.text
        if not text:
            raise ValueError("Empty response from Gemini")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if not text:
            raise ValueError("Response empty after stripping thinking tags")
        return text
