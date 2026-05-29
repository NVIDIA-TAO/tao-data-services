# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video file operations using ffmpeg/ffprobe."""

import base64
import os
import subprocess
import tempfile

from nvidia_tao_ds.core.logging.logging import logging as logger


def get_video_length_sec(video_path):
    """Return the duration of a video in seconds using ffprobe.

    Args:
        video_path (str): Absolute or relative path to the video file.

    Returns:
        float | None: Duration in seconds, or None if the probe fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return float(result.stdout.decode().strip())
    except Exception as e:
        logger.warning("Could not get video length for %s: %s", video_path, e)
        return None


def sample_frames(video_path, duration_sec, sample_fps=0.5, max_frames=60):
    """Extract evenly-spaced JPEG frames from a video via ffmpeg.

    Frames are sampled at regular intervals and returned as base64 strings.

    Args:
        video_path (str): Path to the source video file.
        duration_sec (float): Total duration of the video in seconds.
        sample_fps (float): Target frames per second for sampling.
        max_frames (int): Upper bound on the number of frames to extract.

    Returns:
        list[str]: Base64-encoded JPEG strings, one per extracted frame.
    """
    n_frames = max(1, min(int(duration_sec * sample_fps), max_frames))
    interval = duration_sec / (n_frames + 1)
    timestamps = [interval * (i + 1) for i in range(n_frames)]
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, ts in enumerate(timestamps):
            frame_path = os.path.join(tmpdir, f"frame_{i:02d}.jpg")
            cmd = [
                "ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                "-frames:v", "1", "-q:v", "2", frame_path,
            ]
            try:
                subprocess.run(
                    cmd, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                continue
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode("utf-8"))
    return frames


def split_video_into_chunks(input_path, output_dir, chunk_duration=5):
    """Split a video into fixed-duration MP4 chunks using ffmpeg.

    Each chunk is re-encoded with libx264/aac and written to *output_dir*.

    Args:
        input_path (str): Path to the source video file.
        output_dir (str): Directory where chunk files will be written.
            Created automatically if it does not exist.
        chunk_duration (int): Target duration of each chunk in seconds.

    Returns:
        list[str]: Paths to the successfully created chunk files.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_length_sec(input_path)
    if duration is None:
        logger.warning("Skipping %s: could not determine duration.", input_path)
        return []

    chunks = []
    start_time = 0
    chunk_index = 0

    while start_time + 1 <= duration:
        end_time = min(start_time + chunk_duration, duration)
        output_path = os.path.join(output_dir, f"chunk_{chunk_index:03d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            output_path,
        ]
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            chunks.append(output_path)
        except subprocess.CalledProcessError as e:
            logger.warning(
                "Error creating chunk %d for %s: %s",
                chunk_index, input_path, e.stderr.decode(),
            )
        start_time += chunk_duration
        chunk_index += 1

    return chunks


def extract_highlight_clip(video_path, anomaly_time_sec, output_path,
                           before_sec=3.0, after_sec=3.0):
    """Extract a short clip centered on an anomaly timestamp.

    The clip spans from ``anomaly_time_sec - before_sec`` to
    ``anomaly_time_sec + after_sec``, clamped to the video boundaries.
    A minimum 2-second duration is enforced.

    Args:
        video_path (str): Path to the source video file.
        anomaly_time_sec (float): Timestamp of the anomaly in seconds.
        output_path (str): Destination path for the extracted clip.
        before_sec (float): Seconds to include before the anomaly.
        after_sec (float): Seconds to include after the anomaly.

    Returns:
        tuple[str | None, float | None, float | None]:
            ``(output_path, start_time, end_time)`` on success, or
            ``(None, None, None)`` on failure.
    """
    video_length = get_video_length_sec(video_path)
    if video_length is None:
        return None, None, None

    start_time = max(0, anomaly_time_sec - before_sec)
    end_time = min(video_length, anomaly_time_sec + after_sec)

    if end_time - start_time < 2:
        start_time = max(0, end_time - 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path,
    ]
    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return output_path, start_time, end_time
    except subprocess.CalledProcessError as e:
        logger.warning("Error extracting highlight clip: %s", e.stderr.decode())
        return None, None, None


def select_chunk_duration(video_length_sec, options=None, max_chunks=10):
    """Select the smallest chunk duration that keeps total chunks within a limit.

    Iterates through *options* (ascending) and returns the first duration
    where ``video_length_sec / duration <= max_chunks``. Falls back to
    the largest option if none satisfy the constraint.

    Args:
        video_length_sec (float | None): Video length in seconds. If None,
            the first (smallest) option is returned.
        options (list[int] | None): Candidate chunk durations in seconds,
            sorted ascending. Defaults to ``[5, 10, 15, 20, 30]``.
        max_chunks (int): Maximum number of chunks allowed.

    Returns:
        int: Selected chunk duration in seconds.
    """
    if options is None:
        options = [5, 10, 15, 20, 30]
    if video_length_sec is None:
        return options[0]
    for d in options:
        if video_length_sec / d <= max_chunks:
            return d
    return options[-1]
