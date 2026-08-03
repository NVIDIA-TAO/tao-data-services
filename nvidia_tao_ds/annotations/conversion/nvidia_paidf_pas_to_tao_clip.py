# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert NVIDIA PAIDF PAS SDG output into TAO CLIP fine-tuning data.

The V3.1-compatible input is a dataset root whose ``train`` directory, and
optionally ``val`` and ``eval``/``test`` directories, each contain
``augmented_dataset``. It writes shared ``images`` and ``captions``
directories with root-level ``<split>_list.txt`` and ``<split>_pairs.json``
files. In this mode, ``unique_name`` identifies the converted JPEG while
``image_path`` retains the source-relative PAIDF image path.

The experimental legacy input may instead provide one hosted run::

    <raw_output_dir>/<run-id>/augmented_dataset/<scene>/
      raw/<image>.jpg|jpeg|png
      sidecars/person_attribute_search/bundle_attributes.json
      sidecars/person_attribute_search/bundle_queries.json

Training images produce either all nine PAS captions or the three captions
from one requested query level. Validation and test retain the first query per
selected level. Output image-list and pairs rows stay split-aligned for the
TAO CLIP custom dataloader.
"""

from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import warnings

from PIL import Image

from nvidia_tao_ds.core.decorators import experimental


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REQUIRED_ATTRIBUTES = {
    "top outer type",
    "top outer color",
    "bottom type",
    "bottom color",
    "shoe type",
    "shoe color",
    "viewpoint",
    "accessories",
}
VECTOR_ATTRIBUTES = (
    "top outer color",
    "top outer type",
    "bottom color",
    "bottom type",
    "shoe color",
    "shoe type",
    "viewpoint",
)
QUERY_LEVELS = ("easy", "medium", "hard")
CAPTION_POLICIES = ("all", *QUERY_LEVELS)
INPUT_LAYOUTS = ("single_run", "split_dataset")
TEXT_ATTR_WIDTH_BY_QUERY_TYPE = {"easy": 4, "medium": 6, "hard": 7}
UNCONSTRAINED_ATTR_LABELS = {"missing", "not visible"}
# Internal sdg_manifest.json schema version; independent of TAO-FT V3.1.1.
DATASET_FORMAT_VERSION = 4
CONVERTER_OWNED_PATHS = (
    "images",
    "captions",
    "sdg_image_list.txt",
    "sdg_pairs.json",
    "train_list.txt",
    "train_pairs.json",
    "val_list.txt",
    "val_pairs.json",
    "test_list.txt",
    "test_pairs.json",
    "attribute_vocab.json",
    "sdg_manifest.json",
)
STAGING_PREFIX = ".nvidia-paidf-stage-"
INCOMPLETE_MARKER = ".nvidia-paidf-incomplete"
INCOMPLETE_MARKER_TEMP = f"{INCOMPLETE_MARKER}.tmp"
PREVIOUS_OUTPUT_DIR = ".nvidia-paidf-previous"
PROMOTION_MARKER_VERSION = 1
TAO_OUTPUT_BOOKKEEPING = {
    "status.json",
    INCOMPLETE_MARKER,
    INCOMPLETE_MARKER_TEMP,
}
# ENOTSUP and EOPNOTSUPP are aliases on Linux.
UNSUPPORTED_FLOCK_ERRNOS = {
    errno.ENOLCK,
    errno.ENOSYS,
    errno.ENOTSUP,
    errno.EOPNOTSUPP,
}


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_people(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate a PAIDF PAS sidecar ``people`` mapping."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    people = payload.get("people")
    if not isinstance(people, dict):
        raise ValueError(f"{path} must contain a people object")
    return people


def _raw_images(scene_dir: Path) -> list[Path]:
    """Return supported raw images with unique output stems for one scene."""
    raw_dir = scene_dir / "raw"
    if not raw_dir.is_dir():
        raise ValueError(f"Missing raw image directory: {raw_dir}")
    images = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    images_by_stem: dict[str, list[str]] = {}
    for image in images:
        images_by_stem.setdefault(image.stem, []).append(image.name)
    collisions = {
        stem: names
        for stem, names in images_by_stem.items()
        if len(names) > 1
    }
    if collisions:
        raise ValueError(
            f"Scene {scene_dir.name} has source image stems that would "
            f"collide after JPEG conversion: {collisions}"
        )
    return images


def _resolve_record(
    people: dict[str, dict[str, Any]],
    scene_name: str,
    image_name: str,
    record_kind: str,
    scene_image_count: int,
) -> tuple[str, dict[str, Any]]:
    """Resolve an image record using the producer's supported key variants."""
    exact_key = f"{scene_name}/{image_name}"
    if exact_key in people:
        return exact_key, people[exact_key]

    scene_record = people.get(scene_name)
    if scene_record is not None and len(people) == 1:
        if scene_image_count != 1:
            raise ValueError(
                f"Scene-keyed {record_kind} record {scene_name!r} requires "
                f"exactly one raw image, found {scene_image_count}"
            )
        return scene_name, scene_record

    suffix_matches = [
        (key, value)
        for key, value in people.items()
        if key.endswith(f"/{image_name}") or key == image_name
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise ValueError(
        f"Could not resolve {record_kind} record for {exact_key} "
        f"from {sorted(people)}"
    )


def _normalize_attributes(
    record: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Validate attributes and normalize barefoot shoe color."""
    attributes = record.get("attributes")
    if not isinstance(attributes, dict):
        raise ValueError(f"Attribute record {key} has no attributes object")
    attributes = dict(attributes)
    shoe_type = str(attributes.get("shoe type", "")).strip().lower()
    if (
        "shoe color" not in attributes and
        shoe_type == "barefoot"
    ):
        attributes["shoe color"] = "none"
    missing = sorted(REQUIRED_ATTRIBUTES - set(attributes))
    if missing:
        raise ValueError(f"Attribute record {key} missing fields: {missing}")
    return attributes


def _normalize_text(value: Any) -> str:
    """Normalize attribute labels for vocabulary lookup."""
    return " ".join(
        re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split()
    )


def _load_attribute_vocab(
    path: Path,
) -> dict[str, dict[str, int]]:
    """Load the canonical ordered scalar attribute vocabulary."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    attributes = payload.get("attributes")
    value_to_id = payload.get("value_to_id")
    if attributes != list(VECTOR_ATTRIBUTES) or not isinstance(value_to_id, dict):
        raise ValueError(
            f"{path} must define the canonical PAS attributes {VECTOR_ATTRIBUTES}"
        )

    vocab: dict[str, dict[str, int]] = {}
    for attribute in VECTOR_ATTRIBUTES:
        mapping = value_to_id.get(attribute)
        if not isinstance(mapping, dict):
            raise ValueError(f"{path} is missing vocabulary for {attribute!r}")
        normalized_mapping: dict[str, int] = {}
        for value, value_id in mapping.items():
            if isinstance(value_id, bool) or not isinstance(value_id, int):
                raise ValueError(
                    f"{path} maps {attribute!r} value {value!r} "
                    f"to non-integer ID {value_id!r}"
                )
            normalized_value = _normalize_text(value)
            if normalized_value in normalized_mapping:
                raise ValueError(
                    f"{path} has duplicate normalized values for "
                    f"{attribute!r}: {value!r}"
                )
            normalized_mapping[normalized_value] = value_id
        vocab[attribute] = normalized_mapping
    return vocab


def _attribute_vector(
    attributes: dict[str, Any],
    vocab: dict[str, dict[str, int]],
) -> list[int]:
    """Encode the seven canonical image attributes."""
    values: list[int] = []
    for attribute in VECTOR_ATTRIBUTES:
        value = _normalize_text(attributes[attribute])
        try:
            values.append(vocab[attribute][value])
        except KeyError as exc:
            raise ValueError(
                f"Value {attributes[attribute]!r} is not in the "
                f"{attribute!r} vocabulary"
            ) from exc
    return values


def _unconstrained_attribute_ids(
    vocab: dict[str, dict[str, int]],
) -> list[set[int]]:
    """Return per-attribute IDs that must not constrain text matching."""
    return [
        {
            value_id
            for label, value_id in vocab[attribute].items()
            if label in UNCONSTRAINED_ATTR_LABELS
        }
        for attribute in VECTOR_ATTRIBUTES
    ]


def _compose_text_attribute_vector(
    image_attr_values: list[int],
    unconstrained_attribute_ids: list[set[int]],
    query_type: str,
) -> list[int]:
    """Compose query-level text constraints from image attribute IDs."""
    if query_type not in TEXT_ATTR_WIDTH_BY_QUERY_TYPE:
        raise ValueError(f"Unsupported query type: {query_type}")
    if len(image_attr_values) != len(VECTOR_ATTRIBUTES):
        raise ValueError(
            f"Expected {len(VECTOR_ATTRIBUTES)} image attributes, "
            f"got {len(image_attr_values)}"
        )
    if len(unconstrained_attribute_ids) != len(VECTOR_ATTRIBUTES):
        raise ValueError(
            f"Expected {len(VECTOR_ATTRIBUTES)} unconstrained attribute sets, "
            f"got {len(unconstrained_attribute_ids)}"
        )

    width = TEXT_ATTR_WIDTH_BY_QUERY_TYPE[query_type]
    values: list[int] = []
    for index, image_value_id in enumerate(image_attr_values):
        value_id = int(image_value_id)
        if index >= width or value_id in unconstrained_attribute_ids[index]:
            values.append(-1)
        else:
            values.append(value_id)
    return values


def _select_captions(
    record: dict[str, Any],
    key: str,
    policy: str,
) -> list[tuple[str, int, str]]:
    """Validate all query arrays and select captions for one policy."""
    queries = record.get("queries")
    if not isinstance(queries, dict):
        raise ValueError(f"Query record {key} has no queries object")
    for level in QUERY_LEVELS:
        values = queries.get(level)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(
                f"Query record {key} must contain exactly 3 {level} queries"
            )
        if not all(str(value).strip() for value in values):
            raise ValueError(f"Query record {key} contains an empty {level} query")
    if policy not in CAPTION_POLICIES:
        raise ValueError(f"caption_policy must be one of {CAPTION_POLICIES}")

    selected_levels = QUERY_LEVELS if policy == "all" else (policy,)
    return [
        (level, query_index, str(caption).strip())
        for level in selected_levels
        for query_index, caption in enumerate(queries[level])
    ]


def _discover_sources(
    raw_root: Path,
    input_layout: str,
) -> list[tuple[str, str, Path]]:
    """Resolve explicit single-run or split-dataset input directories."""
    if input_layout == "split_dataset":
        train_dir = raw_root / "train"
        if not (train_dir / "augmented_dataset").is_dir():
            raise ValueError(
                "Expected V3.1 split_dataset input with required "
                f"train/augmented_dataset under {raw_root}"
            )
        if all(
            (raw_root / split / "augmented_dataset").is_dir()
            for split in ("eval", "test")
        ):
            raise ValueError(
                "PAIDF input cannot contain both eval and test split directories"
            )
        sources = [("train", "train", train_dir)]
        for source_split, output_split in (
            ("val", "val"),
            ("eval", "test"),
            ("test", "test"),
        ):
            source_dir = raw_root / source_split
            if not source_dir.exists():
                continue
            if not (source_dir / "augmented_dataset").is_dir():
                raise ValueError(
                    f"Expected {source_split}/augmented_dataset under {raw_root}"
                )
            sources.append((output_split, source_split, source_dir))
        return sources

    if (raw_root / "augmented_dataset").is_dir():
        run_dir = raw_root
    else:
        run_dirs = sorted(
            path
            for path in raw_root.iterdir()
            if path.is_dir() and (path / "augmented_dataset").is_dir()
        )
        if len(run_dirs) != 1:
            raise ValueError(
                "Expected exactly one PAIDF PAS run containing "
                f"augmented_dataset in {raw_root}, found {len(run_dirs)}: "
                f"{[path.name for path in run_dirs]}"
            )
        run_dir = run_dirs[0]
    return [("train", "train", run_dir)]


def _existing_result(
    output_root: Path,
    policy: str,
    input_layout: str,
    raw_root: Path,
    source_inputs: dict[str, dict[str, str]],
    input_vocab_sha256: str,
) -> dict[str, Any] | None:
    """Return a compatible complete result or miss on a relocated source."""
    manifest_path = output_root / "sdg_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dataset_format_version") != DATASET_FORMAT_VERSION:
        raise ValueError(
            f"PAS SDG normalized output uses a legacy dataset format: {output_root}"
        )
    if manifest.get("caption_policy") != policy:
        raise ValueError(
            f"PAS SDG normalized output at {output_root} was built with "
            f"caption_policy={manifest.get('caption_policy')!r}, not {policy!r}"
        )
    if manifest.get("input_layout") != input_layout:
        raise ValueError(
            f"PAS SDG normalized output at {output_root} was built with "
            f"input_layout={manifest.get('input_layout')!r}, not "
            f"{input_layout!r}; use overwrite=True to rebuild it"
        )
    manifest_raw_root = manifest.get("raw_output_dir")
    if (
        not isinstance(manifest_raw_root, str) or
        Path(manifest_raw_root).resolve() != raw_root.resolve() or
        manifest.get("source_inputs") != source_inputs
    ):
        return None

    splits = manifest.get("splits")
    attribute_vocab_file = manifest.get("attribute_vocab_file")
    if (
        not isinstance(splits, dict) or
        not splits or
        not isinstance(attribute_vocab_file, str)
    ):
        raise ValueError(f"Incomplete PAS SDG normalized output: {output_root}")
    attribute_vocab_path = _manifest_output_path(
        output_root,
        attribute_vocab_file,
    )
    if not attribute_vocab_path.is_file():
        raise ValueError(f"Incomplete PAS SDG normalized output: {output_root}")

    expected_artifacts = {"attribute_vocab.json"}
    artifact_paths = {
        "attribute_vocab.json": attribute_vocab_path,
    }
    for split_name, split in splits.items():
        if not isinstance(split, dict):
            raise ValueError(
                f"PAS SDG normalized output has invalid {split_name} metadata: "
                f"{output_root}"
            )
        expected_pairs = split.get("num_pairs")
        image_list_file = split.get("image_list_file")
        pairs_file = split.get("pairs_file")
        if (
            isinstance(expected_pairs, bool) or
            not isinstance(expected_pairs, int) or
            expected_pairs <= 0 or
            not isinstance(image_list_file, str) or
            not isinstance(pairs_file, str)
        ):
            raise ValueError(
                "PAS SDG normalized output has incomplete pair metadata: "
                f"{output_root}"
            )
        image_list_path = _manifest_output_path(output_root, image_list_file)
        pairs_path = _manifest_output_path(output_root, pairs_file)
        if not image_list_path.is_file() or not pairs_path.is_file():
            raise ValueError(f"Incomplete PAS SDG normalized output: {output_root}")
        image_count = _validate_reusable_media(
            output_root,
            image_list_path,
        )
        if image_count != expected_pairs or pairs_path.stat().st_size <= 3:
            raise ValueError(
                "PAS SDG normalized output has incomplete pair metadata: "
                f"{output_root}"
            )
        for path in (image_list_path, pairs_path):
            expected_artifacts.add(path.name)
            artifact_paths[path.name] = path

    artifact_sha256 = manifest.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, dict) or
        set(artifact_sha256) != expected_artifacts
    ):
        raise ValueError(
            f"PAS SDG normalized output lacks reuse integrity metadata: "
            f"{output_root}; use overwrite=True to rebuild it"
        )
    actual_artifact_sha256 = {
        name: _sha256_file(path) for name, path in artifact_paths.items()
    }
    if (
        actual_artifact_sha256 != artifact_sha256 or
        artifact_sha256["attribute_vocab.json"] != input_vocab_sha256
    ):
        raise ValueError(
            f"PAS SDG normalized output failed reuse integrity validation: "
            f"{output_root}; use overwrite=True to rebuild it"
        )
    return manifest


def _manifest_output_path(output_root: Path, relative_path: str) -> Path:
    """Resolve one portable manifest artifact path under the output root."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "PAS SDG normalized output has a non-portable artifact path: "
            f"{relative_path!r}"
        )
    return output_root / path


def _paths_overlap(first_path: Path, second_path: Path) -> bool:
    """Return whether two resolved paths have a parent/child relationship."""
    try:
        first_path.relative_to(second_path)
        return True
    except ValueError:
        pass
    try:
        second_path.relative_to(first_path)
        return True
    except ValueError:
        return False


def _conversion_lock_path(output_root: Path) -> Path:
    """Return the persistent sibling lock for one output directory."""
    resolved_output_root = output_root.resolve()
    return (
        resolved_output_root.parent /
        f".{resolved_output_root.name}.nvidia-paidf.lock"
    )


@contextmanager
def _conversion_lock(output_root: Path):
    """Acquire a non-blocking exclusive lock for one conversion destination."""
    lock_path = _conversion_lock_path(output_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        lock_acquired = False
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise FileExistsError(
                    f"Another PAIDF PAS conversion is using {output_root}"
                ) from exc
            if exc.errno not in UNSUPPORTED_FLOCK_ERRNOS:
                raise
            warnings.warn(
                f"Filesystem does not support advisory locking for "
                f"{output_root}; continuing without protection against "
                "concurrent conversions",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            yield
        finally:
            if lock_acquired:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _single_writer(conversion_function):
    """Prevent overlapping conversion calls from sharing one output directory."""
    @wraps(conversion_function)
    def wrapped(
        raw_output_dir: str,
        output_dir: str,
        attribute_vocab_path: str,
        input_layout: str = "split_dataset",
        caption_policy: str = "all",
        overwrite: bool = False,
    ) -> str:
        if not output_dir:
            raise ValueError("output_dir must be provided")
        raw_root = Path(raw_output_dir)
        output_root = Path(output_dir)
        if _paths_overlap(raw_root.resolve(), output_root.resolve()):
            raise ValueError("output_dir must not overlap raw_output_dir")
        with _conversion_lock(output_root):
            return conversion_function(
                raw_output_dir,
                output_dir,
                attribute_vocab_path,
                input_layout,
                caption_policy,
                overwrite,
            )

    return wrapped


def _validate_reusable_media(
    output_root: Path,
    image_list_path: Path,
) -> int:
    """Check every reusable list entry still has its converted image and caption."""
    image_count = 0
    with image_list_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_name = line.strip()
            if not image_name:
                continue
            relative_image_path = Path(image_name)
            if (
                relative_image_path.is_absolute() or
                ".." in relative_image_path.parts
            ):
                raise ValueError(
                    "PAS SDG normalized output has an invalid image-list entry: "
                    f"{image_name!r}"
                )
            image_path = output_root / "images" / relative_image_path
            caption_path = (
                output_root /
                "captions" /
                relative_image_path.with_suffix(".txt")
            )
            if not image_path.is_file() or not caption_path.is_file():
                raise ValueError(
                    "PAS SDG normalized output is missing converted media for "
                    f"{image_name!r}: {output_root}"
                )
            image_count += 1
    return image_count


def _converter_output_entries(output_root: Path) -> list[Path]:
    """Return non-TAO entries currently present in the output directory."""
    if not output_root.exists():
        return []
    return sorted(
        path
        for path in output_root.iterdir()
        if path.name not in TAO_OUTPUT_BOOKKEEPING
    )


def _remove_path(path: Path) -> None:
    """Remove one file, symlink, or directory when it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _path_exists(path: Path) -> bool:
    """Return whether a path exists, including an unresolved symlink."""
    return path.is_symlink() or path.exists()


def _clear_converter_output(output_root: Path) -> None:
    """Remove converter-owned artifacts without deleting TAO bookkeeping."""
    for relative_path in CONVERTER_OWNED_PATHS:
        _remove_path(output_root / relative_path)


def _previous_converter_paths(output_root: Path) -> tuple[str, ...]:
    """Return the complete output paths eligible for overwrite rollback."""
    manifest_path = output_root / "sdg_manifest.json"
    if not manifest_path.is_file():
        return ()
    ordered_paths = (
        "sdg_manifest.json",
        *(
            path
            for path in CONVERTER_OWNED_PATHS
            if path != "sdg_manifest.json"
        ),
    )
    return tuple(
        relative_path
        for relative_path in ordered_paths
        if _path_exists(output_root / relative_path)
    )


def _write_promotion_marker(
    output_root: Path,
    staging_root: Path,
    previous_paths: tuple[str, ...],
) -> None:
    """Record an atomic rollback ledger before mutating live output paths."""
    marker_path = output_root / INCOMPLETE_MARKER
    temporary_marker_path = output_root / INCOMPLETE_MARKER_TEMP
    marker = {
        "version": PROMOTION_MARKER_VERSION,
        "staging_dir": staging_root.name,
        "previous_paths": list(previous_paths),
    }
    temporary_marker_path.write_text(
        json.dumps(marker, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary_marker_path.replace(marker_path)


def _incomplete_promotion(
    output_root: Path,
) -> tuple[Path, set[str]] | None:
    """Return a validated rollback ledger, if the marker uses this schema."""
    marker_path = output_root / INCOMPLETE_MARKER
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(marker, dict):
        return None
    stage_name = marker.get("staging_dir")
    previous_paths = marker.get("previous_paths")
    if (
        marker.get("version") != PROMOTION_MARKER_VERSION or
        not isinstance(stage_name, str) or
        not isinstance(previous_paths, list) or
        not stage_name.startswith(STAGING_PREFIX) or
        Path(stage_name).name != stage_name
    ):
        return None
    if any(
        not isinstance(path, str) or path not in CONVERTER_OWNED_PATHS
        for path in previous_paths
    ):
        return None
    return output_root / stage_name, set(previous_paths)


def _recover_incomplete_output(output_root: Path) -> None:
    """Restore an interrupted overwrite or clean incomplete first-run output."""
    if not output_root.is_dir():
        return

    marker_path = output_root / INCOMPLETE_MARKER
    promotion = _incomplete_promotion(output_root)
    if promotion is not None:
        staging_root, previous_paths = promotion
        previous_output = staging_root / PREVIOUS_OUTPUT_DIR
        for relative_path in CONVERTER_OWNED_PATHS:
            output_path = output_root / relative_path
            if relative_path not in previous_paths:
                _remove_path(output_path)
                continue
            previous_path = previous_output / relative_path
            if _path_exists(previous_path):
                _remove_path(output_path)
                previous_path.replace(output_path)
    elif marker_path.is_file() and not (output_root / "sdg_manifest.json").is_file():
        _clear_converter_output(output_root)
    if marker_path.is_file():
        marker_path.unlink()
    _remove_path(output_root / INCOMPLETE_MARKER_TEMP)

    for path in output_root.iterdir():
        if path.name.startswith(STAGING_PREFIX):
            _remove_path(path)


def _promote_staged_output(
    staging_root: Path,
    output_root: Path,
    generated_paths: tuple[str, ...],
) -> None:
    """Promote staged output and retain the prior dataset for crash recovery."""
    previous_paths = _previous_converter_paths(output_root)
    _write_promotion_marker(output_root, staging_root, previous_paths)
    if previous_paths:
        previous_output = staging_root / PREVIOUS_OUTPUT_DIR
        previous_output.mkdir()
        for relative_path in previous_paths:
            (output_root / relative_path).replace(previous_output / relative_path)
    else:
        _clear_converter_output(output_root)
    for relative_path in generated_paths:
        if relative_path == "sdg_manifest.json":
            continue
        (staging_root / relative_path).replace(output_root / relative_path)
    (staging_root / "sdg_manifest.json").replace(output_root / "sdg_manifest.json")
    (output_root / INCOMPLETE_MARKER).unlink()


def _materialize_pair_image(
    rgb_image: Image.Image,
    output_image: Path,
    encoded_image_path: Path | None,
) -> Path:
    """Encode once, then hard-link or copy the image for later caption pairs."""
    if encoded_image_path is None:
        rgb_image.save(output_image, format="JPEG", quality=95)
        return output_image
    try:
        os.link(encoded_image_path, output_image)
    except OSError:
        shutil.copyfile(encoded_image_path, output_image)
    return encoded_image_path


def _validate_written_pair(
    output_root: Path,
    row_index: int,
    image_name: str,
    pair: dict[str, Any],
) -> None:
    """Validate one TAO list, caption, image, and pair row."""
    if pair.get("unique_name") != image_name:
        raise ValueError(
            f"PAS SDG pair {row_index} is not aligned with its image-list row"
        )
    image_path = output_root / "images" / image_name
    caption_path = (
        output_root / "captions" / Path(image_name).with_suffix(".txt")
    )
    if not image_path.is_file() or not caption_path.is_file():
        raise ValueError(
            f"PAS SDG pair {row_index} is missing its image or caption file"
        )
    if caption_path.read_text(encoding="utf-8").strip() != pair["caption"]:
        raise ValueError(
            f"PAS SDG pair {row_index} caption file does not match metadata"
        )
    for field in ("image_attr_values", "text_attr_values"):
        values = pair.get(field)
        if not isinstance(values, list) or len(values) != len(VECTOR_ATTRIBUTES):
            raise ValueError(
                f"PAS SDG pair {row_index} has invalid {field}"
            )


@_single_writer
def prepare_pas_sdg_tao_data(
    raw_output_dir: str,
    output_dir: str,
    attribute_vocab_path: str,
    input_layout: str = "split_dataset",
    caption_policy: str = "all",
    overwrite: bool = False,
) -> str:
    """Convert hosted PAIDF PAS SDG output into a TAO CLIP dataset.

    ``input_layout="split_dataset"`` is the default V3.1-compatible mode. It
    requires ``train/augmented_dataset`` and accepts optional
    ``val/augmented_dataset`` plus one of ``eval/augmented_dataset`` or
    ``test/augmented_dataset``. It writes shared ``images`` and ``captions``
    plus root-level ``train_*``, ``val_*``, and ``test_*`` metadata. PAIDF
    ``eval`` maps to TAO ``test``. In this mode, ``image_path`` resolves as
    ``raw_output_dir / source_split / image_path``. ``single_run`` remains a
    legacy experimental mode with ``sdg_*`` metadata. Train retains every
    selected caption; validation and test retain query index zero from each
    selected difficulty. Pair metadata uses the base source identity for
    ``person_id`` and the scene/augmentation directory name for ``person_key``.

    Output is produced in a staging directory and exposed only after every
    selected image-caption pair has been validated. An interrupted overwrite
    restores the prior complete converter output on the next invocation. With
    ``overwrite=False``, a compatible completed manifest is reused. A changed
    source location is treated as a cache miss and rebuilt atomically; other
    non-empty converter output is rejected. With ``overwrite=True``, only
    converter-owned paths are replaced, leaving launcher bookkeeping such as
    ``status.json`` intact.

    Args:
        raw_output_dir: Root matching the structure selected by
            ``input_layout``.
        output_dir: Directory that receives the TAO CLIP filesystem dataset.
        attribute_vocab_path: Canonical TAO-FT V3.1.1 scalar attribute
            vocabulary. The complete file is copied into the output.
        input_layout: ``split_dataset`` for V3.1-compatible named ``train``
            inputs with optional ``val`` and ``eval``/``test`` inputs, or
            ``single_run`` for one legacy ``augmented_dataset`` run.
        caption_policy: ``all`` for nine pairs per training image, or
            ``easy``, ``medium``, or ``hard`` for three. Validation and test
            use one caption per selected level.
        overwrite: Rebuild converter-owned output artifacts when ``True``.

    Returns:
        String path to the completed ``sdg_manifest.json``.

    Raises:
        FileNotFoundError: If the raw root or attribute vocabulary is absent.
        FileExistsError: If non-reusable converter output exists and overwrite
            is disabled.
        ValueError: If the producer layout, sidecars, query arrays, vocabulary,
            attributes, or requested policy violates the conversion contract.
    """
    raw_root = Path(raw_output_dir)
    output_root = Path(output_dir)
    vocab_path = Path(attribute_vocab_path)
    _recover_incomplete_output(output_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"PAS SDG output directory not found: {raw_root}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"TAO-FT attribute vocabulary not found: {vocab_path}")
    if input_layout not in INPUT_LAYOUTS:
        raise ValueError(f"input_layout must be one of {INPUT_LAYOUTS}")
    if caption_policy not in CAPTION_POLICIES:
        raise ValueError(f"caption_policy must be one of {CAPTION_POLICIES}")
    vocab_text = vocab_path.read_text(encoding="utf-8")
    input_vocab_sha256 = hashlib.sha256(vocab_text.encode("utf-8")).hexdigest()
    vocab = _load_attribute_vocab(vocab_path)
    unconstrained_attribute_ids = _unconstrained_attribute_ids(vocab)
    manifest_path = output_root / "sdg_manifest.json"

    split_layout = input_layout == "split_dataset"
    split_sources = _discover_sources(raw_root, input_layout)

    prepared_sources: list[dict[str, Any]] = []
    for output_split, source_split, source_dir in split_sources:
        scenes = sorted(
            path
            for path in (source_dir / "augmented_dataset").iterdir()
            if path.is_dir()
        )
        if not scenes:
            raise ValueError(
                "No scene directories found under augmented_dataset in " f"{source_dir}"
            )
        scene_images = [(scene_dir, _raw_images(scene_dir)) for scene_dir in scenes]
        if not any(images for _scene_dir, images in scene_images):
            raise ValueError(f"No supported source images found in {source_dir}")
        prepared_sources.append(
            {
                "output_split": output_split,
                "source_split": source_split,
                "source_dir": source_dir,
                "scene_images": scene_images,
            }
        )

    source_inputs = {
        source["output_split"]: {
            "source_split": source["source_split"],
            "source_dir": str(source["source_dir"].resolve()),
        }
        for source in prepared_sources
    }

    if not overwrite:
        existing = _existing_result(
            output_root,
            caption_policy,
            input_layout,
            raw_root,
            source_inputs,
            input_vocab_sha256,
        )
        if existing is not None:
            print(f"Using existing PAS SDG dataset: {manifest_path}")
            return str(manifest_path)
        if manifest_path.is_file():
            print(
                "PAIDF source location changed; rebuilding PAS SDG dataset: "
                f"{manifest_path}"
            )
        elif _converter_output_entries(output_root):
            raise FileExistsError(
                "Output directory contains data but has no complete manifest: "
                f"{output_root}"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=STAGING_PREFIX,
            dir=output_root,
        )
    )
    try:
        image_root = staging_root / "images"
        caption_root = staging_root / "captions"
        image_root.mkdir(parents=True)
        caption_root.mkdir(parents=True)

        seen_image_names: set[str] = set()
        total_source_images = 0
        total_pairs = 0
        artifact_names: list[str] = []
        split_manifests: dict[str, dict[str, Any]] = {}
        query_levels = (
            list(QUERY_LEVELS) if caption_policy == "all" else [caption_policy]
        )

        for source in prepared_sources:
            output_split = source["output_split"]
            source_split = source["source_split"]
            source_dir = source["source_dir"]
            scene_images = source["scene_images"]
            file_prefix = output_split if split_layout else "sdg"
            image_list_name = (
                f"{file_prefix}_list.txt" if split_layout else "sdg_image_list.txt"
            )
            pairs_name = f"{file_prefix}_pairs.json"
            image_list_path = staging_root / image_list_name
            pairs_path = staging_root / pairs_name
            artifact_names.extend((image_list_name, pairs_name))

            source_image_count = 0
            pair_count = 0
            first_pair = True
            with (
                image_list_path.open(
                    "w",
                    encoding="utf-8",
                ) as image_list_handle,
                pairs_path.open("w", encoding="utf-8") as pairs_handle,
            ):
                pairs_handle.write("[\n")
                for scene_dir, raw_images in scene_images:
                    attributes_path = (
                        scene_dir /
                        "sidecars" /
                        "person_attribute_search" /
                        "bundle_attributes.json"
                    )
                    queries_path = (
                        scene_dir /
                        "sidecars" /
                        "person_attribute_search" /
                        "bundle_queries.json"
                    )
                    attributes_people = _load_people(attributes_path)
                    queries_people = _load_people(queries_path)

                    for source_image in raw_images:
                        source_key = f"{scene_dir.name}/{source_image.name}"
                        source_image_path = (
                            source_image.relative_to(source_dir).as_posix()
                            if split_layout
                            else None
                        )
                        attribute_key, attribute_record = _resolve_record(
                            attributes_people,
                            scene_dir.name,
                            source_image.name,
                            "attribute",
                            len(raw_images),
                        )
                        query_key, query_record = _resolve_record(
                            queries_people,
                            scene_dir.name,
                            source_image.name,
                            "query",
                            len(raw_images),
                        )
                        if attribute_key != query_key:
                            raise ValueError(
                                "Attribute and query records for "
                                f"{source_key} resolved different people keys: "
                                f"{attribute_key!r} and {query_key!r}"
                            )
                        attributes = _normalize_attributes(
                            attribute_record,
                            source_key,
                        )
                        image_attr_values = _attribute_vector(
                            attributes,
                            vocab,
                        )
                        selected_captions = _select_captions(
                            query_record,
                            source_key,
                            caption_policy,
                        )
                        if split_layout and output_split in {"val", "test"}:
                            selected_captions = [
                                selected
                                for selected in selected_captions
                                if selected[1] == 0
                            ]
                        source_image_count += 1
                        person_id = re.sub(
                            r"_aug\d+$",
                            "",
                            scene_dir.name,
                        )
                        is_augmented = bool(re.search(r"_aug\d+$", scene_dir.name))

                        with Image.open(source_image) as image:
                            rgb_image = image.convert("RGB")
                            encoded_image_path: Path | None = None
                            for (
                                query_type,
                                query_index,
                                caption,
                            ) in selected_captions:
                                if split_layout:
                                    unique_name = f"{output_split}_{pair_count:08d}.jpg"
                                    relative_name = Path(unique_name)
                                else:
                                    sample_stem = (
                                        f"{source_image.stem}__"
                                        f"{query_type}_{query_index}"
                                    )
                                    relative_name = (
                                        Path(scene_dir.name) / f"{sample_stem}.jpg"
                                    )
                                    unique_name = relative_name.as_posix()
                                if unique_name in seen_image_names:
                                    raise ValueError(
                                        "Duplicate PAS SDG output name: "
                                        f"{unique_name}"
                                    )
                                seen_image_names.add(unique_name)

                                output_image = image_root / relative_name
                                output_caption = (
                                    caption_root / relative_name.with_suffix(".txt")
                                )
                                output_image.parent.mkdir(
                                    parents=True,
                                    exist_ok=True,
                                )
                                output_caption.parent.mkdir(
                                    parents=True,
                                    exist_ok=True,
                                )
                                encoded_image_path = _materialize_pair_image(
                                    rgb_image,
                                    output_image,
                                    encoded_image_path,
                                )
                                output_caption.write_text(
                                    caption + "\n",
                                    encoding="utf-8",
                                )

                                pair = {
                                    "idx": pair_count,
                                    "unique_name": unique_name,
                                    "caption": caption,
                                    "image_path": (
                                        source_image_path
                                        if split_layout
                                        else f"images/{unique_name}"
                                    ),
                                    "dataset": "PAS_SDG",
                                    "query_type": query_type,
                                    "person_id": person_id,
                                    "person_key": scene_dir.name,
                                    "source_split": source_split,
                                    "source_collection": "PAS_SDG",
                                    "is_augmented": is_augmented,
                                    "image_attr_values": image_attr_values,
                                    "text_attr_values": (
                                        _compose_text_attribute_vector(
                                            image_attr_values,
                                            unconstrained_attribute_ids,
                                            query_type,
                                        )
                                    ),
                                }
                                _validate_written_pair(
                                    staging_root,
                                    pair_count,
                                    unique_name,
                                    pair,
                                )
                                image_list_handle.write(unique_name + "\n")
                                if not first_pair:
                                    pairs_handle.write(",\n")
                                json.dump(
                                    pair,
                                    pairs_handle,
                                    separators=(",", ":"),
                                )
                                first_pair = False
                                pair_count += 1
                pairs_handle.write("\n]\n")

            total_source_images += source_image_count
            total_pairs += pair_count
            queries_per_level = (
                1 if split_layout and output_split in {"val", "test"} else 3
            )
            tao_dataset = {
                "image_dir": "images",
                "caption_dir": "captions",
                "image_list_file": image_list_name,
                "caption_file_suffix": ".txt",
            }
            if output_split == "train":
                tao_dataset["train_pairs_file"] = pairs_name
            else:
                tao_dataset["attribute_pairs_file"] = pairs_name
            split_manifests[output_split] = {
                "source_split": source_split,
                "source_dir": str(source_dir.resolve()),
                "image_list_file": image_list_name,
                "pairs_file": pairs_name,
                "query_levels": query_levels,
                "queries_per_level": queries_per_level,
                "pairs_per_source_image": (len(query_levels) * queries_per_level),
                "num_source_images": source_image_count,
                "num_images": pair_count,
                "num_pairs": pair_count,
                "tao_dataset": tao_dataset,
            }

        output_vocab_path = staging_root / "attribute_vocab.json"
        output_vocab_path.write_text(vocab_text, encoding="utf-8")
        artifact_names.append("attribute_vocab.json")
        artifact_sha256 = {
            name: _sha256_file(staging_root / name) for name in artifact_names
        }

        manifest = {
            "dataset_format_version": DATASET_FORMAT_VERSION,
            "raw_output_dir": str(raw_root.resolve()),
            "normalized_dir": ".",
            "image_dir": "images",
            "caption_dir": "captions",
            "attribute_vocab_file": "attribute_vocab.json",
            "input_layout": input_layout,
            "caption_policy": caption_policy,
            "source_inputs": source_inputs,
            "splits": split_manifests,
            "num_source_images": total_source_images,
            "num_images": total_pairs,
            "num_pairs": total_pairs,
            "artifact_sha256": artifact_sha256,
            "tao_dataset": {
                split: split_manifest["tao_dataset"]
                for split, split_manifest in split_manifests.items()
            },
        }
        if not split_layout:
            legacy_split = split_manifests["train"]
            source_dir = prepared_sources[0]["source_dir"]
            manifest.update(
                {
                    "source_run_id": source_dir.name,
                    "image_list_file": legacy_split["image_list_file"],
                    "pairs_file": legacy_split["pairs_file"],
                    "query_levels": legacy_split["query_levels"],
                    "queries_per_level": legacy_split["queries_per_level"],
                    "pairs_per_source_image": (legacy_split["pairs_per_source_image"]),
                    "tao_dataset": legacy_split["tao_dataset"],
                }
            )
        (staging_root / "sdg_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        generated_paths = (
            "images",
            "captions",
            *artifact_names,
            "sdg_manifest.json",
        )
        _promote_staged_output(
            staging_root,
            output_root,
            generated_paths,
        )
    finally:
        if (
            staging_root.is_dir() and
            not (output_root / INCOMPLETE_MARKER).is_file()
        ):
            shutil.rmtree(staging_root)
            _remove_path(output_root / INCOMPLETE_MARKER_TEMP)

    print(
        f"Wrote PAS SDG dataset: {output_root} "
        f"({total_source_images} source images, {total_pairs} text-image pairs)"
    )
    return str(manifest_path)


@experimental("NVIDIA PAIDF PAS to TAO CLIP conversion is experimental")
def convert_nvidia_paidf_pas_to_tao_clip(cfg=None, verbose: bool = False) -> str:
    """Run PAIDF PAS conversion from the TAO annotations configuration."""
    if cfg is None:
        raise ValueError("config is not provided")
    if verbose:
        print("Running experimental NVIDIA_PAIDF_PAS to TAO_CLIP conversion")
    pas_cfg = cfg.nvidia_paidf_pas
    return prepare_pas_sdg_tao_data(
        raw_output_dir=pas_cfg.raw_output_dir,
        output_dir=cfg.results_dir,
        attribute_vocab_path=pas_cfg.attribute_vocab_path,
        input_layout=pas_cfg.input_layout,
        caption_policy=pas_cfg.caption_policy,
        overwrite=pas_cfg.overwrite,
    )
