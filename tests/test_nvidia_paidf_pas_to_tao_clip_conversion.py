"""Tests for experimental NVIDIA PAIDF PAS to TAO CLIP conversion."""

import errno
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from nvidia_tao_ds.annotations.conversion import (
    nvidia_paidf_pas_to_tao_clip as converter,
)
from nvidia_tao_ds.config.annotations.default_config import (
    NVIDIAPAIDFPASConfig,
)


SCENE_NAME = "00000_ICFG_PEDES_aug0"
IMAGE_NAME = "crop.png"
SECOND_IMAGE_NAME = "second.png"
PERSON_KEY = SCENE_NAME
EXACT_PERSON_KEY = f"{SCENE_NAME}/{IMAGE_NAME}"


def _minimal_vocab():
    """Return a complete minimal seven-attribute vocabulary."""
    values = {
        "top outer color": "black",
        "top outer type": "hoodie",
        "bottom color": "blue",
        "bottom type": "jeans",
        "shoe color": "none",
        "shoe type": "barefoot",
        "viewpoint": "side view",
    }
    return {
        "metadata_version": 1,
        "attributes": list(converter.VECTOR_ATTRIBUTES),
        "value_to_id": {
            attribute: {"__missing__": 0, value: 1}
            for attribute, value in values.items()
        },
        "id_to_value": {
            attribute: ["__missing__", value]
            for attribute, value in values.items()
        },
    }


def _queries():
    """Return three distinct captions for every PAS query level."""
    return {
        level: [
            f"{level} caption zero",
            f"{level} caption one",
            f"{level} caption two",
        ]
        for level in converter.QUERY_LEVELS
    }


def _write_fixture(tmp_path: Path):
    """Write one authoritative-layout source image and both sidecars."""
    raw_root = tmp_path / "raw"
    scene_dir = (
        raw_root /
        "run-001" /
        "augmented_dataset" /
        SCENE_NAME
    )
    raw_dir = scene_dir / "raw"
    sidecar_dir = (
        scene_dir / "sidecars" / "person_attribute_search"
    )
    raw_dir.mkdir(parents=True)
    sidecar_dir.mkdir(parents=True)
    Image.new("RGBA", (3, 2), (10, 20, 30, 128)).save(
        raw_dir / IMAGE_NAME
    )

    attributes_path = sidecar_dir / "bundle_attributes.json"
    attributes_path.write_text(
        json.dumps(
            {
                "chunk_id": "0",
                "n_people": 1,
                "people": {
                    PERSON_KEY: {
                        "track_id": 0,
                        "attributes": {
                            "top outer type": "hoodie",
                            "top outer color": "black",
                            "bottom type": "jeans",
                            "bottom color": "blue",
                            "shoe type": "barefoot",
                            "viewpoint": "side view",
                            "accessories": [],
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    queries_path = sidecar_dir / "bundle_queries.json"
    queries_path.write_text(
        json.dumps(
            {
                "chunk_id": "0",
                "n_people": 1,
                "people": {
                    PERSON_KEY: {
                        "image_filename": IMAGE_NAME,
                        "queries": _queries(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    vocab_path = tmp_path / "attribute_vocab.json"
    vocab_path.write_text(
        json.dumps(_minimal_vocab(), indent=2) + "\n",
        encoding="utf-8",
    )
    return raw_root, vocab_path, attributes_path, queries_path


def _convert(
    raw_root: Path,
    output_dir: Path,
    vocab_path: Path,
    input_layout: str = "single_run",
    caption_policy: str = "all",
    overwrite: bool = False,
):
    """Run the pure converter using fixture paths."""
    return converter.prepare_pas_sdg_tao_data(
        raw_output_dir=str(raw_root),
        output_dir=str(output_dir),
        attribute_vocab_path=str(vocab_path),
        input_layout=input_layout,
        caption_policy=caption_policy,
        overwrite=overwrite,
    )


def _write_split_fixture(
    tmp_path: Path,
    splits: tuple[str, ...] = ("train", "val", "eval"),
):
    """Write selected split inputs with direct augmented_dataset children."""
    dataset_root = tmp_path / "dataset"
    vocab_path = None
    for split in splits:
        raw_root, split_vocab, _attributes, _queries_path = _write_fixture(
            tmp_path / f"{split}-fixture"
        )
        split_dir = dataset_root / split
        split_dir.mkdir(parents=True)
        (raw_root / "run-001" / "augmented_dataset").rename(
            split_dir / "augmented_dataset"
        )
        if vocab_path is None:
            vocab_path = split_vocab
    assert vocab_path is not None
    return dataset_root, vocab_path


def test_experimental_single_run_writes_legacy_sdg_artifacts(tmp_path):
    """Legacy single-run conversion retains its self-contained metadata."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"

    manifest_path = Path(
        _convert(raw_root, output_dir, vocab_path, input_layout="single_run")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_names = [
        f"{SCENE_NAME}/crop__{level}_{query_index}.jpg"
        for level in converter.QUERY_LEVELS
        for query_index in range(3)
    ]
    image_names = (
        (output_dir / "sdg_image_list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    pairs_text = (output_dir / "sdg_pairs.json").read_text(encoding="utf-8")
    pairs = json.loads(pairs_text)

    assert image_names == expected_names
    assert [pair["unique_name"] for pair in pairs] == expected_names
    assert manifest["dataset_format_version"] == converter.DATASET_FORMAT_VERSION
    assert manifest["input_layout"] == "single_run"
    assert manifest["source_run_id"] == "run-001"
    assert manifest["caption_policy"] == "all"
    assert manifest["query_levels"] == ["easy", "medium", "hard"]
    assert manifest["queries_per_level"] == 3
    assert manifest["pairs_per_source_image"] == 9
    assert manifest["num_source_images"] == 1
    assert manifest["num_pairs"] == 9

    expected_text_vectors = {
        "easy": [1, 1, 1, 1, -1, -1, -1],
        "medium": [1, 1, 1, 1, 1, 1, -1],
        "hard": [1, 1, 1, 1, 1, 1, 1],
    }
    for pair_index, pair in enumerate(pairs):
        query_type = converter.QUERY_LEVELS[pair_index // 3]
        query_index = pair_index % 3
        assert pair["caption"] == f"{query_type} caption {('zero', 'one', 'two')[query_index]}"
        assert pair["query_type"] == query_type
        assert pair["dataset"] == "PAS_SDG"
        assert pair["person_id"] == "00000_ICFG_PEDES"
        assert pair["person_key"] == SCENE_NAME
        assert pair["source_split"] == "train"
        assert pair["source_collection"] == "PAS_SDG"
        assert pair["is_augmented"] is True
        assert pair["image_path"] == f"images/{pair['unique_name']}"
        assert pair["image_attr_values"] == [1] * 7
        assert pair["text_attr_values"] == expected_text_vectors[query_type]

        image_path = output_dir / "images" / pair["unique_name"]
        caption_path = (
            output_dir /
            "captions" /
            Path(pair["unique_name"]).with_suffix(".txt")
        )
        assert image_path.is_file()
        assert caption_path.read_text(encoding="utf-8") == (
            pair["caption"] + "\n"
        )

    with Image.open(output_dir / "images" / image_names[0]) as output_image:
        assert output_image.format == "JPEG"
        assert output_image.mode == "RGB"
        assert output_image.size == (3, 2)
    image_inodes = {
        (output_dir / "images" / image_name).stat().st_ino
        for image_name in image_names
    }
    assert len(image_inodes) == 1
    assert json.loads(
        (output_dir / "attribute_vocab.json").read_text(encoding="utf-8")
    ) == _minimal_vocab()


def test_v31_split_dataset_writes_shared_root_tao_ft_files(tmp_path):
    """V3.1 split conversion retains source identity in root metadata files."""
    raw_root, vocab_path = _write_split_fixture(tmp_path)
    output_dir = tmp_path / "output"

    manifest = json.loads(
        Path(
            _convert(
                raw_root,
                output_dir,
                vocab_path,
                input_layout="split_dataset",
            )
        ).read_text(encoding="utf-8")
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "attribute_vocab.json",
        "captions",
        "images",
        "sdg_manifest.json",
        "test_list.txt",
        "test_pairs.json",
        "train_list.txt",
        "train_pairs.json",
        "val_list.txt",
        "val_pairs.json",
    ]

    expected_counts = {"train": 9, "val": 3, "test": 3}
    expected_source_splits = {
        "train": "train",
        "val": "val",
        "test": "eval",
    }
    for split, expected_count in expected_counts.items():
        image_names = (
            (output_dir / f"{split}_list.txt").read_text(encoding="utf-8").splitlines()
        )
        pairs = json.loads(
            (output_dir / f"{split}_pairs.json").read_text(encoding="utf-8")
        )
        assert image_names == [
            f"{split}_{index:08d}.jpg" for index in range(expected_count)
        ]
        assert [pair["unique_name"] for pair in pairs] == image_names
        assert [pair["idx"] for pair in pairs] == list(range(expected_count))
        assert {pair["source_split"] for pair in pairs} == {
            expected_source_splits[split]
        }
        expected_source_path = (
            f"augmented_dataset/{SCENE_NAME}/raw/{IMAGE_NAME}"
        )
        assert {pair["image_path"] for pair in pairs} == {expected_source_path}
        assert all(
            (
                raw_root /
                pair["source_split"] /
                pair["image_path"]
            ).is_file()
            for pair in pairs
        )
        assert all(
            pair["image_path"] != f"images/{pair['unique_name']}"
            for pair in pairs
        )
        assert all(
            (output_dir / "images" / image_name).is_file() for image_name in image_names
        )
        assert all(
            (output_dir / "captions" / Path(image_name).with_suffix(".txt")).is_file()
            for image_name in image_names
        )

    assert [
        pair["caption"]
        for pair in json.loads(
            (output_dir / "val_pairs.json").read_text(encoding="utf-8")
        )
    ] == [
        "easy caption zero",
        "medium caption zero",
        "hard caption zero",
    ]
    assert manifest["input_layout"] == "split_dataset"
    assert set(manifest["splits"]) == {"train", "val", "test"}
    assert manifest["num_source_images"] == 3
    assert manifest["num_pairs"] == 15
    assert manifest["normalized_dir"] == "."
    assert manifest["image_dir"] == "images"
    assert manifest["caption_dir"] == "captions"
    assert manifest["attribute_vocab_file"] == "attribute_vocab.json"
    for split, split_manifest in manifest["splits"].items():
        assert split_manifest["image_list_file"] == f"{split}_list.txt"
        assert split_manifest["pairs_file"] == f"{split}_pairs.json"
        tao_dataset = manifest["tao_dataset"][split]
        assert tao_dataset["image_dir"] == "images"
        assert tao_dataset["caption_dir"] == "captions"
        assert tao_dataset["image_list_file"] == f"{split}_list.txt"
        pairs_key = (
            "train_pairs_file"
            if split == "train"
            else "attribute_pairs_file"
        )
        assert tao_dataset[pairs_key] == f"{split}_pairs.json"


def test_v31_split_dataset_requires_train_before_writing_output(tmp_path):
    """V3.1 output cannot be built from validation-only source data."""
    raw_root, vocab_path = _write_split_fixture(tmp_path, splits=("val",))
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="required train/augmented_dataset"):
        _convert(
            raw_root,
            output_dir,
            vocab_path,
            input_layout="split_dataset",
        )

    assert not output_dir.exists()


def test_v31_split_dataset_is_default_and_allows_train_only(tmp_path):
    """The default V3.1 mode permits training-only dataset preparation."""
    raw_root, vocab_path = _write_split_fixture(tmp_path, splits=("train",))
    output_dir = tmp_path / "output"

    manifest = json.loads(
        Path(
            converter.prepare_pas_sdg_tao_data(
                raw_output_dir=str(raw_root),
                output_dir=str(output_dir),
                attribute_vocab_path=str(vocab_path),
            )
        ).read_text(encoding="utf-8")
    )

    assert set(manifest["splits"]) == {"train"}
    assert (output_dir / "train_list.txt").is_file()
    assert (output_dir / "train_pairs.json").is_file()
    assert not (output_dir / "val_list.txt").exists()
    assert not (output_dir / "test_list.txt").exists()


def test_v31_split_dataset_rejects_malformed_optional_split(tmp_path):
    """An optional split must be absent or contain PAIDF source data."""
    raw_root, vocab_path = _write_split_fixture(tmp_path, splits=("train",))
    (raw_root / "val").mkdir()
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="Expected val/augmented_dataset"):
        _convert(
            raw_root,
            output_dir,
            vocab_path,
            input_layout="split_dataset",
        )

    assert not output_dir.exists()


def test_rejects_output_directory_that_overlaps_source_input(tmp_path):
    """Converted output cannot become part of the PAIDF input tree."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )

    for output_dir in (raw_root / "converted", raw_root.parent):
        with pytest.raises(ValueError, match="must not overlap"):
            _convert(raw_root, output_dir, vocab_path)


def test_rejects_empty_output_directory_before_touching_cwd(
    tmp_path,
    monkeypatch,
):
    """An empty reusable output path cannot overwrite converter names in CWD."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path / "fixture")
    )
    working_dir = tmp_path / "working"
    sentinel_path = working_dir / "images" / "sentinel.txt"
    sentinel_path.parent.mkdir(parents=True)
    sentinel_path.write_text("keep-me\n", encoding="utf-8")
    monkeypatch.chdir(working_dir)

    with pytest.raises(ValueError, match="^output_dir must be provided$"):
        converter.prepare_pas_sdg_tao_data(
            raw_output_dir=str(raw_root),
            output_dir="",
            attribute_vocab_path=str(vocab_path),
            input_layout="single_run",
            overwrite=True,
        )

    assert sentinel_path.read_text(encoding="utf-8") == "keep-me\n"
    assert sorted(path.name for path in working_dir.iterdir()) == ["images"]
    assert not (tmp_path / ".working.nvidia-paidf.lock").exists()
    assert not list(working_dir.glob(f"{converter.STAGING_PREFIX}*"))


def test_rejects_concurrent_conversion_for_the_same_output(tmp_path):
    """A second writer fails before it can touch shared converter output."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"

    with converter._conversion_lock(output_dir):
        with pytest.raises(FileExistsError, match="Another PAIDF PAS conversion"):
            _convert(raw_root, output_dir, vocab_path)

    assert not output_dir.exists()


def test_unsupported_filesystem_lock_warns_and_converts(tmp_path, monkeypatch):
    """Shared mounts without flock support retain conversion functionality."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    lock_operations = []

    def _unsupported_flock(_file_descriptor, operation):
        lock_operations.append(operation)
        raise OSError(errno.ENOTSUP, "operation not supported")

    monkeypatch.setattr(converter.fcntl, "flock", _unsupported_flock)

    with pytest.warns(RuntimeWarning, match="without protection"):
        _convert(raw_root, output_dir, vocab_path)

    assert lock_operations == [converter.fcntl.LOCK_EX | converter.fcntl.LOCK_NB]
    assert (output_dir / "sdg_image_list.txt").is_file()
    assert (output_dir / "sdg_pairs.json").is_file()


def test_pair_images_fall_back_to_copy_when_hard_links_are_unavailable(
    tmp_path,
    monkeypatch,
):
    """Unsupported hard links retain nine-path TAO compatibility via copies."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    link_calls = []

    def _unsupported_link(source, destination):
        link_calls.append((source, destination))
        raise OSError("hard links are unavailable")

    monkeypatch.setattr(converter.os, "link", _unsupported_link)
    _convert(
        raw_root,
        output_dir,
        vocab_path,
        caption_policy="medium",
    )

    image_names = (
        (output_dir / "sdg_image_list.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    image_payloads = {
        (output_dir / "images" / image_name).read_bytes()
        for image_name in image_names
    }
    assert len(link_calls) == 2
    assert len(image_names) == 3
    assert len(image_payloads) == 1


def test_rejects_multiple_runs_before_writing_output(tmp_path):
    """One conversion output is scoped to exactly one hosted PAIDF run."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    (raw_root / "run-002" / "augmented_dataset").mkdir(parents=True)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="Expected exactly one PAIDF PAS run"):
        _convert(raw_root, output_dir, vocab_path)

    assert not output_dir.exists()


def test_rejects_unknown_input_layout(tmp_path):
    """The public converter rejects ambiguous layout configuration."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="input_layout must be one of"):
        _convert(
            raw_root,
            output_dir,
            vocab_path,
            input_layout="auto",
        )

    assert not output_dir.exists()


def test_rejects_source_image_stem_collisions_before_writing_output(tmp_path):
    """Different source extensions must not collapse to one output name."""
    raw_root, vocab_path, attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    scene_dir = attributes_path.parents[2]
    Image.new("RGB", (3, 2), (10, 20, 30)).save(
        scene_dir / "raw" / "crop.jpg"
    )
    output_dir = tmp_path / "output"

    with pytest.raises(
        ValueError,
        match="source image stems that would collide",
    ):
        _convert(raw_root, output_dir, vocab_path)

    assert not output_dir.exists()


def test_exact_sidecar_key_resolves_in_multi_image_scene():
    """An exact per-image key remains valid when a scene has multiple images."""
    record = {"value": "record"}

    assert converter._resolve_record(
        {EXACT_PERSON_KEY: record},
        SCENE_NAME,
        IMAGE_NAME,
        "attribute",
        2,
    ) == (EXACT_PERSON_KEY, record)


def test_rejects_scene_key_for_multiple_raw_images(tmp_path):
    """A scene-only record cannot silently label multiple source images."""
    raw_root, vocab_path, attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    scene_dir = attributes_path.parents[2]
    Image.new("RGB", (3, 2), (30, 20, 10)).save(
        scene_dir / "raw" / SECOND_IMAGE_NAME
    )
    output_dir = tmp_path / "output"

    with pytest.raises(
        ValueError,
        match="Scene-keyed attribute record .* requires exactly one raw image",
    ):
        _convert(raw_root, output_dir, vocab_path)

    assert not (output_dir / "sdg_manifest.json").exists()


def test_rejects_different_attribute_and_query_people_keys(tmp_path):
    """Both sidecars must identify a source image with the same actual key."""
    raw_root, vocab_path, _attributes_path, queries_path = (
        _write_fixture(tmp_path)
    )
    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    payload["people"] = {
        EXACT_PERSON_KEY: payload["people"][PERSON_KEY],
    }
    queries_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="resolved different people keys"):
        _convert(raw_root, output_dir, vocab_path)

    assert not (output_dir / "sdg_manifest.json").exists()


def test_single_level_policy_writes_three_pairs_and_safe_overwrite(tmp_path):
    """A single query level emits three pairs and removes stale owned output."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    status_path = output_dir / "status.json"
    status_path.write_text("keep-me\n", encoding="utf-8")
    _convert(raw_root, output_dir, vocab_path)

    with pytest.raises(ValueError, match="caption_policy='all', not 'medium'"):
        _convert(
            raw_root,
            output_dir,
            vocab_path,
            caption_policy="medium",
        )

    manifest_path = Path(
        _convert(
            raw_root,
            output_dir,
            vocab_path,
            caption_policy="medium",
            overwrite=True,
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = json.loads(
        (output_dir / "sdg_pairs.json").read_text(encoding="utf-8")
    )

    assert manifest["caption_policy"] == "medium"
    assert manifest["query_levels"] == ["medium"]
    assert manifest["pairs_per_source_image"] == 3
    assert manifest["num_pairs"] == 3
    assert [pair["query_type"] for pair in pairs] == ["medium"] * 3
    assert not list((output_dir / "images").rglob("*__easy_*.jpg"))
    assert not list((output_dir / "images").rglob("*__hard_*.jpg"))
    assert status_path.read_text(encoding="utf-8") == "keep-me\n"


def test_complete_manifest_is_reused_without_reopening_images(
    tmp_path,
    monkeypatch,
):
    """A compatible complete manifest is the idempotent completion marker."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    expected = _convert(raw_root, output_dir, vocab_path)

    def _unexpected_image_open(*_args, **_kwargs):
        raise AssertionError("conversion should have reused its manifest")

    monkeypatch.setattr(converter.Image, "open", _unexpected_image_open)
    assert _convert(raw_root, output_dir, vocab_path) == expected


def test_moved_completed_output_is_reused(tmp_path, monkeypatch):
    """Relative manifest artifacts remain reusable after an output remount."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    original_output = tmp_path / "original-output"
    _convert(raw_root, original_output, vocab_path)
    moved_output = tmp_path / "mounted-output"
    original_output.rename(moved_output)

    def _unexpected_image_open(*_args, **_kwargs):
        raise AssertionError("a moved completed output should be reused")

    monkeypatch.setattr(converter.Image, "open", _unexpected_image_open)
    manifest_path = moved_output / "sdg_manifest.json"

    assert _convert(raw_root, moved_output, vocab_path) == str(manifest_path)


@pytest.mark.parametrize("changed_identity", ("root", "run"))
def test_manifest_source_location_change_rebuilds(
    tmp_path,
    changed_identity,
):
    """A relocated source is an atomic cache miss instead of a hard failure."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path / "original")
    )
    output_dir = tmp_path / "output"
    _convert(raw_root, output_dir, vocab_path)

    if changed_identity == "root":
        current_raw_root, _other_vocab, _attributes, _queries = (
            _write_fixture(tmp_path / "other")
        )
        expected_run_id = "run-001"
    else:
        (raw_root / "run-001").rename(raw_root / "run-002")
        current_raw_root = raw_root
        expected_run_id = "run-002"

    manifest = json.loads(
        Path(
            _convert(current_raw_root, output_dir, vocab_path)
        ).read_text(encoding="utf-8")
    )

    assert manifest["raw_output_dir"] == str(current_raw_root.resolve())
    assert manifest["source_run_id"] == expected_run_id


def test_manifest_reuse_rejects_a_changed_vocabulary(tmp_path):
    """Changing the requested vocabulary invalidates completed output reuse."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    _convert(raw_root, output_dir, vocab_path)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab["value_to_id"]["top outer color"]["green"] = 2
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    with pytest.raises(ValueError, match="reuse integrity validation"):
        _convert(raw_root, output_dir, vocab_path)


def test_manifest_reuse_rejects_corrupted_pairs_metadata(tmp_path):
    """A checksum mismatch prevents reuse of corrupted pair metadata."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    _convert(raw_root, output_dir, vocab_path)
    (output_dir / "sdg_pairs.json").write_text("[{}]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reuse integrity validation"):
        _convert(raw_root, output_dir, vocab_path)


@pytest.mark.parametrize(
    ("directory", "suffix"),
    (("images", ".jpg"), ("captions", ".txt")),
)
def test_manifest_reuse_rejects_missing_converted_media(
    tmp_path,
    directory,
    suffix,
):
    """Reuse validates the converted media required by the TAO image list."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    _convert(raw_root, output_dir, vocab_path)
    image_name = (
        (output_dir / "sdg_image_list.txt")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    (output_dir / directory / Path(image_name).with_suffix(suffix)).unlink()

    with pytest.raises(ValueError, match="missing converted media"):
        _convert(raw_root, output_dir, vocab_path)


def test_failed_conversion_cleans_staging_and_retry_succeeds(
    tmp_path,
    monkeypatch,
):
    """A mid-write failure leaves no live partial dataset and can be retried."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    status_path = output_dir / "status.json"
    status_path.write_text("keep-me\n", encoding="utf-8")

    state = {"calls": 0, "fail": True}
    original_materialize = converter._materialize_pair_image

    def _flaky_materialize(*args, **kwargs):
        state["calls"] += 1
        if state["fail"] and state["calls"] == 2:
            raise OSError("injected JPEG write failure")
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(
        converter,
        "_materialize_pair_image",
        _flaky_materialize,
    )
    with pytest.raises(OSError, match="injected JPEG write failure"):
        _convert(raw_root, output_dir, vocab_path)

    assert [path.name for path in output_dir.iterdir()] == ["status.json"]
    assert status_path.read_text(encoding="utf-8") == "keep-me\n"

    state["fail"] = False
    manifest_path = Path(_convert(raw_root, output_dir, vocab_path))
    assert manifest_path.is_file()
    assert not list(output_dir.glob(f"{converter.STAGING_PREFIX}*"))
    assert status_path.read_text(encoding="utf-8") == "keep-me\n"


def test_failed_overwrite_preserves_existing_dataset(tmp_path):
    """Invalid replacement input cannot remove the existing valid dataset."""
    raw_root, vocab_path, _attributes_path, queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    manifest_path = Path(_convert(raw_root, output_dir, vocab_path))
    original_manifest = manifest_path.read_bytes()
    original_pairs = (output_dir / "sdg_pairs.json").read_bytes()
    original_image_list = (output_dir / "sdg_image_list.txt").read_bytes()

    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    payload["people"][PERSON_KEY]["queries"]["hard"] = []
    queries_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 3 hard queries"):
        _convert(
            raw_root,
            output_dir,
            vocab_path,
            overwrite=True,
        )

    assert manifest_path.read_bytes() == original_manifest
    assert (output_dir / "sdg_pairs.json").read_bytes() == original_pairs
    assert (
        output_dir / "sdg_image_list.txt"
    ).read_bytes() == original_image_list
    assert not list(output_dir.glob(f"{converter.STAGING_PREFIX}*"))


def test_recovers_interrupted_output_before_new_input_validation(tmp_path):
    """A broken retry still restores the prior output transaction first."""
    raw_root, vocab_path, _attributes_path, _queries_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"
    manifest_path = Path(_convert(raw_root, output_dir, vocab_path))
    original_manifest = manifest_path.read_bytes()
    original_pairs = (output_dir / "sdg_pairs.json").read_bytes()
    previous_paths = converter._previous_converter_paths(output_dir)
    staging_root = output_dir / f"{converter.STAGING_PREFIX}interrupted"
    previous_output = staging_root / converter.PREVIOUS_OUTPUT_DIR
    previous_output.mkdir(parents=True)
    converter._write_promotion_marker(
        output_dir,
        staging_root,
        previous_paths,
    )
    for relative_path in previous_paths:
        (output_dir / relative_path).replace(previous_output / relative_path)

    vocab_path.unlink()

    with pytest.raises(FileNotFoundError, match="attribute vocabulary not found"):
        _convert(raw_root, output_dir, vocab_path)

    assert manifest_path.read_bytes() == original_manifest
    assert (output_dir / "sdg_pairs.json").read_bytes() == original_pairs
    assert not (output_dir / converter.INCOMPLETE_MARKER).exists()


def test_interrupted_overwrite_restores_previous_dataset(tmp_path, monkeypatch):
    """Rollback remains recoverable when promotion or recovery is interrupted."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    manifest_path = Path(_convert(raw_root, output_dir, vocab_path))
    original_manifest = manifest_path.read_bytes()
    original_pairs = (output_dir / "sdg_pairs.json").read_bytes()
    status_path = output_dir / "status.json"
    status_path.write_text("keep-me\n", encoding="utf-8")

    original_replace = Path.replace

    def _fail_new_pairs_promotion(path, target):
        target_path = Path(target)
        if (
            path.name == "sdg_pairs.json"
            and path.parent.name.startswith(converter.STAGING_PREFIX)
            and target_path == output_dir / "sdg_pairs.json"
        ):
            raise OSError("injected promotion failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", _fail_new_pairs_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        _convert(
            raw_root,
            output_dir,
            vocab_path,
            caption_policy="medium",
            overwrite=True,
        )

    assert not manifest_path.exists()
    assert (output_dir / converter.INCOMPLETE_MARKER).is_file()
    assert list(output_dir.glob(f"{converter.STAGING_PREFIX}*"))

    rollback_failed = {"value": False}

    def _fail_rollback_once(path, target):
        target_path = Path(target)
        if (
            not rollback_failed["value"]
            and path.parent.name == converter.PREVIOUS_OUTPUT_DIR
            and path.name == "sdg_pairs.json"
            and target_path == output_dir / "sdg_pairs.json"
        ):
            rollback_failed["value"] = True
            raise OSError("injected rollback failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", _fail_rollback_once)
    with pytest.raises(OSError, match="injected rollback failure"):
        _convert(raw_root, output_dir, vocab_path)

    assert rollback_failed["value"] is True
    assert (output_dir / converter.INCOMPLETE_MARKER).is_file()

    monkeypatch.setattr(Path, "replace", original_replace)
    assert _convert(raw_root, output_dir, vocab_path) == str(manifest_path)

    assert manifest_path.read_bytes() == original_manifest
    assert (output_dir / "sdg_pairs.json").read_bytes() == original_pairs
    assert status_path.read_text(encoding="utf-8") == "keep-me\n"
    assert not (output_dir / converter.INCOMPLETE_MARKER).exists()
    assert not list(output_dir.glob(f"{converter.STAGING_PREFIX}*"))


def test_recovers_converter_owned_partial_promotion(tmp_path):
    """A marker permits safe cleanup of partial promoted converter output."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    status_path = output_dir / "status.json"
    status_path.write_text("keep-me\n", encoding="utf-8")
    partial_image_dir = output_dir / "images"
    partial_image_dir.mkdir()
    (partial_image_dir / "partial.jpg").write_bytes(b"partial")
    (output_dir / converter.INCOMPLETE_MARKER).write_text(
        "stale-stage\n",
        encoding="utf-8",
    )
    stale_stage = output_dir / f"{converter.STAGING_PREFIX}stale"
    stale_stage.mkdir()
    (stale_stage / "partial").write_text("partial", encoding="utf-8")

    manifest_path = Path(_convert(raw_root, output_dir, vocab_path))

    assert manifest_path.is_file()
    assert not (output_dir / converter.INCOMPLETE_MARKER).exists()
    assert not stale_stage.exists()
    assert not (partial_image_dir / "partial.jpg").exists()
    assert status_path.read_text(encoding="utf-8") == "keep-me\n"


def test_rejects_invalid_query_count(tmp_path):
    """Every query level remains required even for a single-level export."""
    raw_root, vocab_path, _attributes_path, queries_path = (
        _write_fixture(tmp_path)
    )
    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    payload["people"][PERSON_KEY]["queries"]["easy"] = ["only one"]
    queries_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 3 easy queries"):
        _convert(
            raw_root,
            tmp_path / "output",
            vocab_path,
            caption_policy="medium",
        )


def test_rejects_missing_accessories(tmp_path):
    """Accessories are required even though they are not scalar-vector fields."""
    raw_root, vocab_path, attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    payload = json.loads(attributes_path.read_text(encoding="utf-8"))
    del payload["people"][PERSON_KEY]["attributes"]["accessories"]
    attributes_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields: \\['accessories'\\]"):
        _convert(raw_root, tmp_path / "output", vocab_path)


def test_rejects_attribute_value_absent_from_vocab(tmp_path):
    """Unknown producer labels fail instead of silently mapping to missing."""
    raw_root, vocab_path, attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    payload = json.loads(attributes_path.read_text(encoding="utf-8"))
    payload["people"][PERSON_KEY]["attributes"]["top outer color"] = "purple"
    attributes_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="'purple' is not in the 'top outer color' vocabulary",
    ):
        _convert(raw_root, tmp_path / "output", vocab_path)


def test_missing_attribute_is_unconstrained_in_text_metadata(tmp_path):
    """A requested missing value remains an image ID but is a text wildcard."""
    raw_root, vocab_path, attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    payload = json.loads(attributes_path.read_text(encoding="utf-8"))
    payload["people"][PERSON_KEY]["attributes"]["top outer color"] = (
        "__missing__"
    )
    attributes_path.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "output"

    _convert(
        raw_root,
        output_dir,
        vocab_path,
        caption_policy="easy",
    )

    pairs = json.loads(
        (output_dir / "sdg_pairs.json").read_text(encoding="utf-8")
    )
    assert [pair["image_attr_values"][0] for pair in pairs] == [0, 0, 0]
    assert [pair["text_attr_values"][0] for pair in pairs] == [-1, -1, -1]


def test_experimental_tao_entrypoint_and_config(tmp_path):
    """The public TAO adapter warns while the pure converter stays reusable."""
    raw_root, vocab_path, _attributes_path, _queries_path = (
        _write_fixture(tmp_path)
    )
    output_dir = tmp_path / "output"
    config = SimpleNamespace(
        nvidia_paidf_pas=SimpleNamespace(
            raw_output_dir=str(raw_root),
            attribute_vocab_path=str(vocab_path),
            input_layout="single_run",
            caption_policy="all",
            overwrite=False,
        ),
        results_dir=str(output_dir),
    )

    with pytest.warns(UserWarning, match="experimental function"):
        manifest_path = converter.convert_nvidia_paidf_pas_to_tao_clip(config)

    assert Path(manifest_path).is_file()
    default_config = NVIDIAPAIDFPASConfig()
    assert default_config.input_layout == "split_dataset"
    assert default_config.caption_policy == "all"
    assert default_config.overwrite is False
    config_fields = {field.name: field for field in fields(NVIDIAPAIDFPASConfig)}
    assert config_fields["input_layout"].metadata["valid_options"] == (
        "split_dataset,single_run"
    )
    assert config_fields["caption_policy"].metadata["valid_options"] == (
        "all,easy,medium,hard"
    )
