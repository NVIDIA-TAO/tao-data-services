# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Held-out identities must never be published as DINOv3 training data."""

import pandas as pd
import pytest

from nvidia_tao_ds.mining.dinov3.benchmark import BenchmarkGuard
from nvidia_tao_ds.mining.dinov3.materialize import materialize_manifest


@pytest.fixture
def benchmark(tmp_path):
    """Declare all three independent benchmark identity dimensions."""
    path = tmp_path / "benchmark.parquet"
    pd.DataFrame({
        "sample_id": ["held-out"], "unit": ["held-out-unit"],
        "content_sha256": ["a" * 64],
    }).to_parquet(path, index=False)
    return path


def clean_frame():
    """Return a fully described, disjoint file locator."""
    return pd.DataFrame({
        "sample_id": ["clean"], "unit": ["clean-unit"],
        "content_sha256": ["b" * 64], "storage_type": ["file"],
        "path": ["/data/clean.jpg"],
    })


@pytest.mark.parametrize("column,value", [
    ("sample_id", "held-out"), ("unit", "held-out-unit"),
    ("content_sha256", "sha256:" + "A" * 64),
])
@pytest.mark.parametrize("parent", [False, True])
def test_materialize_rejects_benchmark_before_sealing(tmp_path, benchmark, column, value, parent):
    """Matching any identity in delta or parent rejects before publication."""
    frame = clean_frame()
    frame[column] = value
    contaminated = tmp_path / "contaminated.parquet"
    frame.to_parquet(contaminated, index=False)
    delta = tmp_path / "delta.parquet"
    clean_frame().assign(sample_id="new").to_parquet(delta, index=False)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="overlaps the sealed benchmark"):
        materialize_manifest(
            delta_path=delta if parent else contaminated,
            previous_path=contaminated if parent else None,
            output_dir=output, benchmark_units_path=benchmark,
            acquisition_unit_column="unit",
        )
    assert not (output / "_SUCCESS").exists()
    assert not (output / "artifact.json").exists()
    assert not (output / "training_manifest.parquet").exists()


@pytest.mark.parametrize("column", ["sample_id", "unit", "content_sha256"])
@pytest.mark.parametrize("invalid", [None, "", "missing"])
def test_benchmark_missing_identity_fails_closed(benchmark, column, invalid):
    """Missing metadata cannot silently weaken the declared isolation policy."""
    frame = clean_frame()
    if invalid == "missing":
        frame = frame.drop(columns=column)
    else:
        frame[column] = invalid
    error = "lacks benchmark identity" if invalid == "missing" else "null or empty benchmark identity"
    with pytest.raises(ValueError, match=error):
        BenchmarkGuard(benchmark, "unit").reject(frame, label="Training")


@pytest.mark.parametrize("column,value", [
    ("sample_id", " held-out "), ("unit", "held-out-unit\t"), ("content_sha256", " " + "a" * 64),
])
def test_surrounding_whitespace_does_not_hide_identity(benchmark, column, value):
    """Padding is formatting, not a different sample or acquisition unit."""
    frame = clean_frame()
    frame[column] = value
    with pytest.raises(ValueError, match="overlaps the sealed benchmark"):
        BenchmarkGuard(benchmark, "unit").reject(frame, label="Training")


def test_identities_are_case_sensitive_except_hashes(benchmark):
    """IDs and units are exact; only hex digests ignore case."""
    frame = clean_frame()
    frame["sample_id"], frame["unit"] = "HELD-OUT", "Held-Out-Unit"
    assert not BenchmarkGuard(benchmark, "unit").matches(frame, label="Training").any()


def test_integer_identities_match_their_text(tmp_path):
    """Integer and string spellings of an integer identity are the same."""
    path = tmp_path / "benchmark.parquet"
    pd.DataFrame({"sample_id": [7], "unit": ["u"]}).to_parquet(path, index=False)
    frame = pd.DataFrame({"sample_id": ["7", "8"], "unit": ["x", "y"]})
    assert BenchmarkGuard(path, "unit").matches(frame, label="Training").tolist() == [True, False]


@pytest.mark.parametrize("value", [7.0, b"held-out-unit", True])
def test_non_text_identities_are_rejected(benchmark, value):
    """Floats, bytes and booleans have no reliable text identity."""
    frame = clean_frame()
    frame["unit"] = [value]
    with pytest.raises(ValueError, match="^Training identity 'unit' must be string or integer$"):
        BenchmarkGuard(benchmark, "unit").reject(frame, label="Training")


def test_sidecar_errors_name_the_sidecar(tmp_path):
    """Sidecar messages read as one phrase, not "Benchmark benchmark"."""
    sidecar = tmp_path / "benchmark.parquet"
    pd.DataFrame({"sample_id": ["held-out"], "unit": [7.5]}).to_parquet(sidecar)
    with pytest.raises(ValueError, match="^Benchmark sidecar identity 'unit' must be string or integer$"):
        BenchmarkGuard(sidecar, "unit")


@pytest.mark.parametrize("held_out,other", [("held-out-unit", "clean-unit"), (7, 8)])
def test_categorical_identities_round_trip_through_parquet(tmp_path, held_out, other):
    """Dictionary-encoded columns are judged by their values in both read paths."""
    sidecar = tmp_path / "benchmark.parquet"
    pd.DataFrame({"sample_id": ["held-out"], "unit": [held_out]}).astype("category").to_parquet(sidecar)
    targets = tmp_path / "targets.parquet"
    pd.DataFrame({"sample_id": ["a", "b"], "unit": [other, held_out]}).astype("category").to_parquet(targets)
    with pytest.raises(ValueError, match="overlaps the sealed benchmark: count=1"):
        BenchmarkGuard(sidecar, "unit").reject_parquet(targets, label="Training")


def test_float_categories_are_still_rejected(benchmark):
    """Decoding categories does not let float identities through."""
    frame = clean_frame()
    frame["unit"] = pd.Series([7.5], dtype="category")
    with pytest.raises(ValueError, match="must be string or integer"):
        BenchmarkGuard(benchmark, "unit").reject(frame, label="Training")


def test_nullable_integer_identity_reports_the_null(tmp_path, benchmark):
    """A null read back as float64 is reported as null, not as a float."""
    path = tmp_path / "targets.parquet"
    clean_frame().assign(unit=pd.array([None], dtype="Int64")).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="null or empty benchmark identity 'unit'"):
        BenchmarkGuard(benchmark, "unit").reject_parquet(path, label="Training")


@pytest.mark.parametrize("sidecar,error", [
    ({"sample_id": [], "unit": []}, "must not be empty"),
    ({"sample_id": ["a"], "unit": ["u"], "content_sha256": ["not-a-hash"]}, "invalid content_sha256"),
    ({"sample_id": ["a", "b"], "unit": ["u", "v"], "content_sha256": ["a" * 64, None]},
     "null or empty benchmark identity 'content_sha256'"),
    ({"sample_id": ["a"]}, "lacks benchmark identity column 'unit'"),
])
def test_invalid_sidecar_is_rejected(tmp_path, sidecar, error):
    """A sidecar that cannot describe every held-out row is unusable."""
    path = tmp_path / "benchmark.parquet"
    pd.DataFrame(sidecar).to_parquet(path, index=False)
    with pytest.raises(ValueError, match=error):
        BenchmarkGuard(path, "unit")


def test_clean_materialization_preserves_rows_and_binds_benchmark(tmp_path, benchmark):
    """Clean data remains usable and the identity sidecar is in artifact lineage."""
    delta = tmp_path / "delta.parquet"
    clean_frame().to_parquet(delta, index=False)
    artifact = materialize_manifest(
        delta_path=delta, output_dir=tmp_path / "output",
        benchmark_units_path=benchmark, acquisition_unit_column="unit",
    )
    assert artifact["payload"]["row_count"] == 1
    assert any(item.get("role") == "benchmark_acquisition_units" for item in artifact["inputs"])


def test_no_benchmark_keeps_existing_locator_contract(tmp_path):
    """Unrelated callers need no new identity metadata."""
    delta = tmp_path / "delta.parquet"
    clean_frame().drop(columns=["unit", "content_sha256"]).to_parquet(delta, index=False)
    assert materialize_manifest(delta_path=delta, output_dir=tmp_path / "output")["payload"]["row_count"] == 1


def test_nonempty_parquet_without_any_identity_is_rejected(tmp_path, benchmark):
    """A zero-column projection is not an empty dataset."""
    path = tmp_path / "missing-identities.parquet"
    pd.DataFrame({"unrelated": [1, 2]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="lacks benchmark identity"):
        BenchmarkGuard(benchmark, "unit").reject_parquet(path, label="Training")
