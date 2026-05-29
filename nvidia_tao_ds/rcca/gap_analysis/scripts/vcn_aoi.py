# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gap analysis for SDA pipelines.

Compares model predictions against ground-truth labels and emits
a parquet of FP/FN samples for downstream augmentation.

- :func:`analyze_vcn_inference_gaps` — VCN Classify (TAO inference CSV).
"""

import logging
import math
from dataclasses import dataclass
from os import getenv
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

from nvidia_tao_ds.config.rcca.gap_analysis.vcn_aoi import GapAnalysisConfig
from nvidia_tao_ds.core.hydra.hydra_runner import hydra_runner


def compute_vcn_optimal_threshold(
    results_dir: str,
    output_path: str,
    min_recall: float,
) -> float:
    """Find the best classification threshold from VCN inference results.

    Sweeps all unique ``siamese_score`` values in the inference CSV
    to find the highest threshold that maintains at least
    ``min_recall`` recall on the NO_PASS class (maximising F1, then
    precision, as tie-breakers).

    The threshold is persisted to ``output_path`` so the next SDA
    iteration can read it for gap analysis, and is returned so the
    current iteration can feed it into the eval config
    (``model.classify.eval_margin``).

    Args:
        results_dir: Directory containing TAO VCN's ``inference.csv``.
        output_path: File path where the threshold value will
                     be written (plain text, single float).
        min_recall:  Minimum NO_PASS recall a candidate threshold
                     must achieve (0.0 – 1.0, default 1.0).

    Returns:
        The optimal threshold.

    Raises:
        FileNotFoundError: If ``{results_dir}/inference.csv`` is missing.
        ValueError: If no threshold achieves the required recall.
    """
    @dataclass(frozen=True)
    class _Row:
        """One inference CSV row reduced to the fields used by threshold sweeping.

        Attributes:
            is_pass: True if the ground-truth label is "PASS", False otherwise
                (i.e. the sample is actually NO_PASS).
            score:   The model's ``siamese_score`` for this sample. A sample is
                     predicted NO_PASS when ``score > threshold``.
        """

        is_pass: bool
        score: float

    @dataclass(frozen=True)
    class _Metrics:
        """NO_PASS-class classification metrics at a single candidate threshold.

        Used to rank thresholds during the sweep; the winner is picked by
        ``(f1, precision, threshold)`` with ``recall`` serving as a feasibility
        filter (must meet ``min_recall``).

        Attributes:
            threshold: The candidate classification threshold.
            precision: NO_PASS-class precision at this threshold.
            recall:    NO_PASS-class recall at this threshold.
            f1:        NO_PASS-class F1 at this threshold.
        """

        threshold: float
        precision: float
        recall: float
        f1: float

    csv_path = Path(results_dir) / "inference.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected inference CSV at {csv_path}"
        )

    df = pd.read_csv(csv_path)
    rows = [
        _Row(
            is_pass=str(r["label"]).strip().upper() == "PASS",
            score=float(r["siamese_score"]),
        )
        for _, r in df.iterrows()
    ]
    if not rows:
        raise ValueError(
            f"Inference CSV at {csv_path} contains no rows."
        )

    # Sweep every unique score as a candidate threshold, plus one
    # value just below the minimum so we also evaluate the case where
    # every sample is predicted NO_PASS (i.e. the most permissive
    # boundary).
    unique_scores = sorted({r.score for r in rows})
    first = math.nextafter(unique_scores[0], float("-inf"))
    thresholds = [first, *unique_scores]
    metrics = []

    # For each candidate, compute NO_PASS-class precision/recall/F1.
    # Decision rule: predict NO_PASS when siamese_score > threshold.
    for thr in thresholds:
        tp = fp = tn = fn = 0
        for r in rows:
            no_pass_actual = not r.is_pass
            no_pass_pred = r.score > thr
            if no_pass_actual and no_pass_pred:
                tp += 1
            elif not no_pass_actual and no_pass_pred:
                fp += 1
            elif not no_pass_actual and not no_pass_pred:
                tn += 1
            else:
                fn += 1

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        denom = prec + rec
        f1 = (2.0 * prec * rec) / denom if denom else 0.0

        # Only keep thresholds that meet the minimum recall constraint.
        if (tp + fn) > 0 and rec >= min_recall - 1e-12:
            metrics.append(_Metrics(
                threshold=thr,
                precision=prec,
                recall=rec,
                f1=f1,
            ))

    if not metrics:
        raise ValueError(
            f"No threshold achieves {min_recall:.0%} recall "
            "on the NO_PASS class."
        )

    # Pick the threshold with the best F1, breaking ties by
    # precision then threshold value.
    best = max(
        metrics,
        key=lambda m: (m.f1, m.precision, m.threshold),
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(str(best.threshold))

    logger.info(
        "Optimal threshold: %s (F1=%.4f, Precision=%.4f)",
        best.threshold, best.f1, best.precision,
    )
    logger.info("Threshold written to %s", out_path)
    return best.threshold


def analyze_vcn_inference_gaps(
    results_dir: str, gaps_parquet: str,
    kpi_media_path: str, train_config: str,
    threshold: float,
    top_k_per_label: int = 50,
) -> str:
    """Pick the weakest VCN Classify samples per ground-truth label.

    Computes a continuous ``weakness`` per sample as the signed distance
    from the decision threshold *in the wrong direction*:

    - PASS rows (``label == 'PASS'``):
      ``weakness = siamese_score - threshold`` — the model is wrong
      when ``score > threshold``, so larger positive values mean
      "more confidently wrong".
    - NO_PASS rows (``label != 'PASS'``):
      ``weakness = threshold - siamese_score`` — the model is wrong
      when ``score <= threshold``, so larger positive values mean
      "more confidently wrong".

    Positive weakness => misclassified.  Negative weakness => correctly
    classified, with magnitude indicating how far from the boundary
    the model is.

    **Input file:** ``{results_dir}/inference.csv`` — the KPI
    inference output TAO writes.  Uses ``input_path``,
    ``object_name``, ``label``, and ``siamese_score``.

    **Output file:** ``gaps_parquet`` with columns ``filepath``
    (absolute image path per lighting), ``label`` (ground truth),
    ``siamese_score``, and ``weakness``.

    Args:
        results_dir:      Directory containing TAO VCN's
                          ``inference.csv``.
        gaps_parquet:     Output path for the weak-sample parquet.
        kpi_media_path:   Root prepended to relative image paths.
        train_config:     VCN train YAML; reads
                          ``dataset.classify.input_map`` and
                          ``dataset.classify.image_ext``.
        threshold:        Decision boundary (predict NO_PASS when
                          ``siamese_score > threshold``).
        top_k_per_label:  Maximum number of weakest samples to keep
                          per label group.  Acts as a per-label
                          augmentation budget.

    Returns:
        The path to the written parquet file (gaps_parquet).

    Raises:
        FileNotFoundError: If ``{results_dir}/inference.csv`` is
            missing.
    """
    csv_path = Path(results_dir) / "inference.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected inference CSV at {csv_path}"
        )

    df = pd.read_csv(csv_path)

    # Compute per-sample weakness: signed distance from the threshold
    # in the WRONG direction.  Positive weakness means the model
    # disagrees with the ground truth; negative means it's correct
    # with the magnitude indicating its margin.
    label_upper = df["label"].astype(str).str.strip().str.upper()
    score = df["siamese_score"].astype(float)
    pass_weakness = score - threshold        # PASS rows
    no_pass_weakness = threshold - score      # NO_PASS rows
    df = df.assign(
        _label_upper=label_upper,
        weakness=pass_weakness.where(label_upper == "PASS", no_pass_weakness),
    )

    # Top-K weakest per ground-truth label.  Sorting first then
    # groupby(...).head(K) preserves the descending-weakness order
    # within each group, so each group's K kept rows are its weakest.
    weak_samples_df = (
        df.sort_values("weakness", ascending=False)
        .groupby("_label_upper", sort=False)
        .head(top_k_per_label)
        .drop(columns=["_label_upper"])
    )

    # Read lighting conditions and image extension from the train
    # config so we can expand each CSV row into per-image filepaths
    # matching the TAO dataset convention.
    config_data = yaml.safe_load(Path(train_config).read_text())
    lightings = config_data["dataset"]["classify"]["input_map"]
    ext = config_data["dataset"]["classify"]["image_ext"]

    # Expand each weak sample into one filepath per lighting,
    # carrying the ground-truth label and weakness for routing /
    # downstream filtering.
    media_root = Path(kpi_media_path)
    gap_records = []
    for _, row in weak_samples_df.iterrows():
        base = media_root / str(row["input_path"])
        obj = str(row["object_name"])
        label = str(row["label"]).strip()
        s = float(row["siamese_score"])
        weak = float(row["weakness"])
        for lighting in lightings:
            gap_records.append({
                "filepath": str(base / f"{obj}_{lighting}{ext}"),
                "label": label,
                "siamese_score": s,
                "weakness": weak,
            })

    # Pass columns explicitly so a no-gap result still produces a
    # parquet with the documented schema, not a zero-column DataFrame.
    gaps_df = pd.DataFrame(
        gap_records,
        columns=["filepath", "label", "siamese_score", "weakness"],
    )

    gaps_path = Path(gaps_parquet)
    gaps_path.parent.mkdir(parents=True, exist_ok=True)
    gaps_df.to_parquet(gaps_path, index=False)

    # Per-label breakdown: count, share of total, and how many of the
    # kept rows are actually misclassified (positive weakness).
    label_groups = (
        weak_samples_df.assign(
            _label=weak_samples_df["label"]
            .astype(str).str.strip().str.upper(),
        )
        .groupby("_label", sort=False)
    )
    breakdown_lines = [
        f"Weak samples breakdown by label "
        f"(threshold={threshold}, top_k_per_label={top_k_per_label})",
        f"Total KPI samples: {len(df)}",
        f"Total weak samples kept: {len(weak_samples_df)}",
        "",
    ]
    total_kept = max(len(weak_samples_df), 1)
    for label, group in label_groups:
        count = len(group)
        misclassified = int((group["weakness"] > 0).sum())
        pct = 100.0 * count / total_kept
        breakdown_lines.append(
            f"  {label}: {count} ({pct:.1f}%) — "
            f"{misclassified} misclassified, "
            f"{count - misclassified} marginal"
        )

    breakdown_text = "\n".join(breakdown_lines)
    breakdown_path = gaps_path.parent / "weak_samples_breakdown.txt"
    breakdown_path.write_text(breakdown_text + "\n")

    logger.info(
        "Selected %d weak samples (%d per-lighting filepaths) from "
        "%d total (threshold=%s, top_k_per_label=%d)",
        len(weak_samples_df), len(gaps_df), len(df),
        threshold, top_k_per_label,
    )
    logger.info("\n%s", breakdown_text)
    logger.info("Breakdown saved to %s", breakdown_path)
    return gaps_parquet


spec_root = Path(__file__).resolve().parent


@hydra_runner(
    config_path=str(spec_root / ".." / "experiment_specs"),
    config_name="vcn_aoi",
    schema=GapAnalysisConfig
)
def main(cfg: GapAnalysisConfig):
    """CLI entrypoint for VCN AOI KPI gap analysis."""
    _log_level = getattr(logging, getenv("TAO_LOGGING_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if cfg.threshold < 0:
        threshold = compute_vcn_optimal_threshold(
            results_dir=cfg.inference_results_dir,
            output_path=str(results_dir / "threshold.txt"),
            min_recall=cfg.min_recall,
        )
    else:
        threshold = float(cfg.threshold)

    analyze_vcn_inference_gaps(
        results_dir=cfg.inference_results_dir,
        gaps_parquet=str(results_dir / "kpi_gaps.parquet"),
        kpi_media_path=cfg.kpi_media_path,
        train_config=cfg.train_config,
        threshold=threshold,
        top_k_per_label=cfg.top_k_per_label,
    )


if __name__ == "__main__":
    main()
