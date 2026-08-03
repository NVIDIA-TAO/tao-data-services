# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert NVIDIA_PAS to TAO CLIP fine-tuning format.

This follows the TAO Data Services annotation conversion pattern used by
converters such as AICity to OVPKL for Sparse4D.

Output layout is compatible with the CLIP custom dataloader (ImageTextDataset).
Use these keys in your experiment spec (CLIPDataPathConfig):
- image_dir: <OUTPUT_DIR>/images  (image files or symlinks; list contains basenames)
- image_list_file: <OUTPUT_DIR>/train_list.txt, val_list.txt, or optional test_list.txt
  (one filename per line)
- caption_dir: <OUTPUT_DIR>/captions  (one .txt per image, same basename)
- caption_file_suffix: .txt
- train_pairs.json / val_pairs.json / optional test_pairs.json: each item includes 'query_type' (easy/medium/hard/natural_caption/original_captions)
  and 'caption' for each row. The CLIP dataloader uses query_type when balance_query_types is True,
  and uses caption too when unique_caption_per_batch is True so it can keep at most one row per caption per batch.
  Use ONLY_NATURAL_AND_ORIGINAL_CAPTIONS / --only-natural-and-original to emit only natural_caption and
  original_captions (omit easy, medium, and hard queries).
  Use EXCLUDE_NATURAL_FROM_AUG_DATASETS / --exclude-natural-from-aug to omit natural_caption and
  original_captions for entries whose dataset name ends with "_Aug" (easy/medium/hard only for aug sources).

QA / different-environment note (FileNotFoundError for images, captions present):
- By default the script creates SYMLINKS in output_dir/images pointing to
  NVIDIA_PAS_ROOT (source images). If preparation was run on another machine or
  NVIDIA_PAS_ROOT points to a path that does not exist where training runs (e.g.
  QA at /data/), those symlinks are broken: caption files exist but opening
  the "image" fails. Fix: run this script ON THE SAME ENVIRONMENT where you
  train, with NVIDIA_PAS_ROOT set to the actual dataset path on that host (e.g.
  prep.NVIDIA_PAS_ROOT = '/data/NVIDIA_PAS' on QA). Alternatively use
  use_symlinks=False and copy images so the output is self-contained (slower,
  more disk).

Train/Val sources:
- Train pairs: all datasets except CUHK_PEDES and ICFG_PEDES (those two canonical names are excluded
  from training). Augmented or renamed variants (different entry["dataset"] string) are not excluded
  unless you add them to EXCLUDED_TRAIN_DATASETS.
- Val pairs: every dataset except those in EXCLUDED_VAL_DATASETS (default: PA-100K omitted from val to
  keep retrieval evaluation tractable). CUHK_PEDES and ICFG_PEDES val images remain in val; their train
  split images still never appear in train pairs. After dedup, val is randomly subsampled at VAL_SAMPLE_FRACTION
  per (dataset Ã— query_type) stratum so overall size drops (~10% by default) while preserving mix of query types
  and datasets.
- Test pairs (optional, --include-test): full, non-subsampled test split for the text-to-image KPI
  datasets used by evaluate_nvidia_pas.py / run_siglip2_local_nvidia_pas.sh:
  CUHK_PEDES, ICFG_PEDES, RSTPReid, and PA-100K.
- Filtering is by entry["dataset"] and split (train vs val vs optional test).

Query Type Handling:
- Image-aligned labels (metadata.per_image_annotations and entry.per_image): each row uses that imageâ€™s
  attributes, queries, natural_caption, and original_captions (trainâ†’valâ†’test order matches per_image).
  If an aligned per_image row is missing text but the entry has it, the script warns and falls back to
  the entry-level text for that query/caption type.
  Hard / natural_caption / original_captions are paired only with that image path (not all images of the ID).
- Legacy ID-level labels (no aligned per_image): hard, natural_caption, original_captions use all images
  in the split for that entry; natural_caption may be a list of stringsâ€”each string paired with those images.
- easy, medium (CONFIRMED attribute-based when USE_ATTRIBUTE_MATCHING=True):
  - Text-image pairs use ATTRIBUTES from the per_image row (flash) or entry (legacy). Many IDs may share
    the same attribute values; one query matches all such images. Same caption can appear on multiple rows;
    the training batch sampler keeps at most one row per caption string per batch.
  - Matching uses the same attribute names as in the queryâ€™s second field (comma-separated). For the
    ``accessories`` attribute, values are a list of lists (e.g. type / color / object per item).
    Accessory matching follows PAS evaluation: extract exact item requirements from the query text; require
    color only when the query says color+item. Other attributes still use equality on the indexed value.
  - USE_ATTRIBUTE_MATCHING=True: attribute-based mapping (default).
  - USE_ATTRIBUTE_MATCHING=False: 1-1 mapping to that image (per_image) or that entryâ€™s images (legacy).
"""

import argparse
import json
import os
import random
import re
import shutil
import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm

# Number of workers for parallel file/symlink writing (I/O-bound).
NUM_WORKERS = min(32, (os.cpu_count() or 8) * 2)
# Chunk size per worker to limit memory and keep progress responsive.
WRITE_CHUNK_SIZE = 50_000
# Batch size for disk-backed pair insertion. Larger batches improve SQLite throughput
# without bringing back the old all-pairs-in-memory behavior.
PAIR_INSERT_CHUNK_SIZE = 250_000
PAIR_STORE_NAME = "_pair_store.sqlite"

# Optional defaults for direct module use. The Data Services conversion entrypoint
# passes explicit paths from the experiment spec.
NVIDIA_PAS_ROOT = None
OUTPUT_DIR = None
# True: create symlinks in output_dir/images (fast, requires source at same path when training).
# False: copy images so output is self-contained (slower, portable across machines).
USE_SYMLINKS = True

QUERY_TYPES = ["easy", "medium", "hard", "natural_caption", "original_captions"]
# Subset: human-written natural_caption + dataset original_captions only (no easy/medium/hard queries).
NATURAL_AND_ORIGINAL_QUERY_TYPES = ["natural_caption", "original_captions"]
# When True, only emit pairs for NATURAL_AND_ORIGINAL_QUERY_TYPES.
ONLY_NATURAL_AND_ORIGINAL_CAPTIONS = False
# When True, entries with dataset name ending in "_Aug" emit only easy/medium/hard (no natural/original).
EXCLUDE_NATURAL_FROM_AUG_DATASETS = False
AUG_DATASET_SUFFIX = "_Aug"
STRUCTURED_QUERY_TYPES = ["easy", "medium", "hard"]

# True: easy/medium use attribute-based matching (many IDs can share same attributes; one query matches all such images).
# False: easy/medium use 1-1 person ID only.
USE_ATTRIBUTE_MATCHING = True

# Train: omit pairs from these datasets (CUHK/ICFG train is held out). Val images for these datasets are
# still included in val (unless also listed in EXCLUDED_VAL_DATASETS).
EXCLUDED_TRAIN_DATASETS: Tuple[str, ...] = ("CUHK_PEDES", "ICFG_PEDES")
# Val: omit pairs from these datasets (e.g. PA-100K is huge and slows GPU retrieval metrics).
EXCLUDED_VAL_DATASETS: Tuple[str, ...] = ("PA-100K",)
# Optional test output mirrors the current PAS text-to-image evaluation script.
TEST_DATASETS: Tuple[str, ...] = ("CUHK_PEDES", "ICFG_PEDES", "RSTPReid", "PA-100K")

# After dedup, keep this fraction of val pairs per (dataset Ã— query_type) stratum (random, stratified).
VAL_SAMPLE_FRACTION = 0.1


def _entry_allowed_for_train(entry: Dict) -> bool:
    """True if this entry may contribute train split pairs."""
    return entry.get("dataset") not in EXCLUDED_TRAIN_DATASETS


def _entry_allowed_for_val(entry: Dict) -> bool:
    """True if this entry may contribute val split pairs."""
    return entry.get("dataset") not in EXCLUDED_VAL_DATASETS


def _entry_allowed_for_test(entry: Dict) -> bool:
    """True if this entry may contribute optional test split pairs."""
    return entry.get("dataset") in TEST_DATASETS


def _is_aug_dataset(dataset_name: str) -> bool:
    """True when entry['dataset'] is an augmented PAS source (name ends with _Aug)."""
    return bool(dataset_name) and str(dataset_name).endswith(AUG_DATASET_SUFFIX)


def _query_types_for_entry(
    entry: Dict,
    active_query_types: List[str],
    exclude_natural_from_aug: bool,
) -> List[str]:
    """Per-entry query types; aug datasets may drop natural_caption / original_captions."""
    if not exclude_natural_from_aug or not _is_aug_dataset(entry.get("dataset", "") or ""):
        return active_query_types
    return [q for q in active_query_types if q in STRUCTURED_QUERY_TYPES]


def _flatten_images_split_order(entry: Dict) -> List[str]:
    """images.train, then val, then test (same flatten order as per_image in NVIDIA_PAS)."""
    imgs = entry.get("images") or {}
    out: List[str] = []
    for split in ("train", "val", "test"):
        out.extend(imgs.get(split) or [])
    return out


def _entry_has_aligned_per_image(entry: Dict) -> bool:
    """True when per_image exists and has one row per flattened image (image-aligned annotations)."""
    per_image = entry.get("per_image")
    if not isinstance(per_image, list) or not per_image:
        return False
    return len(per_image) == len(_flatten_images_split_order(entry))


def _natural_caption_strings(natural_caption) -> List[str]:
    """Normalize natural_caption to a list of non-empty strings (handles str or list of str)."""
    if not natural_caption:
        return []
    if isinstance(natural_caption, list):
        out = []
        for c in natural_caption:
            if not c:
                continue
            out.append(c.strip() if isinstance(c, str) else str(c).strip())
        return [x for x in out if x]
    s = natural_caption.strip() if isinstance(natural_caption, str) else str(natural_caption).strip()
    return [s] if s else []


def _query_dict_has_text(queries, query_types: Optional[List[str]] = None) -> bool:
    if not isinstance(queries, dict):
        return False
    keys = query_types if query_types is not None else list(queries.keys())
    return any(isinstance(queries.get(k), list) and bool(queries.get(k)) for k in keys)


def _queries_for_per_image_row(entry: Dict, per_image_row: Dict) -> Dict:
    """Return per-image queries, falling back per query type to entry-level queries."""
    row_queries = per_image_row.get("queries") if isinstance(per_image_row.get("queries"), dict) else {}
    entry_queries = entry.get("queries") if isinstance(entry.get("queries"), dict) else {}
    if not entry_queries:
        return row_queries
    if not row_queries:
        return entry_queries

    merged = dict(entry_queries)
    for qtype, qlist in row_queries.items():
        if qlist:
            merged[qtype] = qlist
        elif qtype not in merged:
            merged[qtype] = qlist
    return merged


def _natural_captions_for_per_image_row(entry: Dict, per_image_row: Dict) -> List[str]:
    captions = _natural_caption_strings(per_image_row.get("natural_caption"))
    if captions:
        return captions
    return _natural_caption_strings(entry.get("natural_caption"))


def _original_captions_for_per_image_row(entry: Dict, per_image_row: Dict, split: str) -> List[str]:
    captions = _original_caption_strings_from_per_image_cell(per_image_row.get("original_captions"))
    if captions:
        return captions
    original_captions = entry.get("original_captions")
    if isinstance(original_captions, dict):
        return _original_caption_strings_from_per_image_cell(original_captions.get(split, []))
    return _original_caption_strings_from_per_image_cell(original_captions)


def _warn_on_per_image_text_fallbacks(entries: List[Dict], active_query_types: List[str]) -> None:
    """Warn when aligned per_image rows need entry-level query/caption fallback."""
    query_types = [q for q in active_query_types if q in ("easy", "medium", "hard")]
    query_entries = defaultdict(list)
    query_rows = defaultdict(int)
    natural_entries = defaultdict(list)
    natural_rows = defaultdict(int)
    no_query_entries = defaultdict(list)
    no_query_rows = defaultdict(int)

    for entry in entries:
        if not _entry_has_aligned_per_image(entry):
            continue
        ds = entry.get("dataset", "") or ""
        person = str(entry.get("person_key") or entry.get("person_id") or "<unknown>")
        rows = entry.get("per_image") or []
        if query_types and _query_dict_has_text(entry.get("queries"), query_types):
            row_count = sum(1 for pi in rows if not _query_dict_has_text(pi.get("queries"), query_types))
            if row_count:
                query_entries[ds].append(person)
                query_rows[ds] += row_count
        if "natural_caption" in active_query_types and _natural_caption_strings(entry.get("natural_caption")):
            row_count = sum(1 for pi in rows if not _natural_caption_strings(pi.get("natural_caption")))
            if row_count:
                natural_entries[ds].append(person)
                natural_rows[ds] += row_count
        if query_types and not _query_dict_has_text(entry.get("queries"), query_types):
            row_count = sum(1 for pi in rows if not _query_dict_has_text(pi.get("queries"), query_types))
            if row_count:
                no_query_entries[ds].append(person)
                no_query_rows[ds] += row_count

    for ds in sorted(set(query_entries) | set(natural_entries) | set(no_query_entries)):
        if ds in query_entries:
            warnings.warn(
                f"{ds}: {len(query_entries[ds])} aligned per_image entries "
                f"({query_rows[ds]} image rows) have empty per_image queries but entry-level queries exist; "
                f"falling back to entry-level queries. Samples: {', '.join(query_entries[ds][:5])}"
            )
        if ds in natural_entries:
            warnings.warn(
                f"{ds}: {len(natural_entries[ds])} aligned per_image entries "
                f"({natural_rows[ds]} image rows) have empty per_image natural_caption but entry-level "
                f"natural_caption exists; falling back to entry-level natural_caption. "
                f"Samples: {', '.join(natural_entries[ds][:5])}"
            )
        if ds in no_query_entries:
            warnings.warn(
                f"{ds}: {len(no_query_entries[ds])} aligned per_image entries "
                f"({no_query_rows[ds]} image rows) have no usable easy/medium/hard queries at either "
                f"per_image or entry level. Samples: {', '.join(no_query_entries[ds][:5])}"
            )


def _pair_dedup_key(img, cap) -> Tuple:
    """Hashable key for (image_path, caption); guards against list/dict edge types in labels.json."""
    if isinstance(img, list):
        img = tuple(img)
    if isinstance(cap, list):
        cap = tuple(cap)
    elif not isinstance(cap, str):
        cap = str(cap)
    return (img, cap)


def _sample_val_pairs_stratified(
    val_pairs_with_dataset: List[Tuple[str, str, str, str]],
    fraction: float,
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    """Random subsample ``fraction`` of val pairs independently per (dataset, query_type) stratum.

    Preserves approximate relative frequencies across query types (easy/medium/hard/natural/original)
    and across datasets. PA-100K rows must already be omitted upstream (see EXCLUDED_VAL_DATASETS).
    """
    if fraction >= 1.0:
        return [(p[0], p[1], p[2]) for p in val_pairs_with_dataset]
    if fraction <= 0.0:
        return []

    strata: Dict[Tuple[str, str], List[Tuple[str, str, str, str]]] = defaultdict(list)
    for p in val_pairs_with_dataset:
        ds = p[3] if len(p) > 3 else ""
        strata[(ds, p[2])].append(p)

    out: List[Tuple[str, str, str]] = []
    for items in strata.values():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        k = min(n, max(0, int(n * fraction + 0.5)))
        if k <= 0:
            continue
        chosen = rng.sample(items, k=k)
        out.extend((x[0], x[1], x[2]) for x in chosen)
    rng.shuffle(out)
    return out


def extract_query_attributes(query_pair) -> List[str]:
    """Extract attribute names from a query pair (for easy/medium queries).

    Same convention as evaluate_nvidia_pas.py: query_pair = [query_text, "attr1, attr2"];
    we return the list of attribute names for indexing/lookup.
    """
    if not isinstance(query_pair, list) or len(query_pair) < 2:
        return []
    attrs_str = query_pair[1]
    return [a.strip() for a in attrs_str.split(',')]


def _iter_unique_query_attr_keys(queries: Dict):
    """Yield unique easy/medium attribute-key shapes used by one entry/image row."""
    seen = set()
    for qtype in ("easy", "medium"):
        for query_pair in (queries or {}).get(qtype, []):
            attr_names = extract_query_attributes(query_pair)
            if not attr_names:
                continue
            attr_key = tuple(sorted(attr_names))
            if attr_key in seen:
                continue
            seen.add(attr_key)
            yield attr_key


def _hashable_attr_value(v):
    """Convert an attribute value to a hashable type for use as dict key (tuple of attr values)."""
    if isinstance(v, list):
        return tuple(str(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable_attr_value(vv)) for k, vv in v.items()))
    try:
        hash(v)
        return v
    except TypeError:
        return str(v)


ACCESSORY_ATTRIBUTE_NAME = "accessories"


def _normalize_phrase(value) -> str:
    """Normalize labels/query text for exact accessory phrase checks."""
    return re.sub(r"\s+", " ", str(value).strip().lower()) if value is not None else ""


def _contains_exact_phrase(text: str, phrase: str) -> bool:
    phrase = _normalize_phrase(phrase)
    if not phrase:
        return False
    text = _normalize_phrase(text)
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _iter_accessory_parts_from_value(accessories_val) -> List[Tuple[str, str]]:
    """Return normalized (item, color) pairs from structured PAS accessories."""
    out: List[Tuple[str, str]] = []
    if not accessories_val:
        return out
    if isinstance(accessories_val, dict):
        item = _normalize_phrase(
            accessories_val.get("item") or accessories_val.get("object") or accessories_val.get("type") or ""
        )
        color = _normalize_phrase(accessories_val.get("color") or "")
        return [(item, color)] if item else out
    if isinstance(accessories_val, list):
        rows = accessories_val if any(isinstance(x, (list, dict)) for x in accessories_val) else [accessories_val]
        for row in rows:
            if isinstance(row, dict):
                out.extend(_iter_accessory_parts_from_value(row))
                continue
            if isinstance(row, list):
                vals = [_normalize_phrase(x) for x in row if _normalize_phrase(x)]
                if not vals:
                    continue
                item = vals[-1]
                color = vals[-2] if len(vals) >= 2 else ""
                out.append((item, color))
            else:
                item = _normalize_phrase(row)
                if item:
                    out.append((item, ""))
        return out
    item = _normalize_phrase(accessories_val)
    return [(item, "")] if item else out


def _iter_accessory_parts(attributes: Dict) -> List[Tuple[str, str]]:
    return _iter_accessory_parts_from_value((attributes or {}).get(ACCESSORY_ATTRIBUTE_NAME))


def _query_text(query_pair) -> str:
    return str(query_pair[0] if isinstance(query_pair, list) and query_pair else query_pair or "").strip()


def _accessory_requirements(query_attributes: Dict, query_pair) -> List[Tuple[str, str]]:
    """Find exactly mentioned accessory requirements from structured labels and query text."""
    text = _query_text(query_pair)
    requirements: List[Tuple[str, str]] = []
    seen = set()
    for item, color in _iter_accessory_parts(query_attributes):
        color_item = f"{color} {item}" if color else ""
        req = None
        if color_item and _contains_exact_phrase(text, color_item):
            req = (item, color)
        elif _contains_exact_phrase(text, item):
            req = (item, "")
        if req and req not in seen:
            seen.add(req)
            requirements.append(req)
    return requirements


def _matches_accessory_requirements(
    candidate_parts: List[Tuple[str, str]],
    requirements: List[Tuple[str, str]],
) -> bool:
    for required_item, required_color in requirements:
        if required_color:
            if not any(item == required_item and color == required_color for item, color in candidate_parts):
                return False
        elif not any(item == required_item for item, _ in candidate_parts):
            return False
    return True


def _attribute_match_signature(attributes: Dict, query_pair) -> Optional[Tuple]:
    """Stable key for a query's attribute-match bucket; None means row-local fallback."""
    attr_names = extract_query_attributes(query_pair)
    if not attr_names:
        return None

    attr_key = tuple(sorted(attr_names))
    if ACCESSORY_ATTRIBUTE_NAME in attr_names:
        names_no_acc = sorted(n for n in attr_names if n != ACCESSORY_ATTRIBUTE_NAME)
        base_vals = tuple(_hashable_attr_value((attributes or {}).get(name, '')) for name in names_no_acc)
        requirements = tuple(_accessory_requirements(attributes or {}, query_pair))
        if not requirements:
            return None
        return ("accessories", attr_key, base_vals, requirements)

    attr_values = tuple(_hashable_attr_value((attributes or {}).get(name, '')) for name in attr_key)
    return ("attributes", attr_key, attr_values)


def _attribute_query_cache_key(
    split_name: str,
    query_type: str,
    dataset_name: str,
    caption: str,
    attributes: Dict,
    query_pair,
) -> Optional[Tuple]:
    signature = _attribute_match_signature(attributes, query_pair)
    if signature is None:
        return None
    caption = caption.strip() if isinstance(caption, str) else str(caption).strip()
    if not caption:
        return None
    dataset_key = "" if split_name == "train" else str(dataset_name or "")
    return (split_name, str(query_type), dataset_key, caption, signature)


def _add_attribute_index_images(attr_index: Dict, attr_key: Tuple, attributes: Dict, images: List[str]) -> None:
    """Index each image once for a unique easy/medium attribute-key shape."""
    if not attr_key or not images:
        return
    attributes = attributes or {}
    if ACCESSORY_ATTRIBUTE_NAME in attr_key:
        names_no_acc = sorted(n for n in attr_key if n != ACCESSORY_ATTRIBUTE_NAME)
        base_vals = tuple(_hashable_attr_value(attributes.get(name, '')) for name in names_no_acc)
        acc_tokens = tuple(_iter_accessory_parts(attributes))
        attr_index[attr_key][base_vals].extend((img, acc_tokens) for img in images)
        return

    attr_values = tuple(_hashable_attr_value(attributes.get(name, '')) for name in attr_key)
    attr_index[attr_key][attr_values].extend(images)


def build_attribute_index_legacy(entries: List[Dict], split: str) -> Dict[str, Dict[Tuple, List[str]]]:
    """Entry-level attributes + all images in ``split`` (legacy ID-aligned labels)."""
    attr_index = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        if _entry_has_aligned_per_image(entry):
            continue
        if split == "train" and not _entry_allowed_for_train(entry):
            continue
        if split == "val" and not _entry_allowed_for_val(entry):
            continue
        if split == "test" and not _entry_allowed_for_test(entry):
            continue
        images = entry['images'].get(split, [])
        if not images:
            continue

        attributes = entry.get('attributes', {})
        queries = entry.get('queries', {})

        for attr_key in _iter_unique_query_attr_keys(queries):
            _add_attribute_index_images(attr_index, attr_key, attributes, images)

    return attr_index


def build_attribute_index_per_image(entries: List[Dict], split: str) -> Dict[str, Dict[Tuple, List[str]]]:
    """Per-image attributes and queries (flash / PA-100K style labels)."""
    attr_index = defaultdict(lambda: defaultdict(list))

    for entry in entries:
        if not _entry_has_aligned_per_image(entry):
            continue
        if split == "train" and not _entry_allowed_for_train(entry):
            continue
        if split == "val" and not _entry_allowed_for_val(entry):
            continue
        if split == "test" and not _entry_allowed_for_test(entry):
            continue
        for pi in entry.get("per_image") or []:
            if pi.get("split") != split:
                continue
            img = pi.get("image_path")
            if not img:
                continue
            attributes = pi.get("attributes") or {}
            queries = _queries_for_per_image_row(entry, pi)
            for attr_key in _iter_unique_query_attr_keys(queries):
                _add_attribute_index_images(attr_index, attr_key, attributes, [img])

    return attr_index


def merge_attribute_indices(
    *indices: Dict[str, Dict[Tuple, List[str]]],
) -> Dict[str, Dict[Tuple, List[str]]]:
    """Merge split-level attribute lookup tables while preserving image lists."""
    out = defaultdict(lambda: defaultdict(list))
    for idx in indices:
        for attr_key, inner in idx.items():
            for vals, imgs in inner.items():
                out[attr_key][vals].extend(imgs)
    return out


# The explicit returns mirror distinct query-type and fallback branches.
# pylint: disable=too-many-return-statements
def get_matching_images_for_query(
    entry: Dict,
    query_type: str,
    query_idx: int,
    split: str,
    attr_index: Dict
) -> List[str]:
    """Get all matching images for a query (1-1 for hard/natural_caption/original_captions; configurable for easy/medium)."""
    if query_type in ['hard', 'natural_caption', 'original_captions']:
        return entry['images'].get(split, [])

    if not USE_ATTRIBUTE_MATCHING:
        return entry['images'].get(split, [])

    queries = entry.get('queries', {}).get(query_type, [])
    if query_idx >= len(queries):
        return []

    query_pair = queries[query_idx]
    attr_names = extract_query_attributes(query_pair)
    if not attr_names:
        return entry['images'].get(split, [])

    attributes = entry.get('attributes', {})
    attr_key = tuple(sorted(attr_names))

    if 'accessories' in attr_names:
        names_no_acc = sorted(n for n in attr_names if n != 'accessories')
        base_vals = tuple(_hashable_attr_value(attributes.get(name, '')) for name in names_no_acc)
        requirements = _accessory_requirements(attributes, query_pair)
        if not requirements:
            return entry['images'].get(split, [])
        bucket = attr_index.get(attr_key, {}).get(base_vals, [])
        out: List[str] = []
        for item in bucket:
            if isinstance(item, tuple) and len(item) == 2:
                img_path, cand_tokens = item
                if _matches_accessory_requirements(cand_tokens, requirements):
                    out.append(img_path)
        return out

    attr_values = tuple(_hashable_attr_value(attributes.get(name, '')) for name in sorted(attr_names))
    return attr_index.get(attr_key, {}).get(attr_values, [])


def get_matching_images_for_query_per_image(
    per_image_row: Dict,
    query_type: str,
    query_idx: int,
    attr_index: Dict,
    query_pair=None,
) -> List[str]:
    """Easy/medium matching using this row's attributes; falls back to the row's image if index misses."""
    if not USE_ATTRIBUTE_MATCHING:
        img = per_image_row.get("image_path")
        return [img] if img else []

    if query_pair is None:
        queries = (per_image_row.get("queries") or {}).get(query_type, [])
        if query_idx >= len(queries):
            return []
        query_pair = queries[query_idx]
    attr_names = extract_query_attributes(query_pair)
    if not attr_names:
        img = per_image_row.get("image_path")
        return [img] if img else []

    attributes = per_image_row.get("attributes") or {}
    attr_key = tuple(sorted(attr_names))

    if 'accessories' in attr_names:
        names_no_acc = sorted(n for n in attr_names if n != 'accessories')
        base_vals = tuple(_hashable_attr_value(attributes.get(name, '')) for name in names_no_acc)
        requirements = _accessory_requirements(attributes, query_pair)
        if not requirements:
            img = per_image_row.get("image_path")
            return [img] if img else []
        bucket = attr_index.get(attr_key, {}).get(base_vals, [])
        imgs = []
        for item in bucket:
            if isinstance(item, tuple) and len(item) == 2:
                img_path, cand_tokens = item
                if _matches_accessory_requirements(cand_tokens, requirements):
                    imgs.append(img_path)
        if imgs:
            return imgs
        img = per_image_row.get("image_path")
        return [img] if img else []

    attr_values = tuple(_hashable_attr_value(attributes.get(name, '')) for name in sorted(attr_names))
    imgs = attr_index.get(attr_key, {}).get(attr_values, [])
    if imgs:
        return imgs
    img = per_image_row.get("image_path")
    return [img] if img else []


# pylint: enable=too-many-return-statements


def _hard_query_captions(hard_list) -> List[str]:
    """Normalize queries.hard to caption strings (list of str or mixed)."""
    out: List[str] = []
    if not hard_list:
        return out
    for q in hard_list:
        if isinstance(q, list) and q:
            s = (q[0] or "").strip() if isinstance(q[0], str) else str(q[0]).strip()
        elif isinstance(q, str):
            s = q.strip()
        else:
            s = str(q).strip() if q is not None else ""
        if s:
            out.append(s)
    return out


class _SplitPairAppender:
    """List-like append target that writes pairs to a disk-backed pair store."""

    def __init__(self, store: "_DiskPairStore", split_name: str):
        self.store = store
        self.split_name = split_name

    def append(self, item) -> None:
        self.store.add(self.split_name, item)

    def append_images(self, image_paths, caption, query_type, dataset: str = "") -> int:
        return self.store.add_images(self.split_name, image_paths, caption, query_type, dataset)

    def __len__(self) -> int:
        return self.store.raw_count(self.split_name)


class _DiskPairStore:
    """SQLite-backed pair sink with on-disk dedup by (split, image, caption)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                path.unlink()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        # This store is disposable build scratch; favor throughput over crash recovery.
        self.conn.execute("PRAGMA journal_mode=OFF")
        self.conn.execute("PRAGMA synchronous=OFF")
        self.conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        self.conn.execute("PRAGMA temp_store=FILE")
        self.conn.execute("PRAGMA cache_size=-200000")
        self.conn.execute(
            """
            CREATE TABLE pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                split TEXT NOT NULL,
                image_path TEXT NOT NULL,
                caption TEXT NOT NULL,
                query_type TEXT NOT NULL,
                dataset TEXT NOT NULL DEFAULT '',
                UNIQUE(split, image_path, caption)
            )
            """
        )
        self.query_indexes_ready = False
        self.buffer: List[Tuple[str, str, str, str, str]] = []
        self.raw_counts = defaultdict(int)

    def appender(self, split_name: str) -> _SplitPairAppender:
        return _SplitPairAppender(self, split_name)

    def add(self, split_name: str, item) -> None:
        dataset = item[3] if len(item) > 3 else ""
        self.add_images(split_name, (item[0],), item[1], item[2], dataset)

    def add_images(self, split_name: str, image_paths, caption, query_type, dataset: str = "") -> int:
        caption = caption.strip() if isinstance(caption, str) else str(caption).strip()
        if not caption:
            return 0
        query_type = str(query_type)
        dataset = str(dataset or "")
        added = 0
        for image_path in image_paths:
            if not image_path:
                continue
            self.buffer.append((
                split_name,
                str(image_path),
                caption,
                query_type,
                dataset,
            ))
            added += 1
            if len(self.buffer) >= PAIR_INSERT_CHUNK_SIZE:
                self.flush()
        if added:
            self.raw_counts[split_name] += added
            self.query_indexes_ready = False
        return added

    def flush(self) -> None:
        if not self.buffer:
            return
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO pairs(split, image_path, caption, query_type, dataset)
            VALUES (?, ?, ?, ?, ?)
            """,
            self.buffer,
        )
        self.conn.commit()
        self.buffer.clear()

    def ensure_query_indexes(self) -> None:
        if self.query_indexes_ready:
            return
        self.flush()
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pairs_split ON pairs(split)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pairs_split_dataset_qtype ON pairs(split, dataset, query_type)"
        )
        self.conn.commit()
        self.query_indexes_ready = True

    def raw_count(self, split_name: str) -> int:
        return self.raw_counts[split_name]

    def count(self, split_name: str, selected_val: bool = False) -> int:
        self.ensure_query_indexes()
        if split_name == "val" and selected_val:
            return self.conn.execute("SELECT COUNT(*) FROM selected_val").fetchone()[0]
        return self.conn.execute(
            "SELECT COUNT(*) FROM pairs WHERE split = ?", (split_name,)
        ).fetchone()[0]

    def count_by_query_type(self, split_name: str, selected_val: bool = False) -> Dict[str, int]:
        self.ensure_query_indexes()
        if split_name == "val" and selected_val:
            rows = self.conn.execute(
                """
                SELECT p.query_type, COUNT(*)
                FROM pairs p
                JOIN selected_val s ON s.id = p.id
                GROUP BY p.query_type
                """
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT query_type, COUNT(*)
                FROM pairs
                WHERE split = ?
                GROUP BY query_type
                """,
                (split_name,),
            ).fetchall()
        return dict(rows)

    def sample_val(self, fraction: float, rng: random.Random) -> None:
        """Create selected_val table with deterministic stratified sampling by dataset/query_type."""
        self.ensure_query_indexes()
        self.conn.execute("DROP TABLE IF EXISTS selected_val")
        self.conn.execute("CREATE TABLE selected_val (id INTEGER PRIMARY KEY)")
        if fraction <= 0.0:
            raise ValueError("val_sample_fraction must be > 0.0")

        strata = self.conn.execute(
            """
            SELECT dataset, query_type, COUNT(*)
            FROM pairs
            WHERE split = 'val'
            GROUP BY dataset, query_type
            """
        ).fetchall()
        for dataset, query_type, count in strata:
            ids = [
                row[0] for row in self.conn.execute(
                    """
                    SELECT id
                    FROM pairs
                    WHERE split = 'val' AND dataset = ? AND query_type = ?
                    ORDER BY id
                    """,
                    (dataset, query_type),
                )
            ]
            if fraction >= 1.0:
                selected = ids
            else:
                n_keep = max(1, int(count * fraction))
                selected = rng.sample(ids, n_keep) if count > n_keep else ids
            self.conn.executemany(
                "INSERT INTO selected_val(id) VALUES (?)",
                [(i,) for i in selected],
            )
        self.conn.commit()

    def iter_pairs(self, split_name: str, selected_val: bool = False):
        self.ensure_query_indexes()
        if split_name == "val" and selected_val:
            cursor = self.conn.execute(
                """
                SELECT p.image_path, p.caption, p.query_type
                FROM pairs p
                JOIN selected_val s ON s.id = p.id
                ORDER BY p.id
                """
            )
        else:
            cursor = self.conn.execute(
                """
                SELECT image_path, caption, query_type
                FROM pairs
                WHERE split = ?
                ORDER BY id
                """,
                (split_name,),
            )
        yield from cursor

    def close(self, remove_db: bool = False) -> None:
        self.flush()
        self.conn.close()
        if remove_db:
            for suffix in ("", "-wal", "-shm"):
                path = Path(str(self.db_path) + suffix)
                if path.exists():
                    path.unlink()


def _write_chunk(
    chunk: List[Tuple[str, str, str]],
    start_idx: int,
    split_name: str,
    out_dir: Path,
    src_root: str,
    use_symlinks: bool,
) -> List[Dict]:
    """Write symlinks and caption files for one chunk of pairs. Returns list of pair dicts for JSON.
    Each chunk item is (img_path, caption, query_type). query_type is written to JSON for balanced sampling.
    """
    captions_dir = out_dir / "captions"
    images_dir = out_dir / "images"
    pairs_json_chunk = []
    for i, item in enumerate(chunk):
        img_path, caption, query_type = item[0], item[1], item[2]
        idx = start_idx + i
        img_ext = Path(img_path).suffix or '.jpg'
        unique_name = f"{split_name}_{idx:08d}{img_ext}"
        pairs_json_chunk.append({
            'idx': idx,
            'unique_name': unique_name,
            'image_path': img_path,
            'caption': caption,
            'query_type': query_type,
        })
        image_out_path = images_dir / unique_name
        actual_path = Path(src_root) / img_path
        if use_symlinks:
            if image_out_path.exists() or image_out_path.is_symlink():
                image_out_path.unlink()
            try:
                image_out_path.symlink_to(actual_path)
            except OSError as e:
                # e.g. target not found, permission, or cross-filesystem on some envs
                warnings.warn(
                    f"Symlink failed for {unique_name} -> {actual_path}: {e}. "
                    "Ensure NVIDIA_PAS_ROOT exists on this machine or use use_symlinks=False to copy."
                )
        else:
            try:
                shutil.copy2(actual_path, image_out_path)
            except OSError as e:
                warnings.warn(f"Copy failed for {unique_name} from {actual_path}: {e}")
        caption_basename = f"{split_name}_{idx:08d}"
        caption_file = captions_dir / f"{caption_basename}.txt"
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(caption)
    return pairs_json_chunk


def write_split(
    pairs: List[Tuple[str, str, str]],
    split_name: str,
    out_dir: str,
    src_root: str,
    use_symlinks: bool = True
):
    """
    Write image list and caption files for a split (parallelized).
    Each pair is (img_path, caption, query_type). query_type is written to <split>_pairs.json.
    Produces layout expected by clip.dataloader.custom_loader:
    - <out_dir>/<split>_list.txt: one image basename per line
    - <out_dir>/images/<basename>: image (or symlink)
    - <out_dir>/captions/<basename>.txt: caption text
    """
    out_path = Path(out_dir)
    captions_dir = out_path / "captions"
    images_dir = out_path / "images"
    captions_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Split into chunks for parallel writing
    chunks = []
    for start in range(0, len(pairs), WRITE_CHUNK_SIZE):
        chunks.append((start, pairs[start:start + WRITE_CHUNK_SIZE]))

    pairs_json = [None] * len(chunks)  # hold results in order
    chunk_sizes = [len(c) for _, c in chunks]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(
                _write_chunk,
                chunk,
                start_idx,
                split_name,
                out_path,
                src_root,
                use_symlinks,
            ): i
            for i, (start_idx, chunk) in enumerate(chunks)
        }
        with tqdm(total=len(pairs), desc=f"Writing {split_name}", unit="it") as pbar:
            for future in as_completed(futures):
                chunk_idx = futures[future]
                pairs_json[chunk_idx] = future.result()
                pbar.update(chunk_sizes[chunk_idx])

    # Flatten in order and write list + JSON
    pairs_json_flat = []
    for chunk_result in pairs_json:
        pairs_json_flat.extend(chunk_result)

    image_list = [p["unique_name"] for p in pairs_json_flat]
    with open(out_path / f"{split_name}_list.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(image_list))

    with open(out_path / f"{split_name}_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(pairs_json_flat, f, indent=2)

    if split_name in ("val", "test"):
        unique_captions = list(set(p[1] for p in pairs))
        with open(out_path / f"{split_name}_class_names.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(unique_captions))


def _iter_pair_chunks(pair_iter, chunk_size: int):
    start_idx = 0
    chunk = []
    for pair in pair_iter:
        chunk.append(pair)
        if len(chunk) >= chunk_size:
            yield start_idx, chunk
            start_idx += len(chunk)
            chunk = []
    if chunk:
        yield start_idx, chunk


def _write_split_from_iter(
    pair_iter,
    total_pairs: int,
    split_name: str,
    out_dir: str,
    src_root: str,
    use_symlinks: bool = True,
):
    """Stream a split from an iterator without materializing all pairs in memory."""
    out_path = Path(out_dir)
    captions_dir = out_path / "captions"
    images_dir = out_path / "images"
    captions_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    image_list_path = out_path / f"{split_name}_list.txt"
    pairs_json_path = out_path / f"{split_name}_pairs.json"
    unique_captions = set() if split_name in ("val", "test") else None
    max_pending = max(1, NUM_WORKERS * 2)
    pending = {}
    next_to_write = 0
    submitted = 0
    first_json = True

    def write_chunk_result(chunk_result, image_list_f, pairs_json_f):
        nonlocal first_json
        for p in chunk_result:
            image_list_f.write(p["unique_name"] + "\n")
            if not first_json:
                pairs_json_f.write(",\n")
            pairs_json_f.write(json.dumps(p, ensure_ascii=False))
            first_json = False
            if unique_captions is not None:
                unique_captions.add(p["caption"])

    with open(image_list_path, 'w', encoding='utf-8') as image_list_f, \
            open(pairs_json_path, 'w', encoding='utf-8') as pairs_json_f, \
            ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        pairs_json_f.write("[\n")
        with tqdm(total=total_pairs, desc=f"Writing {split_name}", unit="it") as pbar:
            for start_idx, chunk in _iter_pair_chunks(pair_iter, WRITE_CHUNK_SIZE):
                pending[submitted] = executor.submit(
                    _write_chunk,
                    chunk,
                    start_idx,
                    split_name,
                    out_path,
                    src_root,
                    use_symlinks,
                )
                submitted += 1
                while len(pending) >= max_pending:
                    future = pending.get(next_to_write)
                    if future is None:
                        break
                    chunk_result = future.result()
                    write_chunk_result(chunk_result, image_list_f, pairs_json_f)
                    pbar.update(len(chunk_result))
                    del pending[next_to_write]
                    next_to_write += 1

            while next_to_write in pending:
                future = pending[next_to_write]
                chunk_result = future.result()
                write_chunk_result(chunk_result, image_list_f, pairs_json_f)
                pbar.update(len(chunk_result))
                del pending[next_to_write]
                next_to_write += 1
        pairs_json_f.write("\n]\n")

    if unique_captions is not None:
        with open(out_path / f"{split_name}_class_names.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(unique_captions)))


def _collect_pairs_legacy_entry(
    entry: Dict,
    train_here: bool,
    active_query_types: List[str],
    train_attr_index: Dict,
    val_attr_index: Dict,
    test_attr_index: Dict,
    train_pairs: List,
    val_pairs: List,
    test_pairs: List,
    stats: Dict,
    include_test: bool,
    emitted_attribute_queries: Set[Tuple],
) -> None:
    """ID-aligned labels: queries and captions at entry level."""
    dataset_name = entry.get("dataset", "") or ""
    queries = entry.get('queries', {})
    attributes = entry.get('attributes', {})
    natural_caption = entry.get('natural_caption', '')
    original_captions = entry.get('original_captions', {})

    for qtype in active_query_types:
        if qtype == 'natural_caption':
            captions = _natural_caption_strings(natural_caption)
            if not captions:
                continue
            query_indices = [0] * len(captions)
        elif qtype == 'original_captions':
            train_orig = original_captions.get('train', [])
            val_orig = original_captions.get('val', [])
            test_orig = original_captions.get('test', [])
            train_caps = []
            for item in train_orig:
                if isinstance(item, list):
                    train_caps.extend(item)
                else:
                    train_caps.append(item)
            val_caps = []
            for item in val_orig:
                if isinstance(item, list):
                    val_caps.extend(item)
                else:
                    val_caps.append(item)
            test_caps = []
            for item in test_orig:
                if isinstance(item, list):
                    test_caps.extend(item)
                else:
                    test_caps.append(item)
            train_images = entry['images'].get('train', [])
            for cap in train_caps:
                if cap and train_here:
                    stats[qtype]['train'] += train_pairs.append_images(train_images, cap, qtype)
            if _entry_allowed_for_val(entry):
                val_images = entry['images'].get('val', [])
                for cap in val_caps:
                    if cap:
                        stats[qtype]['val'] += val_pairs.append_images(val_images, cap, qtype, dataset_name)
            if include_test and _entry_allowed_for_test(entry):
                test_images = entry['images'].get('test', [])
                for cap in test_caps:
                    if cap:
                        stats[qtype]['test'] += test_pairs.append_images(test_images, cap, qtype, dataset_name)
            continue
        else:
            query_list = queries.get(qtype, [])
            captions = []
            query_indices = []
            for idx, q in enumerate(query_list):
                captions.append(q[0] if isinstance(q, list) else q)
                query_indices.append(idx)

        for caption, q_idx in zip(captions, query_indices):
            if not caption:
                continue
            query_pair = queries.get(qtype, [None])[q_idx] if qtype in ("easy", "medium") else None
            if train_here:
                cache_key = (
                    _attribute_query_cache_key("train", qtype, "", caption, attributes, query_pair)
                    if qtype in ("easy", "medium") and USE_ATTRIBUTE_MATCHING
                    else None
                )
                if cache_key is not None and cache_key in emitted_attribute_queries:
                    train_images = []
                else:
                    train_images = get_matching_images_for_query(
                        entry, qtype, q_idx, 'train', train_attr_index,
                    )
                    if cache_key is not None and len(train_images) > 1:
                        emitted_attribute_queries.add(cache_key)
                stats[qtype]['train'] += train_pairs.append_images(train_images, caption, qtype)
            if _entry_allowed_for_val(entry):
                val_images = entry['images'].get('val', [])
                stats[qtype]['val'] += val_pairs.append_images(val_images, caption, qtype, dataset_name)
            if include_test and _entry_allowed_for_test(entry):
                cache_key = (
                    _attribute_query_cache_key("test", qtype, dataset_name, caption, attributes, query_pair)
                    if qtype in ("easy", "medium") and USE_ATTRIBUTE_MATCHING
                    else None
                )
                if cache_key is not None and cache_key in emitted_attribute_queries:
                    test_images = []
                else:
                    test_images = get_matching_images_for_query(
                        entry, qtype, q_idx, 'test', test_attr_index,
                    )
                    if cache_key is not None and len(test_images) > 1:
                        emitted_attribute_queries.add(cache_key)
                stats[qtype]['test'] += test_pairs.append_images(test_images, caption, qtype, dataset_name)


def _original_caption_strings_from_per_image_cell(raw) -> List[str]:
    """Normalize per_image original_captions (list of str, possibly nested) to strings."""
    caps: List[str] = []
    if raw is None:
        return caps
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, list):
                for c in item:
                    if c is not None and str(c).strip():
                        caps.append(str(c).strip())
            elif item is not None and str(item).strip():
                caps.append(str(item).strip())
    elif str(raw).strip():
        caps.append(str(raw).strip())
    return caps


def _collect_pairs_per_image_entry(
    entry: Dict,
    train_here: bool,
    active_query_types: List[str],
    train_attr_index: Dict,
    val_attr_index: Dict,
    test_attr_index: Dict,
    train_pairs: List,
    val_pairs: List,
    test_pairs: List,
    stats: Dict,
    include_test: bool,
    emitted_attribute_queries: Set[Tuple],
) -> None:
    """Image-aligned labels: use each per_image rowâ€™s text and attributes."""
    dataset_name = entry.get("dataset", "") or ""
    for pi in entry.get("per_image") or []:
        spl = pi.get("split")
        if spl not in ("train", "val", "test"):
            continue
        if spl == "test" and not (include_test and _entry_allowed_for_test(entry)):
            continue
        img = pi.get("image_path")
        if not img:
            continue

        for qtype in active_query_types:
            if qtype == "original_captions":
                for cap in _original_captions_for_per_image_row(entry, pi, spl):
                    if spl == "train" and train_here:
                        train_pairs.append((img, cap, qtype))
                        stats[qtype]["train"] += 1
                    elif spl == "val" and _entry_allowed_for_val(entry):
                        val_pairs.append((img, cap, qtype, dataset_name))
                        stats[qtype]["val"] += 1
                    elif spl == "test":
                        test_pairs.append((img, cap, qtype, dataset_name))
                        stats[qtype]["test"] += 1
                continue

            if qtype == "natural_caption":
                for cap in _natural_captions_for_per_image_row(entry, pi):
                    if spl == "train" and train_here:
                        train_pairs.append((img, cap, qtype))
                        stats[qtype]["train"] += 1
                    elif spl == "val" and _entry_allowed_for_val(entry):
                        val_pairs.append((img, cap, qtype, dataset_name))
                        stats[qtype]["val"] += 1
                    elif spl == "test":
                        test_pairs.append((img, cap, qtype, dataset_name))
                        stats[qtype]["test"] += 1
                continue

            if qtype == "hard":
                for cap in _hard_query_captions(_queries_for_per_image_row(entry, pi).get("hard", [])):
                    if spl == "train" and train_here:
                        train_pairs.append((img, cap, qtype))
                        stats[qtype]["train"] += 1
                    elif spl == "val" and _entry_allowed_for_val(entry):
                        val_pairs.append((img, cap, qtype, dataset_name))
                        stats[qtype]["val"] += 1
                    elif spl == "test":
                        test_pairs.append((img, cap, qtype, dataset_name))
                        stats[qtype]["test"] += 1
                continue

            query_list = _queries_for_per_image_row(entry, pi).get(qtype, [])
            for q_idx, q in enumerate(query_list):
                cap = q[0] if isinstance(q, list) else q
                if not cap:
                    continue
                cap = cap.strip() if isinstance(cap, str) else str(cap).strip()
                if not cap:
                    continue
                if spl == "train" and train_here:
                    cache_key = (
                        _attribute_query_cache_key("train", qtype, "", cap, pi.get("attributes") or {}, q)
                        if USE_ATTRIBUTE_MATCHING
                        else None
                    )
                    if cache_key is not None and cache_key in emitted_attribute_queries:
                        mimgs = []
                    else:
                        mimgs = get_matching_images_for_query_per_image(
                            pi, qtype, q_idx, train_attr_index, q,
                        )
                        if cache_key is not None and not (len(mimgs) == 1 and mimgs[0] == img):
                            emitted_attribute_queries.add(cache_key)
                    stats[qtype]["train"] += train_pairs.append_images(mimgs, cap, qtype)
                elif spl == "val" and _entry_allowed_for_val(entry):
                    cache_key = (
                        _attribute_query_cache_key("val", qtype, dataset_name, cap, pi.get("attributes") or {}, q)
                        if USE_ATTRIBUTE_MATCHING
                        else None
                    )
                    if cache_key is not None and cache_key in emitted_attribute_queries:
                        mimgs = []
                    else:
                        mimgs = get_matching_images_for_query_per_image(
                            pi, qtype, q_idx, val_attr_index, q,
                        )
                        if cache_key is not None and not (len(mimgs) == 1 and mimgs[0] == img):
                            emitted_attribute_queries.add(cache_key)
                    stats[qtype]["val"] += val_pairs.append_images(mimgs, cap, qtype, dataset_name)
                elif spl == "test":
                    cache_key = (
                        _attribute_query_cache_key("test", qtype, dataset_name, cap, pi.get("attributes") or {}, q)
                        if USE_ATTRIBUTE_MATCHING
                        else None
                    )
                    if cache_key is not None and cache_key in emitted_attribute_queries:
                        mimgs = []
                    else:
                        mimgs = get_matching_images_for_query_per_image(
                            pi, qtype, q_idx, test_attr_index, q,
                        )
                        if cache_key is not None and not (len(mimgs) == 1 and mimgs[0] == img):
                            emitted_attribute_queries.add(cache_key)
                    stats[qtype]["test"] += test_pairs.append_images(mimgs, cap, qtype, dataset_name)


def prepare_nvclip_data(
    nvidia_pas_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_symlinks: Optional[bool] = None,
    only_natural_and_original: Optional[bool] = None,
    exclude_natural_from_aug: Optional[bool] = None,
    val_sample_fraction: Optional[float] = None,
    include_test: bool = False,
    clean_output: bool = False,
):
    """Main function to prepare TAO CLIP fine-tuning data from NVIDIA_PAS.

    Args:
        nvidia_pas_root: Path to NVIDIA_PAS root (contains labels.json).
        output_dir: Output directory for TAO FT layout.
        use_symlinks: If True, symlink images; if False, copy. Default: module USE_SYMLINKS.
        only_natural_and_original: If True, only natural_caption and original_captions (no easy/medium/hard).
        exclude_natural_from_aug: If True, entries whose dataset ends with "_Aug" use only easy/medium/hard.
        include_test: If True, also write full test_list/test_pairs/test_class_names for PAS KPI evaluation.
        clean_output: If True, delete output_dir before writing to avoid mixing with partial previous runs.
        val_sample_fraction: Fraction of deduped val pairs to keep per (dataset Ã— query_type); default VAL_SAMPLE_FRACTION.
    """
    nvidia_pas_root = nvidia_pas_root if nvidia_pas_root is not None else NVIDIA_PAS_ROOT
    output_dir = output_dir if output_dir is not None else OUTPUT_DIR
    if nvidia_pas_root is None:
        raise ValueError("nvidia_pas_root must be provided")
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    use_symlinks = use_symlinks if use_symlinks is not None else USE_SYMLINKS
    only_natural_and_original = (
        only_natural_and_original
        if only_natural_and_original is not None
        else ONLY_NATURAL_AND_ORIGINAL_CAPTIONS
    )
    exclude_natural_from_aug = (
        exclude_natural_from_aug
        if exclude_natural_from_aug is not None
        else EXCLUDE_NATURAL_FROM_AUG_DATASETS
    )
    val_sample_fraction = (
        val_sample_fraction if val_sample_fraction is not None else VAL_SAMPLE_FRACTION
    )

    active_query_types = (
        NATURAL_AND_ORIGINAL_QUERY_TYPES
        if only_natural_and_original
        else QUERY_TYPES
    )

    output_path = Path(output_dir).expanduser()
    source_path = Path(nvidia_pas_root).expanduser()
    if clean_output and output_path.exists():
        if output_path.resolve() == source_path.resolve():
            raise ValueError(
                f"Refusing to clean output_dir because it is the same as nvidia_pas_root: {output_dir}"
            )
        print(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_path)

    print("=" * 60)
    print("NVIDIA_PAS â†’ TAO CLIP Fine-tuning Data Preparation")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  - Source: {nvidia_pas_root}")
    print(f"  - Output: {output_dir}")
    print(f"  - Pair store: {output_path / PAIR_STORE_NAME} (disk-backed dedup/write)")
    print(f"  - Query types: {active_query_types}")
    if only_natural_and_original:
        print("    (Only natural_caption / original_captions; easy, medium, and hard omitted.)")
    if exclude_natural_from_aug:
        print(
            f"    (Aug datasets (*{AUG_DATASET_SUFFIX}): only {STRUCTURED_QUERY_TYPES}; "
            "natural_caption / original_captions omitted for those entries.)"
        )
    print(
        f"  - Train: omit {EXCLUDED_TRAIN_DATASETS} (CUHK/ICFG base names; val still uses their val images)."
    )
    print(f"  - Val: omit {EXCLUDED_VAL_DATASETS} (e.g. keep retrieval cost manageable).")
    print(
        f"  - Test output: {'enabled' if include_test else 'disabled'} "
        f"(datasets: {TEST_DATASETS})"
    )
    print(
        f"  - Val subsample (after dedup): {val_sample_fraction:.0%} per dataset Ã— query_type stratum "
        "(preserves mix; excludes PA-100K before sampling)."
    )
    print(f"  - Attribute matching for easy/medium: {USE_ATTRIBUTE_MATCHING}")
    if USE_ATTRIBUTE_MATCHING:
        print("    (Easy/medium: text-image pairs by ATTRIBUTES; multiple IDs with same attributes match the same query.)")
    else:
        print("    (Easy/medium: 1-1 person ID mapping only.)")
        print("    WARNING: Evaluation (evaluate_nvidia_pas.py) uses ATTRIBUTE-based matching for easy/medium.")
        print("    For consistent train/eval behavior, set USE_ATTRIBUTE_MATCHING = True and re-run.")

    print(f"\nLoading labels from {nvidia_pas_root}/labels.json...")
    with open(f"{nvidia_pas_root}/labels.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    entries = data['entries']
    print(f"  âœ“ Loaded {len(entries)} entries")
    per_image_n = sum(1 for e in entries if _entry_has_aligned_per_image(e))
    print(
        f"  Entries with aligned per_image (image-level queries/attributes): "
        f"{per_image_n} / {len(entries)}"
    )
    if data.get("metadata", {}).get("datasets_included"):
        print(f"  metadata.datasets_included: {data['metadata']['datasets_included']}")

    # Sanity-check: entry counts by dataset (labels.json uses entry["dataset"])
    dataset_counts = defaultdict(int)
    for e in entries:
        dataset_counts[e.get("dataset", "")] += 1
    print(f"  Entries by dataset: {dict(dataset_counts)}")
    train_included_count = sum(
        c for ds, c in dataset_counts.items() if ds not in EXCLUDED_TRAIN_DATASETS
    )
    train_excluded_count = sum(
        dataset_counts[d] for d in EXCLUDED_TRAIN_DATASETS if d in dataset_counts
    )
    print(f"  Entries contributing to train (not in {EXCLUDED_TRAIN_DATASETS}): {train_included_count}")
    print(f"  Entries train-excluded (CUHK/ICFG; val still emitted if not in val-excluded): {train_excluded_count}")
    val_excluded_count = sum(
        dataset_counts[d] for d in EXCLUDED_VAL_DATASETS if d in dataset_counts
    )
    print(f"  Entries val-excluded ({EXCLUDED_VAL_DATASETS}): {val_excluded_count}")
    test_count = sum(dataset_counts[d] for d in TEST_DATASETS if d in dataset_counts)
    print(f"  Entries in optional test datasets ({TEST_DATASETS}): {test_count}")
    _warn_on_per_image_text_fallbacks(entries, active_query_types)

    os.makedirs(f"{output_dir}/captions", exist_ok=True)

    need_attr_index = any(q in active_query_types for q in ("easy", "medium"))
    if need_attr_index:
        print("\nBuilding attribute indices (legacy + per-image)...")
        train_attr_index = merge_attribute_indices(
            build_attribute_index_legacy(entries, "train"),
            build_attribute_index_per_image(entries, "train"),
        )
        val_attr_index = merge_attribute_indices(
            build_attribute_index_legacy(entries, "val"),
            build_attribute_index_per_image(entries, "val"),
        )
        test_attr_index = (
            merge_attribute_indices(
                build_attribute_index_legacy(entries, "test"),
                build_attribute_index_per_image(entries, "test"),
            )
            if include_test
            else {}
        )
    else:
        print("\nSkipping attribute indices (no easy/medium query types).")
        train_attr_index = {}
        val_attr_index = {}
        test_attr_index = {}

    pair_store = _DiskPairStore(output_path / PAIR_STORE_NAME)
    train_pairs = pair_store.appender("train")
    val_pairs = pair_store.appender("val")
    test_pairs = pair_store.appender("test")
    stats = {qtype: {'train': 0, 'val': 0, 'test': 0} for qtype in active_query_types}
    emitted_attribute_queries: Set[Tuple] = set()

    aug_dataset_names = sorted({
        e.get("dataset", "") for e in entries if _is_aug_dataset(e.get("dataset", "") or "")
    })
    if exclude_natural_from_aug:
        if aug_dataset_names:
            print(f"  Aug datasets ({AUG_DATASET_SUFFIX} suffix, structured queries only): {aug_dataset_names}")
        else:
            warnings.warn(
                f"exclude_natural_from_aug=True but no entry has dataset name ending with {AUG_DATASET_SUFFIX!r}; "
                "all entries will use the full active query-type list."
            )

    print("\nProcessing entries...")
    for entry in tqdm(entries, desc="Processing"):
        train_here = _entry_allowed_for_train(entry)
        entry_query_types = _query_types_for_entry(
            entry, active_query_types, exclude_natural_from_aug,
        )
        if not entry_query_types:
            continue
        if _entry_has_aligned_per_image(entry):
            _collect_pairs_per_image_entry(
                entry,
                train_here,
                entry_query_types,
                train_attr_index,
                val_attr_index,
                test_attr_index,
                train_pairs,
                val_pairs,
                test_pairs,
                stats,
                include_test,
                emitted_attribute_queries,
            )
        else:
            _collect_pairs_legacy_entry(
                entry,
                train_here,
                entry_query_types,
                train_attr_index,
                val_attr_index,
                test_attr_index,
                train_pairs,
                val_pairs,
                test_pairs,
                stats,
                include_test,
                emitted_attribute_queries,
            )

    print("\n" + "=" * 60)
    print("Query Type Statistics")
    print("=" * 60)
    print(f"{'Query Type':<20} {'Train Pairs':>15} {'Val Pairs':>15} {'Test Pairs':>15}")
    print("-" * 60)
    for qtype in active_query_types:
        print(
            f"{qtype:<20} {stats[qtype]['train']:>15,} "
            f"{stats[qtype]['val']:>15,} {stats[qtype]['test']:>15,}"
        )
    print("-" * 60)
    print(f"{'TOTAL':<20} {len(train_pairs):>15,} {len(val_pairs):>15,} {len(test_pairs):>15,}")
    print(
        "(Val/test counts above are raw before dedup / stratified subsampling; "
        "see â€˜Val split sizesâ€™ below for final val size.)"
    )

    print("\nFinalizing disk-backed pair store (dedup by split/image/caption)...")
    pair_store.flush()
    train_count = pair_store.count("train")
    val_dedup_count = pair_store.count("val")
    test_count = pair_store.count("test") if include_test else 0

    rng = random.Random(42)
    pair_store.sample_val(val_sample_fraction, rng)
    val_count = pair_store.count("val", selected_val=True)

    print("\nVal split sizes (after dedup â†’ after stratified subsample):")
    print(f"  Deduped val pairs: {val_dedup_count:,}")
    print(f"  Written val pairs: {val_count:,} ({val_sample_fraction:.0%} per stratum)")
    val_final_by_q = pair_store.count_by_query_type("val", selected_val=True)
    print("  Final val by query_type:", dict(val_final_by_q))
    if include_test:
        test_final_by_q = pair_store.count_by_query_type("test")
        print("\nTest split sizes (after dedup; no subsampling):")
        print(f"  Written test pairs: {test_count:,}")
        print("  Final test by query_type:", dict(test_final_by_q))

    print("\nWriting output files...")
    _write_split_from_iter(
        pair_store.iter_pairs("train"),
        train_count,
        "train",
        output_dir,
        nvidia_pas_root,
        use_symlinks=use_symlinks,
    )
    _write_split_from_iter(
        pair_store.iter_pairs("val", selected_val=True),
        val_count,
        "val",
        output_dir,
        nvidia_pas_root,
        use_symlinks=use_symlinks,
    )
    if include_test:
        _write_split_from_iter(
            pair_store.iter_pairs("test"),
            test_count,
            "test",
            output_dir,
            nvidia_pas_root,
            use_symlinks=use_symlinks,
        )
    pair_store.close(remove_db=True)

    print("\n" + "=" * 60)
    print("âœ“ Data preparation complete!")
    print("=" * 60)
    print("\nUse in experiment spec (keys match custom dataloader / CLIPDataPathConfig):")
    print("  dataset.train.datasets[0]:")
    print(f"    image_dir: {output_dir}/images")
    print(f"    image_list_file: {output_dir}/train_list.txt")
    print(f"    caption_dir: {output_dir}/captions")
    print("    caption_file_suffix: .txt")
    print("  dataset.val.datasets[0]:")
    print(f"    image_dir: {output_dir}/images")
    print(f"    image_list_file: {output_dir}/val_list.txt")
    print(f"    caption_dir: {output_dir}/captions")
    print("    caption_file_suffix: .txt")
    print(f"  dataset.val.classnames_file: {output_dir}/val_class_names.txt")
    if include_test:
        print("\nFor PAS benchmark evaluation, point the TAO eval split at:")
        print(f"  dataset.val.datasets[0].image_list_file: {output_dir}/test_list.txt")
        print(f"  dataset.val.classnames_file: {output_dir}/test_class_names.txt")


def convert_nvidia_pas_to_tao_clip(cfg=None, verbose=False):
    """Convert NVIDIA_PAS labels to the TAO CLIP custom dataloader layout."""
    if cfg is None:
        raise ValueError("config is not provided")

    pas_cfg = cfg.nvidia_pas
    output_dir = cfg.results_dir
    if not output_dir:
        raise ValueError("results_dir must be provided")

    if verbose:
        print("Running NVIDIA_PAS to TAO_CLIP conversion")

    prepare_nvclip_data(
        nvidia_pas_root=pas_cfg.root,
        output_dir=output_dir,
        use_symlinks=pas_cfg.use_symlinks,
        only_natural_and_original=pas_cfg.only_natural_and_original,
        exclude_natural_from_aug=pas_cfg.exclude_natural_from_aug,
        val_sample_fraction=pas_cfg.val_sample_fraction,
        include_test=pas_cfg.include_test,
        clean_output=pas_cfg.clean_output,
    )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Convert NVIDIA_PAS to TAO CLIP fine-tuning format (ImageTextDataset layout)."
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="NVIDIA_PAS root (labels.json).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory.",
    )
    p.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output directory before writing (recommended after an interrupted prep run).",
    )
    p.add_argument(
        "--no-symlinks",
        action="store_true",
        help="Copy images instead of symlinking (portable output).",
    )
    p.add_argument(
        "--only-natural-and-original",
        action="store_true",
        help="Only include natural_caption and original_captions (omit easy, medium, and hard).",
    )
    p.add_argument(
        "--exclude-natural-from-aug",
        action="store_true",
        help=(
            f"For entries whose dataset name ends with {AUG_DATASET_SUFFIX!r}, omit natural_caption "
            "and original_captions (easy/medium/hard only). Non-aug entries are unchanged."
        ),
    )
    p.add_argument(
        "--only-hard-natural-original",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Fraction of deduped validation pairs to write per (dataset Ã— query_type) stratum "
            f"(default: {VAL_SAMPLE_FRACTION}). Use 1.0 to disable subsampling."
        ),
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help=(
            "Also write test_list.txt, test_pairs.json, and test_class_names.txt for "
            f"PAS KPI datasets {TEST_DATASETS}."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.only_hard_natural_original:
        warnings.warn(
            "--only-hard-natural-original is deprecated; use --only-natural-and-original "
            "(natural_caption + original_captions only; hard queries excluded).",
            DeprecationWarning,
            stacklevel=2,
        )
    only_no = args.only_natural_and_original or args.only_hard_natural_original
    prepare_nvclip_data(
        nvidia_pas_root=args.source,
        output_dir=args.output,
        use_symlinks=not args.no_symlinks if args.no_symlinks else None,
        only_natural_and_original=only_no if only_no else None,
        exclude_natural_from_aug=args.exclude_natural_from_aug if args.exclude_natural_from_aug else None,
        val_sample_fraction=args.val_fraction,
        include_test=args.include_test,
        clean_output=args.clean_output,
    )
