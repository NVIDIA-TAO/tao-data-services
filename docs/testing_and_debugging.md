# Testing and Debugging

This repository mixes static checks, unit tests, command dispatch tests,
container checks, and GPU- and data-sensitive workflows. Pick the smallest test that
covers your change.

## Static Checks

CI runs static checks on pull requests through GitHub Actions
(`.github/workflows/static-tests.yml`), which executes the repository's pre-commit
hooks against the changed files:

* SPDX license headers (`.github/hooks/check_license_header.py`)
* `pylint` (`.pylintrc`), `pydocstyle`, and `flake8`, scoped to
  `nvidia_tao_ds/`
* The generated README drift check
  (`tools/update_readme_supported_commands.py --check`)

CI skips the `trufflehog` and `dependency-guard` hooks (secret scanning runs
in the separate `secret-scan.yml` workflow); both still run in the local
pre-commit hook, and `dependency-guard` requires explicit acknowledgment for
requirements and Docker file changes.

Reproduce locally:

```sh
pip install pre-commit
pre-commit install
pre-commit run --from-ref origin/main --to-ref HEAD
```

Separate workflows enforce DCO sign-off on every commit (`dco.yml`), secret
scanning (`secret-scan.yml`), and PR title format (`pr-title.yml`).
Functional (GPU) tests run through the externally triggered `blossom-ci.yml`
workflow, not on every push.

## Documentation-Only Validation

For documentation-only changes, run these checks:

```sh
python tools/update_readme_supported_commands.py --check
python -m py_compile tools/update_readme_supported_commands.py
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/*.py .pre-commit-config.yaml
rg -n "TBD|PLACEHOLDER|example\\.com" README.md docs
```

GPU, Docker build, private checkpoint, NGC, and full dataset tests are outside
the normal blast radius for documentation-only changes.

## Test Environment Setup

The base development image has pytest and the runtime dependencies
(`docker/requirements-pip.txt`), but **not** the TAO packages themselves. Set
up once per container:

```sh
# On the host
git submodule update --init          # tao-core/ and tao-pytorch/ are empty otherwise
source scripts/envsetup.sh
tao_ds --gpus all -- bash

# Inside the container (repository is mounted at /workspace)
pip install /workspace/tao-core/.    # provides nvidia_tao_core (status callbacks, api_utils)
export PYTHONPATH=/workspace/tao-pytorch:$PYTHONPATH   # for tests that import nvidia_tao_pytorch
pytest tests/test_config_modules.py -q                 # smoke check
```

Notes:

* `nvidia_tao_core` is not in `docker/requirements-pip.txt` (only its
  transitive dependencies are), so tests fail with import errors until you
  install the submodule. This is the step the README historically missed.
* `nvidia_tao_ds` itself is importable because the launcher sets
  `PYTHONPATH=/workspace`; run `python setup.py develop` instead if you need
  the console commands.
* Tests that exercise Grounding DINO or TAO CLIP (`test_text2box.py`,
  `tests/mining/test_image_embeddings.py`) import `nvidia_tao_pytorch` from
  the submodule; the mining tests fall back to stubs when it is absent.
* GPU- and data-heavy tests additionally need the private scratch datasets
  mounted (paths under `/media/scratch_metropolis2/tao_ci/...`).

## Targeted Pytest Map

| Change area | Suggested tests |
| :--- | :--- |
| Config package moves or default spec support | `pytest tests/test_config_modules.py -q` |
| Shared launcher / exit-code behavior | `pytest tests/test_entrypoint_exit_code.py -q` |
| Embedding logic | `pytest tests/mining/test_image_embeddings.py tests/mining/test_text_embeddings.py -q` |
| Nearest-neighbor mining | `pytest tests/mining/test_nearest_neighbors.py tests/mining/test_unique_neighbor_matching.py -q` |
| Annotation merge/slice | `pytest tests/test_merger_slicer.py -q` |
| COCO, KITTI, and ODVG conversion | `pytest tests/test_coco_kitti_conversion.py tests/test_coco_odvg_conversion.py -q` |
| AICity / PAS conversions | `pytest tests/test_aicity_ovpkl_conversion.py tests/test_nvidia_paidf_pas_to_tao_clip_conversion.py -q` |
| QA to LLaVA conversion | `pytest tests/test_qa_to_llava_annotation.py tests/test_llava_merger.py -q` |
| Analytics | `pytest tests/test_data_analytics.py -q` |
| Auto-label VLM workflows | `pytest tests/autolabel -q` |
| Auto-label Grounding DINO | `pytest tests/test_text2box.py -q` |
| Gap analysis | `pytest tests/test_od_gap_analysis.py tests/test_vcn_aoi.py tests/test_vlm_bcq.py -q` |
| Logging changes | `pytest tests/test_dual_logging.py -q` |
| Dataset loading helpers | `pytest tests/test_dataset_loading.py -q` |

There is no `conftest.py` or pytest configuration file. Two skip conventions
matter:

* GPU- and data-heavy tests (`test_data_analytics.py`, `test_augment.py`) skip
  themselves when `CI_PROJECT_DIR` is set, a GitLab-era guard that GitHub
  Actions does not set. They need the private scratch datasets mounted.
* Some analytics tests skip unless `WANDB_API_KEY` is set.

A few tests stub or skip `nvidia_tao_core` imports; initialize the submodules
before running the full suite.

## Common Failures

| Symptom | Likely cause | Where to check |
| :--- | :--- | :--- |
| `Experiment spec file was not found` | The standard entry point ran without a valid `-e` path. | `nvidia_tao_ds/core/entrypoint/entrypoint.py` |
| Hydra cannot find a config | The script `config_name` does not match the YAML name or config path. | The target `scripts/*.py` `@hydra_runner(...)`; for example, `augmentation generate` defaults to `kitti.yaml` |
| `nvidia-smi` assertion failure | Requested `num_gpus` exceeds visible GPUs, or the host has no GPU (the launcher calls `nvidia-smi` unconditionally, even for CPU subtasks). | `launch()` GPU logic in the shared entrypoint |
| Command reports FAIL despite exit 0 | The launcher also reads the last record of `results_dir/status.json`. | `_status_reports_failure` in the shared entrypoint |
| New GPU subtask runs on one GPU only | Multi-GPU parsing is keyed on the subtask name `generate`. | `launch(multigpu_support=...)` |
| Import errors for `nvidia_tao_core` / `nvidia_tao_pytorch` | Uninitialized submodules. | `git submodule update --init`; in containers `envsetup.sh` sets `PYTHONPATH` |
| `default_specs` rejects a module | Only flat functions are supported; mining and RCCA are not. | `nvidia_tao_ds/core/utils/default_specs.py` |
| Docker pull or inspect fails | Local tag missing or `docker/manifest.json` digest stale/inaccessible. | `runner/tao_ds.py`, `docker/manifest.json` |
| Video write fails or codec missing | The base image enforces an LGPL codec allow-list (VP9/mjpeg encode only). | `docker/Dockerfile`, `core/utils/video_utils.py` |
| API action missing | Installed console scripts or tao-core module mappings do not expose it. | `setup.py`, tao-core `api_utils.module_utils` |
| Weights and Biases tests skip | `WANDB_API_KEY` is not set. | `tests/test_data_analytics.py` |

## Debugging Runtime Commands

Run a command through the launcher:

```sh
tao_ds --gpus all -- annotations convert -e nvidia_tao_ds/annotations/experiment_specs/annotations.yaml
```

The shared entrypoint launches each subtask as a fresh Python subprocess, so
breakpoints set in a `scripts/*.py` are never hit through the console command.
Debug in-process by running the script directly:

```sh
python nvidia_tao_ds/<function>/scripts/<subtask>.py \
  --config-path /abs/path/to/spec/dir --config-name <spec_name>
```

Command-line Hydra overrides follow the experiment specification, for example,
`results_dir=/results data.input_format=COCO`.
