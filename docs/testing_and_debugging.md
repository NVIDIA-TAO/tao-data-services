# Testing And Debugging

This repository mixes static checks, unit tests, command dispatch tests,
container checks, and GPU/data-sensitive workflows. Pick the smallest test that
covers your change.

## GitLab Static Tests

`.gitlab-ci.yml` runs the `static_tests` job in the TAO Data Services base
image. The job now checks generated README drift before running
`ci/run_static_tests.py`.

`ci/run_static_tests.py` runs:

| Tool | Scope |
| :--- | :--- |
| `pylint --rcfile .pylintrc` | Modules listed in `ci/utils.py` `TEST_MODULES`. |
| `pydocstyle --ignore=D4,D200,D203,D205,D210,D212,D213,D301,D400,D401` | Same module list. |
| `flake8 --ignore=E24,W504,E501` | Same module list. |

`ci/utils.py` resolves the Docker image from `docker/manifest.json` for local
static runs. In CI it runs directly in the job image.

## Docs-Only Validation

```sh
python tools/update_readme_supported_commands.py --check
python -m py_compile tools/update_readme_supported_commands.py
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/*.py .pre-commit-config.yaml .gitlab-ci.yml
rg -n "TBD|PLACEHOLDER|example\\.com" README.md docs
```

GPU, Docker build, private checkpoint, NGC, and full dataset tests are outside
the normal blast radius for documentation-only changes.

## Targeted Pytest Map

| Change Area | Suggested Tests |
| :--- | :--- |
| Config package moves or default spec support | `pytest tests/test_config_modules.py -q` |
| Mining dispatchers | `pytest tests/mining/test_mining_entrypoints.py -q` |
| Embedding logic | `pytest tests/mining/test_image_embeddings.py -q` |
| Nearest-neighbor mining | `pytest tests/mining/test_nearest_neighbors.py -q` |
| Annotation merge/slice | `pytest tests/test_merger_slicer.py -q` |
| COCO/KITTI conversion | `pytest tests/test_coco_kitti_conversion.py tests/test_coco_odvg_conversion.py -q` |
| AICity conversion | `pytest tests/test_aicity_ovpkl_conversion.py -q` |
| SDU base-image upgrade | `pytest tests/test_sdu_runtime.py tests/test_aicity_ovpkl_conversion.py tests/test_sparse4d_*.py -q` |
| QA to LLaVA conversion | `pytest tests/test_qa_to_llava_annotation.py tests/test_llava_merger.py -q` |
| Analytics | `pytest tests/test_data_analytics.py -q` |
| Auto-label prompt and parsing logic | `pytest tests/autolabel -q` |
| Logging changes | `pytest tests/test_dual_logging.py tests/test_baselogger_recursion.py -q` |

`ci/run_functional_tests.py` runs `pytest tests -v --color=yes`. Use it when a
change spans multiple domains and the environment has the needed dependencies.

The SDU runtime test requires the installed distribution to match the exact pin
in `docker/requirements-pip.txt`. It intentionally fails on an old base image,
even if the legacy SDU API still passes conversion tests. Rebuild and validate
each supported architecture, then update both `docker/manifest.json` and
`release/docker/Dockerfile.release` to the published image digests. The Dockerfile
installs SDU separately with `--no-deps` to preserve the curated TAO package set;
passing API tests does not imply that SDU's declared dependency constraints are
all satisfied. Retain the package inventory and dependency-check output for
Infra review.

## Common Failures

| Symptom | Likely Cause | Where To Check |
| :--- | :--- | :--- |
| `Experiment spec file was not found` | Standard entrypoint was called without a valid `-e` path. | `nvidia_tao_ds/core/entrypoint/entrypoint.py` |
| Hydra cannot find a config | The script `config_name` does not match the YAML name or config path. | The target `scripts/*.py` `@hydra_runner(...)` |
| `nvidia-smi` assertion failure | Requested `num_gpus` exceeds visible GPUs. | `launch()` GPU override logic and spec `gpu_ids` |
| Docker pull or inspect fails | Local tag is missing or `docker/manifest.json` digest is stale/inaccessible. | `runner/tao_ds.py`, `docker/manifest.json` |
| Default spec generation rejects a module | Config package shape is not supported by `default_specs.py`. | `nvidia_tao_ds/core/utils/default_specs.py` |
| API action missing | Installed console scripts or `tao-core` module mappings do not expose it. | `setup.py`, `tao-core/nvidia_tao_core/api_utils/module_utils.py` |
| W&B tests skip | `WANDB_API_KEY` is not set. | `tests/test_data_analytics.py` |

## Debugging Runtime Commands

Print the Docker command without hiding what will run:

```sh
tao_ds --gpus all --volume "$PWD:/workspace" -- annotations convert -e nvidia_tao_ds/annotations/experiment_specs/annotations.yaml
```

For standard entrypoints, command-line Hydra overrides follow the experiment
spec. For mining and RCCA direct dispatchers, pass Hydra args directly to the
subtask script through the dispatcher.

The AICity conversion tests include an optional TAO PyTorch integration check.
With a compatible TAO PyTorch checkout on `PYTHONPATH`, the generated unlabeled
JPEG/HDF5 PKLs, Data Services lazy index, and RT-DETR caches are passed through
the real Sparse4D dataset, image/depth transforms, and cache loader. Minimal
Data Services environments without TAO PyTorch or with an older Sparse4D runtime
that lacks `lazy_load` or `LoadRTDETR2D` skip those two integration cases; the
standalone conversion and CLI tests still run. In the combined environment,
require those capabilities explicitly so a stale runtime cannot silently skip
validation:

```sh
pytest tests/test_aicity_ovpkl_conversion.py -q --require-sparse4d-runtime
```

Once the required capabilities are present, dataset and transform failures
(including the annotation-free `gt_boxes=None` route) remain test failures.
