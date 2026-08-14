# Agent Onboarding

Use this guide to get oriented without disturbing a user-owned worktree. For a
full directory walk, read the [Codebase tour](codebase_tour.md).

## Mental Model

`scripts/envsetup.sh` sets `NV_TAO_DS_TOP` and defines a shell function named
`tao_ds`. That function runs `runner/tao_ds.py`, which starts the base Docker
image from `docker/manifest.json`, mounts this source tree at `/workspace`, and
executes any command passed after `--`.

Inside the container, `setup.py` installs one console script per service:
`annotations`, `augmentation`, `analytics`, `auto_label`, `image`,
`gap_analysis`, `tmm`, and `embedding`. All of them use the shared command dispatcher in
`nvidia_tao_ds/core/entrypoint/entrypoint.py` to discover subtasks from
`scripts/`, require `-e/--experiment_spec_file`, and run the selected script as
a fresh Python subprocess under Hydra.

```text
console command (setup.py)
  -> nvidia_tao_ds/core/entrypoint/entrypoint.py
  -> nvidia_tao_ds/<service>/scripts/<subtask>.py   (fresh subprocess)
  -> Hydra spec validated against nvidia_tao_ds/config/<service>/
  -> service logic (conversion, DALI, model inference, mining, ...)
```

## First Pass

Run these before editing:

```sh
pwd
git -c filter.lfs.process= -c filter.lfs.required=false status --short --branch
git remote -v
find . -maxdepth 2 -type d
sed -n '1,220p' README.md
rg -n "console_scripts|entry_points" setup.py
rg -n "ArgumentParser|manifest.json|--gpus|--tag|--run_as_user" runner scripts docker release
rg -n "hydra_runner|default_specs|get_subtasks|entrypoint|console_scripts" nvidia_tao_ds setup.py
ls .github/workflows/
sed -n '1,80p' .pre-commit-config.yaml
```

Treat `tao-core/` and `tao-pytorch/` as submodules in this checkout. They may
be dirty for reasons unrelated to your task. Do not reset, update, or rewrite
them unless the task explicitly requires it. Initialize them before running
anything:

```sh
git submodule update --init
```

## Source Truths

| Question | Source of truth |
| :--- | :--- |
| Which package commands exist? | `setup.py` `console_scripts` |
| Which host launcher flags exist? | `runner/tao_ds.py` `parse_cli_args` |
| Which base image is pulled? | `docker/manifest.json` |
| Which subtasks exist? | `nvidia_tao_ds/<service>/scripts/*.py` |
| Which example specs exist? | `nvidia_tao_ds/<service>/experiment_specs/*.yaml` |
| Which specification template does a subtask load? | The script's `@hydra_runner(config_name=...)`, not the subtask name |
| Which dataclass schema is used? | `nvidia_tao_ds/config/<service>/...` (`analytics` is the configuration package for `data_analytics/`) |
| Which static tests run in CI? | `.pre-commit-config.yaml` via `.github/workflows/static-tests.yml` |
| Which README content is generated? | `tools/update_readme_supported_commands.py` |

## Common Agent Questions

| Question | Where to look |
| :--- | :--- |
| Where is the shared dispatcher behavior (GPU handling, subprocess, telemetry)? | `nvidia_tao_ds/core/entrypoint/entrypoint.py` |
| Why does a CPU command need a GPU host? | The dispatcher calls `nvidia-smi` unconditionally |
| Why does my new subtask ignore `num_gpus`? | Multi-GPU is keyed on the literal subtask name `generate` |
| Why is there no `status.json` for a subtask? | The script lacks `@monitor_status`; refer to the coverage list in the [Codebase tour](codebase_tour.md) |
| Which model does auto-label use? | `auto_label/scripts/generate.py` dispatches on `cfg.autolabel_type` |
| Where do LLM/VLM calls happen? | `nvidia_tao_ds/core/llm_clients/` |
| How does the API relate to the CLI? | Dev-mode Flask app versus the tao-core microservice; refer to [Architecture](architecture.md) |

## Worktree Safety

Before editing, capture the status and decide which files you own. Avoid broad
cleanup. Do not remove user-created tests, local examples, cache directories
outside your own generated outputs, or submodule changes unless the user asks.

## Targeted Checks

For documentation and generated README changes:

```sh
python tools/update_readme_supported_commands.py --check
python -m py_compile tools/update_readme_supported_commands.py
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/*.py .pre-commit-config.yaml
rg -n "TBD|PLACEHOLDER|example\\.com" README.md docs
```

For source-adjacent command or configuration changes, run the same static
checks CI runs (`pre-commit run` on the changed files) and add focused pytest
runs from [Testing and debugging](testing_and_debugging.md). GPU, Docker, private
checkpoint, NGC, and full dataset tests are usually outside a documentation-only change.

All commits need a DCO sign-off (`git commit -s`); CI enforces it.
