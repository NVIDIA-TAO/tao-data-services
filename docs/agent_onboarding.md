# Agent Onboarding

Use this guide to get oriented without disturbing a user-owned worktree.

## First Pass

Run these before editing:

```sh
pwd
git -c filter.lfs.process= -c filter.lfs.required=false status --short --branch
git remote -v
find . -maxdepth 2 -type d
sed -n '1,220p' README.md
sed -n '1,220p' .gitlab-ci.yml
rg -n "console_scripts|entry_points" setup.py
rg -n "ArgumentParser|manifest.json|--gpus|--tag|--run_as_user" runner scripts docker release
rg -n "hydra_runner|default_specs|get_subtasks|entrypoint|console_scripts" nvidia_tao_ds setup.py
rg -n "pytest|run_static|pre-commit|flake8|pylint" .gitlab-ci.yml ci tests
```

Treat `tao-core/` and `tao-pytorch/` as submodules or vendored source in this
checkout. They may be dirty for reasons unrelated to your task. Do not reset,
update, or rewrite them unless the task explicitly requires it.

## Mental Model

`scripts/envsetup.sh` sets `NV_TAO_DS_TOP` and defines a shell function named
`tao_ds`. That function runs `runner/tao_ds.py`, which starts the base Docker
image from `docker/manifest.json`, mounts this source tree at `/workspace`, and
executes any command passed after `--`.

Inside the container, `setup.py` installs package console scripts such as
`annotations`, `augmentation`, `analytics`, `auto_label`, `image`,
`gap_analysis`, `tmm`, and `embedding`. Most commands use the shared launcher in
`nvidia_tao_ds/core/entrypoint/entrypoint.py` to discover subtasks from
`scripts/`, require `-e/--experiment_spec_file`, and pass the selected YAML to
Hydra. Mining and RCCA commands use thinner dispatchers that forward Hydra
arguments directly to their selected script.

## Source Truths

| Question | Source Of Truth |
| :--- | :--- |
| Which package commands exist? | `setup.py` `console_scripts` |
| Which host launcher flags exist? | `runner/tao_ds.py` `parse_cli_args` |
| Which base image is pulled? | `docker/manifest.json` |
| Which subtasks exist? | `nvidia_tao_ds/$DOMAIN/scripts/*.py` |
| Which example specs exist? | `nvidia_tao_ds/$DOMAIN/experiment_specs/*.yaml` |
| Which dataclass schema is used? | `nvidia_tao_ds/config/$DOMAIN/...` |
| Which static tests run in GitLab? | `.gitlab-ci.yml` and `ci/run_static_tests.py` |
| Which README content is generated? | `tools/update_readme_supported_commands.py` |

## Worktree Safety

Before editing, capture the status and decide which files you own. The docs
rollout normally touches `README.md`, `docs/`, `tools/update_readme_supported_commands.py`,
`.pre-commit-config.yaml`, and `.gitlab-ci.yml`.

Avoid broad cleanup. Do not remove user-created tests, local examples, cache
directories outside your own generated outputs, or submodule changes unless the
user asks.

## Targeted Checks

For documentation and generated README changes:

```sh
python tools/update_readme_supported_commands.py --check
python -m py_compile tools/update_readme_supported_commands.py
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/*.py .pre-commit-config.yaml .gitlab-ci.yml
rg -n "TBD|PLACEHOLDER|example\\.com" README.md docs
```

For source-adjacent command or config changes, add focused pytest runs from
[testing_and_debugging.md](testing_and_debugging.md). GPU, Docker, private
checkpoint, NGC, and full dataset tests are usually outside a docs-only change.
