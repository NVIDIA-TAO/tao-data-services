# TAO Data Services Source Documentation

This documentation is for contributors, maintainers, coding agents, and
container power users working from the TAO Data Services source tree.

The root `README.md` is intentionally concise. Start here when you need the
repository mental model, command flow, configuration flow, or validation map.

## Start Paths

| Role | Start with | Then read |
| :--- | :--- | :--- |
| New developer picking up the repository | [Codebase tour](codebase_tour.md) | [Architecture](architecture.md) |
| Coding agent or new maintainer | [Agent onboarding](agent_onboarding.md) | [Architecture](architecture.md), [Testing and debugging](testing_and_debugging.md) |
| Feature developer | [Development workflows](development_workflows.md) | [New data-service command](new_data_service_command.md) |
| Container power user | [Container power users](container_power_users.md) | [Architecture](architecture.md) |
| Reviewer | [Architecture](architecture.md) | [Testing and debugging](testing_and_debugging.md) |

## Documentation Map

| Document | Purpose |
| :--- | :--- |
| [Codebase tour](codebase_tour.md) | Annotated repository tree, module map, service inventory, and the sharp edges new developers hit. |
| [Agent onboarding](agent_onboarding.md) | First-pass audit commands, worktree safety, and source-of-truth files. |
| [Architecture](architecture.md) | Runtime dispatch, Hydra config, models and weights, API service, and extension points. |
| [Development workflows](development_workflows.md) | Recipes for common source, configuration, Docker, release, and README changes. |
| [Testing and debugging](testing_and_debugging.md) | CI static checks, targeted pytest commands, GPU-sensitive paths, and failure triage. |
| [Container power users](container_power_users.md) | `tao_ds`, mounts, GPUs, base-image digests, service mode, and direct Docker equivalents. |
| [New data-service command](new_data_service_command.md) | Source-backed guide for adding or extending commands and subtasks. |

## Repository Anchors

| Path | What to look for |
| :--- | :--- |
| `setup.py` | Package metadata and in-container console script entry points. |
| `scripts/envsetup.sh` | `NV_TAO_DS_TOP` setup, the host-side `tao_ds` shell function, and git-hook installation. |
| `runner/tao_ds.py` | Docker launcher, GPU selection, mount handling, service mode, and manifest lookup. |
| `nvidia_tao_ds/core/entrypoint/entrypoint.py` | Shared subtask discovery, experiment spec handling, GPU override handling, and subprocess launch. |
| `nvidia_tao_ds/core/hydra/hydra_runner.py` | Local Hydra wrapper used by script modules. |
| `nvidia_tao_ds/config/` | Dataclass-backed schemas and default-spec sources. |
| `nvidia_tao_ds/*/experiment_specs/` | Example YAML specs used by script subtasks. |
| `nvidia_tao_ds/api/app.py` | Dev-mode Flask API routes, job queue handoff, schema validation, and OpenAPI endpoints. |
| `docker/manifest.json` | Immutable base-image registry, repository, and architecture-specific digests. |
| `.pre-commit-config.yaml` and `.github/workflows/` | Pull-request checks: lint, license headers, DCO, README drift, and secret scan. |

## Command Layers

TAO Data Services has three command layers:

1. `tao_ds` runs on the host after `source scripts/envsetup.sh`. It starts the
   base development container and mounts the repository at `/workspace`.
2. Package console scripts such as `annotations`, `augmentation`, `analytics`,
   and `auto_label` run inside the container after the wheel or editable package
   is available.
3. Subtasks are discovered from each command package's `scripts/` directory and
   normally run through Hydra with an experiment spec.

The generated command table in `README.md` is maintained by
`tools/update_readme_supported_commands.py`; update it whenever command,
launcher, or image-manifest metadata changes.

## Diagrams

Architecture and workflow diagrams are checked in as SVG files under
`docs/assets/` and embedded with normal Markdown image links. The SVG files are
the canonical editable sources.

| Diagram | Source |
| :--- | :--- |
| Runtime dispatch flow | [assets/runtime_flow.svg](assets/runtime_flow.svg) |
| Configuration flow | [assets/config_flow.svg](assets/config_flow.svg) |
| Package module map | [assets/module_map.svg](assets/module_map.svg) |
| Container launch and build flow | [assets/container_flow.svg](assets/container_flow.svg) |
