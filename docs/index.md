# TAO Data Services Source Documentation

This documentation is for contributors, maintainers, coding agents, and
container power users working from the TAO Data Services source tree.

The root `README.md` is intentionally concise. Start here when you need the
repository mental model, command flow, configuration flow, or validation map.

## Start Paths

| Role | Start With | Then Read |
| :--- | :--- | :--- |
| Coding agent or new maintainer | [Agent onboarding](agent_onboarding.md) | [Architecture](architecture.md), [Testing and debugging](testing_and_debugging.md) |
| Feature developer | [Development workflows](development_workflows.md) | [New data-service command](new_data_service_command.md) |
| Container power user | [Container power users](container_power_users.md) | [Architecture](architecture.md) |
| Reviewer | [Architecture](architecture.md) | [Testing and debugging](testing_and_debugging.md) |

## Documentation Map

| Document | Purpose |
| :--- | :--- |
| [agent_onboarding.md](agent_onboarding.md) | First-pass audit commands, worktree safety, and source-of-truth files. |
| [architecture.md](architecture.md) | Runtime dispatch, Hydra config, API service, package layout, and extension points. |
| [development_workflows.md](development_workflows.md) | Recipes for common source, config, Docker, release, and README changes. |
| [testing_and_debugging.md](testing_and_debugging.md) | CI static checks, targeted pytest commands, GPU-sensitive paths, and failure triage. |
| [container_power_users.md](container_power_users.md) | `tao_ds`, mounts, GPUs, base-image digests, service mode, and direct Docker equivalents. |
| [new_data_service_command.md](new_data_service_command.md) | Source-backed guide for adding or extending commands and subtasks. |

## Repository Anchors

| Path | What To Look For |
| :--- | :--- |
| `setup.py` | Package metadata and in-container console script entrypoints. |
| `scripts/envsetup.sh` | `NV_TAO_DS_TOP` setup and the host-side `tao_ds` shell function. |
| `runner/tao_ds.py` | Docker launcher, GPU selection, mount handling, service mode, and manifest lookup. |
| `nvidia_tao_ds/core/entrypoint/entrypoint.py` | Shared subtask discovery, experiment spec handling, GPU override handling, and subprocess launch. |
| `nvidia_tao_ds/core/hydra/hydra_runner.py` | Local Hydra wrapper used by script modules. |
| `nvidia_tao_ds/config/` | Dataclass-backed schemas and default-spec sources. |
| `nvidia_tao_ds/*/experiment_specs/` | Example YAML specs used by script subtasks. |
| `nvidia_tao_ds/api/app.py` | Flask API routes, job queue handoff, schema validation, and OpenAPI endpoints. |
| `docker/manifest.json` | Immutable base-image registry, repository, and architecture-specific digests. |
| `.gitlab-ci.yml` and `ci/` | Merge-request checks and static-test helpers. |

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

| Diagram | Source |
| :--- | :--- |
| Runtime dispatch flow | [assets/runtime_flow.svg](assets/runtime_flow.svg) |
| Container launch and build flow | [assets/container_flow.svg](assets/container_flow.svg) |
