# New Data-Service Command Guide

Use this guide before adding a new console command or a new subtask under an
existing command. It is calibrated against the current source patterns rather
than an idealized skeleton.

## Existing Exemplars

| Exemplar | What It Teaches |
| :--- | :--- |
| `augmentation generate` | Standard shared launcher, `-e` spec handling, `@hydra_runner`, central dataclass schema, multiple example specs, and multi-GPU MPI handling. |
| `annotations convert`, `merge`, `slice`, `qa_to_llava_annotation` | One command can own several script subtasks with different config names, schemas, and even script-local dataclasses. |
| `analytics` | Implementation package can differ from config package name: `data_analytics` uses `config/analytics`. |
| `embedding image_embeddings` and `tmm nearest_neighbors` | Some nested domains use direct dispatchers that forward Hydra args instead of the shared `-e` wrapper. |
| `gap_analysis vcn_aoi` and `vlm_bcq` | RCCA is a nested package and config namespace. |

## Choose The Surface

If the workflow belongs to an existing command, add a subtask under that
command's `scripts/` package. If users need a new package command, add a new
console script with an `entrypoint/` wrapper and register it in `setup.py`.

Do not add a top-level console script only to expose a variant of an existing
domain. Prefer a subtask unless the package, config namespace, tests, and user
mental model are genuinely separate.

## Pattern Matrix

| Pattern | Existing Source | Use When |
| :--- | :--- | :--- |
| Shared launcher, flat package | `augmentation`, `annotations`, `analytics`, `image` | The command should require `-e/--experiment_spec_file` and use common GPU/spec/log handling. |
| Direct dispatcher, nested package | `mining/embedding`, `mining/tmm`, `rcca/gap_analysis` | The command should forward Hydra overrides directly to a nested script package. |
| Central config dataclass | `config/augmentation/default_config.py`, `config/analytics/default_config.py` | The schema is reusable by CLI, default-spec generation, tests, or API code. |
| Task-specific config dataclass | `config/annotations/merge_config.py`, `config/annotations/slice_config.py` | A subtask needs a schema that should live with the package config surface. |
| Script-local config dataclass | `annotations/scripts/qa_to_llava_annotation.py` | A small, self-contained conversion subtask does not need shared config reuse. |
| Nested config namespace | `config/mining/embedding/`, `config/rcca/gap_analysis/` | The command lives below a domain namespace and should not be flattened. |

## Standard Subtask Pattern

For a standard shared-launcher command:

1. Add `nvidia_tao_ds/$DOMAIN/scripts/$SUBTASK.py`.
2. Add or update an example YAML under
   `nvidia_tao_ds/$DOMAIN/experiment_specs/`.
3. Add or update dataclass config under `nvidia_tao_ds/config/$DOMAIN/`, or
   keep a script-local dataclass only for a small subtask that will not be
   shared by default-spec or API code.
4. Decorate the script main function with `@hydra_runner(config_path=..., config_name=..., schema=...)`.
5. Let the existing entrypoint discover the script through `get_subtasks()`.
6. Add tests under `tests/`.
7. Run `python tools/update_readme_supported_commands.py`.

The shared launcher maps `-e specs/experiment.yaml` to Hydra
`--config-path specs --config-name experiment.yaml`. The script's
`config_name` is the direct-script default; shared-launcher commands normally
override it through `-e`. It is not always the only supported spec. For example,
`augmentation generate` defaults to `kitti` but also carries `coco.yaml`, and
`annotations convert` defaults to `annotations` while the package includes
conversion-specific examples such as `coco2odvg.yaml` and `odvg2coco.yaml`.

Before documenting a subtask, record the real script name, default
`config_name`, schema class, and example YAML files together.

| Surface | Rule |
| :--- | :--- |
| Subtask name | Comes from the `scripts/*.py` filename discovered by `get_subtasks()`. |
| Default config name | Comes from the script's `@hydra_runner(config_name=...)`. |
| User-selected spec | Comes from `-e/--experiment_spec_file` for shared-launcher commands. |
| Schema class | Comes from the script's `@hydra_runner(..., schema=...)`, not from the subtask name. |
| Multi-GPU behavior | Comes from `launch()` and the `network` argument passed by the entrypoint wrapper. |

## Known Variants

Annotations has multiple schemas. `convert.py` uses the default annotations
config, `merge.py` and `slice.py` use `MergeConfig` and `SliceConfig`, and
`qa_to_llava_annotation.py` owns a small `QAToLLaVAConfig` dataclass in the
script. Do not force all subtasks into one dataclass if the domain already has
task-specific config classes.

Mining and RCCA use nested package layouts. Their entrypoints build a child
command directly:

```text
python $RUNNER_PATH $HYDRA_OVERRIDES
```

They do not require `-e/--experiment_spec_file` at the wrapper layer. Match that
behavior when adding a subtask to those domains, and add tests like
`tests/mining/test_mining_entrypoints.py` if dispatcher behavior changes.

Analytics is implemented under `nvidia_tao_ds/data_analytics/`, but command and
config surfaces use `analytics`. If a new command has an implementation/config
name mismatch, add an explicit alias where discovery expects it, as
`default_specs.py` does with `CONFIG_TO_DS_MODULE_ALIASES`.

## Adding A Console Command

For a flat shared-launcher command, add this package shape:

```text
nvidia_tao_ds/my_domain/
  __init__.py
  entrypoint/
    __init__.py
    my_command.py
  scripts/
    __init__.py
    my_subtask.py
  experiment_specs/
    my_subtask.yaml
nvidia_tao_ds/config/my_domain/
  __init__.py
  default_config.py
```

For a nested direct-dispatch command, follow the mining/RCCA shape instead:

```text
nvidia_tao_ds/parent_domain/my_command/
  __init__.py
  entrypoint/
    __init__.py
    my_command.py
  scripts/
    __init__.py
    my_subtask.py
  experiment_specs/
    my_subtask.yaml
nvidia_tao_ds/config/parent_domain/my_command/
  __init__.py
  my_subtask.py
```

Register the command in `setup.py`:

```python
entry_points={
    "console_scripts": [
        "my_command=nvidia_tao_ds.my_domain.entrypoint.my_command:main",
    ],
}
```

Then run:

```sh
python tools/update_readme_supported_commands.py
python tools/update_readme_supported_commands.py --check
```

Add or update tests that cover both subtask discovery and the actual dispatch
contract. For nested direct dispatchers, include a command-construction test
like `tests/mining/test_mining_entrypoints.py`.

## API Considerations

The Flask API discovers commands through installed console scripts and
`nvidia_tao_core.api_utils.module_utils`. The current API routes and OpenAPI
text mostly describe flat data-service commands such as `analytics`,
`annotations`, `augmentation`, `auto_label`, and `image`. API-visible commands
may require coordinated changes in `tao-core/`, `nvidia_tao_ds/api/app.py`, and
`nvidia_tao_ds/api/openapi.json`.

Before declaring a command API-ready, verify:

| Surface | Check |
| :--- | :--- |
| Action discovery | `module_utils.get_neural_network_actions()` returns the command and actions. |
| Schema import | The API schema route can import the correct config module; the current flat path is `nvidia_tao_ds.config.$NEURAL_NETWORK_NAME.default_config`. |
| Action-specific schemas | Subtasks with `MergeConfig`, `SliceConfig`, script-local dataclasses, or nested config files need explicit API handling. |
| Aliases | Any implementation/config name mismatch is handled, as with `data_analytics` and `analytics`. |
| OpenAPI | `nvidia_tao_ds/api/openapi.json` matches the route behavior. |

## Test Checklist

Before review:

```sh
python tools/update_readme_supported_commands.py --check
python -m py_compile tools/update_readme_supported_commands.py
pytest tests/test_config_modules.py -q
pytest tests/mining/test_mining_entrypoints.py -q
pytest "$TARGETED_DOMAIN_TEST" -q
```

Also run a source audit:

```sh
rg -n "hydra_runner|config_name|get_subtasks|console_scripts" nvidia_tao_ds setup.py
```

Confirm the guide, README generated table, and tests all describe the same
command names, spec names, and config classes.
