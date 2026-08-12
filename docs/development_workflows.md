# Development Workflows

This guide gives concrete recipes for common source-code changes.

These recipes assume you have already run:

```sh
git submodule update --init
source scripts/envsetup.sh
```

`envsetup.sh` sets `NV_TAO_DS_TOP`, defines the `tao_ds` shell function, and
installs the repository's pre-commit hooks.

## Build or Install the Package

Inside the development container:

```sh
tao_ds -- make build
tao_ds -- make install
```

For editable local work where dependencies are already present:

```sh
python3 setup.py develop
```

`release/python/version.py` owns the package name, package description, version
components, and package metadata used by `setup.py`.

## Trace a Command to Code

Use this when a task mentions a command such as `annotations convert`.

```sh
rg -n "annotations" setup.py
sed -n '1,80p' nvidia_tao_ds/annotations/entrypoint/annotations.py
sed -n '1,120p' nvidia_tao_ds/annotations/scripts/convert.py
rg -n "config_name=" nvidia_tao_ds/annotations/scripts/
```

Then inspect the specifications, schema, and tests:

```sh
find nvidia_tao_ds/annotations/experiment_specs -maxdepth 1 -type f | sort
find nvidia_tao_ds/config/annotations -maxdepth 1 -type f | sort
find tests -name '*annotation*' -o -name '*merger*' | sort
```

## Change a Command or Subtask

1. Find the command in `setup.py`.
2. Open its package under `nvidia_tao_ds/`.
3. Read the package `entrypoint/` wrapper and the target file in `scripts/`.
4. Check the matching example YAML under `experiment_specs/` and the
   `@hydra_runner(config_name=...)` declaration.
5. Check the matching dataclass schema under `nvidia_tao_ds/config/`.
6. Update focused tests under `tests/`.
7. Run `python tools/update_readme_supported_commands.py` if command metadata,
   subtask files, launcher options, or image digests changed.

When adding a subtask, remember two launcher behaviors keyed on names: only
subtasks named `generate` get multi-GPU handling, and only scripts decorated
with `@monitor_status` write `status.json` (which the launcher reads to detect
failures).

## Update Configuration Defaults

Flat command configurations live under `nvidia_tao_ds/config/<module>/`. The analytics
command is implemented in `nvidia_tao_ds/data_analytics/` but its configuration
package is `nvidia_tao_ds/config/analytics/`.

Nested domains use nested, per-subtask configuration packages:

| Domain | Config path |
| :--- | :--- |
| TMM mining | `nvidia_tao_ds/config/mining/tmm/` |
| Embedding mining | `nvidia_tao_ds/config/mining/embedding/` |
| RCCA gap analysis | `nvidia_tao_ds/config/rcca/gap_analysis/` |

Declare new fields with the factories from
`nvidia_tao_ds/config/utils/types.py` (`STR_FIELD`, `INT_FIELD`, ...), including
a `description`; the metadata feeds the API schema layer and default-specification
generation.

After configuration changes, run focused configuration tests:

```sh
pytest tests/test_config_modules.py -q
```

## Update the API Service

Dev-mode API routes live in `nvidia_tao_ds/api/app.py`. The service relies on
installed console scripts and `nvidia_tao_core.api_utils.module_utils` for
action discovery, so API-visible command changes may require coordinated
changes in `tao-core/`.

For schema behavior, check the relevant dataclass configuration and any tests that
exercise API or default-spec imports. For endpoint inventory, compare
`nvidia_tao_ds/api/app.py` and `nvidia_tao_ds/api/openapi.json`. The release
container runs the tao-core microservice app, not this Flask app; refer to
[Architecture](architecture.md).

## Update the Base Development Image

Use `docker/build.sh` for the base image:

```sh
cd "$NV_TAO_DS_TOP/docker"
./build.sh --build --x86
./build.sh --build --arm
./build.sh --build --multiplatform --push
```

Single-platform builds can load locally. Multi-platform builds require `--push`
because Docker buildx cannot load multiple architectures into the local Docker
daemon at once.

**Update every digest reference, not just `docker/manifest.json`.** After
pushing, update the new digest in both places that pin it:

| Location | What it drives |
| :--- | :--- |
| `docker/manifest.json` | The `tao_ds` launcher; this is the source of truth for the per-architecture `x86` and `arm` digests |
| `release/docker/Dockerfile.release` | Default `X86_DIGEST`/`ARM64_DIGEST` build args for the release image `FROM` lines |

Find any stragglers with `grep -rl <old-digest> .` and replace per architecture
(the `x86` and `arm` digests are distinct strings, so a per-digest `sed` is
safe). The README intentionally does **not** carry the digests, because they
are internal `nvstaging` references that are not useful in a public
repository; do not re-add them. Dependency changes to requirements or Docker
files also trip the `dependency-guard` pre-commit hook, which requires
explicit acknowledgment.

## Update the Release Image

Use `release/docker/deploy.sh` for the release image:

```sh
cd "$NV_TAO_DS_TOP/release/docker"
./deploy.sh --build --wheel
```

The script can build the source wheel through `tao_ds -- make build`, build
`release/docker/Dockerfile.release`, push the release image, and clean wheels
when `--wheel` is used.

Any change touching video or codecs must respect the FFmpeg codec allow-list
baked into `docker/Dockerfile` (an LGPL-only build with VP9 and mjpeg encode and no libx264, hevc, or aac). The image build asserts this and fails if restricted codecs
reappear.

## Keep README Generated Content Fresh

`README.md` contains a generated section for launcher options, console scripts,
subtasks, and the base-image manifest pointer. Update or check it with:

```sh
python tools/update_readme_supported_commands.py
python tools/update_readme_supported_commands.py --check
```

The local pre-commit hook and the GitHub Actions `static-tests` workflow run
the `--check` mode.
