# Development Workflows

These recipes assume you have already run:

```sh
source scripts/envsetup.sh
```

## Build Or Install The Package

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

## Change A Command Or Subtask

1. Find the command in `setup.py`.
2. Open its package under `nvidia_tao_ds/`.
3. Read the package `entrypoint/` wrapper and the target file in `scripts/`.
4. Check the matching example YAML under `experiment_specs/`.
5. Check the matching dataclass schema under `nvidia_tao_ds/config/`.
6. Update focused tests under `tests/`.
7. Run `python tools/update_readme_supported_commands.py` if command metadata,
   subtask files, launcher options, or image digests changed.

Standard command wrappers require `-e/--experiment_spec_file` for normal
subtasks. Mining and RCCA direct dispatchers forward Hydra arguments directly,
so mirror their existing tests before changing their CLI behavior.

## Update Config Defaults

Flat command configs live under `nvidia_tao_ds/config/<module>/`. The analytics
command is implemented in `nvidia_tao_ds/data_analytics/` but its config package
is `nvidia_tao_ds/config/analytics/`.

Nested domains use nested config packages:

| Domain | Config Path |
| :--- | :--- |
| TMM mining | `nvidia_tao_ds/config/mining/tmm/` |
| Embedding mining | `nvidia_tao_ds/config/mining/embedding/` |
| RCCA gap analysis | `nvidia_tao_ds/config/rcca/gap_analysis/` |

After config changes, run focused config and command tests:

```sh
pytest tests/test_config_modules.py -q
pytest tests/mining/test_mining_entrypoints.py -q
```

## Update The API Service

API routes live in `nvidia_tao_ds/api/app.py`. The service relies on installed
console scripts and `nvidia_tao_core.api_utils.module_utils` for action
discovery, so API-visible command changes may require coordinated changes in
`tao-core/`.

For schema behavior, check the relevant dataclass config and any tests that
exercise API or default-spec imports. For endpoint inventory, compare
`nvidia_tao_ds/api/app.py` and `nvidia_tao_ds/api/openapi.json`.

## Update The Base Development Image

Use `docker/build.sh` for the base image:

```sh
cd "$NV_TAO_DS_TOP/docker"
./build.sh --build --x86
./build.sh --build --arm
./build.sh --build --multiplatform --push
```

Single-platform builds can load locally. Multi-platform builds require `--push`
because Docker buildx cannot load multiple architectures into the local Docker
daemon at once. After pushing, update `docker/manifest.json` with the digest
printed by the build script.

**Update every digest reference, not just `docker/manifest.json`.** The base-image
digest is hardcoded in several files. A base-image rebuild must update all of them for
the target architecture (`x86` and/or `arm`), or CI, Jenkins, and the release build will
keep pulling the stale image:

| Location | What it drives |
| :--- | :--- |
| `docker/manifest.json` | `tao_ds` launcher + `ci/utils.py` (static/functional tests); per-arch `x86`/`arm` digests — the source of truth |
| `.gitlab-ci.yml` | the `image:` the GitLab static-test jobs run in |
| `Jenkinsfile.dev`, `Jenkinsfile.develop`, `Jenkinsfile.nightly` | the `base-image` container the Jenkins jobs run in |
| `Jenkinsfile.release` | `BUILD_X86` / `BUILD_ARM` release base images |
| `release/docker/Dockerfile.release` | `FROM` base digests for the release image |

Find them all with `grep -rl <old-digest> .` and replace per architecture (the `x86`
and `arm` digests are distinct strings, so a per-digest `sed` is safe). The
`check_base_image_digests_consistent` static test (`ci/run_static_tests.py`) fails the build
if any of these files references a digest not in `docker/manifest.json`, so a partial bump
is caught in CI. The README intentionally does **not** carry the digests (they are internal
`nvstaging` references, not useful in a public repo) — do not re-add them.

## Update The Release Image

Use `release/docker/deploy.sh` for the release image:

```sh
cd "$NV_TAO_DS_TOP/release/docker"
./deploy.sh --build --wheel
```

The script can build the source wheel through `tao_ds -- make build`, build
`release/docker/Dockerfile.release`, push the release image, and clean wheels
when `--wheel` is used.

## Keep README Generated Content Fresh

`README.md` contains a generated section for launcher options, console scripts,
subtasks, and base-image digests. Update or check it with:

```sh
python tools/update_readme_supported_commands.py
python tools/update_readme_supported_commands.py --check
```

The local pre-commit hook and GitLab `static_tests` job run the `--check` mode.
