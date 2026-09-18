# DINOv3 SSL Data Refinement

The `nvidia_tao_ds.mining.dinov3` package provides DINOv3-only data
operations for the DINOv3 SSL DEFT workflow. This is not a generic mining API;
other model families are not supported by this package. DINOv3 scoring and training remain in
TAO PyTorch; the DS-owned `nvidia_tao_ds.mining.dinov3.workflow` package owns
workflow state, execution, recovery and packaged recipes.

## Run DEFT in one allocated DS container

The DS image already includes TAO PyTorch/Core, Torch and the data-stack
dependencies. No host TAO install, Skill Bank runtime, nested Docker, new
Dockerfile or startup pip install is required. The image release must include
the DEFT modules from the companion DS/PyTorch/Core changes.

```bash
python -m nvidia_tao_ds.mining.dinov3.workflow preflight --gpu
python -m nvidia_tao_ds.mining.dinov3.workflow init --recipe grit-score --output run.yaml
# Set real paths and resource requests within the allocated container.
python -m nvidia_tao_ds.mining.dinov3.workflow validate run.yaml
python -m nvidia_tao_ds.mining.dinov3.workflow plan run.yaml
python -m nvidia_tao_ds.mining.dinov3.workflow run run.yaml
```

For one allocation use `execution.backend: local` without `container_images`:
the durable process runner executes mining, scoring and training in the same
runtime. External four-verb runners remain supported for multi-node or separate
image jobs; this controller does not replace the TAO platform launcher.
Preflight checks imports and optional mixed-precision CUDA convolution forward/
backward execution, not GPU FAISS or dataset quality. Review the scoring backend
explicitly. A release image is ready for the single-container flow only after it
contains the companion DS/PyTorch/Core modules and passes compute preflight; do
not install packages at startup or silently alter precision to hide runtime
incompatibilities.

## Local operator contract

These DINOv3-only module commands are internal workflow helpers, not registered
TAO launcher/API commands. They deliberately do not alter other model entrypoints.
An approved DEFT release image is required; source-overlay tests do not establish
that the stock image ships these modules. The optional cuVS indexed-search path
also needs TAO Infra approval and a tested image containing cuVS. Exact search
does not require cuVS. Do not install dependencies at run startup.

Before validation, export the platform-approved `CUDA_VISIBLE_DEVICES` allocation
and a writable node-local `TAO_LOCAL_SCRATCH`; run
`python -m nvidia_tao_ds.mining.dinov3.workflow preflight run.yaml --gpu`.
Use an output directory on a local or demonstrably lock-capable POSIX filesystem.
CIFS with `nobrl` and NFS without working locks are unsupported: an exclusive
directory alone is not a reliable distributed stale-owner protocol.

`run` and `resume` both reconcile committed stages. Cancel is terminal; to
restart after cancellation, approve a new run directory/configuration. Never
delete the cancellation marker or hand-edit state. Local retries are capped by
`execution.max_attempts` (default 3) and recorded with attempt identities.
Status reports controller lock ownership and live job state, including when
resume is needed. If a leader dies while descendants survive, automatic
signaling is refused because a reused process-group ID is unsafe. Have the
allocation owner identify and terminate those exact descendants or terminate
the owned allocation, verify the GPUs are released, then retry reconciliation.
Do not kill every process on a shared host.

Controller/package/Python provenance drift is recorded as a warning and event.
Schema/workflow compatibility changes and model-leaf implementation drift still
require a newly approved run; keep the old directory immutable as evidence.
Registered stores default to sealed-inventory validation on their declared
immutable roots; choose `data.source_store_validation: full_sha256` when the
storage immutability contract is not trusted.

Resume verifies all completed training contracts and their checkpoint bytes,
including historical rounds. Within one execution, the controller and native
actions share digests only while the resolved path, device, inode, size, mtime
and ctime match. Every new run/resume starts with an empty cache; hashes are not
persisted as substitutes for content verification. This removes repeated reads
of unchanged checkpoints within an execution, not the initial historical scan.
Dense-store finalization likewise retains source-shard content verification:
progress markers alone cannot prove that the source bytes stayed unchanged.

### Why this orchestration is in Data Services

Core API jobs own platform allocations and model dispatch; they do not own the
DINOv3 corpus ledger, mined cumulative manifests, per-round acceptance, or
original-checkpoint restart policy. Reusing the platform job API as the local
loop would require an API deployment and credentials inside the existing DS
allocation. The AutoML loop optimizes supervised experiments and has different
sampling/stopping contracts. This workflow therefore keeps DINOv3 policy and its
runner/state implementation together in the same DINOv3 workflow change. The
modules remain separated from the policy controller, not promoted to shared
infrastructure or added to other model entrypoints.
The local runner supervises only children in an already allocated container;
it does not schedule hardware. The external adapter boundary delegates platform
jobs and preserves the existing integration, but no customer runner executable
is shipped or claimed tested by this change. Generic runner extraction is not
required for this workflow and is outside the scope of this change.

## Register an embedding store

```bash
python -m nvidia_tao_ds.mining.dinov3.internal.refinement register-store \
  --store-root /data/cradiov4/parquet \
  --output-dir /data/cradiov4/registered \
  --encoder-name c-radiov4 \
  --encoder-checkpoint-digest sha256:... \
  --input-resolution 512 \
  --normalization imagenet \
  --source-payload-contract /data/source_payload_contract.json
```

The action inventories existing Parquet shards without copying them. Its
`embedding_store.json`, `artifact.json`, and `_SUCCESS` files form the stable
input used by later runs. Every shard is content-hashed. The source-payload
contract names immutable dataset IDs, versions, and local roots; registration
rejects locators outside those roots and records a deterministic locator audit.
For example:

```json
{
  "schema_version": "1.0",
  "immutability": "immutable",
  "datasets": [
    {
      "dataset_id": "wfm-aoi",
      "version": "dss-snapshot-12345",
      "root_uri": "file:///data/wfm-aoi-v12345"
    }
  ]
}
```

Omitting `--source-payload-contract` remains available for standalone legacy
uses, but the DINOv3 SSL DEFT workflow requires this provenance binding.

## Select, search, and publish

`select-grit` consumes a model-produced `grit_score`; Data Services does not
implement the formula. `select-multitask` normalizes weakness within task,
allocates equal per-task budgets, and deduplicates samples across tasks.

`exact-search` scans every declared embedding shard and applies cumulative
exclusions plus two independent C-RADIO thresholds. `min_similarity` is the
initial weak-query relevance radius; `duplicate_similarity` rejects near-exact
copies of the query or any already accepted image. The action retrieves
`candidate_multiplier * top_k` candidates, allocates them round-robin across
queries, and lowers relevance by `similarity_step` only until
`hard_min_similarity`. It accepts an underfilled result rather than cross that
floor. `search_summary.json` records every attempted radius and rejection count.

This is the correctness reference for small pools and search validation.
For full-corpus deployments, `dense-init`, an array of `dense-write`, and
`dense-finalize` materialize one immutable float32 random-access vector store.
`build-ann-index` is an experimental, opt-in persistent multi-GPU cuVS IVF-PQ
index builder. The approved stock DS image does not include cuVS; dense exact
search is the supported dependency-free path. An approved CUDA-matched ANN
runtime and explicit `TAO_DINOV3_EXPERIMENTAL_ANN=1` are required for index
construction and retrieval. A disposable cuVS smoke test is not image approval.
Index construction is a one-time corpus operation; it is not repeated per
refinement round.

Before indexed mining, `audit-ann-recall` accepts `--queries`,
`--query-embedding-contract`, `--dense-store-manifest`, `--ann-index-manifest`,
`--n-probes`, `--ann-candidates`, and `--output-dir`; optional `--top-k`,
`--sample-size`, `--seed`, and `--minimum-recall` control a reproducible exact
top-k comparison. It commits both successful and failed measurements, returning
a nonzero exit code for failure. Only a passing audit can authorize mining.
Choose representative queries and inspect the measured recall, not just the
presence of an audit file.

Dense-store content identity is portable, but its runtime locators and vector
file identity are currently bound to the original mount. Copying a committed
store does not rebind it to the copy. Keep the original approved mount/path;
otherwise rebuild and validate a new store. A safe in-place reseal/relocation
command is not part of this increment. Do not hand-edit committed manifests.

Production indexed search has two resumable actions. `ann-candidates` loads the
index and accepts only the probe count and candidate depth approved by its
committed exact-recall audit. `ann-rerank` consumes that committed candidate
artifact, reads the corresponding float32 vectors, computes exact cosine
scores, and applies the same radius, exclusion, and duplicate rules as exact
search. The runtimes are deliberately separable so cuVS and model-side PyTorch
CUDA dependencies do not need to coexist. ANN underfill is
`search_budget_exhausted`; use `exact-search` when a terminal
`radius_exhausted` proof is required.

`materialize` appends novel neighbors to an immutable cumulative Parquet
training manifest. Images remain at their original `path` or archive
`storage_type`/`path`/`member`; no symlinks or extracted copies are created.
