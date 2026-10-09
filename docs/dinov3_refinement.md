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
Use a standard DS release image whose installed packages contain these modules;
source-overlay tests do not establish that a given image ships them. The optional cuVS indexed-search path
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
Registration hashes each shard and emits the content-verification seal used by
this default; no separate `bind-store-payload` step is required when registering
with `--source-payload-contract`. Metadata-only registration (`--no-hash-content`)
does not produce a seal and cannot be used with sealed-inventory validation.
Seals bind device, inode, mtime and ctime and are local to the registration mount.
Copying shards or changing the mount can invalidate them even with identical
bytes. Re-register into a fresh output directory on the destination mount, or
use `full_sha256` for portable content verification.

Resume verifies all completed training contracts and their checkpoint bytes,
including historical rounds. Within one execution, the controller and native
actions share digests only while the resolved path, device, inode, size, mtime
and ctime match. Files changed within the last two seconds (or with future
timestamps) bypass caching and receive a second content read to detect writes
within one filesystem timestamp tick. This requires coherent POSIX timestamps
with resolution of one second or finer; metadata-caching network mounts are
not a substitute for immutable storage. Every new run/resume starts with an empty cache; hashes are not
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

## Prepare fixed mining embeddings

GRIT scoring/training uses DINOv3, but nearest-neighbor mining uses a separate,
fixed encoder. C-RADIO is **not required**, and no C-RADIO producer is shipped.
Use the existing DS CLIP or SigLIP image-embedding producer for both source and
target images with exactly the same immutable model and processor configuration.
Keep these embeddings fixed across rounds; do not regenerate them from the
round's DINOv3 checkpoint. No additional model package/install is needed.

Prepare separate source and target input Parquets. Every row needs a unique
`filepath` (an absolute local image filename), globally unique `sample_id`,
`path` equal to `filepath`, and `storage_type: file`. Targets additionally need
`task` and `role` (`query` or `reference`), including a reference population for
each query task. Query/reference/source populations must follow your dataset
split policy. The existing producer preserves these extra columns; duplicate
filepaths would multiply rows in its metadata join and must be removed first.
This producer reads individual image files, not tar/zip members; archive-backed
datasets need an explicitly prepared file view before this step.

Run the shipped producer twice (replace `/data/target-input.parquet` and its
output with source paths for the source pass):

```bash
python -m nvidia_tao_ds.mining.embedding.scripts.image_embeddings \
  input_parquet=/data/target-input.parquet \
  output_parquet=/data/target-embeddings.parquet \
  model=CLIP model_path=/models/fixed-clip batch_size=64
```

`/models/fixed-clip` must be a pre-staged Hugging Face model **and processor**
snapshot, available in the allocated container. SigLIP is also supported with
`model=SigLIP` and a matching snapshot. Do not use a moving model revision between
the two passes. The examples below assume a 224-pixel CLIP processor: use the
actual resolution and normalization of your processor, and replace `sha256:...`
with your immutable checkpoint's digest. Record configuration/processor identity
in the normalization descriptor when it differs. These are operator declarations,
not facts recoverable from vectors; matching dimensions alone do not establish
that two encoders are compatible.

## Register an embedding store

```bash
python -m nvidia_tao_ds.mining.dinov3.internal.refinement register-store \
  --store-root /data/source-embeddings \
  --output-dir /data/mining-registered \
  --encoder-name CLIP \
  --encoder-checkpoint-digest sha256:... \
  --input-resolution 224 \
  --normalization clip-default \
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

Write the target contract using the encoder identity actually used for the
target pass, not a different encoder relabeled to match the source:

```bash
python -m nvidia_tao_ds.mining.dinov3.internal.refinement write-target-contract \
  --targets /data/target-embeddings.parquet \
  --source-store-manifest /data/mining-registered/embedding_store.json \
  --output-dir /data/target-contract \
  --encoder-name CLIP \
  --encoder-checkpoint-digest sha256:... \
  --input-resolution 224 \
  --normalization clip-default
```

The writer checks the declared encoder against the registered source, validates
all target vectors and dimensions, and emits `target_embedding_contract.json`
with the source inventory binding required by indexed search. It also records
input digests in `artifact.json` and commits `_SUCCESS`. It does not certify
the producer's identity or validate task/split semantics. The recorded target
digest is lineage, not an enforced binding to the workflow's current
`data.target_manifest`: consumers compare encoder, dimension and source inventory,
but do not compare this recorded target digest. Vector validation happens when
the contract is written; regenerate the contract when targets change and approve
a fresh run. Use a fresh output
directory for each publication. In `run.yaml`, set `data.target_manifest` to
the target output Parquet, `data.source_store_manifest` to the registered store,
and `data.target_embedding_contract` to the generated contract; set the other
checkpoint, training, payload-contract and output paths, then run `validate`.

## Held-out benchmark isolation

Declare held-out benchmark identities in `data.benchmark_acquisition_units`, a
non-empty Parquet sidecar. It is required whenever `data.benchmark_manifest` is
declared, even with evaluation disabled. Only an evaluator with
`actions.evaluate.scope: diagnostic_replay` may omit it, and that run makes no
held-out claim. A declared sidecar always activates isolation, whatever the
scope. Without a held-out benchmark, omit both fields.

`preflight` reports `"contracts": {"benchmark_isolation": 1}`. Older images
accept the sidecar without screening the source pool, so before declaring a
held-out benchmark, stop unless preflight reports version 1 or later.

The sidecar contains `sample_id` and `data.acquisition_unit_column` (default
`acquisition_unit_id`), and optionally `content_sha256` (64 hex digits, optional
`sha256:` prefix). No DS producer writes per-row `content_sha256`:
`register_embedding_store` hashes shard files, not rows. Declare it only when
your embedding job writes it next to `path`; without it, isolation matches IDs
and units only and does not catch a held-out sample copied under a new ID and
unit. A row matches when any declared identity matches. IDs and
units compare exactly and case-sensitively after surrounding whitespace is
removed; hashes compare case-insensitively. Identity columns must hold strings
or integers, not floats.

Targets, parent history, every nonempty source shard, and every training
manifest must carry each declared identity. Declaring `content_sha256` in the
sidecar makes it mandatory on all of them. Missing, null, or empty identities
fail closed. `validate` rejects contaminated targets and parent history and
projects the benchmark onto the source shards before any stage runs; matching
source rows join the search exclusions. Every `validate`, `run`, `resume` and
`adopt-training` therefore reads the identity columns of every source shard. `materialize` rejects a contaminated
cumulative manifest before writing `artifact.json` or `_SUCCESS`, and the
controller rechecks cached, resumed, and adopted training inputs. Custom search
adapters must honour these exclusions; otherwise materialization rejects the
round. Use `sample_id` as the acquisition unit column only when each sample is
its own acquisition unit. A source store whose `id_column` is not `sample_id`
cannot use that column as the unit, because mined rows carry it as `sample_id`.

Isolation trusts the declared identities. It does not recompute hashes, find
semantic near-duplicates, or audit the data used to pretrain the base checkpoint.

## Select, search, and publish

`select-grit` consumes a model-produced `grit_score`; Data Services does not
implement the formula. `select-multitask` normalizes weakness within task,
allocates equal per-task budgets, and deduplicates samples across tasks.

`exact-search` scans every declared embedding shard and applies cumulative
exclusions plus two independent cosine-similarity thresholds. `min_similarity` is the
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
training manifest. With `--benchmark-acquisition-units` and
`--acquisition-unit-column`, it rejects held-out identities before publication. Images remain at their original `path` or archive
`storage_type`/`path`/`member`; no symlinks or extracted copies are created.
