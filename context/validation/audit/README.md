# Follow-up correctness, recovery and scale validation

Changes in PRs #34–36 remove implicit train-as-validation, add episode holdouts
with training-only normalization, balance validation by source/episode, retain
tails, activate mixture weights and replace Python sample maps with cumulative
lengths. Worker input RNG is deterministic and checkpoints resume directly at a
batch cursor. DDP/FSDP save complete training state for the same world/sharding
configuration. Feature preparation is sharded, checksummed and resumable.

## Regression and failure injection

The server full suite with pretrained assets, real data and both A800 GPUs
passed **507 tests, zero skips**. A subsequently added SIGKILL feature-writer
recovery case also passed, together with the complete 20-case cache/split subset.
The current suite therefore contains 508 cases covered by these runs. CPU-only
regression before the last two test additions passed 479 with four resource
skips and 23 integration/GPU deselections. Hosted CI is checked before merge.

DDP and Megatron-FSDP were tested in independent restarted processes with two
workers, validation, AdamW at 1e-3 and checkpoint saves. The full versus resumed
loss sequences matched exactly. The tests exposed and fixed the FSDP validation
entry point (sampling must pass through its wrapper). Killed writers, altered
array bytes with unchanged shape/size, incompatible build contracts and
train/validation episode overlap are covered. An interrupted checkpoint's
unpublished directory no longer blocks retrying that step.

## Real mixed training and recovery

Four episodes each of Kai0 and Agibot were split by seed 123 and fraction 0.25:
three training and one held-out episode per source. Training-only statistics
were shared with validation. Real BF16 frozen DINO features were prepared in
512-row shards: **8,734 training / 3,095 validation samples**.

The complete 2,015.9M-parameter policy trained at batch 64, two workers, 150
action steps, 3 camera slots, 224px images, history one and prefix up to four.
Five hundred complete optimizer updates plus ten restored updates finished
with finite loss and gradient checks. Validation selected 128 samples per
source every 100 updates. At update 500, source reconstruction errors were
Kai0 **0.164353**, Agibot **0.208351**, aggregate **0.1912**. These are two held-out
episodes, not a full-corpus convergence result.

Allocated GPU memory sampled at the training forward hook after warmup ranged
from **44,998.54 to 45,005.41 MiB**. This is not peak reserved memory. The initial
500-update run took 408.33 seconds including validation/checkpoint saving.
Restart and ten updates took 46.50 seconds including model/checkpoint loading;
previous batches were not decoded again. The preserved checkpoint is update
500; the extra ten updates do not create another numbered checkpoint.

```sh
PYTHONPATH=src OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 python scripts/prebuild_features.py \
  --data-mix kai0,agibot --max-episodes 4 --val-fraction 0.25 --feature-split train \
  --batch-size 64 --num-workers 2 --no-mmap --feature-shard-rows 512 \
  --output-dir /scratch/features-train
# Repeat with --feature-split val and --output-dir /scratch/features-val.
PYTHONPATH=src OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
  python context/validation/audit/train_real.py --train-cache /scratch/features-train \
  --val-cache /scratch/features-val --output /scratch/audit-train
```

## Comparable throughput

The prior 11,829-sample real cache, batch 128 and two workers produced
**123.4069 samples/s**, compared with the previous 123.3 samples/s. Measurement
includes loader wait, transfer, forward, backward, clipping and AdamW: 70
updates, first ten excluded. Peak allocated memory was 75,317.82 MiB. This
confirms no material regression on that workload; it is not a new speedup.

## Expanded LIBERO closed loop

The existing trained LIBERO task-0 checkpoint was served with policy RNG seed
2026, simulator seed 17 and official initial states **10–39**, distinct from
the previous evaluation's 0–9. Thirty actual closed-loop rollouts completed:
**23/30 successes**. All seven failures hit the 220-step limit; no exception
was counted as a normal failure and returned actions were checked for finiteness.
Replanning was every five steps with ten diffusion steps. This remains one
task, not a full-suite score; the initial states are not claimed to be independent
of demonstration collection. Seeding makes the protocol reproducible, without
claiming bitwise reproducibility across different hardware/software versions.

```sh
PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python context/validation/audit/serve_seeded.py \
  --seed 2026 --ckpt /checkpoints/libero/3000.pt --robot-type libero \
  --host 127.0.0.1 --port 8124
# In the compatible Python 3.10 LIBERO environment described in ../remaining/:
MUJOCO_GL=egl python context/validation/remaining/eval_libero.py --repo /path/to/lbm \
  --port 8124 --trials 30 --init-offset 10 --seed 17 --output /scratch/libero-audit
```

Only aggregate reports and reproduction drivers are committed. Dataset-derived
normalization, observations, feature arrays and checkpoints remain on the server.
These bounded experiments do not establish multi-day stability or general robot
performance. Legacy caches require rebuilding for episode provenance and payload
checksums; old sampling-version checkpoints require weight-only initialization.

## Outstanding server-only checks

The out-of-RAM stress driver physically copied 16 repetitions of the real
feature cache (288 shards). Its capacity check required the copied bytes to
exceed the host's approximately 200 GiB RAM. These are duplicated real features
for IO stress, not additional independent training examples.

After copying, SSH stopped responding. After access returned, the temporary
cache and logs were absent. The prior stress result cannot be recovered and is
**not claimed as passing**. The current container has a 200 GiB memory limit,
no swap, and a 200 GiB `/dev/shm` mount which consumes that same memory budget.
Its new OOM counters cannot establish what happened before the restart, and
previous kernel logs are inaccessible. OOM remains a hypothesis.

The driver now refuses to copy anything unless it starts alone in a dedicated
cgroup v2 with `memory.max <= 32 GiB`, `memory.swap.max = 0`, and
`memory.oom.group = 1`. Workers inherit that limit. Size is compared with this
smaller budget, so stressing beyond available cache memory does not require
exceeding the entire server's memory. Defaults are batch 16, one worker and
prefetch one. Progress includes cgroup memory/cache/shmem counters and `/dev/shm`
usage, flushed with fsync after each copy group and read batch. Store the report
on a persistent volume; `/tmp` and `/dev/shm` reports are rejected.

The current server mounts cgroup v2 read-only and exposes only its container
root. The isolation guard correctly rejects it before data access. All nine
guard regression tests passed on the server. A delegated writable child cgroup
(or a separately limited test container exposing an appropriate child cgroup)
is required before another capacity stress run. A soft RSS monitor is not a
substitute for this kernel-enforced limit.

Run from the repository root, after entering that isolated cgroup, with a
persistent report destination:

```sh
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES= python context/validation/audit/cache_pressure.py \
  --source /persistent/features-train --output /scratch/pressure-copy \
  --report /persistent/pressure-result.json --copies 2
```

Choose enough copies to exceed the dedicated memory limit. A whole-group OOM
kill cannot execute Python's `finally` cleanup: inspect the persisted progress
and remove only this run's generated output directory afterward. Do not delete
the source feature caches. Successful normal exit records `complete` and
`cleaned`; missing records must not be interpreted as success.

The full-size-model, real-sharded-data FSDP follow-up subsequently passed after
the startup fix described below. The capacity stress run remains blocked.

## Validation after access restoration

PR #38 introduced fail-closed pressure isolation. Recovery checks found that
persistent source datasets, pretrained assets, and the task-0 LIBERO checkpoint
survived; the temporary test worktree and caches did not. A new isolated worktree
was populated without altering the user's checkout. Train and validation feature
caches were rebuilt on the persistent volume with 8,734 and 3,095 rows.

The full-model feature-cache FSDP check exposed an actual startup failure:
Megatron's post-wrap state dictionary contains uneven DTensors, which the dense
encoder fingerprint function cannot reshape/hash. Fresh-run encoder validation
now happens before FSDP wrapping. Resume restores the trusted training checkpoint
and checks its exact cache-manifest signature; the frozen encoder was validated
when that checkpoint's run began. Train and validation cache metadata must also
agree on the encoder fingerprint.

Random batched feature reads now group indices by shard, then return samples in
the original sampler order. This avoids reopening the same shard repeatedly
within a batch, retains the two-shard LRU bound, and preserves duplicates and
negative indices. The seeded wrapper uses bulk reads only for feature caches;
other datasets keep per-sample RNG isolation, including datasets exposing their
own bulk-reading API. Tests check zero/two workers, exact sample values/order,
and a bound of one shard opening per batch per shard.

LIBERO official states 40–49 were evaluated with policy seed 2026 and simulator
seed 17: **9/10 successes**, one timeout at 220 steps, no inference or simulation
exceptions. These extend task-0 initial-state coverage only. The policy RNG was
restarted for this run; this is not one uninterrupted evaluation of states 10–49
or a multi-task score. The policy service was stopped after evaluation. Offline
MuJoCo 3.2.3 dependencies and aggregate logs now live on persistent storage.

The repaired complete 2,015.9M-parameter model completed two optimizer updates
on two A800s (batch eight and two workers per rank), validation for both sources,
and a distributed checkpoint. A new pair of processes loaded update two,
continued through update four, validated both sources again, and saved update
four successfully. Logged losses and gradient norms were finite. Validation
aggregate error was 9.0240 at update two and 8.9877 at update four; four updates
are a functionality check, not evidence of convergence. Exact loss equivalence
is covered separately by the cached FSDP independent-process regression. The
related cache/sampler/distributed suite passed 25 tests. Hosted static and CPU
checks passed for the implementation commit.

### Refreshed real-cache throughput

These measurements use the rebuilt 8,734-row **sharded cache on persistent
JuiceFS storage**, rather than the earlier 11,829-row local-scratch monolithic
cache. All runs use 70 complete updates with the first ten excluded, the full
model, BF16, frozen DINO, 150 actions, and maximum prefix four. Timing includes
loader wait, device transfer, forward/backward, clipping and optimizer update;
it excludes feature preparation, validation and checkpoint saving.

| Configuration | Batch per GPU | Workers per GPU | Global samples/s | Max allocated GiB per GPU |
| --- | ---: | ---: | ---: | ---: |
| Single GPU, before bulk reads | 128 | 2 | 118.90 | 73.55 |
| Single GPU, bulk reads | 128 | 0 | 81.72 | 73.55 |
| Single GPU, bulk reads | 128 | 2 | 121.34 | 73.55 |
| Two-GPU FSDP, bulk reads | 64 | 2 | 194.61 | 48.42 |
| Two-GPU FSDP, bulk reads | 120 | 2 | 227.20 | 74.17 |

The before-change zero-worker run was stopped after observing filesystem-read
waits and did not produce a complete measurement. Its duration is not used as
a speedup denominator. Bulk reads reduce redundant shard openings by construction
and regression tests; the 2% measured difference at two workers is too small to
claim a substantial training speedup from one comparison.

For this measured cached workload, retain **single GPU batch 128/workers two**,
or **FSDP batch 120/workers two per GPU, block group one**. The latter is 16.7%
faster than per-rank batch 64 here. Global FSDP throughput uses
`world * batch * measured_updates / max(rank_seconds)`; both rank timings are
preserved in results.json. This is a recommendation among measured candidates,
not a universal optimum. FSDP uses FP32 master optimizer state and a larger
global batch, so its optimizer numerics are not identical to single-GPU BF16
training. Earlier online-vision results remain in ../real-throughput/; cached
throughput must not be presented as including online encoder computation.

Reproduce with ../real-throughput/run.py, selecting `--mode features`, the new
cache, `--steps 70`, and the batch/workers from the table. Add `--fsdp` under
`torchrun --standalone --nproc_per_node=2` for two GPUs. Each refreshed candidate
had an external 600-second timeout. Logs, feature arrays, and distributed
checkpoints remain in the isolated persistent server validation directory;
only these aggregate metrics are published.
