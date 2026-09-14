# Matched short history validation — 2026-09-14

Eight independent full-model A800 runs: two seeds (123, 456), four input modes,
100 updates each, batch 64, Agibot + Galaxea subtask data, four episodes per
source and 25% parent-episode holdout. Training-only normalization is recomputed
for each paired split. Fixed validation RNG seed 2026 and four validation batches
are used. Initial shared weights and first-three state/action/text/CUDA-RNG
hashes match across modes within each seed. All runs completed with finite loss
and gradient metrics. These short runs are exploratory, not convergence tests.

| Inputs | Seed 123 reconstruction | Seed 456 reconstruction |
|---|---:|---:|
| current | 0.493244 | 0.636223 |
| vision | 0.737480 | 0.448127 |
| state | 0.494201 | 0.594961 |
| both | 0.759760 | 0.408593 |

Visual history changes direction between seeds. State history is near equal for
seed 123 and better for seed 456; two short runs do not establish a robust
quality advantage. Keep current-observation defaults and both history additions
opt-in. Do not choose an architecture from the pooled mean alone. These are
held-out action reconstruction metrics, not robot task success rates.

The action horizon is one second; these values and throughput are not directly
comparable to older long-action-window experiments. Source dimensions can change
with the selected episodes; matching is established within each seed, not by
assuming different seed workloads are identical. Full aggregate evidence is in
quality-summary.json; reproduce with run_quality.sh and existing private assets.
