# Current-code LIBERO deployment check — 2026-09-14

The production evaluator now exposes bounded task and initial-state selection,
refuses to overwrite reports, records exceptions separately from ordinary task
failures, validates action dimensions/finiteness, and closes each environment
in a finally block. Videos use unique run/task/trial paths and can be disabled.

Eight targeted CPU regressions passed for CLI environment routing, installer
failure behavior, initial-state bounds, reset failure, invalid actions and
per-episode history reset/timestamp propagation. Ruff and syntax checks passed.

Actual A800 / MuJoCo 3.2.3 / robosuite 1.4.1 evaluation used the preserved
3,000-update single-task LIBERO checkpoint, latest production policy code,
`serve_paired.py`, 10 diffusion steps and policy seed 2026 + initial-state index.
Task: libero_spatial 0; official initial states 0 and 1; simulator seed 7 + index;
10 settling steps, 220 control-step limit, replan every 5 steps. Trial 0 succeeded
at step 101 (21 requests); trial 1 reached the 220-step limit (44 requests).
**1/2 successes; both trials completed without simulator/inference exceptions.**

This is an old-checkpoint deployment compatibility smoke test. Two trials are
not a quality estimate or a comparison against the previous 8/10 result, whose
policy diffusion RNG was not fixed. Historical conditioning is disabled in this
checkpoint; historical-policy quality remains a separate experiment.

The isolated test source/report is `/tmp/lbm-closed-loop-next` on js_dev_1.
Private checkpoint and normalization files stay on the server. The test server
was stopped after completion. No user environment or checkpoint was modified.

```bash
# Start a seeded server with the preserved checkpoint and matching normalization.
PYTHONPATH=src python context/validation/correctness/serve_paired.py \
  --checkpoint /private/libero/3000.pt --norm-stats /private/libero/norm_stats.json \
  --port 8134 --seed 2026
# In the installed simulator environment with LIBERO_ROOT/CONFIG_PATH configured:
PORT=8134 bash simulation/libero/eval_env.sh --task-ids 0 \
  --num-trials-per-task 2 --no-save-video --result-path /new/output/result.json
```
