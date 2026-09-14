# Historical-policy LIBERO train/resume/deploy check — 2026-09-14

This checks the new historical path with an actual trained checkpoint, separate
from the old-checkpoint compatibility rollout. It is not a policy-quality claim.

The full 2,015.7M-parameter BF16 policy used pretrained DINO/CLIP and a fresh DiT,
with both visual and state history enabled: three observations at 10Hz, including
the current observation. Task 0 of libero_spatial provided 45 demonstrations,
split into 36 training and 9 held-out episodes (3,636/851 samples) with seed 123.
Normalization uses training episodes only. Actions span five seconds / 50 control
positions, batch 32, two workers, AdamW 1e-4 and 10-step warmup.

Training completed 50 updates, saved the standard checkpoint, then restored in a
fresh process and reached 100 updates. Held-out reconstruction on the bounded
64-sample selection was 0.330222 at 50 and 0.341076 at 100. This short, non-monotonic
curve does not establish convergence. The resumed checkpoint was then loaded by
the production LBMPolicy/HTTP server.

Two official initial states (0,1), simulator seed 7+index, policy seed 2026+index,
10 diffusion steps, 10 settling steps and 220 control-step limit were evaluated.
Replan=1 sends every control observation (including its MuJoCo timestamp) so the
three-observation history can actually populate. Server metadata confirmed
history_size=3, timed vision enabled, state_history_length=.3 and required
timestamps. Both trials reached the 220-step limit: **0/2 successes**, 440 policy
requests, **no inference/simulation exceptions**, complete=true, and clean
simulator closure. The test server was stopped after completion.

These trials establish train/resume/deploy functionality for the historical
interface, not sufficient robot skill. They are not comparable to the old
3,000-update checkpoint's score: training length and replanning cadence differ.
Do not infer that history improves or harms task success from these two trials.

Aggregate rollout evidence: libero-rollout.json. Drivers: train_libero.py and
the production simulation/libero/eval_env.sh. The verified update 100 checkpoint,
train_config.json, private norm_stats.json and closed_loop.json are preserved on
js_dev_1's cloud storage under:
`/mnt/kpfs/workspace/jinaoqun/Projects/lbm/checkpoints/validation_20260914/history-libero`.
No checkpoint or normalization arrays are committed to Git.

```bash
PYTHONPATH=src:. python context/validation/history/train_libero.py --output /new/private/run --steps 50
PYTHONPATH=src:. python context/validation/history/train_libero.py --output /new/private/run \
  --steps 100 --resume /new/private/run/last.pt
PYTHONPATH=src:. python context/validation/correctness/serve_paired.py \
  --checkpoint /new/private/run/100.pt --norm-stats /new/private/run/norm_stats.json --port 8135
# In the configured Python 3.10 simulator environment:
PORT=8135 bash simulation/libero/eval_env.sh --task-ids 0 --num-trials-per-task 2 \
  --replan-steps 1 --no-save-video --result-path /new/private/eval.json
```
