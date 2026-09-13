# Subtask data validation — 2026-09-14

Implementation, CPU regression and all three real GPU smoke arms are complete.
Each arm finished ten updates and fresh-process recovery to twenty, with finite
training loss and validation metrics. These short runs demonstrate functional
training/recovery, not convergence or quality improvement.

## Behavior

`--instruction-mode subtask` selects the adapter's subtask reader. The default
`episode` mode remains. Shared interval validation and sampling live in
`custom/instructions.py`; adding a source requires an optional `read_subtasks`
adapter function, rather than another dataset-name branch in training.

Intervals use physical episode coordinates. Gaps are excluded from sampling;
empty, overlapping, unordered or out-of-range intervals fail. Action windows
repeat the last frame of the current interval at the boundary. Historical image
windows repeat its first frame, so they cannot read the preceding subtask.
Physical records and mmap/video coordinates retain their original lengths.
Normalization fitting uses selected intervals, and automatic heldout splitting
fits statistics on training parents only.

Agibot matches annotation `episode_id` to the physical episode and reads
`task_name` for episode mode, `action_text` for subtask mode. Its explicit
conservative training policy is `[start_frame, end_frame)`: end frames and
unlabeled tails are not extended. This is not a verified claim about the
publisher's endpoint ownership. Adjacent intervals therefore never duplicate
a frame. Alpha/Beta matching task/episode IDs share a holdout parent because
[the publisher describes Alpha as a Beta subset](https://github.com/OpenDriveLab/AgiBot-World).

Galaxea uses per-frame `task_index`, verifies contiguous `frame_index`, and
resolves every task ID without a fallback to another task. Its Chinese@English
annotation format selects the supplied English text for CLIP. English-only
text is preserved. No token truncation or model context-length change is made.

## Cache and checkpoint contracts

Agibot and Galaxea adapter scan revision 2 rejects older scan indexes with an
explicit rescan error. Agibot annotation digests also detect source edits,
deletions and newly available annotations. A bounded `--max-episodes` rescan
does not replace the full scan cache. Source image/vector mmap identities remain
unchanged. New feature caches record instruction mode and adapter revisions;
training and validation both reject stale instruction caches. Rebuild task
embeddings/features for the selected sample instead of reusing old empty or
bilingual instructions. Only derived summaries are published here.

Parent identity is separate from language/annotation content. Replay dataset
fingerprints now include language and record metadata, so corrected data cannot
silently resume an old training trajectory. Model state-dict keys are unchanged;
old weight initialization remains supported, while incompatible old exact-resume
contracts must start a fresh run. Fresh-process recovery within this corrected
protocol is part of the real smoke matrix.

## Reproduction and failures

`run_smoke.py` uses the production full-size 2B policy, pretrained DINO/CLIP,
BF16, batch 16, four physical episodes per source, 25% parent holdout, one-second
action windows, ten warmup updates, validation every five updates and two
validation batches. Each source trains to ten steps and resumes to twenty.
The adapter may pin action frequency to its native on-disk delta rate; inspect
the logged effective frequency. Agibot, Galaxea and the combined mix run
serially on one A800. Bulk preprocessing and frame-cache building are disabled;
existing image caches are read. The vector reader may lazily populate small
parquet caches for the selected episodes. No full-source cache rebuild occurs.

```bash
PYTHONPATH=src:. python context/validation/subtasks/run_smoke.py \
  --dataset agibot --output /private/output/agibot
PYTHONPATH=src:. python context/validation/subtasks/run_smoke.py \
  --dataset agibot --output /private/output/agibot --steps 20 \
  --resume /private/output/agibot/10.pt
```

Use `galaxea` and `mixed` for the other two arms. Raw logs/checkpoints are private
under `lbm-validation-20260913/subtasks` on js_dev_1. The first Galaxea attempt
failed because concatenated bilingual text exceeded CLIP's 77-token context;
its output is preserved under `attempts/clip-overflow`. It was not counted as a
completed training run. The corrected run uses source-provided English text.

## Completed results

| Source | Final loss | Final held-out reconstruction error | Peak allocated GPU memory |
| --- | ---: | --- | ---: |
| agibot | 0.9453 | agibot 0.804080 | 18.22 GiB |
| galaxea | 2.1562 | galaxea 6.054416 | 18.22 GiB |
| mixed | 1.1250 | agibot 0.598409, galaxea 7.856572 | 18.45 GiB |

Loss values are from short independent workloads, not a controlled ranking.
All twelve sampled camera-cache jobs were structurally ready; 36 selected
frames decoded successfully. This is not full-corpus integrity or source-pixel
equivalence evidence. See `summary.json` for the recorded protocol and metrics.

Final CPU regression: **531 passed, 27 skipped, 14 deselected**. Ruff and Python
syntax checks pass. The [periodic review](REVIEW.md) establishes PR45 as the
next normalization-review baseline after merge.
