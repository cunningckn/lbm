# DROID dataset

## 2026-09-04 — Registered as `droid`

LeRobot v2.1 dump at `datasets/droid` → `/mnt/open_source_data/Droid/droid_1.0.1/` (~95.6k episodes, 15 fps, `robot_type=Franka`).

- Spec: `src/lbm/dataloader/custom/datasets/droid.py`
- Embodiment: `oxe_droid` (id 17, already reserved)
- Cameras: `exterior_1_left`, `exterior_2_left`, `wrist_left`
- State/action: packed `observation.state` / `action` (8-D joint + gripper)
- Action space: `unimanual_joint()` — arm remaps with `--action-mode` (default delta); gripper stays abs
- Spec is the catalog name (`--dataset droid`). `info.json` `robot_type` is not used to pick a spec.
- Language comes from `meta/episodes.jsonl` `tasks` (episode 0 is empty)

Train: `uv run python scripts/train.py --dataset droid`
