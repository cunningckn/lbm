"""Per-episode joint→EEF cache at ``{dataset_root}/.cache/fk``.

Each record writes ``episode_{i:06d}/{state,action}.npy``. Training loads these
when ``action_kind=eef``; missing files fall back to live FK.
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from lbm.action_space import EEF, JOINT
from lbm.dataloader.custom.common.lerobot import lerobot_of
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.mmap.mmap_io import atomic_save_npy, atomic_write_text, exclusive_cache_lock, read_manifest
from lbm.kinematics import apply_joint_fk, has_chain

CACHE_REL = Path(".cache") / "fk"
VERSION = 1
MANIFEST = "manifest.json"
STATE_NPY = "state.npy"
ACTION_NPY = "action.npy"


def cache_dir(root: Path | str) -> Path:
    return Path(root) / CACHE_REL


def episode_dir(root: Path | str, index: int) -> Path:
    return cache_dir(root) / f"episode_{int(index):06d}"


def source_key(record: EpisodeRecord) -> str:
    dump = lerobot_of(record)
    if dump is None:
        return str(record.path)
    return f"{record.path}#{int(dump.episode_index)}"


def needs_joint_fk(spec) -> bool:
    return any(
        sl.kind == JOINT and not sl.stored and has_chain(spec.embodiment, sl.width)
        for sl in spec.action_space
    )


def is_ready(dest: Path, *, source: str, n_frames: int) -> bool:
    man = read_manifest(dest / MANIFEST)
    if man is None:
        return False
    if int(man.get("version", -1)) != VERSION:
        return False
    if str(man.get("source")) != str(source):
        return False
    if int(man.get("n_frames", -1)) != int(n_frames):
        return False
    return (dest / STATE_NPY).is_file() and (dest / ACTION_NPY).is_file()


def write_fk_episode(
    dest: Path,
    state: np.ndarray,
    action: np.ndarray,
    *,
    source: str,
    n_frames: int,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    st = np.asarray(state, dtype=np.float32)
    act = np.asarray(action, dtype=np.float32)
    payload = {
        "version": VERSION,
        "source": str(source),
        "n_frames": int(n_frames),
        "state_shape": list(st.shape),
        "action_shape": list(act.shape),
    }
    with exclusive_cache_lock(dest / ".lock"):
        atomic_save_npy(dest / STATE_NPY, st)
        atomic_save_npy(dest / ACTION_NPY, act)
        atomic_write_text(dest / MANIFEST, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_fk_episode(
    root: Path | str,
    index: int,
    *,
    source: str,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    dest = episode_dir(root, index)
    if not is_ready(dest, source=source, n_frames=n_frames):
        return None
    state = np.load(dest / STATE_NPY, mmap_mode="r")
    action = np.load(dest / ACTION_NPY, mmap_mode="r")
    return np.asarray(state), np.asarray(action)


def prebuild_fk(dataset, *, workers: int = 0, force: bool = False) -> tuple[int, int]:
    """Write FK state/action for every record. Returns ``(built, reused)``."""
    spec = dataset.spec
    root = getattr(dataset, "root", None)
    records = getattr(dataset, "records", None) or []
    if root is None or not records:
        print(f"[fk] {spec.name}: no on-disk records, skip", flush=True)
        return 0, 0
    if not needs_joint_fk(spec):
        print(f"[fk] {spec.name}: no joint FK, skip", flush=True)
        return 0, 0
    jobs = _jobs(dataset, force=force)
    n_skip = len(records) - len(jobs)
    built = _run_jobs(jobs, workers=workers, spec_name=spec.name)
    _write_root_manifest(root, spec, n_records=len(records))
    print(f"[fk] {spec.name}: built={built} reused={n_skip} -> {cache_dir(root)}", flush=True)
    return built, n_skip


def _jobs(dataset, *, force: bool) -> list[dict[str, Any]]:
    root = Path(dataset.root)
    jobs: list[dict[str, Any]] = []
    for i, rec in enumerate(dataset.records):
        dest = episode_dir(root, i)
        source = source_key(rec)
        if not force and is_ready(dest, source=source, n_frames=rec.n_frames):
            continue
        jobs.append(
            {
                "spec": dataset.spec.name,
                "kind": rec.kind,
                "path": rec.path,
                "n_frames": int(rec.n_frames),
                "lang": rec.lang,
                "extra": rec.extra,
                "dest": str(dest),
                "source": source,
                "action_mode": dataset.action_mode,
                "action_freq": float(dataset.action_freq),
                "force": bool(force),
            }
        )
    return jobs


def _write_root_manifest(root: Path | str, spec, n_records: int) -> None:
    dest = cache_dir(root)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "spec": spec.name,
        "embodiment": spec.embodiment,
        "n_records": int(n_records),
        "complete": True,
    }
    atomic_write_text(dest / MANIFEST, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fk_job(job: dict[str, Any]) -> bool:
    from lbm.dataloader.custom.datasets import CUSTOM_SPECS
    from lbm.dataloader.custom.scan import read_vectors

    spec = CUSTOM_SPECS[job["spec"]]
    record = EpisodeRecord(
        kind=str(job["kind"]),
        path=str(job["path"]),
        n_frames=int(job["n_frames"]),
        lang=str(job["lang"] or ""),
        extra=dict(job.get("extra") or {}),
    )
    dest = Path(job["dest"])
    source = str(job["source"])
    if not job.get("force") and is_ready(dest, source=source, n_frames=record.n_frames):
        return False
    state, action = read_vectors(record, spec, action_freq=job.get("action_freq"))
    st, act, _slices = apply_joint_fk(state, action, spec, job["action_mode"], EEF)
    write_fk_episode(dest, st, act, source=source, n_frames=record.n_frames)
    return True


def _run_jobs(jobs: list[dict[str, Any]], *, workers: int, spec_name: str) -> int:
    if not jobs:
        return 0
    import os

    from lbm.utils.progress import track

    nproc = int(workers) if workers and workers > 0 else max(1, min(32, os.cpu_count() or 8))
    nproc = max(1, min(nproc, len(jobs)))
    desc = f"fk {spec_name}"
    if nproc == 1:
        return sum(int(_fk_job(job)) for job in track(jobs, desc=desc, unit="ep"))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as pool:
        futures = [pool.submit(_fk_job, job) for job in jobs]
        return sum(
            int(fut.result())
            for fut in track(
                as_completed(futures), total=len(futures), desc=f"{desc} x{nproc}", unit="ep"
            )
        )
