"""Episode-index cache at ``{dataset_root}/.cache/episodes``.

Write ``repos.jsonl``, then ``episodes.parquet``, then ``manifest.json`` (commit).
A missing directory is a cold start; anything incomplete or mismatched raises.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from lbm.dataloader.custom.common.lerobot import LerobotCam, LerobotDump, lerobot_of
from lbm.dataloader.custom.record import EpisodeRecord
from lbm.dataloader.custom.spec import CustomSpec
from lbm.dataloader.mmap.mmap_io import atomic_write_text, exclusive_cache_lock

SCANNER_VERSION = 1
CACHE_REL = Path(".cache") / "episodes"
MANIFEST = "manifest.json"
REPOS = "repos.jsonl"
EPISODES = "episodes.parquet"

REPO_KEYS = ("repo", "version", "info", "video_tmpl", "fps")
EPISODE_COLUMNS = (
    "kind",
    "path",
    "n_frames",
    "lang",
    "extra_json",
    "repo_index",
    "episode_index",
    "parquet",
    "chunk",
    "videos_json",
)
DUMP_COLUMNS = ("repo_index", "episode_index", "parquet", "chunk", "videos_json")
CAM_KEYS = ("video_key", "chunk", "file_index", "from_timestamp")


class ScanIndexError(ValueError):
    """``{root}/.cache/episodes`` is incomplete, mismatched, or unreadable."""


def cache_dir(root: Path | str) -> Path:
    return Path(root) / CACHE_REL


def save_scan_index(root: Path | str, spec: CustomSpec, records: list[EpisodeRecord]) -> Path:
    if not records:
        raise ScanIndexError("refusing to write an empty scan index")
    root = Path(root)
    dest = cache_dir(root)
    dest.mkdir(parents=True, exist_ok=True)
    repos, rows = _pack(records)
    with exclusive_cache_lock(dest / ".lock"):
        atomic_write_text(dest / REPOS, "".join(_dumps(row) + "\n" for row in repos))
        tmp = dest / f"{EPISODES}.tmp"
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.replace(dest / EPISODES)
        atomic_write_text(dest / MANIFEST, _dumps(_manifest(root, spec, records), indent=2) + "\n")
    return dest


def load_scan_index(root: Path | str, spec: CustomSpec) -> list[EpisodeRecord] | None:
    """``None`` only when ``.cache/episodes`` does not exist."""
    dest = cache_dir(root)
    if not dest.is_dir():
        return None
    paths = (dest / MANIFEST, dest / REPOS, dest / EPISODES)
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise _fail(dest, f"incomplete scan index, missing {missing}")
    man = json.loads(paths[0].read_text(encoding="utf-8"))
    _check_manifest(man, Path(root), spec, dest)
    records = _unpack(_read_repos(paths[1]), pd.read_parquet(paths[2]))
    n_records = int(_need(man, "n_records"))
    if len(records) != n_records:
        raise _fail(dest, f"episode count {len(records)} != manifest n_records {n_records}")
    for label, rec in (("first", records[0]), ("last", records[-1])):
        path = Path(rec.path)
        if not path.exists():
            raise _fail(dest, f"{label} episode path missing: {path}")
    return records


def _spec_fields(spec: CustomSpec) -> dict[str, Any]:
    return {"name": spec.name, "camera_keys": list(spec.camera_keys), "kind": spec.kind, "fps": float(spec.fps)}


def _manifest(root: Path, spec: CustomSpec, records: list[EpisodeRecord]) -> dict[str, Any]:
    return {
        "scanner_version": SCANNER_VERSION,
        "spec": _spec_fields(spec),
        "root": str(root),
        "n_records": len(records),
        "n_steps": int(sum(max(int(r.n_frames), 0) for r in records)),
        "complete": True,
    }


def _dumps(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True)


def _need(data: dict, key: str) -> Any:
    if key not in data:
        raise ScanIndexError(f"scan index missing {key!r}")
    return data[key]


def _fail(dest: Path, msg: str) -> ScanIndexError:
    return ScanIndexError(f"{msg} ({dest}). Pass rescan=True to rebuild.")


def _blank() -> dict[str, Any]:
    return {name: pd.NA for name in DUMP_COLUMNS}


def _is_na(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value))


def _check_manifest(man: dict, root: Path, spec: CustomSpec, dest: Path) -> None:
    if int(_need(man, "scanner_version")) != SCANNER_VERSION:
        raise _fail(dest, f"scanner_version {man['scanner_version']!r} != {SCANNER_VERSION}")
    if _need(man, "complete") is not True:
        raise _fail(dest, "scan index is not marked complete")
    stored = _need(man, "spec")
    if not isinstance(stored, dict):
        raise _fail(dest, "manifest spec is not an object")
    got = {
        "name": str(_need(stored, "name")),
        "camera_keys": list(_need(stored, "camera_keys")),
        "kind": str(_need(stored, "kind")),
        "fps": float(_need(stored, "fps")),
    }
    expect = _spec_fields(spec)
    if got != expect:
        raise _fail(dest, f"spec {got} != {expect}")
    if str(_need(man, "root")) != str(root):
        raise _fail(dest, f"root {man['root']!r} != {str(root)!r}")
    if int(_need(man, "n_records")) < 1:
        raise _fail(dest, "n_records < 1")


def _read_repos(path: Path) -> list[dict]:
    repos = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in REPO_KEYS:
            _need(row, key)
        repos.append(row)
    return repos


def _intern(dump: LerobotDump, repos: list[dict], index: dict[str, int]) -> int:
    key = str(dump.repo)
    if key not in index:
        index[key] = len(repos)
        repos.append(
            {
                "repo": key,
                "version": dump.version,
                "info": dump.info,
                "video_tmpl": dump.video_tmpl,
                "fps": float(dump.fps),
            }
        )
    return index[key]


def _pack(records: list[EpisodeRecord]) -> tuple[list[dict], list[dict]]:
    repos: list[dict] = []
    index: dict[str, int] = {}
    rows: list[dict] = []
    for rec in records:
        extra = dict(rec.extra)
        dump = lerobot_of(rec)
        row: dict[str, Any] = {
            "kind": rec.kind,
            "path": rec.path,
            "n_frames": int(rec.n_frames),
            "lang": rec.lang,
            **_blank(),
        }
        if dump is not None:
            extra.pop("lerobot")
            row["repo_index"] = _intern(dump, repos, index)
            row["episode_index"] = int(dump.episode_index)
            row["parquet"] = str(dump.parquet)
            row["chunk"] = int(dump.chunk)
            row["videos_json"] = _dumps({cam: asdict(meta) for cam, meta in dump.videos.items()})
        row["extra_json"] = _dumps(extra)
        rows.append(row)
    return repos, rows


def _unpack(repos: list[dict], frame: pd.DataFrame) -> list[EpisodeRecord]:
    missing = [c for c in EPISODE_COLUMNS if c not in frame.columns]
    if missing:
        raise ScanIndexError(f"episodes.parquet missing columns {missing}")
    out: list[EpisodeRecord] = []
    for row in frame.to_dict(orient="records"):
        extra = json.loads(row["extra_json"])
        if not isinstance(extra, dict):
            raise ScanIndexError("extra_json is not an object")
        if not _is_na(row["repo_index"]):
            extra["lerobot"] = _dump_from_row(row, repos)
        out.append(
            EpisodeRecord(
                kind=str(row["kind"]),
                path=str(row["path"]),
                n_frames=int(row["n_frames"]),
                lang=str(row["lang"]),
                extra=extra,
            )
        )
    return out


def _cam(meta: dict) -> LerobotCam:
    for key in CAM_KEYS:
        _need(meta, key)
    return LerobotCam(
        video_key=str(meta["video_key"]),
        chunk=int(meta["chunk"]),
        file_index=int(meta["file_index"]),
        from_timestamp=float(meta["from_timestamp"]),
    )


def _dump_from_row(row: dict, repos: list[dict]) -> LerobotDump:
    if any(_is_na(row[c]) for c in DUMP_COLUMNS):
        raise ScanIndexError(f"lerobot episode {row['path']!r} missing dump fields")
    i = int(row["repo_index"])
    if i < 0 or i >= len(repos):
        raise ScanIndexError(f"repo_index {i} out of range for {len(repos)} interned repos")
    repo = repos[i]
    return LerobotDump(
        repo=Path(repo["repo"]),
        version=str(repo["version"]),
        episode_index=int(row["episode_index"]),
        parquet=Path(row["parquet"]),
        info=repo["info"],
        video_tmpl=str(repo["video_tmpl"]),
        fps=float(repo["fps"]),
        chunk=int(row["chunk"]),
        videos={cam: _cam(meta) for cam, meta in json.loads(row["videos_json"]).items()},
    )
