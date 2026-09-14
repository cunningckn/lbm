"""Resumable orchestration for a complete MolmoMotion cache release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from .archives import build_tar_index
from .common import (
    fsync_directory,
    read_json,
    sha256_file,
    utc_now,
    verify_checksum_manifest,
    write_checksum_manifest,
    write_json,
    write_parquet,
)
from .droid import build_droid_cache
from .generic import GENERIC_SUBSETS, build_generic_cache


SOURCE_DATASETS = ("droid", "egodex", "hdepic", "molmospaces", "stereo4d", "xperience", "ytvis")


def _tree_manifest(source_root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    manifests = sorted((source_root / ".cache" / "huggingface" / "trees").glob("*.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"expected exactly one Hugging Face tree manifest under {source_root}, got {len(manifests)}"
        )
    document = read_json(manifests[0])
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"invalid Hugging Face tree manifest: {manifests[0]}")
    return manifests[0], files


def _expected_size(metadata: dict[str, Any]) -> int:
    return int(metadata.get("lfs_size", metadata["size"]))


def _hash_path(path_text: str) -> tuple[str, str]:
    path = Path(path_text)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return path_text, digest.hexdigest()


def inspect_source_snapshot(
    source_root: str | Path,
    *,
    verify_hashes: bool,
    workers: int,
) -> dict[str, Any]:
    """Validate every path and byte count in the pinned HF local-dir manifest."""

    root = Path(source_root).resolve()
    manifest_path, files = _tree_manifest(root)
    missing: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    present_bytes = 0
    for relative, metadata in sorted(files.items()):
        path = root / relative
        expected = _expected_size(metadata)
        if not path.is_file():
            missing.append(relative)
            continue
        actual = path.stat().st_size
        present_bytes += actual
        if actual != expected:
            size_mismatches.append(
                {"path": relative, "expected_bytes": expected, "actual_bytes": actual}
            )
    hash_mismatches: list[dict[str, str]] = []
    hashed_files = 0
    hash_candidates = [
        (relative, root / relative, str(metadata["lfs_sha256"]))
        for relative, metadata in sorted(files.items())
        if metadata.get("lfs_sha256") and (root / relative).is_file()
    ]
    if verify_hashes and not missing and not size_mismatches:
        # Hashing is storage-bound. Hundreds of readers degrade shared-CFS
        # throughput even when the conversion job was allocated 100 CPUs.
        pool_size = max(1, min(workers, 32, len(hash_candidates)))
        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            futures = {
                executor.submit(_hash_path, str(path)): (relative, expected)
                for relative, path, expected in hash_candidates
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                path_text, actual = future.result()
                relative, expected = futures[future]
                if actual != expected:
                    hash_mismatches.append(
                        {"path": relative, "expected_sha256": expected, "actual_sha256": actual}
                    )
                if completed == 1 or completed % 10 == 0 or completed == len(futures):
                    print(
                        f"[preflight] hashed LFS files {completed}/{len(futures)}",
                        flush=True,
                    )
            hashed_files = len(hash_candidates)
    expected_bytes = sum(_expected_size(metadata) for metadata in files.values())
    status = "passed" if not (missing or size_mismatches or hash_mismatches) else "failed"
    return {
        "status": status,
        "source_root": str(root),
        "source_revision": manifest_path.stem,
        "manifest_path": str(manifest_path),
        "expected_files": len(files),
        "expected_bytes": expected_bytes,
        "present_bytes": present_bytes,
        "missing_files": missing,
        "size_mismatches": size_mismatches,
        "hash_verification_requested": verify_hashes,
        "hashed_lfs_files": hashed_files,
        "hash_mismatches": hash_mismatches,
        "checked_at": utc_now(),
    }


def require_complete_source(result: dict[str, Any]) -> None:
    if result["status"] == "passed":
        return
    raise RuntimeError(
        "MolmoMotion source preflight failed: "
        f"missing={len(result['missing_files'])}, "
        f"size_mismatches={len(result['size_mismatches'])}, "
        f"hash_mismatches={len(result['hash_mismatches'])}"
    )


def verify_release(output_root: str | Path, *, verify_files: bool) -> dict[str, Any]:
    """Validate release markers and, optionally, every delivery file hash."""

    output = Path(output_root).resolve()
    if not (output / "READY.json").is_file():
        raise FileNotFoundError(f"release has no top-level READY.json: {output}")
    components = {
        **{dataset: output / "subsets" / dataset for dataset in SOURCE_DATASETS},
        "assets": output / "assets",
    }
    results: dict[str, dict[str, Any]] = {}
    for name, component in components.items():
        marker = component / "READY.json"
        manifest = component / "SHA256SUMS"
        if not marker.is_file() or not manifest.is_file():
            raise FileNotFoundError(f"incomplete release component: {component}")
        readiness = read_json(marker)
        expected = readiness.get("sha256sums_sha256")
        actual = sha256_file(manifest)
        if expected != actual:
            raise ValueError(
                f"SHA256SUMS digest mismatch for {name}: expected {expected}, got {actual}"
            )
        result: dict[str, Any] = {
            "status": "manifest-verified",
            "sha256sums_sha256": actual,
        }
        if verify_files:
            result.update(verify_checksum_manifest(component))
            result["status"] = "files-verified"
        results[name] = result
    return {
        "status": "passed",
        "output": str(output),
        "verify_files": verify_files,
        "components": results,
        "verified_at": utc_now(),
    }


def _selected_split_rows(path: Path, limit: int | None) -> list[tuple[str, dict[str, Any]]]:
    document = read_json(path)
    rows: list[tuple[str, dict[str, Any]]] = []
    for split in ("train", "test"):
        values = document.get(split)
        if not isinstance(values, list):
            raise ValueError(f"invalid Stereo4D split file: {path}")
        for metadata in values if limit is None else values[:limit]:
            rows.append((split, metadata))
    return rows


def build_stereo4d_metadata(
    source_root: str | Path,
    output: str | Path,
    *,
    limit: int | None,
) -> dict[str, Any]:
    """Convert the only shipped Stereo4D payload: annotations and track indices."""

    root = Path(source_root).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing metadata output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
    try:
        subset = root / "stereo4d"
        rows = _selected_split_rows(subset / "annotations" / "stereo4d_split.json", limit)
        track_document = read_json(subset / "track_index" / "stereo4d_track_index.json")
        track_clips = track_document.get("clips")
        if not isinstance(track_clips, dict):
            raise ValueError("Stereo4D track index has no clips mapping")
        clips: list[dict[str, Any]] = []
        objects: list[dict[str, Any]] = []
        for split, metadata in rows:
            video_id = str(metadata["file"])
            indexed = track_clips.get(video_id)
            if not isinstance(indexed, dict):
                raise KeyError(f"Stereo4D canonical clip missing from track index: {video_id}")
            indexed_objects = indexed.get("objects")
            if not isinstance(indexed_objects, dict):
                raise ValueError(f"Stereo4D clip has no object index: {video_id}")
            clips.append(
                {
                    "sample_id": f"stereo4d/metadata/{video_id}",
                    "video_id": video_id,
                    "split": split,
                    "caption": str(metadata.get("caption", "")),
                    "fps": float(metadata["fps"]),
                    "num_frames": int(metadata["num_frames"]),
                    "num_objects": len(indexed_objects),
                    "origin_shift_json": json.dumps(indexed.get("T0", []), separators=(",", ":")),
                }
            )
            ranges = metadata.get("clips_by_object", {})
            for object_id, row_indices in indexed_objects.items():
                objects.append(
                    {
                        "sample_id": f"stereo4d/metadata/{video_id}",
                        "object_id": str(object_id),
                        "source_row_indices_json": json.dumps(row_indices, separators=(",", ":")),
                        "motion_ranges_json": json.dumps(
                            ranges.get(object_id, []), separators=(",", ":")
                        ),
                    }
                )
        write_parquet(clips, staged / "clips.parquet")
        write_parquet(objects, staged / "track_index.parquet")
        scope = "pilot" if limit is not None else "complete-published-metadata"
        write_json(
            staged / "dataset.json",
            {
                "format": "molmo-motion-cache",
                "format_version": 1,
                "dataset": "stereo4d",
                "build_scope": scope,
                "records": len(clips),
                "trajectory_arrays_present": False,
                "requires_upstream_reconstruction": True,
                "reconstruction_note": (
                    "The MolmoMotion release ships only a Stereo4D track index; tracks, "
                    "camera, and RGB must be reconstructed from the public upstream data."
                ),
                "created_at": utc_now(),
            },
        )
        manifest_hash = write_checksum_manifest(staged)
        marker = "PILOT_READY.json" if limit is not None else "READY.json"
        ready = {
            "status": "metadata-only-ready",
            "dataset": "stereo4d",
            "records": len(clips),
            "sha256sums_sha256": manifest_hash,
            "created_at": utc_now(),
        }
        write_json(staged / marker, ready)
        fsync_directory(staged)
        os.replace(staged, output_path)
        fsync_directory(output_path.parent)
        return {"output": str(output_path), **ready}
    except BaseException as error:
        raise RuntimeError(
            f"Stereo4D metadata build failed; partial staging was preserved at {staged}"
        ) from error


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _asset_kind(member_name: str) -> str:
    if member_name.startswith("videos/"):
        return "video"
    if member_name.startswith("robot_trajectories/"):
        return "robot_trajectory"
    return "other"


def build_portable_assets(
    source_root: str | Path,
    output: str | Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Retain shipped MP4/H5 tar shards and add byte-offset member indices."""

    root = Path(source_root).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing asset output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
    try:
        manifest_path, files = _tree_manifest(root)
        archive_paths = sorted(
            [
                *(root / "molmospaces" / "videos").glob("videos-*.tar"),
                *(root / "molmospaces" / "robot_trajectories").glob(
                    "robot_trajectories-*.tar"
                ),
            ]
        )
        index = build_tar_index(root, archive_paths, workers=workers)
        archive_modes: dict[str, str] = {}
        for source in archive_paths:
            relative = source.relative_to(root)
            target = staged / "archives" / relative
            archive_modes[relative.as_posix()] = _link_or_copy(source, target)
        asset_rows = [
            {
                "asset_kind": _asset_kind(reference.member_name),
                "member_name": reference.member_name,
                "archive": f"archives/{reference.tar_relpath}",
                "offset": reference.offset,
                "size": reference.size,
            }
            for reference in sorted(index.values(), key=lambda value: value.member_name)
        ]
        write_parquet(asset_rows, staged / "assets_index.parquet")

        metadata_files = [
            relative
            for relative in sorted(files)
            if not relative.endswith(".tar") and (root / relative).is_file()
        ]
        for relative in metadata_files:
            _link_or_copy(root / relative, staged / "source_metadata" / relative)
        archive_manifest = []
        for source in archive_paths:
            relative = source.relative_to(root).as_posix()
            metadata = files[relative]
            archive_manifest.append(
                {
                    "path": f"archives/{relative}",
                    "size_bytes": source.stat().st_size,
                    "source_lfs_sha256": metadata.get("lfs_sha256"),
                    "materialization": archive_modes[relative],
                }
            )
        write_parquet(archive_manifest, staged / "archive_manifest.parquet")
        write_json(
            staged / "dataset.json",
            {
                "format": "molmo-motion-portable-assets",
                "format_version": 1,
                "source_revision": manifest_path.stem,
                "archives": len(archive_paths),
                "indexed_assets": len(asset_rows),
                "metadata_files": len(metadata_files),
                "runtime_paths_relative": True,
                "created_at": utc_now(),
            },
        )
        trusted_hashes = {
            f"archives/{relative}": str(files[relative]["lfs_sha256"])
            for relative in archive_modes
            if files[relative].get("lfs_sha256")
        }
        manifest_hash = write_checksum_manifest(staged, trusted_hashes)
        ready = {
            "status": "ready",
            "archives": len(archive_paths),
            "indexed_assets": len(asset_rows),
            "sha256sums_sha256": manifest_hash,
            "created_at": utc_now(),
        }
        write_json(staged / "READY.json", ready)
        fsync_directory(staged)
        os.replace(staged, output_path)
        fsync_directory(output_path.parent)
        return {"output": str(output_path), **ready}
    except BaseException as error:
        raise RuntimeError(
            f"portable asset build failed; partial staging was preserved at {staged}"
        ) from error


def _ready(path: Path, *, pilot: bool) -> bool:
    marker = path / ("PILOT_READY.json" if pilot else "READY.json")
    return marker.is_file()


def _build_or_resume(
    target: Path,
    *,
    pilot: bool,
    build: Any,
) -> dict[str, Any]:
    if _ready(target, pilot=pilot):
        return {"output": str(target), "status": "resumed-existing-ready"}
    if target.exists():
        raise FileExistsError(
            f"existing output has no readiness marker and will not be overwritten: {target}"
        )
    return build()


def build_full_release(
    source_root: str | Path,
    output_root: str | Path,
    *,
    workers: int,
    shard_size: int,
    checks: int,
    limit_per_subset: int | None,
    verify_source_hashes: bool,
    checksums: bool,
) -> dict[str, Any]:
    """Build or resume all seven published MolmoMotion subset deliverables."""

    if workers <= 0 or shard_size <= 0:
        raise ValueError("workers and shard_size must be positive")
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pilot = limit_per_subset is not None
    top_marker = output / ("PILOT_READY.json" if pilot else "READY.json")
    if top_marker.exists():
        return {"output": str(output), "status": "resumed-existing-ready"}

    preflight = inspect_source_snapshot(
        source,
        verify_hashes=verify_source_hashes,
        workers=workers,
    )
    write_json(output / "source_preflight.json", preflight)
    require_complete_source(preflight)
    subset_root = output / "subsets"
    subset_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    droid_target = subset_root / "droid"
    results["droid"] = _build_or_resume(
        droid_target,
        pilot=pilot,
        build=lambda: build_droid_cache(
            source,
            droid_target,
            limit=limit_per_subset,
            shard_size=shard_size,
            checks=checks,
        ),
    )
    for dataset in GENERIC_SUBSETS:
        target = subset_root / dataset
        results[dataset] = _build_or_resume(
            target,
            pilot=pilot,
            build=lambda dataset=dataset, target=target: build_generic_cache(
                source,
                target,
                dataset,
                limit_per_track_kind=limit_per_subset,
                shard_size=shard_size,
                workers=workers,
                checks=checks,
                checksums=checksums,
            ),
        )
    stereo_target = subset_root / "stereo4d"
    results["stereo4d"] = _build_or_resume(
        stereo_target,
        pilot=pilot,
        build=lambda: build_stereo4d_metadata(source, stereo_target, limit=limit_per_subset),
    )
    if not pilot:
        asset_target = output / "assets"
        results["assets"] = _build_or_resume(
            asset_target,
            pilot=False,
            build=lambda: build_portable_assets(source, asset_target, workers=workers),
        )

    summary = {
        "format": "molmo-motion-cache-release",
        "format_version": 1,
        "status": "pilot-ready" if pilot else "ready",
        "source_revision": preflight["source_revision"],
        "source_files": preflight["expected_files"],
        "source_bytes": preflight["expected_bytes"],
        "subsets": list(SOURCE_DATASETS),
        "results": results,
        "portable": True,
        "limitations": [
            "Stereo4D publishes metadata/track indices only; numeric tracks, camera, and RGB require reconstruction.",
            "Xperience publishes tracks only; camera and RGB require gated upstream reconstruction.",
            "DROID, EgoDex, HD-EPIC, and YTVIS RGB require their documented upstream reconstruction.",
        ],
        "created_at": utc_now(),
    }
    write_json(output / "dataset.json", summary)
    write_json(top_marker, summary)
    fsync_directory(output)
    return {"output": str(output), **summary}
