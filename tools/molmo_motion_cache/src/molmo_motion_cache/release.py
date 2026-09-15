"""Resumable orchestration for a complete MolmoMotion cache release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from .archives import build_tar_index
from .common import (
    completion_marker_name,
    fsync_directory,
    read_completion_marker,
    read_json,
    utc_now,
    verify_completion_manifest,
    verify_checksum_manifest,
    write_checksum_manifest,
    write_completion_marker,
    write_json,
    write_parquet,
)
from .droid import build_droid_cache
from .common import GENERIC_SUBSETS
from .generic import build_generic_cache
from .identity import (
    component_identity,
    load_source_snapshot,
    require_snapshot_identity,
    validate_component_identity,
    validate_snapshot_identity,
    with_paired_component_fingerprints,
)


SOURCE_DATASETS = ("droid", "egodex", "hdepic", "molmospaces", "stereo4d", "xperience", "ytvis")


def _expected_size(metadata: Mapping[str, Any]) -> int:
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
    snapshot = load_source_snapshot(root)
    files = snapshot.files
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
        "source_revision": snapshot.revision,
        "manifest_path": str(snapshot.manifest_path),
        "source_identity": snapshot.public_identity(),
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


def _verify_release_source_identity(
    output: Path, components: Mapping[str, Path], *, require_source_identity: bool
) -> dict[str, Any]:
    release = read_json(output / "dataset.json")
    release_identity = release.get("source_identity")
    if release_identity is None:
        if require_source_identity:
            raise ValueError(
                "release predates source identity metadata; it cannot prove component pairing. "
                "Rebuild into a new output directory before cross-component use."
            )
        return {
            "status": "legacy-unverified",
            "warning": (
                "this legacy release has no source/component fingerprints; component pairing "
                "cannot be verified"
            ),
        }
    require_snapshot_identity(release_identity, label="release")
    expected_fingerprints = release.get("component_fingerprints")
    if not isinstance(expected_fingerprints, Mapping):
        raise ValueError("release source identity has no component fingerprint map")
    verified: dict[str, str] = {}
    for name, component in components.items():
        metadata = read_json(component / "dataset.json")
        identity = metadata.get("source_identity")
        validate_snapshot_identity(identity, release_identity, label=f"release component {name}")
        expected_fingerprint = expected_fingerprints.get(name)
        if not isinstance(expected_fingerprint, str):
            raise ValueError(f"release has no expected component fingerprint for {name!r}")
        actual = identity.get("component_fingerprint") if isinstance(identity, Mapping) else None
        if actual != expected_fingerprint:
            raise ValueError(
                f"release component fingerprint mismatch for {name!r}; "
                "do not pair components by video ID alone"
            )
        verified[name] = expected_fingerprint
    return {
        "status": "verified",
        "snapshot_revision": release_identity["snapshot_revision"],
        "snapshot_fingerprint": release_identity["snapshot_fingerprint"],
        "component_fingerprints": verified,
    }


def verify_release(
    output_root: str | Path, *, verify_files: bool, require_source_identity: bool = False
) -> dict[str, Any]:
    """Validate release markers and, optionally, every delivery file hash."""

    output = Path(output_root).resolve()
    read_completion_marker(output, pilot=False)
    components = {
        **{dataset: output / "subsets" / dataset for dataset in SOURCE_DATASETS},
        "assets": output / "assets",
    }
    results: dict[str, dict[str, Any]] = {}
    for name, component in components.items():
        _marker, readiness = read_completion_marker(component, pilot=False)
        actual = verify_completion_manifest(component, readiness)
        result: dict[str, Any] = {
            "status": "manifest-verified",
            "sha256sums_sha256": actual,
        }
        if verify_files:
            result.update(verify_checksum_manifest(component))
            result["status"] = "files-verified"
        results[name] = result
    identity_result = _verify_release_source_identity(
        output, components, require_source_identity=require_source_identity
    )
    return {
        "status": "passed",
        "output": str(output),
        "verify_files": verify_files,
        "components": results,
        "source_identity": identity_result,
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
    source_identity: Mapping[str, Any] | None = None,
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
        dataset_metadata: dict[str, Any] = {
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
        }
        if source_identity is not None:
            dataset_metadata["source_identity"] = dict(source_identity)
        write_json(staged / "dataset.json", dataset_metadata)
        manifest_hash = write_checksum_manifest(staged)
        ready = {
            "status": "metadata-only-ready",
            "dataset": "stereo4d",
            "records": len(clips),
            "sha256sums_sha256": manifest_hash,
            "created_at": utc_now(),
        }
        write_completion_marker(staged, ready, pilot=limit is not None)
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
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Retain shipped MP4/H5 tar shards and add byte-offset member indices."""

    root = Path(source_root).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing asset output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.partial-", dir=output_path.parent))
    try:
        snapshot = load_source_snapshot(root)
        files = snapshot.files
        identity = (
            dict(source_identity)
            if source_identity is not None
            else component_identity(snapshot, "assets")
        )
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
                "source_revision": snapshot.revision,
                "source_identity": identity,
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
        write_completion_marker(staged, ready, pilot=False)
        fsync_directory(staged)
        os.replace(staged, output_path)
        fsync_directory(output_path.parent)
        return {"output": str(output_path), **ready}
    except BaseException as error:
        raise RuntimeError(
            f"portable asset build failed; partial staging was preserved at {staged}"
        ) from error


def _build_or_resume(
    target: Path,
    *,
    pilot: bool,
    build: Any,
    expected_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        read_completion_marker(target, pilot=pilot)
    except FileNotFoundError:
        pass
    else:
        metadata = read_json(target / "dataset.json")
        validate_component_identity(
            metadata.get("source_identity"),
            expected_source_identity,
            label=f"existing component {target}",
        )
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
    if not checksums:
        raise ValueError(
            "build-release requires SHA-256 manifests; use a single-subset command for unchecked experiments"
        )
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pilot = limit_per_subset is not None
    snapshot = load_source_snapshot(source)
    release_identity = snapshot.public_identity()
    top_marker = output / completion_marker_name(pilot=pilot)
    if top_marker.is_file():
        read_completion_marker(output, pilot=pilot)
        existing_summary = read_json(output / "dataset.json")
        validate_snapshot_identity(
            existing_summary.get("source_identity"),
            release_identity,
            label=f"existing release {output}",
        )
        if not pilot:
            verify_release(output, verify_files=False, require_source_identity=True)
        return {"output": str(output), "status": "resumed-existing-ready"}

    preflight = inspect_source_snapshot(
        source,
        verify_hashes=verify_source_hashes,
        workers=workers,
    )
    write_json(output / "source_preflight.json", preflight)
    require_complete_source(preflight)
    component_identities = {
        name: component_identity(snapshot, name)
        for name in (*SOURCE_DATASETS, "assets")
    }
    component_identities["molmospaces"] = with_paired_component_fingerprints(
        component_identities["molmospaces"],
        {"assets": component_identities["assets"]["component_fingerprint"]},
    )
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
            source_identity=component_identities["droid"],
        ),
        expected_source_identity=component_identities["droid"],
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
                source_identity=component_identities[dataset],
            ),
            expected_source_identity=component_identities[dataset],
        )
    stereo_target = subset_root / "stereo4d"
    results["stereo4d"] = _build_or_resume(
        stereo_target,
        pilot=pilot,
        build=lambda: build_stereo4d_metadata(
            source,
            stereo_target,
            limit=limit_per_subset,
            source_identity=component_identities["stereo4d"],
        ),
        expected_source_identity=component_identities["stereo4d"],
    )
    if not pilot:
        asset_target = output / "assets"
        results["assets"] = _build_or_resume(
            asset_target,
            pilot=False,
            build=lambda: build_portable_assets(
                source,
                asset_target,
                workers=workers,
                source_identity=component_identities["assets"],
            ),
            expected_source_identity=component_identities["assets"],
        )

    summary = {
        "format": "molmo-motion-cache-release",
        "format_version": 1,
        "status": "pilot-ready" if pilot else "ready",
        "source_revision": preflight["source_revision"],
        "source_identity": release_identity,
        "component_fingerprints": {
            name: component_identities[name]["component_fingerprint"] for name in results
        },
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
    write_completion_marker(output, summary, pilot=pilot)
    fsync_directory(output)
    return {"output": str(output), **summary}
