"""Strict, explicit publication and independent-copy exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .common import (
    checksum_entries,
    read_json,
    sha256_file,
    verify_checksum_manifest,
    write_checksum_manifest,
    write_json,
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_fingerprint():
    return fingerprint({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def release_manifest(root, components):
    """Only explicit published content; never include job logs or pilot archives."""
    entries = {}
    for name in ("dataset.json", "source_preflight.json", "build_contract.json"):
        entries[name] = sha256_file(root / name)
    for relative in components:
        component = root / relative
        entries.update({f"{relative}/{name}": digest for name, digest in checksum_entries(component).items()})
        for name in ("SHA256SUMS", "READY.json", "PILOT_READY.json"):
            if (component / name).is_file():
                entries[f"{relative}/{name}"] = sha256_file(component / name)
    with (root / "SHA256SUMS").open("w", encoding="utf-8", newline="\n") as handle:
        for name, digest in sorted(entries.items()):
            handle.write(f"{digest}  {name}\n")
    return sha256_file(root / "SHA256SUMS")


def audit_component(root, *, verify_files=False):
    root = Path(root)
    markers = [p for p in (root / "READY.json", root / "PILOT_READY.json") if p.is_file()]
    if len(markers) != 1 or markers[0].is_symlink():
        raise ValueError("expected exactly one regular readiness marker")
    marker = read_json(markers[0])
    if marker.get("format_version") != 1 or marker.get("status") not in {"ready", "pilot-ready"}:
        raise ValueError("unknown component schema/status")
    if marker.get("sha256sums_sha256") != sha256_file(root / "SHA256SUMS"):
        raise ValueError("READY manifest mismatch")
    result = verify_checksum_manifest(root, verify_files=verify_files)
    entries = checksum_entries(root)
    for name, digest in entries.items():
        if name.endswith((".json", ".parquet")) and sha256_file(root / name) != digest:
            raise ValueError(f"metadata mismatch: {name}")
    return result


def export_component(source, destination):
    """Export a component including pilot components; never overwrite or hardlink."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(source):
        raise ValueError("destination must be outside source")
    audit_component(source, verify_files=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=destination.parent))
    names = [*checksum_entries(source), "SHA256SUMS"]
    marker = "READY.json" if (source / "READY.json").exists() else "PILOT_READY.json"
    for name in names:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    verify_checksum_manifest(staging)
    shutil.copyfile(source / marker, staging / marker)
    audit_component(staging, verify_files=True)
    os.rename(staging, destination)
    return {
        "destination": str(destination),
        "files": len(names) + 1,
        "content_verified": True,
        "independent_copy": True,
    }


def export_release(source, destination):
    from .release import SOURCE_DATASETS, verify_release

    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("export needs a new destination outside source")
    verify_release(source, verify_files=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".release-export-", dir=destination.parent))
    for name in [*(f"subsets/{s}" for s in SOURCE_DATASETS), "assets"]:
        export_component(source / name, staging / name)
    write_json(
        staging / "capabilities.json",
        {
            "schema": 1,
            "state": "annotations-only",
            "full_multimodal": False,
            "rgb": "MolmoSpaces source MP4 archives only; decoded RGB cache not included",
            "stereo4d": "metadata/index-only",
            "export_code": code_fingerprint(),
        },
    )
    digest = write_checksum_manifest(staging)
    verify_checksum_manifest(staging)
    write_json(
        staging / "READY.json",
        {"format_version": 1, "status": "ready", "sha256sums_sha256": digest, "publication_contract": 2},
    )
    os.rename(staging, destination)
    return {"destination": str(destination), "content_verified": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "export-component", "export-release"])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", nargs="?", type=Path)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    result = (
        audit_component(args.source, verify_files=args.verify_files)
        if args.command == "audit"
        else (
            export_release(args.source, args.destination)
            if args.command == "export-release"
            else export_component(args.source, args.destination)
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
