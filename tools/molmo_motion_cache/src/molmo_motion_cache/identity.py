"""Stable source-snapshot identities for portable MolmoMotion components."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import read_json


IDENTITY_FORMAT_VERSION = 1


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_size(metadata: Mapping[str, Any]) -> int:
    return int(metadata.get("lfs_size", metadata["size"]))


def _file_identity(relative: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    digest = metadata.get("lfs_sha256")
    return {
        "path": relative,
        "size_bytes": _expected_size(metadata),
        "sha256": str(digest) if digest else None,
    }


@dataclass(frozen=True)
class SourceSnapshot:
    """Pinned Hugging Face tree metadata without a machine-specific source path."""

    manifest_path: Path
    revision: str
    fingerprint: str
    files: Mapping[str, Mapping[str, Any]]

    def public_identity(self) -> dict[str, Any]:
        return {
            "format_version": IDENTITY_FORMAT_VERSION,
            "snapshot_revision": self.revision,
            "snapshot_fingerprint": self.fingerprint,
        }


def load_source_snapshot(source_root: str | Path) -> SourceSnapshot:
    """Load the single pinned local-dir manifest and fingerprint its declared files."""

    root = Path(source_root).resolve()
    manifests = sorted((root / ".cache" / "huggingface" / "trees").glob("*.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"expected exactly one Hugging Face tree manifest under {root}, got {len(manifests)}"
        )
    document = read_json(manifests[0])
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"invalid Hugging Face tree manifest: {manifests[0]}")
    normalized = {str(path): dict(metadata) for path, metadata in files.items()}
    records = [_file_identity(path, metadata) for path, metadata in sorted(normalized.items())]
    fingerprint = _canonical_sha256(
        {
            "format_version": IDENTITY_FORMAT_VERSION,
            "snapshot_revision": manifests[0].stem,
            "files": records,
        }
    )
    return SourceSnapshot(manifests[0], manifests[0].stem, fingerprint, normalized)


def component_source_paths(snapshot: SourceSnapshot, component: str) -> tuple[str, ...]:
    """Return the manifest paths materialized by one release component."""

    if component == "assets":
        paths = [
            path
            for path in snapshot.files
            if not path.endswith(".tar")
            or path.startswith("molmospaces/videos/videos-")
            or path.startswith("molmospaces/robot_trajectories/robot_trajectories-")
        ]
    else:
        prefix = f"{component}/"
        paths = [path for path in snapshot.files if path.startswith(prefix)]
    if not paths:
        raise ValueError(f"source snapshot has no files for component {component!r}")
    return tuple(sorted(paths))


def component_identity(
    snapshot: SourceSnapshot, component: str, paths: Iterable[str] | None = None
) -> dict[str, Any]:
    """Fingerprint a component's declared source files within one snapshot."""

    selected = component_source_paths(snapshot, component) if paths is None else tuple(sorted(paths))
    unknown = [path for path in selected if path not in snapshot.files]
    if unknown:
        raise ValueError(
            f"component {component!r} references paths absent from the source snapshot: {unknown[:3]}"
        )
    records = [_file_identity(path, snapshot.files[path]) for path in selected]
    identity = {
        **snapshot.public_identity(),
        "component": component,
        "component_fingerprint": _canonical_sha256(
            {
                "format_version": IDENTITY_FORMAT_VERSION,
                "snapshot_fingerprint": snapshot.fingerprint,
                "component": component,
                "files": records,
            }
        ),
    }
    return identity


def with_paired_component_fingerprints(
    identity: Mapping[str, Any], fingerprints: Mapping[str, str]
) -> dict[str, Any]:
    """Attach explicit fingerprints required when a reader pairs components."""

    result = dict(identity)
    result["paired_component_fingerprints"] = dict(sorted(fingerprints.items()))
    return result


def _require_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} has no source identity; rebuild it before pairing components")
    for key in ("snapshot_revision", "snapshot_fingerprint", "component", "component_fingerprint"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"{label} source identity is missing {key!r}")
    return value


def require_snapshot_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    """Validate the shared snapshot portion of an identity object."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} has no source identity")
    for key in ("snapshot_revision", "snapshot_fingerprint"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"{label} source identity is missing {key!r}")
    return value


def validate_snapshot_identity(
    actual_identity: Any, expected_identity: Mapping[str, Any], *, label: str
) -> dict[str, str]:
    """Require a result to come from the exact pinned source snapshot."""

    actual = require_snapshot_identity(actual_identity, label=label)
    expected = require_snapshot_identity(expected_identity, label="expected source")
    if actual["snapshot_fingerprint"] != expected["snapshot_fingerprint"]:
        raise ValueError(
            f"{label} was built from a different source snapshot: "
            f"{actual['snapshot_revision']} != {expected['snapshot_revision']}"
        )
    return {
        "snapshot_revision": str(actual["snapshot_revision"]),
        "snapshot_fingerprint": str(actual["snapshot_fingerprint"]),
    }


def validate_component_identity(
    actual_identity: Any, expected_identity: Mapping[str, Any], *, label: str
) -> dict[str, str]:
    """Require a reusable component to match the intended source payload exactly."""

    actual = _require_identity(actual_identity, label=label)
    expected = _require_identity(expected_identity, label="expected component")
    validate_snapshot_identity(actual, expected, label=label)
    if actual["component"] != expected["component"]:
        raise ValueError(
            f"{label} component kind differs: {actual['component']!r} != {expected['component']!r}"
        )
    if actual["component_fingerprint"] != expected["component_fingerprint"]:
        raise ValueError(f"{label} component fingerprint differs; refuse an ambiguous resume")
    return {
        "snapshot_revision": str(actual["snapshot_revision"]),
        "snapshot_fingerprint": str(actual["snapshot_fingerprint"]),
        "component": str(actual["component"]),
        "component_fingerprint": str(actual["component_fingerprint"]),
    }


def validate_component_pair(
    primary_identity: Any,
    secondary_identity: Any,
    *,
    secondary_component: str,
) -> dict[str, str]:
    """Reject same-ID components built from a different snapshot or payload set."""

    primary = _require_identity(primary_identity, label="primary component")
    secondary = _require_identity(secondary_identity, label="secondary component")
    if secondary["component"] != secondary_component:
        raise ValueError(
            f"expected paired component {secondary_component!r}, got {secondary['component']!r}"
        )
    if primary["snapshot_fingerprint"] != secondary["snapshot_fingerprint"]:
        raise ValueError(
            "paired components come from different source snapshots: "
            f"{primary['snapshot_revision']} != {secondary['snapshot_revision']}"
        )
    expected = primary.get("paired_component_fingerprints", {})
    if not isinstance(expected, Mapping) or expected.get(secondary_component) != secondary[
        "component_fingerprint"
    ]:
        raise ValueError(
            f"paired component fingerprint mismatch for {secondary_component!r}; "
            "do not match components by video ID alone"
        )
    return {
        "snapshot_revision": str(primary["snapshot_revision"]),
        "snapshot_fingerprint": str(primary["snapshot_fingerprint"]),
        "primary_component": str(primary["component"]),
        "secondary_component": str(secondary["component"]),
    }
