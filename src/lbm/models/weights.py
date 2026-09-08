"""Checkpoint IO for encoder towers: safetensors / pickle, optional HTTP fetch."""

from __future__ import annotations

import json
import os
import struct
import urllib.request
from pathlib import Path

import torch

_SAFETENSORS_DTYPE = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
}


def hf_endpoints() -> list[str]:
    seen: list[str] = []
    for item in (os.environ.get("HF_ENDPOINT"), "https://huggingface.co", "https://hf-mirror.com"):
        if not item:
            continue
        item = item.rstrip("/")
        if item not in seen:
            seen.append(item)
    return seen


def hf_file_url(repo: str, filename: str) -> str:
    return f"{hf_endpoints()[0]}/{repo}/resolve/main/{filename}"


def download_if_missing(url: str, path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "lbm"})
    print(f"downloading {url}\n  -> {path}", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=60) as src, tmp.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def _head_ok(url: str, timeout: float = 8) -> bool | None:
    """True if HEAD is 2xx/3xx, False if clearly dead, None if HEAD is inconclusive."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "lbm"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if 200 <= status < 400:
                return True
            if status in {404, 410, 502, 503, 504}:
                return False
            return None
    except Exception:
        return False


def download_hf_file(repo: str, filename: str, dest: Path, *, url_override: str | None = None) -> Path:
    dest = Path(dest).expanduser()
    if dest.is_file():
        return dest
    if url_override:
        return download_if_missing(url_override, dest)
    errors: list[str] = []
    for endpoint in hf_endpoints():
        url = f"{endpoint}/{repo}/resolve/main/{filename}"
        reachable = _head_ok(url)
        if reachable is False:
            errors.append(f"{url}: unreachable")
            continue
        try:
            return download_if_missing(url, dest)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"failed to download {repo}/{filename}:\n" + "\n".join(errors))


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        blob = f.read()
    out: dict[str, torch.Tensor] = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = _SAFETENSORS_DTYPE[info["dtype"]]
        tensor = torch.frombuffer(bytearray(blob[start:end]), dtype=dtype).clone()
        out[key] = tensor.reshape(info["shape"])
    return out


def read_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    path = Path(path).expanduser()
    if path.suffix == ".safetensors":
        return read_safetensors(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and _is_tensor_dict(ckpt["model"]):
        return ckpt["model"]
    if _is_tensor_dict(ckpt):
        return ckpt
    raise RuntimeError(f"unrecognized checkpoint: {path}")


def _is_tensor_dict(value) -> bool:
    return isinstance(value, dict) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def strip_prefixes(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        out[key] = value
    return out


def first_existing(*candidates: Path) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def local_checkpoint_candidates(subdir: str, *names: str) -> list[Path]:
    """Search ``lbm/checkpoints/<subdir>``, then the checkpoints root, then ``~/.cache/<subdir>``."""
    from lbm.config import default_checkpoints_dir

    root = default_checkpoints_dir()
    home = Path.home() / ".cache" / subdir
    out: list[Path] = []
    for name in names:
        out.append(root / subdir / name)
        out.append(root / name)
        out.append(home / name)
    return out


def env_paths(*names: str) -> list[Path]:
    """Resolve checkpoint paths from env vars. ``lbm_DINO`` also checks ``LBM_DINO``."""
    out: list[Path] = []
    seen: set[str] = set()
    for name in names:
        aliases = [name]
        if name != name.upper():
            aliases.append(name.upper())
        if name.startswith("LBM_") and name != name.lower():
            aliases.append("lbm_" + name[4:])
        for alias in aliases:
            if alias in seen:
                continue
            seen.add(alias)
            value = os.environ.get(alias)
            if value:
                out.append(Path(value).expanduser())
    return out


def ensure_hf_cached(
    resolve,
    dest: Path,
    *,
    repo: str,
    filename: str,
    url_env: str,
) -> Path:
    path = resolve()
    if path is not None:
        return path
    return download_hf_file(repo, filename, dest, url_override=os.environ.get(url_env))


def cat_keys(state: dict[str, torch.Tensor], *keys: str, dim: int = 0) -> torch.Tensor:
    return torch.cat([state[key] for key in keys], dim=dim)


def layer_ids(state: dict[str, torch.Tensor], prefix: str, field: int) -> list[int]:
    return sorted({int(key.split(".")[field]) for key in state if key.startswith(prefix)})


def load_into(
    module,
    state: dict[str, torch.Tensor],
    *,
    name: str,
    skip_missing: tuple[str, ...] = (),
):
    missing, unexpected = module.load_state_dict(state, strict=False)
    if skip_missing:
        missing = [key for key in missing if not key.endswith(skip_missing)]
    if missing:
        raise RuntimeError(f"missing {name} keys: {missing[:8]}")
    if unexpected:
        raise RuntimeError(f"unexpected {name} keys: {unexpected[:8]}")
    return state, unexpected
