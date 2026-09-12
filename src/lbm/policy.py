"""Load an LBM checkpoint and run ``sample_actions`` on sim observations."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lbm.action_space import (
    action_space_from_payload,
    apply_action_space,
    invert_action_space,
)
from lbm.config import ClipConfig, DiTConfig, FlowConfig
from lbm.dataloader.custom.datasets import CUSTOM_SPECS
from lbm.dataloader.paths import resolve_dataset
from lbm.models.clip import CLIPTextEmbedder
from lbm.models.dit import DiTPolicy, load_pretrained
from lbm.temporal import n_steps
from lbm.utils.preprocess import (
    load_norm_stats,
    norm_stats_filename,
    normalize,
    resize_pad_normalize,
    unnormalize,
)


def dit_config_from_mapping(raw: dict[str, Any]) -> DiTConfig:
    """Build ``DiTConfig`` from ``train_config.json``'s ``model`` object."""
    allowed = {f.name for f in fields(DiTConfig)}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in allowed:
            continue
        if key == "camera_keys":
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return DiTConfig(**kwargs)


def load_train_model_config(path: Path) -> tuple[DiTConfig, int]:
    """Return (model config, diffusion steps) from a saved ``train_config.json``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    model_raw = raw["model"] if "model" in raw else raw
    steps = int(raw["flow"]["num_diffusion_steps"])
    return dit_config_from_mapping(model_raw), steps


def apply_spec(config: DiTConfig, robot_type: str) -> DiTConfig:
    """Fill cameras / dims / fps from the LBM dataset spec when known."""
    spec = CUSTOM_SPECS[robot_type]
    kwargs: dict[str, Any] = {
        "camera_keys": tuple(spec.camera_keys),
        "state_dim": int(spec.state_dim),
        "action_dim": int(spec.action_dim),
    }
    if spec.fps > 0:
        kwargs["action_freq"] = float(spec.fps)
    return replace(config, **kwargs)


def _identity_stats(dim: int) -> dict[str, np.ndarray]:
    z = np.zeros(dim, dtype=np.float32)
    o = np.ones(dim, dtype=np.float32)
    return {"mean": z, "std": o, "min": z, "max": o, "count": 0}


def _stats_dimension(stats):
    return int(np.asarray(stats['q01'] if 'q01' in stats else stats['mean']).size)


def _fit_vector(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, int(x.size))
    out[:n] = x[:n]
    return out


def _hwc_to_chw(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] in {1, 3, 4}:
        return np.transpose(arr, (2, 0, 1))
    return arr


def resolve_norm_stats_path(
    *,
    explicit: Path | None,
    ckpt: Path,
    robot_type: str,
    action_freq: float | None = None,
    action_length: float | None = None,
    slices: tuple = (),
) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(Path(explicit))
    ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent
    candidates.append(ckpt_dir / "norm_stats.json")
    if robot_type and action_freq is not None:
        dump = resolve_dataset(robot_type, required=False)
        if dump is not None:
            candidates.append(
                dump / norm_stats_filename(action_freq, action_length, slices=slices)
            )
    for path in candidates:
        if path.is_file():
            return path
    return None


class LBMPolicy:
    """In-process LBM policy: sim obs ``{state, images, prompt}`` → action chunk."""

    def __init__(
        self,
        model: DiTPolicy,
        *,
        config: DiTConfig,
        device: torch.device,
        norm_stats: dict[str, Any],
        diffusion_steps: int = 10,
        clip: ClipConfig | None = None,
        embodiment_id: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.model_config = config
        self.config = type("_Cfg", (), {"model": config})()
        self.device = device
        self.dtype = dtype
        self.norm_stats = norm_stats
        self.action_space = ()
        self.camera_keys = tuple(config.camera_keys)
        self.state_dim = _stats_dimension(norm_stats['state'])
        self.action_dim = _stats_dimension(norm_stats['actions'])
        self.output_steps = config.chunk_length
        if self.state_dim > config.state_dim or self.action_dim > config.action_dim:
            raise ValueError('source normalization dimensions exceed checkpoint dimensions')
        self.diffusion_steps = int(diffusion_steps)
        self.embodiment_id = int(embodiment_id)
        self._clip: CLIPTextEmbedder | None = None
        self._clip_cfg = clip or ClipConfig()
        self._task_cache: dict[str, torch.Tensor] = {}
        self._history: dict[str, deque[np.ndarray]] = {
            cam: deque(maxlen=max(1, config.history_size)) for cam in config.camera_keys
        }
        self.task_vec = torch.zeros(1, config.task_embed_dim, device=device, dtype=dtype)

    @classmethod
    def from_checkpoint(
        cls,
        ckpt: str | Path,
        *,
        robot_type: str = "",
        config_path: str | Path | None = None,
        norm_stats_path: str | Path | None = None,
        device: str = "auto",
        diffusion_steps: int | None = None,
    ) -> LBMPolicy:
        ckpt_path = Path(ckpt).expanduser()
        if ckpt_path.is_dir():
            pt = sorted(ckpt_path.glob("*.pt"))
            if not pt:
                raise FileNotFoundError(f"no .pt checkpoint in {ckpt_path}")
            ckpt_path = pt[-1]

        cfg_file = Path(config_path) if config_path else ckpt_path.parent / "train_config.json"
        if cfg_file.is_file():
            config, saved_steps = load_train_model_config(cfg_file)
        else:
            config, saved_steps = DiTConfig(), FlowConfig().num_diffusion_steps
        if not robot_type:
            raise ValueError("from_checkpoint requires robot_type")
        # A saved config describes trained parameter shapes, including mixture padding.
        if not cfg_file.is_file():
            config = apply_spec(config, robot_type)
        if not set(CUSTOM_SPECS[robot_type].camera_keys) <= set(config.camera_keys):
            raise ValueError('source cameras are not present in checkpoint camera slots')
        saved_freq = float(config.action_freq)
        saved_length = float(config.action_length)
        lang = getattr(config, "language_encoder", "none") or "none"
        if lang != "none":
            raise ValueError(
                "LBMPolicy eval encodes the prompt with CLIP as task_vec_clip "
                f"(train --language-encoder none); got {lang!r}"
            )

        steps = int(diffusion_steps or saved_steps)
        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(device)

        spec = CUSTOM_SPECS[robot_type]
        from lbm.action_space import resolve_action_space

        stats_path = resolve_norm_stats_path(
            explicit=Path(norm_stats_path) if norm_stats_path else None,
            ckpt=ckpt_path,
            robot_type=robot_type,
            action_freq=min(saved_freq, spec.fps),
            action_length=saved_length,
            slices=resolve_action_space(spec, "delta"),
        )
        space = ()
        raw = {}
        if stats_path is not None:
            raw = json.loads(Path(stats_path).read_text(encoding="utf-8"))
            stats = load_norm_stats(stats_path)
            space = action_space_from_payload(raw)
        else:
            stats = {
                "state": _identity_stats(spec.state_dim),
                "actions": _identity_stats(spec.action_dim),
            }
        if not space:
            space = tuple(spec.action_space)
            if not space:
                raise ValueError(f"{robot_type}: checkpoint and spec have no action_space")

        dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
        model = DiTPolicy(config)
        load_pretrained(model, ckpt_path)
        model = model.to(device=torch_device, dtype=dtype).eval()

        embodiment_id = int(spec.embodiment_id)
        policy = cls(
            model,
            config=config,
            device=torch_device,
            norm_stats=stats,
            diffusion_steps=steps,
            embodiment_id=embodiment_id,
            dtype=dtype,
        )
        policy.action_space = space
        policy.camera_keys = tuple(spec.camera_keys)
        source_freq = float(raw.get('action_freq', min(config.action_freq, spec.fps)))
        policy.output_steps = min(config.chunk_length, n_steps(config.action_length, source_freq))
        return policy

    def reset(self) -> None:
        for buf in self._history.values():
            buf.clear()

    def metadata(self) -> dict[str, Any]:
        cfg = self.model_config
        return {
            "camera_keys": list(self.camera_keys),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "chunk_length": int(self.output_steps),
            "history_size": int(cfg.history_size),
            "diffusion_steps": int(self.diffusion_steps),
            "embodiment_id": int(self.embodiment_id),
        }

    def _ensure_clip(self) -> CLIPTextEmbedder:
        if self._clip is None:
            self._clip = CLIPTextEmbedder(self._clip_cfg, device="cpu")
        return self._clip

    def _task_vec(self, prompt: str) -> torch.Tensor:
        if prompt not in self._task_cache:
            vec = self._ensure_clip().encode([prompt]).to(device=self.device, dtype=self.dtype)
            self._task_cache[prompt] = vec
        return self._task_cache[prompt]

    def _prep_frame(self, image: np.ndarray) -> torch.Tensor:
        chw = _hwc_to_chw(image)
        return resize_pad_normalize(chw).unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def _images(self, images: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        hist = max(1, self.model_config.history_size)
        for cam in self.camera_keys:
            if cam not in images:
                raise KeyError(f"missing camera {cam!r}; got {tuple(images)}")
            self._history[cam].append(np.asarray(images[cam]))
            frames = list(self._history[cam])
            while len(frames) < hist:
                frames.insert(0, frames[0])
            frames = frames[-hist:]
            stacked = torch.stack([self._prep_frame(f).squeeze(0) for f in frames], dim=0)
            out[cam] = stacked[-1].unsqueeze(0) if hist == 1 else stacked.unsqueeze(0)
        template = next(iter(out.values()))
        return {cam: out[cam] if cam in out else torch.zeros_like(template)
                for cam in self.model_config.camera_keys}

    def normalized_action_prefix(self, action_prefix: np.ndarray, prefix_length: int) -> np.ndarray:
        chunk = int(self.model_config.chunk_length)
        dim = int(self.action_dim)
        out = np.zeros((chunk, self.model_config.action_dim), dtype=np.float32)
        prefix = np.asarray(action_prefix, dtype=np.float32)
        if prefix.ndim == 1:
            prefix = prefix[None]
        n = min(int(prefix_length), chunk, prefix.shape[0])
        if n <= 0:
            return out
        fitted = np.stack([_fit_vector(row, dim) for row in prefix[:n]], axis=0)
        ref = getattr(self, "_last_raw_state", None)
        if ref is not None and self.action_space:
            fitted = apply_action_space(fitted, ref, self.action_space)
        out[:n, :dim] = normalize(fitted, self.norm_stats["actions"], self.action_space)
        return out

    @torch.no_grad()
    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        if obs.get("reset"):
            self.reset()
        raw_state = _fit_vector(obs["state"], self.state_dim)
        self._last_raw_state = raw_state
        state = normalize(raw_state, self.norm_stats["state"], self.action_space, field="state")
        state = _fit_vector(state, self.model_config.state_dim)
        prompt = str(obs.get("prompt") or "")
        batch = {
            "state": torch.from_numpy(state).unsqueeze(0).to(device=self.device, dtype=self.dtype),
            "images": self._images(obs["images"]),
            "task_vec_clip": self._task_vec(prompt),
            "embodiment_id": torch.tensor([self.embodiment_id], device=self.device, dtype=torch.long),
        }
        batch['camera_mask'] = torch.tensor(
            [[cam in self.camera_keys for cam in self.model_config.camera_keys]],
            device=self.device, dtype=torch.bool,
        )
        self.task_vec = batch["task_vec_clip"]
        actions = self.model.sample_actions(batch, num_steps=self.diffusion_steps)
        chunk = actions[0, :self.output_steps, :self.action_dim].float().cpu().numpy()
        chunk = unnormalize(chunk, self.norm_stats["actions"], self.action_space).astype(np.float32)
        if self.action_space:
            chunk = invert_action_space(chunk, raw_state, self.action_space)
        return chunk.astype(np.float32)
