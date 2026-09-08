"""Factories for optional vision and language towers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torch import nn

from lbm.config import ClipConfig, DiTConfig
from lbm.models.dino import DinoVisionBackbone, load_dinov3, resolve_dinov3_path
from lbm.models.siglip import SiglipVisionBackbone, ensure_siglip_weights, load_siglip, resolve_siglip_path


def build_vision_backbone(config: DiTConfig) -> nn.Module:
    if config.vision_encoder == "dino":
        return DinoVisionBackbone(config)
    if config.vision_encoder == "siglip":
        return SiglipVisionBackbone(config)
    raise ValueError(f"unknown vision_encoder {config.vision_encoder!r}")


def build_language_encoder(config: DiTConfig) -> nn.Module | None:
    if config.language_encoder in {"none", "", None}:
        return None
    if config.language_encoder == "clip":
        from lbm.models.clip import CLIPLanguageEncoder

        return CLIPLanguageEncoder(ClipConfig())
    if config.language_encoder == "t5":
        from lbm.models.t5 import T5LanguageEncoder

        return T5LanguageEncoder(config)
    raise ValueError(f"unknown language_encoder {config.language_encoder!r}")


def _dist_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _dist_barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _is_vit_b16(config: DiTConfig) -> bool:
    return (config.vit_embed_dim, config.vit_depth, config.vit_num_heads) == (768, 12, 12)


def _is_t5_small(config: DiTConfig) -> bool:
    return (
        config.t5_vocab_size == 32128
        and config.t5_d_model == 512
        and config.t5_d_ff == 2048
        and config.t5_num_layers == 6
        and config.t5_num_heads == 8
    )


def _require_ckpt(
    resolve: Callable[[], Path | None],
    *,
    download: bool,
    rank: int,
    ensure: Callable[[], Path],
    missing: str,
    validate: Callable[[], None] | None = None,
) -> Path:
    path = resolve()
    if path is None and download and rank == 0:
        if validate is not None:
            validate()
        path = ensure()
    _dist_barrier()
    path = path or resolve()
    if path is None:
        raise FileNotFoundError(missing)
    return path


def _check_or_none(ok: bool, message: str) -> Callable[[], None] | None:
    if ok:
        return None

    def _raise() -> None:
        raise ValueError(message)

    return _raise


def load_encoder_weights(model, *, download: bool = True) -> dict[str, str]:
    """Load pretrained towers selected in ``model.config``.

    DINOv3 is loaded only when a local ViT-B/16 checkpoint is already on disk.
    SigLIP-B and t5-small download to ``lbm/checkpoints/siglip`` /
    ``lbm/checkpoints/t5`` on first use when ``download=True`` and the config
    matches those official sizes.
    CLIP text weights are loaded in ``CLIPLanguageEncoder``. Rank 0 downloads;
    every rank then reads ``lbm/checkpoints``.
    """
    cfg: DiTConfig = model.config
    loaded: dict[str, str] = {}
    rank = _dist_rank()

    if cfg.vision_encoder == "dino":
        path = resolve_dinov3_path()
        if path is not None and _is_vit_b16(cfg):
            load_dinov3(model.img_backbone, path)
            loaded["dino"] = str(path)
    elif cfg.vision_encoder == "siglip":
        path = _require_ckpt(
            resolve_siglip_path,
            download=download,
            rank=rank,
            ensure=ensure_siglip_weights,
            missing=(
                "SigLIP checkpoint not found. Set lbm_SIGLIP or allow download "
                "(HF_ENDPOINT / lbm_SIGLIP_URL)."
            ),
            validate=_check_or_none(
                _is_vit_b16(cfg),
                "official SigLIP-B weights require vit_embed_dim=768, "
                "vit_depth=12, vit_num_heads=12",
            ),
        )
        load_siglip(model.img_backbone, path)
        loaded["siglip"] = str(path)

    if cfg.language_encoder == "t5":
        from lbm.models.t5 import ensure_t5_weights, load_t5, resolve_t5_path

        path = _require_ckpt(
            resolve_t5_path,
            download=download,
            rank=rank,
            ensure=ensure_t5_weights,
            missing=(
                "T5 checkpoint not found. Set lbm_T5 or allow download "
                "(HF_ENDPOINT / lbm_T5_URL)."
            ),
            validate=_check_or_none(
                _is_t5_small(cfg),
                "official t5-small weights require t5_vocab_size=32128, "
                "t5_d_model=512, t5_d_ff=2048, t5_num_layers=6, t5_num_heads=8",
            ),
        )
        load_t5(model.language_encoder, path)
        loaded["t5"] = str(path)
    elif cfg.language_encoder == "clip":
        from lbm.models.clip import ensure_clip_weights, load_clip, resolve_clip_path

        path = _require_ckpt(
            resolve_clip_path,
            download=download,
            rank=rank,
            ensure=ensure_clip_weights,
            missing="CLIP checkpoint not found. Set lbm_CLIP or allow download.",
        )
        load_clip(model.language_encoder, path)
        loaded["clip"] = str(path)
    return loaded
