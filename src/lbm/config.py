"""Architecture and training-recipe configs for LBM (Large Behavior Cloning Model)."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from lbm.temporal import n_steps


def lbm_repo_root() -> Path:
    """Repo root that contains ``src/lbm`` (this file lives in ``src/lbm``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "lbm").is_dir():
            return parent
    return here.parents[2]


def default_checkpoints_dir() -> Path:
    """``lbm/checkpoints`` (override with ``LBM_CHECKPOINTS`` / ``lbm_CHECKPOINTS``)."""
    env = os.environ.get("LBM_CHECKPOINTS") or os.environ.get("lbm_CHECKPOINTS")
    if env:
        return Path(env).expanduser()
    return lbm_repo_root() / "checkpoints"


@dataclass
class OptimConfig:
    """AdamW with a linear-warmup-then-constant LR schedule."""

    learning_rate: float = 1e-4
    lr_warmup_steps: int = 1000
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 10.0
    vision_lr_scale: float = 1.0


@dataclass
class FlowConfig:
    """Rectified-flow matching + action-prefix conditioning."""

    mask_state_ratio: float = 0.1
    max_action_prefix: int = 4
    prefix_conditioning_prob: float = 1.0
    prefix_noise_scale: float = 0.05
    num_diffusion_steps: int = 10


@dataclass
class ClipConfig:
    """CLIP ViT-B/32 text asset locations (downloaded into ``lbm/checkpoints/clip``)."""

    cache_dir: str = field(default_factory=lambda: str(default_checkpoints_dir() / "clip"))
    model_url: str = (
        "https://openaipublic.azureedge.net/clip/models/"
        "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
    )
    bpe_url: str = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
    model_name: str = "ViT-B-32.pt"
    bpe_name: str = "bpe_simple_vocab_16e6.txt.gz"


@dataclass
class ParallelConfig:
    """Megatron-FSDP sharding on a dense DP mesh. FSDP always shards last."""

    zero_dp_strategy: str = "optim_grads_params"
    dp_shard_dim: str = "dp_shard"
    tp_dim: str = "tp"
    overlap_param_gather: bool = True
    overlap_grad_reduce: bool = True
    fsdp_double_buffer: bool = True
    shard_dino: bool = False
    average_in_collective: bool = True
    main_params_dtype: str = "fp32"
    main_grads_dtype: str = "auto"
    grad_comm_dtype: str = "bf16"
    # Wrap granularity. ``block`` = each DiTBlock; ``attn_mlp`` = attn / cross /
    # MLP as separate units (asymmetric; needs max-pool double buffer).
    fsdp_unit: str = "block"
    # Consecutive DiTBlocks per FSDP unit. 1 = no grouping. Must divide depth
    # for symmetric double-buffering (32 → 1,2,4,8,16,32).
    fsdp_group_size: int = 1
    # Duplicate DP process group so all-gather can overlap reduce-scatter.
    independent_ag_group: bool = False
    # Megatron-FSDP unshards per submodule instead of the whole unit.
    fine_grained_param_gather: bool = False
    # All-gather prefetch budget in elements. None = library default (and a
    # 1e9 floor).  When double-buffer is off we override to 2× unit numel so
    # prefetch is "current + next" instead of that floor.
    ag_prefetch_elements: int | None = None
    # None = auto (on when units are asymmetric: VisionTokenPool vs DiTBlock,
    # sharded DINO, attn/mlp split, uneven groups).
    maxpool_double_buffer: bool | None = None


@dataclass
class DiTConfig:
    """LBM architecture defaults (flow-matching DiT)."""

    hidden_size: int = 1536
    depth: int = 32
    num_heads: int = 24
    mlp_ratio: float = 4.0
    state_dim: int = 14
    action_dim: int = 14
    # Temporal windows: (length in seconds) × (frequency in Hz) → sample count.
    # chunk_length = round(action_length * action_freq); history_size similarly.
    action_length: float = 5.0
    action_freq: float = 10.0
    history_length: float = 0.0
    history_freq: float = 10.0
    camera_keys: tuple[str, ...] = ("top", "left", "right")
    task_embed_dim: int = 512

    vit_embed_dim: int = 768
    vit_depth: int = 12
    vit_num_heads: int = 12
    vision_pool_num_queries: int = 12
    vision_pool_num_heads: int = 8
    vision_pool_mlp_ratio: int = 4

    vision_encoder: str = "dino"
    language_encoder: str = "none"
    language_max_length: int = 77
    t5_vocab_size: int = 32128
    t5_d_model: int = 512
    t5_d_ff: int = 2048
    t5_num_layers: int = 6
    t5_num_heads: int = 8
    t5_dropout: float = 0.1

    train_vision_encoder: bool = False
    train_language_encoder: bool = False

    @property
    def chunk_length(self) -> int:
        return n_steps(self.action_length, self.action_freq)

    @property
    def history_size(self) -> int:
        return n_steps(self.history_length, self.history_freq)


def validate_model_config(model: DiTConfig) -> list[str]:
    model_dims = [
        model.hidden_size,
        model.depth,
        model.num_heads,
        model.mlp_ratio,
        model.state_dim,
        model.action_dim,
        model.chunk_length,
        model.action_length,
        model.action_freq,
        model.history_freq,
        model.task_embed_dim,
        model.vit_embed_dim,
        model.vit_depth,
        model.vit_num_heads,
        model.vision_pool_num_queries,
        model.vision_pool_num_heads,
        model.vision_pool_mlp_ratio,
    ]
    errors = []
    if min(model_dims) <= 0 or not model.camera_keys:
        errors.append("model dimensions and camera_keys must be positive/non-empty")
    if model.history_length < 0:
        errors.append("history_length must be >= 0 (0 = current frame only)")
    if model.vision_encoder not in {"dino", "siglip"}:
        errors.append("vision_encoder must be 'dino' or 'siglip'")
    if model.language_encoder not in {"none", "clip", "t5"}:
        errors.append("language_encoder must be 'none', 'clip', or 't5'")
    if model.language_max_length <= 0:
        errors.append("language_max_length must be positive")
    if model.t5_d_model % model.t5_num_heads:
        errors.append("t5_d_model must be divisible by t5_num_heads")
    if (
        model.hidden_size % model.num_heads
        or model.vit_embed_dim % model.vit_num_heads
        or model.vit_embed_dim % model.vision_pool_num_heads
        or model.hidden_size % 2
        or (model.vit_embed_dim // model.vit_num_heads) % 4
    ):
        errors.append("attention dimensions must be compatible with their head counts")
    return errors


_ZERO_STRATEGIES = {"no_shard", "optim", "optim_grads", "optim_grads_params"}
_DTYPE_NAMES = {"fp32", "bf16", "fp16", "auto"}
_FSDP_UNITS = {"block", "attn_mlp"}


def validate_parallel_config(parallel: ParallelConfig) -> list[str]:
    errors = []
    if parallel.zero_dp_strategy not in _ZERO_STRATEGIES:
        errors.append(
            f"zero_dp_strategy must be one of {sorted(_ZERO_STRATEGIES)}, "
            f"got {parallel.zero_dp_strategy!r}"
        )
    for name in ("main_params_dtype", "main_grads_dtype", "grad_comm_dtype"):
        value = getattr(parallel, name)
        if value not in _DTYPE_NAMES:
            errors.append(f"{name} must be one of {sorted(_DTYPE_NAMES)}, got {value!r}")
    if not parallel.dp_shard_dim:
        errors.append("dp_shard_dim must be non-empty")
    if not parallel.tp_dim:
        errors.append("tp_dim must be non-empty (use a size-1 TP mesh if not tensor-parallel)")
    if parallel.fsdp_unit not in _FSDP_UNITS:
        errors.append(
            f"fsdp_unit must be one of {sorted(_FSDP_UNITS)}, got {parallel.fsdp_unit!r}"
        )
    if parallel.fsdp_group_size < 1:
        errors.append("fsdp_group_size must be >= 1")
    if parallel.fsdp_unit == "attn_mlp" and parallel.fsdp_group_size != 1:
        errors.append("fsdp_unit=attn_mlp cannot be combined with fsdp_group_size > 1")
    if parallel.ag_prefetch_elements is not None and parallel.ag_prefetch_elements < 1:
        errors.append("ag_prefetch_elements must be >= 1 when set")
    return errors


def add_encoder_arguments(parser: argparse.ArgumentParser) -> None:
    """Vision/language encoder choice and which towers to train."""
    parser.add_argument(
        "--vision-encoder",
        default="dino",
        choices=("dino", "siglip"),
        help="vision tower: DINOv3 ViT-B/16 or SigLIP ViT-B/16",
    )
    parser.add_argument(
        "--language-encoder",
        default="none",
        choices=("none", "clip", "t5"),
        help="none = use batch task_vec_clip; clip = CLIP text tower; t5 = T5 encoder",
    )
    parser.add_argument(
        "--train-vision-encoder",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="train the vision tower (default: frozen). --train-vision-encoder enables training",
    )
    parser.add_argument(
        "--train-language-encoder",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="train the in-graph language encoder (default: frozen; no-op when --language-encoder none)",
    )


_ENCODER_BOOLS = (
    "train_vision_encoder",
    "train_language_encoder",
)


def apply_encoder_args(config: DiTConfig, args: argparse.Namespace) -> DiTConfig:
    config.vision_encoder = str(args.vision_encoder)
    config.language_encoder = str(args.language_encoder)
    for name in _ENCODER_BOOLS:
        setattr(config, name, bool(getattr(args, name, getattr(config, name))))
    if config.language_encoder == "t5":
        config.task_embed_dim = config.t5_d_model
        config.language_max_length = min(config.language_max_length, 128)
    elif config.language_encoder == "clip":
        config.task_embed_dim = 512
        config.language_max_length = 77
    return config


def encoder_train_summary(config: DiTConfig) -> str:
    return (
        f"vision={config.vision_encoder} language={config.language_encoder} "
        f"train_vision_encoder={int(config.train_vision_encoder)} "
        f"train_language_encoder={int(config.train_language_encoder)}"
    )


def add_fsdp_wrap_arguments(parser: argparse.ArgumentParser) -> None:
    """Prefetch / wrap-granularity knobs for Megatron-FSDP."""
    parser.add_argument(
        "--fsdp-unit",
        default="block",
        choices=("block", "attn_mlp"),
        help="FSDP unit: whole DiTBlock, or attn/cross/MLP separately",
    )
    parser.add_argument(
        "--fsdp-group-size",
        type=int,
        default=1,
        help="consecutive DiTBlocks per FSDP unit (1,2,4,8,16,32 for depth=32)",
    )
    parser.add_argument(
        "--independent-ag",
        action="store_true",
        help="duplicate DP process group so param all-gather can overlap grad reduce-scatter",
    )
    parser.add_argument(
        "--fine-grained-gather",
        action="store_true",
        help="unshard each submodule of an FSDP unit instead of the whole unit",
    )
    parser.add_argument(
        "--ag-prefetch-elements",
        type=int,
        default=0,
        help="all-gather prefetch budget in elements; 0 = auto (2× unit when no double-buffer)",
    )
    parser.add_argument(
        "--maxpool-double-buffer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="MaxPool FSDP double-buffer for asymmetric units "
        "(VisionTokenPool vs DiTBlock). "
        "Default: auto (on; units are always size-asymmetric)",
    )


def apply_fsdp_wrap_args(parallel: ParallelConfig, args: argparse.Namespace) -> ParallelConfig:
    if hasattr(args, "fsdp_unit"):
        parallel.fsdp_unit = str(args.fsdp_unit)
    if hasattr(args, "fsdp_group_size"):
        parallel.fsdp_group_size = int(args.fsdp_group_size)
    if hasattr(args, "independent_ag"):
        parallel.independent_ag_group = bool(args.independent_ag)
    if hasattr(args, "fine_grained_gather"):
        parallel.fine_grained_param_gather = bool(args.fine_grained_gather)
    if hasattr(args, "ag_prefetch_elements") and int(args.ag_prefetch_elements) > 0:
        parallel.ag_prefetch_elements = int(args.ag_prefetch_elements)
    if getattr(args, "maxpool_double_buffer", None) is not None:
        parallel.maxpool_double_buffer = bool(args.maxpool_double_buffer)
    return parallel


def fsdp_wrap_summary(parallel: ParallelConfig) -> str:
    prefetch = (
        "auto" if parallel.ag_prefetch_elements is None else str(parallel.ag_prefetch_elements)
    )
    if parallel.maxpool_double_buffer is None:
        maxpool = "auto"
    else:
        maxpool = str(int(parallel.maxpool_double_buffer))
    return (
        f"unit={parallel.fsdp_unit} group={parallel.fsdp_group_size} "
        f"overlap_ag={int(parallel.overlap_param_gather)} "
        f"overlap_rs={int(parallel.overlap_grad_reduce)} "
        f"double_buffer={int(parallel.fsdp_double_buffer)} "
        f"maxpool={maxpool} "
        f"independent_ag={int(parallel.independent_ag_group)} "
        f"fine_gather={int(parallel.fine_grained_param_gather)} "
        f"prefetch={prefetch}"
    )


def find_lerobot_dataset(name: str | None = None) -> str:
    """Find a LeRobot tree with ``meta/info.json`` under ``datasets/``."""
    from lbm.dataloader.paths import datasets_root, resolve_dataset

    if name:
        path = resolve_dataset(name, required=False)
        if path is not None and (path / "meta" / "info.json").is_file():
            return str(path)
        return ""
    bases: list[Path] = [datasets_root(), Path.cwd() / "datasets"]
    seen: set[Path] = set()
    for base in bases:
        try:
            base = base.resolve()
        except OSError:
            continue
        if base in seen or not base.is_dir():
            continue
        seen.add(base)
        for child in sorted(base.iterdir()):
            if (child / "meta" / "info.json").is_file():
                return str(child)
    return ""


def default_rmbench_dataset() -> str:
    """Back-compat wrapper around :func:`find_lerobot_dataset`."""
    return find_lerobot_dataset("rmbench")


@dataclass
class DataConfig:
    """Dump location and IO knobs."""

    data_root_dir: str = ""
    data_mix: str = ""
    dataset: str = ""
    val_dataset: str = ""
    robot_type: str = ""
    video_backend: str = "decord"
    lerobot_version: str = "v2.0"
    action_mode: str = "delta"
    action_kind: str | None = None
    action_format: str = ""
    use_mmap: bool = True
    use_mmap_frames: bool = True
    mmap_prebuild: bool = True
    mmap_prebuild_workers: int = 0
    include_state: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    max_action_dim: int | None = None
    max_state_dim: int | None = None
    override_action_freq: bool = False
    max_episodes: int | None = None
    rescan: bool = False


@dataclass
class TrainConfig:
    """End-to-end LBM training: real LeRobot data or synthetic batches."""

    seed: int = 123
    batch_size: int = 4
    num_workers: int = 8
    train_steps: int = 75_000
    output_dir: str = field(default_factory=lambda: str(default_checkpoints_dir()))
    fake_data: bool = False

    load_pretrained: str = ""
    pretrained_encoders: bool = True
    compile: bool = False
    bf16: bool = True
    fsdp: bool = False

    log_every: int = 20
    val_every: int = 2500
    val_batches: int = 4
    ckpt_every: int = 5000
    log_wandb: bool = False
    wandb_project: str = "lbm"
    # Rank-0 dump of the first real loader batch under ``output_dir/first_batch``.
    dump_batch: bool = True

    optim: OptimConfig = field(default_factory=OptimConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    model: DiTConfig = field(default_factory=DiTConfig)
    data: DataConfig = field(default_factory=DataConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)


def add_temporal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--action-length",
        type=float,
        default=None,
        help="action window duration in seconds (chunk_length = action_length * action_freq)",
    )
    parser.add_argument(
        "--action-freq",
        type=float,
        default=None,
        help="action sampling frequency in Hz",
    )
    parser.add_argument(
        "--history-length",
        type=float,
        default=None,
        help="image history duration in seconds (0 = current frame only)",
    )
    parser.add_argument(
        "--history-freq",
        type=float,
        default=None,
        help="image history sampling frequency in Hz",
    )


def apply_temporal_args(config: DiTConfig, args: argparse.Namespace) -> DiTConfig:
    if getattr(args, "action_length", None) is not None:
        config.action_length = float(args.action_length)
    if getattr(args, "action_freq", None) is not None:
        config.action_freq = float(args.action_freq)
    if getattr(args, "history_length", None) is not None:
        config.history_length = float(args.history_length)
    if getattr(args, "history_freq", None) is not None:
        config.history_freq = float(args.history_freq)
    return config


def temporal_summary(config: DiTConfig) -> str:
    return (
        f"action={config.action_length:g}s@{config.action_freq:g}Hz→{config.chunk_length} "
        f"history={config.history_length:g}s@{config.history_freq:g}Hz→{config.history_size}"
    )


def data_cfg_from_train(config: TrainConfig) -> dict:
    """Dict consumed by custom dataset constructors."""
    from lbm.dataloader.paths import datasets_root

    m = config.model
    d = config.data
    return {
        "lerobot_version": d.lerobot_version,
        "video_backend": d.video_backend,
        "use_mmap": d.use_mmap,
        "use_mmap_frames": d.use_mmap_frames,
        "mmap_prebuild": d.mmap_prebuild,
        "mmap_prebuild_workers": d.mmap_prebuild_workers,
        "action_mode": d.action_mode,
        "action_kind": d.action_kind,
        "action_format": d.action_format or None,
        "include_state": d.include_state,
        "action_length": m.action_length,
        "action_freq": m.action_freq if d.override_action_freq else None,
        "history_length": m.history_length,
        "history_freq": m.history_freq,
        "data_root_dir": d.data_root_dir or str(datasets_root()),
        "data_mix": d.data_mix,
        "per_device_batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": d.pin_memory,
        "persistent_workers": d.persistent_workers,
        "prefetch_factor": d.prefetch_factor,
        "max_action_dim": d.max_action_dim,
        "max_state_dim": d.max_state_dim,
        "max_episodes": d.max_episodes,
        "rescan": d.rescan,
    }
