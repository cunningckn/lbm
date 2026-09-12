"""CLI for LBM training."""

from __future__ import annotations

import argparse

from lbm.config import (
    TRAIN_DEFAULTS,
    TrainConfig,
    add_encoder_arguments,
    add_fsdp_wrap_arguments,
    add_temporal_arguments,
    apply_encoder_args,
    apply_fsdp_wrap_args,
    apply_temporal_args,
    default_checkpoints_dir,
)


def add_dump_list_arguments(parser: argparse.ArgumentParser) -> None:
    """Ops scripts: empty ``--dataset`` selects every registered dataset."""
    parser.add_argument(
        "--dataset",
        default="",
        help="comma list of dump names (default: all registered datasets)",
    )
    parser.add_argument("--data-root", default="", help="parent of dump folders (default: <repo>/datasets)")


def add_dataset_location_arguments(parser: argparse.ArgumentParser, *, dataset_default: str = "") -> None:
    parser.add_argument(
        "--dataset",
        default=dataset_default,
        help="dataset name under datasets/ (kai0, rmbench, …) or a filesystem path",
    )
    parser.add_argument("--data-root", default="", help="mixture root (default: <repo>/datasets)")
    parser.add_argument(
        "--data-mix",
        default="",
        help="named mix from mixes.py (all) or comma list (kai0,libero)",
    )
    parser.add_argument("--robot-type", default="")


def apply_dataset_location_args(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    from lbm.dataloader.paths import datasets_root

    cfg.data.dataset = str(getattr(args, "dataset", "") or "")
    cfg.data.data_mix = str(getattr(args, "data_mix", "") or "")
    cfg.data.robot_type = str(getattr(args, "robot_type", "") or "")
    cfg.data.data_root_dir = str(getattr(args, "data_root", "") or "") or str(datasets_root())
    max_episodes = getattr(args, "max_episodes", None)
    if max_episodes:
        cfg.data.max_episodes = int(max_episodes)
    workers = getattr(args, "workers", None)
    if workers is not None:
        cfg.data.mmap_prebuild_workers = int(workers)
    return cfg


def apply_dump_list_args(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    """Bind ``--data-root`` / caps for ops scripts. Spec names come from ``dumps_from_args``."""
    from lbm.dataloader.paths import datasets_root

    cfg.data.data_root_dir = str(getattr(args, "data_root", "") or "") or str(datasets_root())
    max_episodes = getattr(args, "max_episodes", None)
    if max_episodes:
        cfg.data.max_episodes = int(max_episodes)
    workers = getattr(args, "workers", None)
    if workers is not None:
        cfg.data.mmap_prebuild_workers = int(workers)
    return cfg


def dumps_from_args(args: argparse.Namespace, *, default: tuple[str, ...] | None = None) -> tuple[str, ...]:
    from lbm.dataloader.catalog import select_dumps
    from lbm.dataloader.custom.datasets import dataset_names

    if default is None:
        default = dataset_names()
    try:
        return select_dumps(getattr(args, "dataset", "") or "", default=default)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def parse_args(
    argv: list[str] | None = None,
    *,
    default_dataset: str | None = None,
    default_fake: bool = False,
) -> argparse.Namespace:
    if default_dataset is None:
        default_dataset = ""
    p = argparse.ArgumentParser(description="Train LBM on LeRobot data (or synthetic batches).")
    p.add_argument("--batch-size", type=int, default=TRAIN_DEFAULTS.batch_size)
    p.add_argument("--steps", type=int, default=None, help="train steps (default: 4 if fake, 75000 if real)")
    p.add_argument("--lr", type=float, default=TRAIN_DEFAULTS.learning_rate)
    p.add_argument("--seed", type=int, default=TRAIN_DEFAULTS.seed)
    p.add_argument("--num-workers", type=int, default=TRAIN_DEFAULTS.num_workers)
    p.add_argument("--output-dir", default=str(default_checkpoints_dir()))
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--ckpt", default="", help="optional pretrained policy checkpoint")
    initialization.add_argument("--resume", default="", help="resume a trusted training checkpoint")
    p.add_argument("--max-episodes", type=int, default=None, help="optional episode cap per source for experiments")
    p.add_argument("--feature-cache", default="", help="prebuilt frozen-vision policy inputs")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--bf16", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--log-every", type=int, default=TRAIN_DEFAULTS.log_every)
    p.add_argument("--val-every", type=int, default=TRAIN_DEFAULTS.val_every)
    p.add_argument("--val-batches", type=int, default=TRAIN_DEFAULTS.val_batches)
    p.add_argument("--ckpt-every", type=int, default=None, help="checkpoint interval (default: 5000 for real data)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="lbm")
    p.add_argument(
        "--dump-batch",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="rank 0: dump first loader batch (images/videos/plots/lang) under output_dir/first_batch",
    )

    add_dataset_location_arguments(p, dataset_default=default_dataset)
    p.add_argument("--val-dataset", default="")
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="episode holdout fraction; recomputes training-only normalization")
    p.add_argument("--video-backend", default=TRAIN_DEFAULTS.video_backend)
    p.add_argument("--action-mode", default=TRAIN_DEFAULTS.action_mode, help="delta (default), rel, or abs")
    p.add_argument(
        "--action-kind",
        default=None,
        help="eef: run FK on joint groups (aloha/vx300s, franka). omit to keep spec kinds",
    )
    p.add_argument(
        "--action-format",
        default=None,
        help="eef packed pose: default, xyz+rotvec, xyz+rot6d, xyz+quat",
    )
    p.add_argument("--no-mmap", action="store_true")
    p.add_argument(
        "--rescan",
        action="store_true",
        help="rebuild <dataset>/.cache/episodes instead of loading it",
    )
    p.add_argument(
        "--mmap-prebuild",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="rank 0 multi-process JPEG mmap build before training (default: on)",
    )
    p.add_argument(
        "--mmap-prebuild-workers",
        type=int,
        default=0,
        help="prebuild process count (0 = min(32, CPU count))",
    )
    p.add_argument(
        "--fake-data",
        action="store_true",
        default=default_fake,
        help="force synthetic batches",
    )

    add_temporal_arguments(p)
    add_encoder_arguments(p)
    add_fsdp_wrap_arguments(p)
    p.add_argument(
        "--pretrained-encoders",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="load DINOv3 / SigLIP / t5-small for the selected towers",
    )
    return p.parse_args(argv)


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    real = bool(args.dataset or args.data_mix or args.feature_cache) and not args.fake_data
    cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_steps=int(
            args.steps if args.steps is not None
            else (TRAIN_DEFAULTS.train_steps_real if real else TRAIN_DEFAULTS.train_steps_fake)
        ),
        output_dir=args.output_dir,
        fake_data=not real,
        feature_cache=args.feature_cache,
        load_pretrained=args.ckpt,
        resume=args.resume,
        pretrained_encoders=bool(args.pretrained_encoders),
        compile=bool(args.compile),
        bf16=bool(args.bf16),
        fsdp=bool(args.fsdp),
        log_every=args.log_every,
        val_every=args.val_every,
        val_batches=args.val_batches,
        ckpt_every=args.ckpt_every if args.ckpt_every is not None else TRAIN_DEFAULTS.ckpt_every,
        log_wandb=bool(args.wandb),
        wandb_project=args.wandb_project,
        dump_batch=bool(args.dump_batch),
    )
    cfg.optim.learning_rate = args.lr
    apply_dataset_location_args(cfg, args)
    cfg.data.val_dataset = args.val_dataset
    cfg.data.val_fraction = args.val_fraction
    cfg.data.video_backend = args.video_backend
    cfg.data.action_mode = args.action_mode
    cfg.data.action_kind = getattr(args, "action_kind", None)
    cfg.data.action_format = str(getattr(args, "action_format", None) or "")
    cfg.data.use_mmap = not args.no_mmap
    cfg.data.use_mmap_frames = not args.no_mmap
    cfg.data.mmap_prebuild = bool(args.mmap_prebuild) and not args.no_mmap
    cfg.data.mmap_prebuild_workers = int(args.mmap_prebuild_workers)
    cfg.data.rescan = bool(args.rescan)
    apply_encoder_args(cfg.model, args)
    apply_temporal_args(cfg.model, args)
    apply_fsdp_wrap_args(cfg.parallel, args)
    if getattr(args, "action_freq", None) is not None:
        cfg.data.override_action_freq = True
    if not real:
        cfg.log_every = 1
        cfg.val_every = args.val_every
        if args.ckpt_every is None:
            cfg.ckpt_every = cfg.train_steps + 1
    return cfg
