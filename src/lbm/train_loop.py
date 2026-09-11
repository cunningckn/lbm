"""Train LBM: real dumps (or synthetic), validation, DDP / FSDP."""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from lbm.batch import infer_policy_io, policy_batch_from_loader
from lbm.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from lbm.config import (
    TrainConfig,
    encoder_train_summary,
    fsdp_wrap_summary,
    temporal_summary,
    validate_loader_config,
    validate_model_config,
    validate_parallel_config,
)
from lbm.models.clip import CLIPTextEmbedder
from lbm.models.dit import DiTPolicy, load_pretrained
from lbm.models.encoders import load_encoder_weights
from lbm.optim import build_adamw, count_trainable
from lbm.utils.fake_data import FakeActionDataset, collate_samples, move_batch_to_device


def _is_distributed() -> bool:
    return "RANK" in os.environ


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    module = model.module if isinstance(model, DDP) else model
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def _t5_tokenize_fn(max_length: int):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("google-t5/t5-small")

    def _tokenize(texts: list[str]):
        encoded = tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"].float()

    return _tokenize


def _make_loader(
    dataset,
    *,
    config: TrainConfig,
    distributed: bool,
    train: bool,
    collate_fn,
    drop_last: bool = True,
    num_workers: int | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    workers = config.num_workers if num_workers is None else num_workers
    errors = validate_loader_config(config.data, num_workers=workers)
    if errors:
        raise ValueError("Invalid loader config: " + "; ".join(errors))
    if distributed:
        sampler = DistributedSampler(
            dataset, shuffle=train, seed=config.seed, drop_last=drop_last
        )
    samples_per_rank = len(sampler) if sampler is not None else len(dataset)
    if train and (samples_per_rank == 0 or (drop_last and samples_per_rank < config.batch_size)):
        world_size = sampler.num_replicas if sampler is not None else 1
        raise ValueError(
            "Training loader has no batches: "
            f"dataset_size={len(dataset)}, samples_per_rank={samples_per_rank}, "
            f"batch_size={config.batch_size}, world_size={world_size}, drop_last={drop_last}. "
            "Reduce batch_size or world_size, or provide more training samples."
        )
    kwargs: dict = {
        "batch_size": config.batch_size,
        "sampler": sampler,
        "shuffle": sampler is None and train,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": config.data.pin_memory,
        "drop_last": drop_last,
    }
    if train:
        kwargs["generator"] = torch.Generator().manual_seed(config.seed)
    if workers > 0:
        kwargs["persistent_workers"] = config.data.persistent_workers
        kwargs["prefetch_factor"] = config.data.prefetch_factor
        if not config.fake_data:
            from lbm.dataloader.pad import dataloader_worker_init_fn

            kwargs["worker_init_fn"] = dataloader_worker_init_fn
    return DataLoader(dataset, **kwargs), sampler


def _action_error_stats(pred, actions, action_mask=None) -> torch.Tensor:
    """Squared-error sum and valid element count, additive across batches/ranks."""
    error = (pred.float() - actions.float()).square()
    if action_mask is None:
        count = error.new_tensor(error.numel())
    else:
        valid = torch.broadcast_to(action_mask.to(device=error.device, dtype=torch.bool), error.shape)
        error = error.masked_fill(~valid, 0.0)
        count = valid.sum().to(dtype=error.dtype)
    return torch.stack((error.sum(), count))


def _dump_norm_stats(dataset):
    return getattr(dataset, "norm_stats", None)


def _log_and_copy_norm_stats(train_ds, output_dir: Path) -> None:
    inner = list(getattr(train_ds, "datasets", [train_ds]))
    missing = [
        getattr(getattr(ds, "spec", None), "name", "?")
        for ds in inner
        if _dump_norm_stats(ds) is None
    ]
    quantile = all(
        stats is not None
        and "q01" in stats.get("state", {})
        and "q99" in stats.get("state", {})
        and "q01" in stats.get("actions", {})
        and "q99" in stats.get("actions", {})
        for stats in (_dump_norm_stats(ds) for ds in inner)
    )
    if missing:
        print(f"warning: no norm_stats for {missing}; state/action left raw")
    else:
        kind = "q01/q99→[-1,1]" if quantile else "mean/std"
        print(f"norm={kind} dumps={len(inner)}")
    if len(inner) == 1:
        from lbm.action_space import resolve_action_space
        from lbm.utils.preprocess import dump_norm_stats_path

        ds = inner[0]
        root = getattr(ds, "root", None)
        slices = (
            resolve_action_space(
                ds.spec,
                ds.action_mode,
                action_kind=getattr(ds, "action_kind", None),
                action_format=getattr(ds, "action_format", None),
            )
            if hasattr(ds, "spec")
            else ()
        )
        src = (
            dump_norm_stats_path(root, ds.action_freq, ds.action_length, slices)
            if root is not None
            else None
        )
        if src is not None and src.is_file():
            dest = output_dir / "norm_stats.json"
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _prebuild_mmap_caches(
    datasets: list,
    config: TrainConfig,
    *,
    rank: int,
    distributed: bool,
) -> None:
    unique = []
    seen: set[int] = set()
    for dataset in datasets:
        if dataset is None or not hasattr(dataset, "prebuild_mmap_caches"):
            continue
        key = id(dataset)
        if key in seen:
            continue
        seen.add(key)
        unique.append(dataset)
    if not unique or not (config.data.use_mmap or config.data.use_mmap_frames):
        return
    if config.data.mmap_prebuild:
        if rank == 0:
            for dataset in unique:
                dataset.prebuild_mmap_caches(workers=config.data.mmap_prebuild_workers)
        if distributed:
            dist.barrier()
        for dataset in unique:
            if hasattr(dataset, "set_mmap_allow_build"):
                dataset.set_mmap_allow_build(False)


def _resume_signature(config, train_loader, device):
    # Output paths and total steps may change when extending a run. Settings
    # affecting samples, optimization or validation RNG consumption may not.
    fields = (
        "model", "optim", "flow", "data", "seed", "batch_size", "num_workers", "fake_data",
        "bf16", "compile", "val_every", "val_batches", "dump_batch",
    )
    values = asdict(config)
    return {
        **{key: values[key] for key in fields},
        "batches_per_epoch": len(train_loader),
        "dataset_length": len(train_loader.dataset),
        "device_type": device.type,
        "distributed": _is_distributed(),
    }


def main(config: TrainConfig) -> None:
    errors = validate_model_config(config.model)
    errors.extend(validate_loader_config(config.data, num_workers=config.num_workers))
    if config.fsdp:
        errors.extend(validate_parallel_config(config.parallel))
    if (
        min(config.batch_size, config.train_steps, config.log_every, config.val_every, config.ckpt_every)
        <= 0
        or config.val_batches < 0
    ):
        errors.append("batch size, step intervals must be positive; val_batches >= 0")
    if config.resume:
        if config.load_pretrained:
            errors.append("--resume and --ckpt are mutually exclusive")
        if config.fsdp or _is_distributed():
            errors.append("--resume currently requires single-process training (no DDP/FSDP)")
        if config.num_workers != 0:
            errors.append("--resume requires --num-workers 0; worker RNG/prefetch state is not checkpointed")
        if not Path(config.resume).is_file():
            errors.append(f"resume checkpoint is not a file: {config.resume}")
    if errors:
        raise ValueError("Invalid training config:\n  - " + "\n  - ".join(errors))

    torch.set_float32_matmul_precision("high")
    distributed = _is_distributed()
    fsdp = bool(config.fsdp)
    if distributed:
        from lbm.distributed import init_distributed

        local_rank, rank, world = init_distributed()
        device = torch.device("cuda", local_rank)
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if fsdp:
            raise ValueError("--fsdp requires torchrun / RANK")

    random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    use_bf16 = bool(config.bf16) and device.type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    output_dir = Path(config.output_dir)
    tokenize_fn = None
    embedder = None
    collate_fn = collate_samples
    train_ds = val_ds = None

    if config.fake_data:
        n_train = max(config.batch_size * world * 8, 64)
        n_val = max(config.batch_size * world * config.val_batches, 8)
        train_ds = FakeActionDataset(config.model, length=n_train, seed=config.seed)
        val_ds = FakeActionDataset(config.model, length=n_val, seed=config.seed + 1)
    else:
        from lbm.dataloader.pad import collate_fn as dump_collate

        collate_fn = dump_collate
        from lbm.dataloader.custom import CustomMixtureDataset
        from lbm.dataloader.mixture import load_dataset

        train_ds = load_dataset(config, mode="train")
        if config.data.val_dataset:
            val_ds = load_dataset(
                config, mode="val", dataset_path=config.data.val_dataset
            )
        else:
            val_ds = CustomMixtureDataset(
                [(ds, 1.0) for ds in train_ds.datasets],
                mode="val",
                seed=config.seed,
            )
        io = infer_policy_io(train_ds)
        config.model.camera_keys = io["camera_keys"]
        action_dim = int(io["action_dim"])
        if config.data.max_action_dim:
            action_dim = max(action_dim, int(config.data.max_action_dim))
        config.model.action_dim = action_dim
        if io["state_dim"]:
            state_dim = int(io["state_dim"])
            if config.data.max_state_dim:
                state_dim = max(state_dim, int(config.data.max_state_dim))
            config.model.state_dim = state_dim
        elif rank == 0:
            print("warning: dataset has no state keys; state_dim left at config default")
        chunk = int(io.get("chunk_length") or 0)
        if chunk and config.model.action_length > 0:
            config.model.action_freq = chunk / float(config.model.action_length)
        _prebuild_mmap_caches(
            [train_ds] + ([val_ds] if config.data.val_dataset else []),
            config,
            rank=rank,
            distributed=distributed,
        )

    train_loader, train_sampler = _make_loader(
        train_ds, config=config, distributed=distributed, train=True, collate_fn=collate_fn
    )
    val_indices = list(
        range(rank, min(len(val_ds), max(config.val_batches, 1) * config.batch_size * world), world)
    )
    if val_indices:
        val_subset = torch.utils.data.Subset(val_ds, val_indices)
        val_loader, _ = _make_loader(
            val_subset,
            config=config,
            distributed=False,
            train=False,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=min(2, config.num_workers),
        )
    else:
        val_loader = None

    model = DiTPolicy(config.model)
    if config.load_pretrained:
        load_pretrained(model, config.load_pretrained)
    elif config.pretrained_encoders and not config.resume:
        loaded = load_encoder_weights(model, download=True)
        if rank == 0 and loaded:
            print("pretrained:", ", ".join(f"{name}={path}" for name, path in loaded.items()))
    model = model.to(device=device, dtype=dtype)
    if use_bf16 and hasattr(model.img_backbone, "set_bfloat16"):
        model.img_backbone.set_bfloat16(True)

    optimizer = build_adamw(model, config.optim)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / max(config.optim.lr_warmup_steps, 1), 1.0)
    )

    step, epoch, batch_cursor = 0, 0, 0
    signature = _resume_signature(config, train_loader, device)
    resume_payload = None
    if config.resume:
        resume_payload = load_checkpoint(config.resume, model, optimizer, scheduler, signature=signature)
        step, epoch, batch_cursor = (resume_payload[key] for key in ("step", "epoch", "batch_in_epoch"))
        if config.train_steps < step:
            raise ValueError("--steps is the total target and cannot be less than the saved step")
        if rank == 0:
            print(f"resumed {config.resume} at step={step} epoch={epoch} batch={batch_cursor}")

    if config.compile:
        model = torch.compile(model, fullgraph=True)
        if rank == 0:
            print("torch.compile(fullgraph=True) enabled")

    if fsdp:
        from lbm.distributed import wrap_policy_fsdp

        model, optimizer = wrap_policy_fsdp(model, optimizer, config.parallel, device=device)
    elif distributed:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    module = _unwrap(model)

    lang = config.model.language_encoder
    if not config.fake_data:
        if distributed and rank != 0:
            dist.barrier()
        if lang == "t5":
            tokenize_fn = _t5_tokenize_fn(config.model.language_max_length)
        else:
            embedder = CLIPTextEmbedder(config.clip, device="cpu")
        if distributed and rank == 0:
            dist.barrier()

    def to_policy(raw, *, train: bool):
        if config.fake_data:
            batch = move_batch_to_device(raw, device, non_blocking=True)
            if dtype != torch.float32:
                for key, value in list(batch.items()):
                    if key == "images":
                        batch[key] = {cam: img.to(dtype=dtype) for cam, img in value.items()}
                    elif torch.is_tensor(value) and value.is_floating_point():
                        batch[key] = value.to(dtype=dtype)
            return batch
        return policy_batch_from_loader(
            raw,
            camera_keys=tuple(config.model.camera_keys),
            device=device,
            dtype=dtype,
            embedder=embedder,
            language_encoder=lang,
            tokenize_fn=tokenize_fn,
            mask_state_ratio=config.flow.mask_state_ratio if train else 0.0,
            train=train,
            state_dim=config.model.state_dim,
        )

    wandb = None
    if config.log_wandb and rank == 0:
        try:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(project=config.wandb_project, config=asdict(config))
        except Exception as exc:
            print(f"wandb disabled: {exc}")

    n_train, n_params = count_trainable(model)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        import json as _json

        (output_dir / "train_config.json").write_text(
            _json.dumps(asdict(config), indent=2, default=str)
        )
        print(
            f"device={device} world={world} params={n_params / 1e6:.1f}M "
            f"trainable={n_train / 1e6:.1f}M bf16={use_bf16} fsdp={int(fsdp)} "
            f"{temporal_summary(config.model)} {encoder_train_summary(config.model)}"
        )
        if fsdp:
            print(fsdp_wrap_summary(config.parallel))
        if not config.fake_data:
            print(
                f"cameras={config.model.camera_keys} "
                f"state_dim={config.model.state_dim} action_dim={config.model.action_dim} "
                f"train_len={len(train_ds)} val_len={len(val_ds)}"
            )
            _log_and_copy_norm_stats(train_ds, output_dir)

    model.train()
    t_last = time.monotonic()
    dumped_batch = bool(config.resume and step > 0)
    while step < config.train_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        train_loader.generator.manual_seed(config.seed + epoch)
        epoch_rng = resume_payload["epoch_rng"] if resume_payload else capture_rng_state()
        if resume_payload:
            restore_rng_state(epoch_rng)
        iterator = iter(train_loader)
        # Replay data reads up to the saved cursor, then restore training RNG
        # after all setup and skipped reads, before obtaining the next batch.
        for _ in range(batch_cursor):
            next(iterator)
        if resume_payload:
            restore_rng_state(resume_payload["rng_state"])
            resume_payload = None
        for batch_index, raw in enumerate(iterator, start=batch_cursor):
            if step >= config.train_steps:
                break
            if (
                rank == 0
                and config.dump_batch
                and not config.fake_data
                and not dumped_batch
                and isinstance(raw, dict)
                and "image" in raw
            ):
                from lbm.utils.batch_dump import save_loader_batch, video_fps_from_config

                dump_dir = output_dir / "first_batch"
                save_loader_batch(raw, dump_dir, fps=video_fps_from_config(config, train_ds))
                print(f"saved first batch -> {dump_dir}")
                dumped_batch = True
            batch = to_policy(raw, train=True)
            loss = model(
                batch,
                max_action_prefix=config.flow.max_action_prefix,
                prefix_conditioning_prob=config.flow.prefix_conditioning_prob,
                prefix_noise_scale=config.flow.prefix_noise_scale,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optim.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            step += 1

            if step % config.log_every == 0:
                loss_d = loss.detach()
                if distributed:
                    dist.all_reduce(loss_d, op=dist.ReduceOp.AVG)
                if rank == 0:
                    dt = time.monotonic() - t_last
                    t_last = time.monotonic()
                    sps = config.log_every / dt
                    lr = scheduler.get_last_lr()[0]
                    gnorm = float(grad_norm.detach()) if torch.is_tensor(grad_norm) else float(grad_norm)
                    print(
                        f"step {step:6d}  loss {loss_d.item():.4f}  "
                        f"lr {lr:.2e}  gnorm {gnorm:.3f}  {sps:.2f} it/s"
                    )
                    if wandb:
                        wandb.log(
                            {
                                "loss": loss_d.item(),
                                "lr": lr,
                                "grad_norm": gnorm,
                                "steps_per_s": sps,
                            },
                            step=step,
                        )

            if step % config.val_every == 0:
                model.eval()
                stats = torch.zeros(2, device=device, dtype=torch.float64)
                if val_loader is not None:
                    for vb in val_loader:
                        vb = to_policy(vb, train=False)
                        with torch.no_grad():
                            pred = module.sample_actions(
                                vb, num_steps=config.flow.num_diffusion_steps
                            )
                            stats += _action_error_stats(pred, vb["actions"], vb.get("action_mask"))
                if distributed:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                model.train()
                if rank == 0:
                    if stats[1].item() > 0:
                        recon = (stats[0] / stats[1]).item()
                        print(f"step {step:6d}  val_recon_error {recon:.4f}")
                        if wandb:
                            wandb.log({"val_recon_error": recon}, step=step)
                    else:
                        print(f"step {step:6d}  val skipped (no valid action elements in full validation batches)")
                t_last = time.monotonic()

            if step % config.ckpt_every == 0 and (rank == 0 or fsdp):
                ckpt_dir = output_dir / f"{step}"
                if fsdp:
                    from lbm.distributed import save_fsdp_checkpoint

                    save_fsdp_checkpoint(ckpt_dir, model, optimizer)
                    if rank == 0:
                        print(f"saved {ckpt_dir}")
                elif rank == 0:
                    path = output_dir / f"{step}.pt"
                    for checkpoint_path in (path, output_dir / "last.pt"):
                        save_checkpoint(
                            checkpoint_path, _unwrap(model), optimizer, scheduler,
                            step=step, epoch=epoch, batch_in_epoch=batch_index + 1,
                            epoch_rng=epoch_rng, signature=signature,
                        )
                    print(f"saved {path}")
        epoch += 1
        batch_cursor = 0

    if distributed:
        dist.barrier()
        dist.destroy_process_group()
