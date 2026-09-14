"""Matched live/cached historical training on a bounded real Agibot subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from context.validation.correctness.train_comparison import memory_sample

from lbm import train_loop
from lbm.config import TrainConfig
from lbm.dataloader import mixture


def digest(tensor):
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("current", "vision", "state", "both"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--feature-cache", default="")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--open-shards", type=int, default=2)
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint-every", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA GPU required")
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    cfg = TrainConfig(
        output_dir=str(output),
        seed=123,
        feature_cache=args.feature_cache,
        resume=args.resume,
        feature_cache_open_shards=args.open_shards,
        batch_size=args.batch,
        num_workers=args.workers,
        train_steps=args.steps,
        val_every=args.checkpoint_every,
        val_batches=2,
        ckpt_every=args.checkpoint_every,
        log_every=5,
        dump_batch=False,
    )
    cfg.data.dataset = "agibot"
    cfg.data.instruction_mode = "subtask"
    cfg.data.rescan = True
    cfg.data.max_episodes = 2
    cfg.data.val_fraction = 0.0
    cfg.data.mmap_prebuild = False
    cfg.model.action_length = 1.0
    cfg.model.action_freq = 15.0
    cfg.model.history_time_encoding = args.mode in ("vision", "both")
    cfg.model.history_length = 0.3 if cfg.model.history_time_encoding else 0.0
    cfg.model.history_freq = 10.0
    cfg.model.state_history_length = 0.3 if args.mode in ("state", "both") else 0.0
    cfg.model.state_history_freq = 10.0
    cfg.optim.lr_warmup_steps = 10
    record = dict(
        mode=args.mode,
        resumed=bool(args.resume),
        initial_hash_scope="model construction before optional resume restore",
        steps=args.steps,
        batch=args.batch,
        train=[],
        validation=[],
        input_hashes=[],
        scope="matched real-data throughput and finite-loss stability; not convergence",
        feature_cache=bool(args.feature_cache),
        workers=args.workers,
        open_shards=args.open_shards,
        resources=[],
    )
    original_load = mixture.load_dataset

    def load(*a, **kw):
        ds = original_load(*a, **kw)
        ds.set_mmap_allow_build(False)
        return ds

    mixture.load_dataset = load
    original_policy = train_loop.DiTPolicy

    class CheckedPolicy(original_policy):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            combined = hashlib.sha256()
            for name, value in self.state_dict().items():
                if not name.startswith("state_history_pool."):
                    combined.update(name.encode())
                    combined.update(digest(value).encode())
            record["shared_initial_weights_sha256"] = combined.hexdigest()

        def forward(self, batch, *a, **kw):
            # Hashes establish common data/RNG without publishing dataset arrays.
            if self.training and len(record["input_hashes"]) < 3:
                record["input_hashes"].append(
                    {
                        **{key: digest(batch[key]) for key in ("state", "actions", "task_vec_clip")},
                        "cuda_rng": digest(torch.cuda.get_rng_state()),
                    }
                )
            loss = super().forward(batch, *a, **kw)
            if not torch.isfinite(loss).all():
                raise ValueError("nonfinite history training loss")
            return loss

    train_loop.DiTPolicy = CheckedPolicy
    original_train = train_loop.log_train_metrics
    original_val = train_loop.log_validation_metrics

    def log_train(**kw):
        original_train(**kw)
        record['resources'].append(dict(step=int(kw['step']), **memory_sample()))
        record["train"].append({key: float(kw[key]) for key in ("step", "loss", "grad_norm", "steps_per_s")})

    def log_val(stats, *, step, logger=None):
        original_val(stats, step=step, logger=logger)
        record["validation"].append(dict(step=step, squared_error=float(stats[0]), count=float(stats[1])))

    train_loop.log_train_metrics = log_train
    train_loop.log_validation_metrics = log_val
    train_loop.main(cfg)
    record["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    record["gpu"] = torch.cuda.get_device_name()
    (output / f"summary-{args.steps}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
