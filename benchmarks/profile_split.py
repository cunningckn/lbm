#!/usr/bin/env python3
"""1-GPU compute profile: where a train step spends time.

CUDA-event split of the full step (fwd / bwd / Adam), then isolated vision vs
DiT timings. ``--kernels`` prints torch.profiler top CUDA ops.

This is the main 1-GPU optimization signal. For FSDP wrap/prefetch use
``profile_fsdp.py``.

    PYTHONPATH=src python benchmarks/profile_split.py --batch-size 32
    PYTHONPATH=src python benchmarks/profile_split.py --batch-size 32 --kernels
"""

from __future__ import annotations

import argparse

import torch
from _common import cuda_ms, profile_train_step, resolve_device

from lbm import DiTConfig, DiTPolicy, make_fake_batch, validate_model_config
from lbm.models.attention import configure_torch_sdp, set_use_flash_attn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--kernels", action="store_true", help="print torch.profiler top CUDA ops")
    return p.parse_args()


def nparams(module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def main() -> None:
    args = parse_args()
    device = resolve_device("auto")
    if device.type != "cuda":
        raise SystemExit("profile_split requires CUDA")
    config = DiTConfig()
    errors = validate_model_config(config)
    if errors:
        raise SystemExit("\n".join(errors))
    dtype = torch.bfloat16
    configure_torch_sdp(mode="auto")
    set_use_flash_attn(None)

    model = DiTPolicy(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print(
        f"bs={args.batch_size} total={nparams(model):.1f}M dino={nparams(model.img_backbone):.1f}M "
        f"dit_blocks={nparams(model.blocks):.1f}M apool={nparams(model.apool):.1f}M"
    )
    model.train()
    bs = args.batch_size
    batch = make_fake_batch(config, bs, device=device, dtype=dtype)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)
        loss.backward()
        optimizer.step()
        return loss

    split = profile_train_step(
        model, optimizer, batch, device, warmup=args.warmup, iters=args.iters
    )
    total = sum(split.values())
    print(
        f"step_ms fwd={split['fwd']:.1f} ({100 * split['fwd'] / total:.0f}%) "
        f"bwd={split['bwd']:.1f} ({100 * split['bwd'] / total:.0f}%) "
        f"adam={split['step']:.1f} ({100 * split['step'] / total:.0f}%) "
        f"total={total:.1f}"
    )

    vision = model.build_vision_tokens(batch["images"]).detach()
    x_t = torch.randn(bs, config.chunk_length, config.action_dim, device=device, dtype=dtype)
    t = torch.rand(bs, device=device, dtype=dtype)
    task_h = model.encode_task(batch).detach()
    c = model.compute_cond(batch["state"], task_h, t).detach()

    def vision_fwd():
        return model.build_vision_tokens(batch["images"])

    def dit_fwd():
        return model.predict_velocity(x_t, c, vision)

    def cond_fwd():
        return model.compute_cond(batch["state"], task_h, t)

    def vision_fb():
        optimizer.zero_grad(set_to_none=True)
        model.build_vision_tokens(batch["images"]).sum().backward()

    def dit_fb():
        optimizer.zero_grad(set_to_none=True)
        vt = vision.detach().requires_grad_(True)
        cc = c.detach().requires_grad_(True)
        xt = x_t.detach().requires_grad_(True)
        model.predict_velocity(xt, cc, vt).sum().backward()

    def stacked_dino_fwd():
        imgs = torch.cat([batch["images"][cam] for cam in model.camera_keys], dim=0)
        return model.img_backbone.encode_image_tokens(imgs)

    def loop_dino_fwd():
        return [model.img_backbone.encode_image_tokens(batch["images"][cam]) for cam in model.camera_keys]

    kw = dict(device=device, warmup=args.warmup, iters=args.iters)
    rows = [
        ("vision_fwd", cuda_ms(vision_fwd, **kw)),
        ("dit_fwd", cuda_ms(dit_fwd, **kw)),
        ("cond_fwd", cuda_ms(cond_fwd, **kw)),
        ("dino_loop_3cam", cuda_ms(loop_dino_fwd, **kw)),
        ("dino_stacked_3cam", cuda_ms(stacked_dino_fwd, **kw)),
        ("vision_fwd+bwd", cuda_ms(vision_fb, **kw)),
        ("dit_fwd+bwd", cuda_ms(dit_fb, **kw)),
    ]
    print(f"{'section':<20} {'ms':>8}  share_of_fwd+bwd")
    compute = split["fwd"] + split["bwd"]
    for name, ms in rows:
        print(f"{name:<20} {ms:8.1f}  {100 * ms / compute:5.1f}% of fwd+bwd")

    if not args.kernels:
        return

    orig_vis = model.build_vision_tokens
    orig_dit = model.predict_velocity

    def vis_wrap(images):
        with torch.autograd.profiler.record_function("vision"):
            return orig_vis(images)

    def dit_wrap(x_t_, c_, vision_tokens):
        with torch.autograd.profiler.record_function("dit_blocks"):
            return orig_dit(x_t_, c_, vision_tokens)

    model.build_vision_tokens = vis_wrap
    model.predict_velocity = dit_wrap
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    for _ in range(2):
        train_step()
    torch.cuda.synchronize(device)
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        train_step()
    torch.cuda.synchronize(device)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
