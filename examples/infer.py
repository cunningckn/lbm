#!/usr/bin/env python3
"""Minimal inference (`sample_actions`) on a synthetic observation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lbm import DiTConfig, DiTPolicy, describe_batch, make_fake_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--device", default="auto", help="cpu | cuda | cuda:N | auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    config = DiTConfig()
    model = DiTPolicy(config).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    batch = make_fake_batch(
        config, args.batch_size, device=device, include_actions=False
    )
    print(f"device={device}  params={n_params / 1e6:.1f}M  steps={args.diffusion_steps}")
    print("fake observation:")
    print(describe_batch(batch))

    with torch.no_grad():
        actions = model.sample_actions(batch, num_steps=args.diffusion_steps)
    print(
        f"sample_actions: {tuple(actions.shape)}  "
        f"finite={bool(torch.isfinite(actions).all())}"
    )


if __name__ == "__main__":
    main()
