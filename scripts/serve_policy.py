#!/usr/bin/env python3
"""Serve an LBM checkpoint over HTTP for simulation eval clients.

    uv run python scripts/serve_policy.py --ckpt checkpoints/<run>/<step>.pt --robot-type rmbench
    uv run python scripts/serve_policy.py --ckpt checkpoints/<run> --robot-type libero --port 8000

The sim venv talks to this process with ``simulation/client.py`` (stdlib HTTP).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lbm.policy import LBMPolicy
from lbm.serve import serve_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an LBM policy for sim eval")
    parser.add_argument("--ckpt", required=True, help="checkpoint .pt file or directory of .pt files")
    parser.add_argument(
        "--robot-type",
        default="",
        help="dataset spec to apply (libero, rmbench, …). Sets cameras / state-action dims.",
    )
    parser.add_argument("--config", default=None, help="train_config.json (default: next to --ckpt)")
    parser.add_argument("--norm-stats", default=None, help="norm_stats.json (default: ckpt dir or datasets/<robot>)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--diffusion-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    policy = LBMPolicy.from_checkpoint(
        args.ckpt,
        robot_type=args.robot_type,
        config_path=args.config,
        norm_stats_path=args.norm_stats,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
    )
    logging.info("loaded policy metadata: %s", policy.metadata())
    serve_policy(policy, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
