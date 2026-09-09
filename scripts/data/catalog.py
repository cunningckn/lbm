#!/usr/bin/env python3
"""Official download URLs and how each mix dump becomes the LBM layout.

``process`` is only official-download → ``datasets/<name>``. Scan / FK / mmap /
norm stay in ``scripts/prebuild_*.py`` and ``scripts/compute_norm.py``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from typing import Any

# Mix names match ``lbm.dataloader.custom.datasets.mixes.ALL``.
DUMPS: dict[str, dict[str, Any]] = {
    "abc": {
        "title": "ABC-130k (MCAP)",
        "urls": ["https://huggingface.co/datasets/XDOF/ABC-130k"],
        "hf": [{"repo": "XDOF/ABC-130k"}],
        "process": "none",
        "process_note": "HF dump is already episode.mcap under data/{train,val}/.",
        "local": "/mnt/open_source_data/ABC-130K",
    },
    "agibot": {
        "title": "AgiBot World Alpha + Beta (HDF5 proprio + mp4)",
        "urls": [
            "https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha",
            "https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta",
        ],
        "hf": [
            {"repo": "agibot-world/AgiBotWorld-Alpha", "subdir": "alpha"},
            {"repo": "agibot-world/AgiBotWorld-Beta", "subdir": "beta"},
        ],
        "process": "agibot_layout",
        "process_note": "Place Alpha/Beta as AgiBotWorld_alpha and AgiBotWorld_beta (native HDF5, not LeRobot).",
        "local": "/mnt/open_source_data/AgiBotWorld",
    },
    "das_gripper": {
        "title": "DAS Gripper",
        "urls": [
            "https://huggingface.co/datasets/genrobot2025/DAS-Sample-Data",
            "https://huggingface.co/datasets/genrobot2025/10Kh-RealOmin-OpenData",
        ],
        "hf": [{"repo": "genrobot2025/DAS-Sample-Data"}],
        "process": "das_slim",
        "process_note": (
            "LBM dump is slim episode.hdf5 + wrist mp4. Default download is the public "
            "HDF5 sample; full 10Kh corpus is MCAP and is not converted here."
        ),
        "local": "/mnt/open_source_data/DASGripper_slim",
    },
    "droid": {
        "title": "DROID 1.0.1 (LeRobot v2.1)",
        "urls": ["https://huggingface.co/datasets/lerobot/droid_1.0.1"],
        "hf": [{"repo": "lerobot/droid_1.0.1"}],
        "process": "none",
        "process_note": "Already LeRobot (exterior_1_left / exterior_2_left / wrist_left).",
        "local": "/mnt/open_source_data/Droid/droid_1.0.1",
    },
    "egoverse": {
        "title": "EgoVerse Aria zarr",
        "urls": ["https://github.com/GaTech-RL2/EgoVerse"],
        "hf": [],
        "s3_cmd": (
            "python egomimic/scripts/data_download/sync_s3.py "
            "--local-dir DEST --filters aria-all"
        ),
        "process": "none",
        "process_note": "S3 zarr is the dump. VRS→zarr is upstream (egomimic/scripts/aria_process).",
        "local": "/mnt/open_source_data/EgoVerse/aria",
    },
    "galaxea": {
        "title": "Galaxea Open-World (LeRobot task tarballs)",
        "urls": ["https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset"],
        "hf": [{"repo": "OpenGalaxea/Galaxea-Open-World-Dataset", "include": ["lerobot"]}],
        "process": "galaxea_extract",
        "process_note": "Extract lerobot/<task>.tar.gz into one LeRobot folder per task.",
        "local": "/mnt/open_source_data/Galaxea/lerobot",
    },
    "hifi_umi": {
        "title": "HiFi-UMI-2K (LeRobot v3 chunks)",
        "urls": ["https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K"],
        "hf": [{"repo": "simple-world-lab/HiFi-UMI-2K"}],
        "process": "none",
        "process_note": "Already LeRobot v3 chunk-*/part-* layout.",
        "local": "/mnt/open_source_data/HIFI-UMI-2K",
    },
    "hy_lance": {
        "title": "Hy-Embodied-0.5-VLA-Data (Lance)",
        "urls": ["https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data"],
        "hf": [{"repo": "tencent/Hy-Embodied-0.5-VLA-Data"}],
        "process": "none",
        "process_note": "Already Lance table_*/ layout.",
        "local": "/mnt/open_source_data/Hy-Embodied-0.5-VLA-Data",
    },
    "kai0": {
        "title": "Kai0 (nested LeRobot v2.1)",
        "urls": ["https://huggingface.co/datasets/OpenDriveLab-org/Kai0"],
        "hf": [{"repo": "OpenDriveLab-org/Kai0"}],
        "process": "none",
        "process_note": "Already Task_{A,B,C}/{base,dagger} LeRobot repos.",
        "local": "/mnt/open_source_data/Kai0",
    },
    "libero": {
        "title": "LIBERO (LeRobot conversion on HF)",
        "urls": ["https://huggingface.co/datasets/physical-intelligence/libero"],
        "hf": [{"repo": "physical-intelligence/libero"}],
        "process": "none",
        "process_note": "HF dump is already the merged LeRobot corpus LBM reads.",
        "local": "/mnt/open_source_data/libero",
    },
    "rmbench": {
        "title": "RMBench demos (HDF5 demo_clean → LeRobot)",
        "urls": ["https://huggingface.co/datasets/TianxingChen/RMBench"],
        "hf": [{"repo": "TianxingChen/RMBench", "include": ["data/*/demo_clean/**"]}],
        "process": "rmbench_hdf5",
        "process_note": "Official files are data/<task>/demo_clean HDF5 (+ videos). Convert to one LeRobot repo.",
        "local": "/mnt/open_source_data/rmbench",
    },
    "robotwin": {
        "title": "RoboTwin unified (LeRobot v3)",
        "urls": [
            "https://huggingface.co/datasets/lerobot/robotwin_unified",
            "https://huggingface.co/datasets/TianxingChen/RoboTwin2.0",
        ],
        "hf": [{"repo": "lerobot/robotwin_unified"}],
        "process": "none",
        "process_note": (
            "Default download is lerobot/robotwin_unified (already LeRobot, cam_high / wrists). "
            "Official HDF5 is TianxingChen/RoboTwin2.0 (dataset/); convert with RoboTwin/XPolicyLab scripts."
        ),
        "local": "",
    },
}

NAMES: tuple[str, ...] = tuple(DUMPS.keys())


def dump(name: str) -> dict[str, Any]:
    try:
        return DUMPS[name]
    except KeyError as exc:
        raise SystemExit(f"unknown dump {name!r}; expected one of {', '.join(NAMES)}") from exc


def emit_bash(name: str) -> str:
    spec = dump(name)
    lines = [
        f"DUMP_NAME={shlex.quote(name)}",
        f"DUMP_URL={shlex.quote(spec['urls'][0])}",
        "DUMP_URLS=(%s)" % " ".join(shlex.quote(u) for u in spec["urls"]),
        f"PROCESS={shlex.quote(spec['process'])}",
        f"PROCESS_NOTE={shlex.quote(spec.get('process_note') or '')}",
        f"LOCAL_DUMP={shlex.quote(spec.get('local') or '')}",
        f"S3_CMD={shlex.quote(spec.get('s3_cmd') or '')}",
        "HF_REPO=()",
        "HF_SUBDIR=()",
        "HF_INCLUDE=()",
    ]
    for item in spec.get("hf") or ():
        lines.append(f"HF_REPO+=({shlex.quote(item['repo'])})")
        lines.append(f"HF_SUBDIR+=({shlex.quote(item.get('subdir') or '')})")
        lines.append(f"HF_INCLUDE+=({shlex.quote(' '.join(item.get('include') or ()))})")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Print dump catalog (bash or json)")
    p.add_argument("name", nargs="?", help="dump name")
    p.add_argument("--bash", action="store_true", help="emit bash assignments for download.sh")
    p.add_argument("--list", action="store_true", help="print dump names")
    args = p.parse_args(argv)
    if args.list:
        sys.stdout.write("\n".join(NAMES) + "\n")
        return 0
    if not args.name:
        json.dump(DUMPS, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if args.bash:
        sys.stdout.write(emit_bash(args.name))
        return 0
    json.dump(dump(args.name), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
