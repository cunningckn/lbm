"""Submit the complete MolmoMotion cache build as a CPU-only TI-ONE task.

The request shape follows ``mindon_pi0_dev/scripts/tc_cloud/submit_job.py``.
Code is snapshotted from the clean Git commit before submission so a queued
task cannot observe later edits to the live checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_KEY_INFO = Path(
    os.environ.get(
        "TC_KEY_INFO_PATH",
        "/home/tione/workspace/kainingchen/mindon_pi0_dev/scripts/tc_cloud/key_info.json",
    )
)
DEFAULT_SOURCE = Path("/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m")
DEFAULT_OUTPUT = Path("/home/tione/workspace/kainingchen/Datasets/molmo-motion-1m-mmap/v1")
DEFAULT_IMAGE = (
    "ccr.ccs.tencentyun.com/taco/taco-train:"
    "dtk26.04-torch2.7.1-py3.11-hccpd1-hccl-v1.0-vla-openpi-torchcodec"
)
DEFAULT_REGION = "ap-beijing"
DEFAULT_RESOURCE_GROUP_ID = "rsg-d45m9fwc"
CPU_CORES = 100
GPU_COUNT = 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--submit", action="store_true", help="create the TI-ONE task")
    result.add_argument("--dry-run", action="store_true", help="print request JSON only")
    result.add_argument("--job-name", default="")
    result.add_argument("--key-info", type=Path, default=DEFAULT_KEY_INFO)
    result.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--workers", type=int, default=100)
    result.add_argument("--shard-size", type=int, default=256)
    result.add_argument("--checks", type=int, default=32)
    result.add_argument("--memory-gb", type=int, default=625)
    result.add_argument("--resource-group-id", default=DEFAULT_RESOURCE_GROUP_ID)
    result.add_argument("--image-url", default=DEFAULT_IMAGE)
    result.add_argument("--region", default=DEFAULT_REGION)
    return result


def _load_key_info(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Tencent Cloud key file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    for key in ("TENCENTCLOUD_SECRET_ID", "TENCENTCLOUD_SECRET_KEY"):
        if not value.get(key):
            raise ValueError(f"Tencent Cloud key file has no {key}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments], text=True
    ).strip()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return value or "user"


def resolve_job_name(requested: str, user_name: str, timestamp: str) -> str:
    if requested:
        name = _safe_name(requested)
    else:
        name = f"{_safe_name(user_name)}-molmo-motion-mmap-{timestamp}"
    if len(name) > 60:
        name = name[:60].rstrip("-_")
    return name


def require_clean_commit() -> tuple[str, str]:
    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError("refusing to snapshot a dirty Git worktree")
    return branch, commit


def snapshot_code(destination: Path) -> tuple[str, str]:
    branch, commit = require_clean_commit()
    if destination.exists():
        raise FileExistsError(f"snapshot destination already exists: {destination}")
    destination.mkdir(parents=True)
    archive = subprocess.Popen(
        ["git", "-C", str(REPO_ROOT), "archive", "--format=tar", "HEAD"],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    try:
        subprocess.run(
            ["tar", "-x", "-C", str(destination)],
            stdin=archive.stdout,
            check=True,
        )
    finally:
        archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError("git archive failed")
    return branch, commit


def start_command(
    code_root: Path,
    source_root: Path,
    output_root: Path,
    log_file: Path,
    *,
    workers: int,
    shard_size: int,
    checks: int,
) -> str:
    values = (code_root, source_root, output_root, log_file)
    if any("\n" in str(value) for value in values):
        raise ValueError("paths may not contain newlines")
    code = shlex.quote(str(code_root))
    source = shlex.quote(str(source_root))
    output = shlex.quote(str(output_root))
    log = shlex.quote(str(log_file))
    log_parent = shlex.quote(str(log_file.parent))
    return (
        "set -Eeuo pipefail\n"
        f"cd {code}\n"
        f"mkdir -p {log_parent}\n"
        f"SOURCE_ROOT={source} OUTPUT_ROOT={output} "
        f"WORKERS={workers} SHARD_SIZE={shard_size} CHECKS={checks} "
        "MODE=full VERIFY_SOURCE_HASHES=1 CHECKSUMS=1 PYTHON_BIN=python3 "
        f"bash scripts/tc_cloud/run_molmo_motion_full.sh >{log} 2>&1\n"
    )


def build_params(
    args: argparse.Namespace,
    key_info: dict[str, str],
    job_name: str,
    code_root: Path,
    log_file: Path,
) -> dict[str, Any]:
    if args.workers <= 0 or args.shard_size <= 0 or args.checks < 0:
        raise ValueError("workers/shard-size must be positive and checks non-negative")
    if args.memory_gb <= 0:
        raise ValueError("memory-gb must be positive")
    resource = {
        "Role": "WORKER",
        "Cpu": CPU_CORES * 1000,
        "Memory": args.memory_gb * 1024,
        "GpuType": "HCC-BW1000",
        "Gpu": GPU_COUNT * 100,
        "InstanceNum": 1,
    }
    return {
        "Name": job_name,
        "ChargeType": "PREPAID",
        "TrainingMode": "DDP",
        "ResourceConfigInfos": [resource],
        "ResourceGroupId": args.resource_group_id,
        "ImageInfo": {
            "ImageType": "CCR",
            "ImageUrl": args.image_url,
            "RegistryRegion": "ap-guangzhou",
        },
        "StartCmdInfo": {
            "StartCmd": start_command(
                code_root,
                args.source_root.resolve(),
                args.output_root.resolve(),
                log_file,
                workers=args.workers,
                shard_size=args.shard_size,
                checks=args.checks,
            )
        },
        "DataConfigs": [
            {
                "MappingPath": "/home/tione/workspace",
                "DataSourceUsage": "OTHER",
                "DataSourceType": "PUBLIC_DATA_SOURCE",
                "PublicDataSource": {
                    "DataSourceId": "dsrc-Cfs-chcn5v7uliio",
                    "SubPath": "/",
                },
                "ReadOnly": False,
            },
            {
                "MappingPath": "/home/tione/data",
                "DataSourceUsage": "OTHER",
                "DataSourceType": "PUBLIC_DATA_SOURCE",
                "PublicDataSource": {
                    "DataSourceId": "dsrc-Cfs-ch5rmhi5rk00",
                    "SubPath": "/",
                },
                "ReadOnly": False,
            },
        ],
        "Envs": [
            {"Name": "PYTHONUNBUFFERED", "Value": "1"},
            {"Name": "OMP_NUM_THREADS", "Value": "1"},
            {"Name": "OPENBLAS_NUM_THREADS", "Value": "1"},
            {"Name": "MKL_NUM_THREADS", "Value": "1"},
            {"Name": "MOLMO_MOTION_BUILD", "Value": "full-v1"},
            {"Name": "REQUESTED_BY", "Value": key_info.get("USER_NAME", "user")},
        ],
        "Remark": (
            "MolmoMotion-1M portable mmap conversion; CPU-only 100-core job; "
            "strict source preflight and resumable subset outputs"
        ),
    }


def make_client(key_info: dict[str, str], region: str) -> Any:
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.tione.v20211111 import tione_client

    cred = credential.Credential(
        key_info["TENCENTCLOUD_SECRET_ID"],
        key_info["TENCENTCLOUD_SECRET_KEY"],
    )
    http_profile = HttpProfile()
    http_profile.endpoint = "tione.tencentcloudapi.com"
    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile
    return tione_client.TioneClient(cred, region, client_profile)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.submit == args.dry_run:
        raise SystemExit("choose exactly one of --submit or --dry-run")
    key_info = _load_key_info(args.key_info.resolve())
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    job_name = resolve_job_name(args.job_name, key_info.get("USER_NAME", "user"), timestamp)
    run_root = args.output_root.resolve() / "_jobs" / job_name
    code_root = run_root / "code"
    log_file = run_root / "build.log"

    if args.dry_run:
        code_root = REPO_ROOT
        params = build_params(args, key_info, job_name, code_root, log_file)
        print(json.dumps(params, indent=2, ensure_ascii=False))
        print("[dry-run] no snapshot created and no task submitted")
        return 0

    branch, commit = snapshot_code(code_root)
    metadata = {
        "job_name": job_name,
        "branch": branch,
        "commit": commit,
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "cpu_cores": CPU_CORES,
        "gpu_count": GPU_COUNT,
        "workers": args.workers,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "submission.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    params = build_params(args, key_info, job_name, code_root, log_file)
    (run_root / "request.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(params, indent=2, ensure_ascii=False))

    from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
        TencentCloudSDKException,
    )
    from tencentcloud.tione.v20211111 import models

    try:
        client = make_client(key_info, args.region)
        request = models.CreateTrainingTaskRequest()
        request.from_json_string(json.dumps(params))
        response = client.CreateTrainingTask(request)
    except TencentCloudSDKException as error:
        raise SystemExit(f"TI-ONE submission failed: {error}") from error
    response_json = json.loads(response.to_json_string())
    (run_root / "response.json").write_text(
        json.dumps(response_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(response_json, indent=2, ensure_ascii=False))
    print(f"[log] tail -F {log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
