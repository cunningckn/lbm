"""
Install:
pip install kingsoftcloud-sdk-python -i https://mirrors.cloud.tencent.com/pypi/simple/ --trusted-host mirrors.cloud.tencent.com

Usage:
python submit_job.py --train_job_name "test-train-job-api" --description "test train job api" --num_gpu 4

Reference:
https://apiexplorer.ksyun.com/#/api/239/CreateTrainJob/2024-06-12/1246
"""

import os
import json
import re
import time
import argparse
import subprocess
from datetime import datetime
from ksyun.common import credential
from ksyun.common.profile.client_profile import ClientProfile
from ksyun.common.profile.http_profile import HttpProfile
from ksyun.common.exception.ksyun_sdk_exception import KsyunSDKException
from ksyun.client.aicp.v20240612 import client as aicp_client, models as aicp_models
from ksyun.client.iam.v20151101 import client as iam_client

with open("scripts/ks_cloud/key_info.json", "r") as f:
    key_info = json.load(f)

os.environ["http_proxy"] = "http://10.0.0.222:8888"
os.environ["https_proxy"] = "http://10.0.0.222:8888"

PRIORITY_MAP = {1: "kaic-high", 2: "kaic-normal", 3: "kaic-low"}

# Tracked paths not needed for training; rsync --files-from needs /*** for dirs.
SNAPSHOT_EXTRA_EXCLUDES = ["assets/web/***"]

PRETRAINED_PATH_RE = re.compile(r"(model\.pretrained_model_path\s*=\s*\")([^\"]+)(\")")

parser = argparse.ArgumentParser()
parser.add_argument("--train_job_name", "-t", type=str, default="")
parser.add_argument("--description", type=str, default="")
parser.add_argument("--num_gpu", "-g", type=int, default=0)
parser.add_argument("--num_cpu", type=int, default=-1)
parser.add_argument("--num_node", "-n", type=int, default=1)
parser.add_argument("--memory", type=int, default=-1)
parser.add_argument("--run_shell_script", "-r", type=str, default="scripts/ks_cloud/train_ks.sh")
parser.add_argument("--priority", "-p", type=int, default=1, choices=[1, 2, 3])
parser.add_argument(
    "--max_gpus",
    type=int,
    default=8,
    help="若已有 running/pending 任务占用 GPU 总数 >= 该值，则自动降为 priority 2（默认 8）",
)
parser.add_argument(
    "--no-snapshot",
    action="store_true",
    help="不创建代码快照，从 live 仓库直接运行（旧行为，便于快速调试）",
)
parser.add_argument("--queue_name", "-q", type=str, default="a800-gpu")
args = parser.parse_args()


def get_exp_name(train_script):
    with open(train_script, "r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            # Skip fully commented and empty lines.
            if not stripped or stripped.startswith("#"):
                continue

            # Ignore inline comments, e.g. exp_name="foo"  # backup config
            line = re.sub(r"\s+#.*$", "", raw_line)
            match = re.search(r'exp_name\s*=\s*"([^"]+)"', line)
            if match:
                return match.group(1)

    raise ValueError(f"Could not find active exp_name in {train_script}")


def make_client(service, endpoint, region, scheme="http", method="POST"):
    cred = credential.Credential(key_info["KSYUN_SECRET_ID"], key_info["KSYUN_SECRET_KEY"])
    httpProfile = HttpProfile()
    httpProfile.endpoint = endpoint
    httpProfile.reqMethod = method
    httpProfile.reqTimeout = 60
    httpProfile.scheme = scheme
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    return service(cred, region, profile=clientProfile)


def resolve_api_user_name():
    """Find IAM UserName that owns KSYUN_SECRET_ID."""
    iam = make_client(iam_client.IamClient, "iam.api.ksyun.com", "cn-beijing-6", "https", "GET")
    secret_id = key_info["KSYUN_SECRET_ID"]
    users = json.loads(iam.call("ListUsers", {"MaxItems": 100}))["ListUserResult"]["Users"]["member"]
    if isinstance(users, dict):
        users = [users]
    for user in users:
        aks = (
            json.loads(iam.call("ListAccessKeys", {"UserName": user["UserName"]}))
            .get("ListAccessKeyResult", {})
            .get("AccessKeyMetadata", {})
            .get("member", [])
        )
        if isinstance(aks, dict):
            aks = [aks]
        if any(ak.get("AccessKeyId") == secret_id for ak in (aks or [])):
            return user["UserName"]
    raise RuntimeError(f"Cannot resolve IAM user for {secret_id}")


def _job_gpu_count(job) -> int:
    """Per-replica GPUNumber * replica count (Master/Worker/...)."""
    gpu_per_replica = int(job.get("GPUNumber") or 0)
    replicas = job.get("FrameworkReplicas") or {}
    n_replicas = sum(int(replicas.get(k) or 0) for k in ("Master", "Worker", "Chief", "Evaluator", "PS"))
    return gpu_per_replica * max(n_replicas, 1)


def count_active_gpus(aicp, api_user):
    """Sum GPUs of this API user's running/pending jobs."""
    data = json.loads(aicp.call("DescribeTrainJob", {"MaxResults": 100}))
    total = 0
    for job in data.get("TrainJobSet", []):
        if job.get("CreateUserName") != api_user:
            continue
        state = job.get("Status", {}).get("State", "").lower()
        if state not in ("running", "pending"):
            continue
        gpus = _job_gpu_count(job)
        print(
            f"[priority-check] active job: {job.get('TrainJobName')} "
            f"({state}, priority={job.get('Priority')}, gpus={gpus})"
        )
        total += gpus
    return total


def absolutize_pretrained_path(path: str, code_dir: str) -> str:
    if path == "none" or path.startswith("/"):
        return path
    return os.path.join(code_dir, path)


def prepare_train_script(src: str, dst: str, code_dir: str) -> None:
    out_lines = []
    with open(src, encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                out_lines.append(raw_line)
                continue
            line_body = re.sub(r"\s+#.*$", "", raw_line.rstrip("\n"))
            match = PRETRAINED_PATH_RE.search(line_body)
            if not match:
                out_lines.append(raw_line)
                continue
            old_path = match.group(2)
            new_path = absolutize_pretrained_path(old_path, code_dir)
            if new_path != old_path:
                print(f"[snapshot] pretrained_model_path: {old_path} -> {new_path}")
            new_body = PRETRAINED_PATH_RE.sub(
                lambda m, np=new_path: m.group(1) + np + m.group(3),
                line_body,
                count=1,
            )
            suffix = "\n" if raw_line.endswith("\n") else ""
            out_lines.append(new_body + suffix)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("".join(out_lines))


def write_meta(run_dir: str, code_dir: str) -> None:
    os.makedirs(run_dir, exist_ok=True)
    meta = {}
    for key, cmd in [
        ("branch", ["git", "-C", code_dir, "rev-parse", "--abbrev-ref", "HEAD"]),
        ("commit", ["git", "-C", code_dir, "rev-parse", "HEAD"]),
        ("status", ["git", "-C", code_dir, "status", "--porcelain"]),
    ]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            meta[key] = result.stdout.strip()
        except subprocess.CalledProcessError:
            meta[key] = None
    meta["dirty"] = bool(meta.get("status"))
    meta["snapshot_time"] = datetime.now().isoformat()
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[snapshot] meta: branch={meta.get('branch')} commit={meta.get('commit')} dirty={meta['dirty']}")


def snapshot_code(code_dir: str, snapshot_dest: str) -> None:
    """Copy working tree via git ls-files (respects .gitignore incl. !negation)."""
    os.makedirs(snapshot_dest, exist_ok=True)
    raw = subprocess.check_output(["git", "-C", code_dir, "ls-files", "-z", "-c", "-o", "--exclude-standard"])
    # Skip index entries deleted on disk so rsync doesn't fail.
    files = b"\0".join(p for p in raw.split(b"\0") if p and os.path.lexists(os.path.join(code_dir, os.fsdecode(p))))
    cmd = ["rsync", "-a", "--files-from=-", "--from0"]
    for pat in SNAPSHOT_EXTRA_EXCLUDES:
        cmd.extend(["--exclude", pat])
    cmd.extend([code_dir + "/", snapshot_dest])
    print(f"[snapshot] git-aware rsync {code_dir} -> {snapshot_dest}")
    subprocess.run(cmd, input=files + b"\0", check=True)


def build_run_command(
    run_shell_script: str,
    use_snapshot: bool,
    enable_multi_nodes: bool,
) -> tuple[str, str, str]:
    """Return (cloud_command, log_file, exp_name)."""
    code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_script_src = os.path.join(code_dir, run_shell_script)
    exp_name = get_exp_name(train_script_src)
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    ckpt_exp_dir = os.path.join(code_dir, "checkpoints", exp_name)
    os.makedirs(ckpt_exp_dir, exist_ok=True)
    log_file = os.path.join(ckpt_exp_dir, f"server_{ts}.log")
    multi_nodes = int(enable_multi_nodes)

    if not use_snapshot:
        print("[snapshot] disabled (--no-snapshot), using live repo")
        train_script_run = os.path.join(ckpt_exp_dir, f"train_ks_{ts}.sh")
        prepare_train_script(train_script_src, train_script_run, code_dir)
        command = f"cd {code_dir}\nENABLE_MULTI_NODES={multi_nodes} bash {train_script_run} 2>&1 | tee {log_file}\n"
    else:
        print(f"[snapshot] enabled, run_dir=checkpoints/{exp_name}/runs/{ts}")
        run_dir = os.path.join(ckpt_exp_dir, "runs", ts)
        snapshot_dir = os.path.join(run_dir, "mindon_pi0")
        train_script_run = os.path.join(run_dir, "train_ks.sh")

        write_meta(run_dir, code_dir)
        t0 = time.time()
        snapshot_code(code_dir, snapshot_dir)
        print(f"[snapshot] rsync done in {time.time() - t0:.1f}s")
        prepare_train_script(train_script_src, train_script_run, code_dir)

        command = (
            f"SNAP={snapshot_dir}\n"
            f"LIVE={code_dir}\n"
            f"cd $SNAP\n"
            f"export LOCAL_DATA_PATH=$SNAP\n"
            f"export DEV_PATH=$(dirname $LIVE)\n"
            f"ENABLE_MULTI_NODES={multi_nodes} bash {train_script_run} 2>&1 | tee {log_file}\n"
        )

    return command, log_file, exp_name


def maybe_downgrade_priority(aicp, priority: int, max_gpus: int) -> int:
    """If priority is 1 and active GPUs >= max_gpus, downgrade to 2."""
    if priority != 1:
        return priority
    for attempt in range(1, 6):
        try:
            api_user = resolve_api_user_name()
            n_gpus = count_active_gpus(aicp, api_user)
            print(f"[priority-check] API user={api_user}, active_gpus={n_gpus}, max={max_gpus}")
            if n_gpus >= max_gpus:
                print(f"[priority-check] {n_gpus} >= {max_gpus}, downgrade priority 1 -> 2")
                return 2
            return 1
        except Exception as err:
            print(f"[priority-check] attempt {attempt}/5 failed: {err}")
            if attempt == 5:
                print("[priority-check] skip after 5 failures")
            else:
                time.sleep(2)
    return priority


def resolve_train_job_name(train_job_name: str, exp_name: str, user_name: str) -> str:
    """Return cloud job name; default {user}-{exp_name}, clipped to 64 chars."""
    if train_job_name:
        return train_job_name
    name = f"{user_name}-{exp_name}"
    if len(name) > 64:
        return name[:60] + "..."
    return name


def main():
    # 分配资源
    assert args.num_gpu in [0, 1, 2, 3, 4, 6, 7, 8], "num_gpus must be restricted"
    node_config_dict = {
        0: {"num_cpu": 50, "memory": 440},
        1: {"num_cpu": 12, "memory": 110},
        2: {"num_cpu": 24, "memory": 220},
        3: {"num_cpu": 36, "memory": 330},
        4: {"num_cpu": 48, "memory": 440},
        6: {"num_cpu": 72, "memory": 660},
        7: {"num_cpu": 84, "memory": 770},
        8: {"num_cpu": 96, "memory": 880},
    }
    if args.num_cpu == -1:
        args.num_cpu = node_config_dict[args.num_gpu]["num_cpu"]
    if args.memory == -1:
        args.memory = node_config_dict[args.num_gpu]["memory"]
    print(f"Using {args.num_gpu} GPUs, {args.num_cpu} CPUs, {args.memory}GB memory")

    # 创建AICP客户端
    aicpClient = make_client(aicp_client.AicpClient, "aicp.api.ksyun.com", "cn-northwest-3")

    # 分配任务优先级
    args.priority = maybe_downgrade_priority(aicpClient, args.priority, args.max_gpus)
    job_priority = PRIORITY_MAP[args.priority]

    # 构建运行命令
    run_command, log_file, exp_name = build_run_command(
        args.run_shell_script,
        not args.no_snapshot,
        args.num_node > 1,
    )

    # 构建任务名字
    train_job_name = resolve_train_job_name(args.train_job_name, exp_name, key_info["USER_NAME"])

    # 构建任务配置
    config = {
        "TrainJobName": train_job_name,
        "ResourcePoolId": "9210ffe0-b529-4ca9-a996-b509c9d7722d",
        "QueueName": args.queue_name,
        "Priority": job_priority,
        "Description": args.description,
        "Command": run_command,
        "Framework": "pytorch",
        "ImageSource": "Personal",
        "FrameworkReplicas": {"Worker": args.num_node - 1, "Chief": 0, "Evaluator": 0, "PS": 0, "Master": 1},
        "RestartPolicy": "Never",
        "Envs": [{"Name": "PYTHONUNBUFFERED", "Value": "1"}],
        "SupportTensorboard": False,
        "ImageId": "b73ee1b2-9bea-4fdb-ae2d-f824084e4fc4",
        "ImageRepoId": "xinyu",
        "ImageTagId": "fix-ib",
        "GPUType": "GM302",
        "GPUNumber": args.num_gpu,
        "CPUNum": args.num_cpu,
        "Memory": args.memory,
        "StorageConfigs": [
            {
                "StorageConfigId": "8f1a83d7-85a8-4abf-9c12-123f1fa82fbf",
                "MountPath": "/mnt/kpfs/data",
                "StorageConfigType": "DataSet",
                "MountType": "DataSet",
            },
            {
                "StorageConfigId": "6781b761-abbc-4692-80df-4e02cae92a19",
                "MountPath": "/mnt/kpfs/workspace",
                "StorageConfigType": "DataSet",
                "MountType": "DataSet",
            },
            {
                "StorageConfigId": "d4927d17-9bde-4ac9-b5eb-881e30294f61",
                "MountPath": "/mnt/open_source_data",
                "StorageConfigType": "DataSet",
                "MountType": "DataSet",
            },
        ],
        "AccessType": "QueueMember",
        "MaxRuntime": 720,
        "SelfHealing": False,
        "RunOnCPU": False,
    }

    print(json.dumps(config, indent=4))

    try:
        print(aicpClient.call_json("CreateTrainJob", config))
    except KsyunSDKException as err:
        print(err)

    subprocess.run(["tail", "-F", log_file])


if __name__ == "__main__":
    main()
