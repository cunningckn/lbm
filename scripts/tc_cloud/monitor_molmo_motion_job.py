"""Query one TI-ONE MolmoMotion conversion task and append a safe audit record."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from submit_molmo_motion_full import (
    DEFAULT_API_PROXY,
    DEFAULT_KEY_INFO,
    DEFAULT_REGION,
    _load_key_info,
    make_client,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-id", required=True)
    result.add_argument("--key-info", type=Path, default=DEFAULT_KEY_INFO)
    result.add_argument("--region", default=DEFAULT_REGION)
    result.add_argument("--api-proxy", default=DEFAULT_API_PROXY)
    result.add_argument("--audit-log", type=Path, default=None)
    return result


def summarize(document: dict[str, Any]) -> dict[str, Any]:
    detail = document.get("TrainingTaskDetail") or {}
    return {
        "task_id": detail.get("Id"),
        "name": detail.get("Name"),
        "status": detail.get("Status"),
        "message": detail.get("Message"),
        "failure_reason": detail.get("FailureReason"),
        "create_time": detail.get("CreateTime"),
        "start_time": detail.get("StartTime"),
        "end_time": detail.get("EndTime"),
        "runtime_seconds": detail.get("RuntimeInSeconds"),
        "latest_instance_id": detail.get("LatestInstanceId"),
        "request_id": document.get("RequestId"),
        "observed_at": datetime.now().astimezone().isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    key_info = _load_key_info(args.key_info.resolve())
    from tencentcloud.tione.v20211111 import models

    request = models.DescribeTrainingTaskRequest()
    request.Id = args.task_id
    response = make_client(key_info, args.region, args.api_proxy).DescribeTrainingTask(request)
    result = summarize(json.loads(response.to_json_string()))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.audit_log is not None:
        path = args.audit_log.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
