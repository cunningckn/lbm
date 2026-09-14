import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


def test_failed_locked_install_stops_and_preserves_existing_environment(tmp_path):
    source = tmp_path / "libero"
    source.mkdir()
    venv = tmp_path / "custom venv"
    venv.mkdir()
    sentinel = venv / "keep.txt"
    sentinel.write_text("existing user file")
    log = tmp_path / "uv.jsonl"
    uv = tmp_path / "fake uv"
    executable(
        uv,
        """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['LBM_TEST_LOG'], 'a') as f:
    f.write(json.dumps(args)+'\\n')
if args[0] == 'venv':
    p = Path(args[-1])/'bin'/'python'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('#!/bin/sh\\nexit 0\\n')
    p.chmod(0o755)
if args[:2] == ['pip', 'install'] and '-r' in args:
    raise SystemExit(19)
""",
    )
    env = dict(
        os.environ,
        LIBERO_ROOT=str(source),
        LIBERO_VENV=str(venv),
        LIBERO_PYTHON="chosen-interpreter",
        UV_BIN=str(uv),
        LBM_TEST_LOG=str(log),
    )
    result = subprocess.run(
        ["bash", str(ROOT / "simulation/libero/install_env.sh")], env=env, capture_output=True, text=True
    )
    assert result.returncode == 19
    assert sentinel.read_text() == "existing user file"
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 2  # No reduced fallback, editable install or success announcement.
    assert calls[0][-1] == str(venv)
    assert "chosen-interpreter" in calls[0]
    assert Path(calls[1][calls[1].index("-r") + 1]).name == "requirements.txt"
    assert "Environment ready" not in result.stdout


def test_eval_uses_same_explicit_environment(tmp_path):
    venv = tmp_path / "custom venv"
    (venv / "bin").mkdir(parents=True)
    log = tmp_path / "eval.json"
    executable(
        venv / "bin/python",
        """
import json, os, sys
with open(os.environ['LBM_TEST_LOG'], 'w') as f:
    json.dump(sys.argv[1:], f)
""",
    )
    env = dict(os.environ, LIBERO_VENV=str(venv), LBM_TEST_LOG=str(log))
    result = subprocess.run(
        ["bash", str(ROOT / "simulation/libero/eval_env.sh"), "--replan-steps", "1"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    argv = json.loads(log.read_text())
    assert Path(argv[0]).name == "main.py"
    assert argv[-2:] == ["--replan-steps", "1"]
