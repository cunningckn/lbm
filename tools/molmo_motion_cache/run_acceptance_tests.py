"""Persist machine-readable standalone test evidence outside immutable caches."""

import argparse
import io
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from molmo_motion_cache.common import write_json
from molmo_motion_cache.delivery import code_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    stream = io.StringIO()
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "skipped": len(result.skipped),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "elapsed_seconds": time.perf_counter() - start,
        "code_fingerprint": code_fingerprint(),
        "output": stream.getvalue(),
    }
    write_json(args.report, report)
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
