"""Sample GPU busy / SM(CU) / wave occupancy during a bench window.

Hygon DCU (this cluster): ``hy-smi`` / ``rocm-smi``.
NVIDIA fallback: ``nvidia-smi``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Any

_HCU_UTIL = re.compile(r"HCU\[(\d+)\].*HCU util in last second\s*:\s*([\d.]+)%")
_CU_UTIL = re.compile(r"HCU\[(\d+)\].*CU util in last second\s*:\s*([\d.]+)%")
_WAVE_UTIL = re.compile(r"HCU\[(\d+)\].*wave util in last second\s*:\s*([\d.]+)%")
_HCU_USE = re.compile(r"HCU\[(\d+)\].*HCU use \(%\):\s*([\d.]+)")
_MEM_USE = re.compile(r"HCU\[(\d+)\].*HCU memory use \(%\):\s*([\d.]+)")
_PKG_PWR = re.compile(r"HCU\[(\d+)\].*Average Graphics Package Power \(W\):\s*([\d.]+)")
_GFX_PWR = re.compile(r"HCU\[(\d+)\].*Average GFX Core Power \(W\):\s*([\d.]+)")
_SCLK = re.compile(r"HCU\[(\d+)\].*sclk clock level:.*\((\d+)Mhz\)")
_MEMBW = re.compile(
    r"HCU\[(\d+)\].*Bandwidth:.*R:\s*([\d.]+)\s*MB/s.*W:\s*([\d.]+)\s*MB/s.*R_W:\s*([\d.]+)\s*MB/s"
)
_NV_CSV = re.compile(r"^\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*$")


@dataclass
class GpuSample:
    """One poll of a single device. None means that metric was not in the dump."""

    gpu: float | None = None
    sm: float | None = None
    wave: float | None = None
    mem: float | None = None
    power_w: float | None = None
    gfx_w: float | None = None
    sclk_mhz: float | None = None
    membw_gbps: float | None = None


def _set_device(rows: dict[int, GpuSample], idx: int) -> GpuSample:
    return rows.setdefault(idx, GpuSample())


_SIMPLE_FIELDS = (
    (_HCU_UTIL, "gpu"),
    (_CU_UTIL, "sm"),
    (_WAVE_UTIL, "wave"),
    (_MEM_USE, "mem"),
    (_PKG_PWR, "power_w"),
    (_GFX_PWR, "gfx_w"),
    (_SCLK, "sclk_mhz"),
)


def parse_smi_text(text: str) -> dict[int, GpuSample]:
    """Parse hy-smi / rocm-smi stdout into per-device samples."""
    rows: dict[int, GpuSample] = {}
    for pattern, field in _SIMPLE_FIELDS:
        for match in pattern.finditer(text):
            setattr(_set_device(rows, int(match.group(1))), field, float(match.group(2)))
    for match in _HCU_USE.finditer(text):
        sample = _set_device(rows, int(match.group(1)))
        if sample.gpu is None:
            sample.gpu = float(match.group(2))
    for match in _MEMBW.finditer(text):
        _set_device(rows, int(match.group(1))).membw_gbps = float(match.group(4)) / 1024.0
    return rows


def parse_nvidia_smi_csv(text: str, device_index: int) -> GpuSample | None:
    for line in text.splitlines():
        match = _NV_CSV.match(line.strip())
        if not match:
            continue
        return GpuSample(
            gpu=float(match.group(1)),
            mem=float(match.group(2)),
            power_w=float(match.group(3)),
            sclk_mhz=float(match.group(4)),
        )
    return None


def _run(cmd: list[str], timeout: float) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


def _hy_smi() -> str | None:
    return shutil.which("hy-smi") or shutil.which("rocm-smi")


def poll_device(device_index: int, *, sm_metrics: bool) -> GpuSample | None:
    hy = _hy_smi()
    if hy:
        cmd = [hy, "-d", str(device_index), "-u", "--showmemuse", "-P"]
        if sm_metrics:
            cmd.extend(["--showhcuutil", "--showcuutil", "--showwaveutil", "--showmembw", "-g"])
        timeout = 12.0 if sm_metrics else 2.0
        rows = parse_smi_text(_run(cmd, timeout))
        if device_index in rows:
            return rows[device_index]
        if len(rows) == 1:
            return next(iter(rows.values()))
        return None
    nv = shutil.which("nvidia-smi")
    if nv:
        text = _run(
            [
                nv,
                f"--id={device_index}",
                "--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            timeout=2.0,
        )
        return parse_nvidia_smi_csv(text, device_index)
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(samples: list[GpuSample]) -> GpuSample:
    out = GpuSample()
    for name in ("gpu", "sm", "wave", "mem", "power_w", "gfx_w", "sclk_mhz", "membw_gbps"):
        vals = [getattr(s, name) for s in samples if getattr(s, name) is not None]
        setattr(out, name, _mean(vals))
    return out


def merge_rank_summaries(summaries: list[GpuSample | None]) -> GpuSample:
    return summarize([s for s in summaries if s is not None])


def format_util(sample: GpuSample | None) -> str:
    if sample is None:
        return "gpu=nan sm=nan wave=nan mem=nan pwr=nan sclk=nan membw=nan n=0"

    def fmt(name: str, nd: int = 1) -> str:
        val = getattr(sample, name)
        return f"{val:.{nd}f}" if val is not None else "nan"

    return (
        f"gpu={fmt('gpu')} sm={fmt('sm')} wave={fmt('wave')} mem={fmt('mem')} "
        f"pwr={fmt('power_w')} gfx={fmt('gfx_w')} sclk={fmt('sclk_mhz', 0)} "
        f"membw={fmt('membw_gbps')}"
    )


class GpuUtilSampler:
    """Background polls of local-device util while a bench window runs."""

    def __init__(self, device_index: int):
        self.device_index = int(device_index)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._fast: list[GpuSample] = []
        self._slow: list[GpuSample] = []
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> GpuUtilSampler:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        self._stop.clear()
        fast = threading.Thread(target=self._loop_fast, name="gpu-util-fast", daemon=True)
        slow = threading.Thread(target=self._loop_slow, name="gpu-util-sm", daemon=True)
        self._threads = [fast, slow]
        for thread in self._threads:
            thread.start()

    def stop(self) -> GpuSample | None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=15.0)
        self._threads = []
        return self.summary()

    def summary(self) -> GpuSample | None:
        with self._lock:
            fast = list(self._fast)
            slow = list(self._slow)
        if not fast and not slow:
            return None
        merged = summarize(fast + slow)
        # Prefer dedicated SM/wave/gpu-busy samples from the slow hy-smi path.
        slow_sum = summarize(slow) if slow else GpuSample()
        for name in ("gpu", "sm", "wave", "membw_gbps", "sclk_mhz"):
            val = getattr(slow_sum, name)
            if val is not None:
                setattr(merged, name, val)
        if merged.gpu is None:
            merged.gpu = summarize(fast).gpu
        return merged

    def n_samples(self) -> tuple[int, int]:
        with self._lock:
            return len(self._fast), len(self._slow)

    def _loop_fast(self) -> None:
        while not self._stop.is_set():
            sample = poll_device(self.device_index, sm_metrics=False)
            if sample is not None:
                with self._lock:
                    self._fast.append(sample)
            self._stop.wait(0.25)

    def _loop_slow(self) -> None:
        while not self._stop.is_set():
            sample = poll_device(self.device_index, sm_metrics=True)
            if sample is not None:
                with self._lock:
                    self._slow.append(sample)
            # hy-smi already blocks ~3s; a short pause avoids back-to-back pileup
            self._stop.wait(0.05)


def gather_util(local: GpuSample | None, *, enabled: bool) -> GpuSample | None:
    """All-gather per-rank summaries; return the cluster mean."""
    if not enabled:
        return local
    try:
        import torch

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return local
        world = torch.distributed.get_world_size()
        gathered: list[GpuSample | None] = [None] * world
        torch.distributed.all_gather_object(gathered, local)
        return merge_rank_summaries(gathered)
    except Exception:
        return local
