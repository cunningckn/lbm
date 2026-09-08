from __future__ import annotations

import pytest
import torch

from lbm import ClipConfig, CLIPTextEmbedder, DiTConfig, DiTPolicy, load_dinov3, make_fake_batch, resolve_dinov3_path

PROMPT = "put the bottles in the bin"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: tests that need real datasets on disk")
    config.addinivalue_line("markers", "gpu: tests that need a CUDA device")


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def config() -> DiTConfig:
    return DiTConfig()


@pytest.fixture(scope="session")
def dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


@pytest.fixture(scope="session")
def clip(device: torch.device) -> CLIPTextEmbedder:
    embedder = CLIPTextEmbedder(ClipConfig(), device=device)
    if device.type == "cuda":
        embedder.set_bfloat16(True)
    return embedder


@pytest.fixture(scope="session")
def task_vec(clip: CLIPTextEmbedder) -> torch.Tensor:
    return clip.encode(PROMPT)


@pytest.fixture(scope="session")
def model(device: torch.device, config: DiTConfig, dtype: torch.dtype) -> DiTPolicy:
    ckpt = resolve_dinov3_path()
    if ckpt is None:
        pytest.skip(
            "DINOv3 checkpoint not found. Set LBM_DINO / lbm_DINO or place "
            "weights in lbm/checkpoints/dinov3."
        )
    torch.set_float32_matmul_precision("high")
    policy = DiTPolicy(config).to(device=device, dtype=dtype)
    load_dinov3(policy.img_backbone, ckpt)
    if device.type == "cuda":
        policy.img_backbone.set_bfloat16(True)
    return policy


@pytest.fixture
def batch(config: DiTConfig, device: torch.device, task_vec: torch.Tensor, dtype: torch.dtype) -> dict:
    out = make_fake_batch(config, batch_size=1, device=device, dtype=dtype)
    out["task_vec_clip"] = task_vec.to(device=device, dtype=dtype)
    return out
