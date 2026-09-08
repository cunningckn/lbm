import torch

from lbm.batch import _as_str, _current_vector, _normalize_images, policy_batch_from_loader
from lbm.config import DiTConfig
from lbm.utils.fake_data import make_fake_batch
from lbm.utils.preprocess import imagenet_normalize


def test_current_vector_takes_last_step():
    x = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    out = _current_vector(x)
    assert out.shape == (2, 2)
    assert torch.equal(out[0], x[0, -1])


def test_as_str_flattens_numpy():
    assert _as_str(b"pick up") == "pick up"
    assert _as_str(["stack blocks"]) == "stack blocks"


def test_normalize_images_single_and_history():
    keys = ("cam_high", "cam_left")
    single = torch.randint(0, 256, (2, 2, 1, 8, 8, 3), dtype=torch.uint8)
    out = _normalize_images(single, keys)
    assert set(out) == set(keys)
    assert out["cam_high"].shape == (2, 3, 8, 8)
    expected = imagenet_normalize(single[:, 0, -1].permute(0, 3, 1, 2).float() / 255.0)
    torch.testing.assert_close(out["cam_high"], expected)

    hist = torch.randint(0, 256, (2, 2, 4, 8, 8, 3), dtype=torch.uint8)
    out_h = _normalize_images(hist, keys)
    assert out_h["cam_high"].shape == (2, 4, 3, 8, 8)


def test_policy_batch_from_loader_shapes():
    cfg = DiTConfig(action_length=8.0, action_freq=1.0, camera_keys=("cam_high", "cam_left_wrist"))
    fake = make_fake_batch(cfg, batch_size=2)
    image = torch.zeros(2, 2, 1, 224, 224, 3, dtype=torch.uint8)
    raw = {
        "image": image,
        "action": fake["actions"],
        "state": fake["state"].unsqueeze(1),
        "lang": ["pick the cup", "place the cup"],
        "robot_tag": ["aloha", "franka"],
        "action_mask": torch.ones_like(fake["actions"], dtype=torch.bool),
        "camera_mask": torch.ones(2, 2, dtype=torch.bool),
    }

    class _Emb:
        def encode(self, texts):
            return torch.randn(len(texts), cfg.task_embed_dim)

    batch = policy_batch_from_loader(
        raw,
        camera_keys=cfg.camera_keys,
        device=torch.device("cpu"),
        dtype=torch.float32,
        embedder=_Emb(),
        language_encoder="none",
        mask_state_ratio=0.0,
        train=True,
    )
    assert batch["state"].shape == (2, cfg.state_dim)
    assert batch["actions"].shape == (2, cfg.chunk_length, cfg.action_dim)
    assert batch["images"]["cam_high"].shape == (2, 3, 224, 224)
    assert batch["task_vec_clip"].shape == (2, cfg.task_embed_dim)
    assert batch["embodiment_id"].tolist() == [7, 25]
    assert batch["action_mask"].shape == batch["actions"].shape
