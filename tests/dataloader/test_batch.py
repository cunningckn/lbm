import pytest
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


def test_single_source_batch_pads_to_mixed_model_and_masks_padding():
    from tests.helpers import tiny_dit_config

    from lbm.models.dit import DiTPolicy

    cfg = tiny_dit_config(state_dim=20, action_dim=22, action_length=4,
                          camera_keys=("other", "present"))
    raw = {"image": torch.zeros(2, 1, 1, 32, 32, 3, dtype=torch.uint8),
           "camera_keys": ("present",), "state": torch.ones(2, 14),
           "action": torch.ones(2, 2, 14), "lang": ["test", "test"]}
    batch = policy_batch_from_loader(raw, camera_keys=cfg.camera_keys,
                                    device=torch.device("cpu"), dtype=torch.float32,
                                    state_dim=20, action_dim=22, action_steps=4)
    assert batch["state"].shape == (2, 20)
    assert not batch["state"][:, 14:].any()
    assert batch["actions"].shape == (2, 4, 22)
    assert batch["action_mask"][:, :2, :14].all()
    assert not batch["action_mask"][:, 2:].any()
    assert not batch["action_mask"][..., 14:].any()
    assert batch["camera_mask"].tolist() == [[False, True], [False, True]]
    assert not batch["images"]["other"].any()
    batch["task_vec_clip"] = torch.zeros(2, cfg.task_embed_dim)
    model = DiTPolicy(cfg)
    loss = model(batch)
    assert torch.isfinite(loss)
    loss.backward()


def test_mixed_model_rejects_truncating_observed_dimensions():
    import pytest
    raw = {"image": torch.zeros(1, 1, 1, 32, 32, 3, dtype=torch.uint8),
           "state": torch.ones(1, 20), "action": torch.ones(1, 2, 22), "lang": ["test"]}
    with pytest.raises(ValueError, match="state_dim"):
        policy_batch_from_loader(raw, camera_keys=("cam",), device=torch.device("cpu"),
                                 dtype=torch.float32, state_dim=14)
    with pytest.raises(ValueError, match="action_dim"):
        policy_batch_from_loader(raw, camera_keys=("cam",), device=torch.device("cpu"),
                                 dtype=torch.float32, action_dim=14)


def test_shorter_model_horizon_is_not_silently_truncated():
    import pytest
    raw = {"image": torch.zeros(1, 1, 1, 32, 32, 3, dtype=torch.uint8),
           "state": torch.ones(1, 14), "action": torch.ones(1, 4, 14), "lang": ["test"]}
    with pytest.raises(ValueError, match="action_steps"):
        policy_batch_from_loader(raw, camera_keys=("cam",), device=torch.device("cpu"),
                                 dtype=torch.float32, action_steps=2)


@pytest.mark.gpu
@pytest.mark.parametrize('history', [1, 3])
def test_cuda_image_normalization_matches_cpu(history):
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    raw = dict(image=torch.randint(0, 256, (2, 2, history, 16, 16, 3), dtype=torch.uint8).pin_memory(),
               action=torch.zeros(2, 4, 3), state=torch.zeros(2, 3), lang=['task', 'task'])
    class Embedder:
        def encode(self, texts):
            return torch.zeros(len(texts), 8)
    kwargs = dict(camera_keys=('one', 'two'), dtype=torch.bfloat16, train=False, embedder=Embedder())
    cpu = policy_batch_from_loader(raw, device=torch.device('cpu'), **kwargs)
    cuda = policy_batch_from_loader(raw, device=torch.device('cuda'), **kwargs)
    for key in cpu['images']:
        torch.testing.assert_close(cuda['images'][key].cpu(), cpu['images'][key], rtol=0, atol=0)
