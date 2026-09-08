import torch

from lbm.config import OptimConfig
from lbm.models.dit import DiTPolicy
from lbm.optim import build_adamw, count_trainable
from lbm.utils.fake_data import make_fake_batch
from tests.helpers import tiny_dit_config


def _tiny(**kwargs):
    return tiny_dit_config(**kwargs)


def _grads(module) -> list[torch.Tensor]:
    return [p.grad.detach() for p in module.parameters() if p.grad is not None]


def test_history_frames_concat_vision_tokens():
    config = _tiny(history_length=0.3, history_freq=10.0)  # 3 frames
    assert config.history_size == 3
    model = DiTPolicy(config)
    batch = make_fake_batch(config, batch_size=2)
    t_hist = config.history_size
    batch["images"] = {
        cam: img.unsqueeze(1).expand(-1, t_hist, -1, -1, -1).contiguous()
        for cam, img in batch["images"].items()
    }
    tokens = model.build_vision_tokens(batch["images"])
    single = {
        cam: img[:, 0] for cam, img in batch["images"].items()
    }
    tokens_1 = model.build_vision_tokens(single)
    assert tokens.shape[0] == 2
    assert tokens.shape[1] == t_hist * tokens_1.shape[1]
    assert tokens.shape[2] == tokens_1.shape[2]
    loss = model(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_frozen_encoder_skips_vision_grads():
    config = _tiny(train_vision_encoder=False)
    model = DiTPolicy(config)
    loss = model(make_fake_batch(config, batch_size=2))
    loss.backward()
    assert all(p.grad is None for p in model.img_backbone.parameters())
    pool_grads = _grads(model.apool)
    assert pool_grads and any(g.abs().sum() > 0 for g in pool_grads)
    model.train()
    assert not model.img_backbone.training


def test_pool_proj_always_trainable():
    config = _tiny(train_vision_encoder=False)
    model = DiTPolicy(config)
    assert all(p.requires_grad for p in model.apool.parameters())
    assert model.vision_tokens_proj.weight.requires_grad
    assert model.vision_camera_embed.weight.requires_grad
    model(make_fake_batch(config, batch_size=2)).backward()
    assert all(p.grad is None for p in model.img_backbone.parameters())
    assert any(p.grad is not None for p in model.apool.parameters())
    assert any(p.grad is not None for p in model.blocks.parameters())


def test_remap_vision_pool_checkpoint_keys():
    from lbm.models.dit import _remap_vision_pool_keys

    sd = {
        "apool_queries.top": torch.zeros(1),
        "apool.top.q.weight": torch.zeros(1),
        "vision_tokens_proj.weight": torch.zeros(1),
        "vision_camera_embed.weight": torch.zeros(1),
        "blocks.0.attn.qkv.weight": torch.zeros(1),
    }
    out = _remap_vision_pool_keys(sd)
    assert set(out) == {
        "vision_pool.queries.top",
        "vision_pool.apool.top.q.weight",
        "vision_pool.proj.weight",
        "vision_pool.camera_embed.weight",
        "blocks.0.attn.qkv.weight",
    }
    assert _remap_vision_pool_keys(out) == out


def test_language_pool_proj_always_trainable():
    config = _tiny(language_encoder="t5", train_language_encoder=False, task_embed_dim=64)
    model = DiTPolicy(config)
    assert all(p.requires_grad for p in model.language_pool_proj.parameters())
    model(make_fake_batch(config, batch_size=2)).backward()
    assert all(p.grad is None for p in model.language_encoder.parameters())
    assert any(p.grad is not None for p in model.language_pool_proj.parameters())
    assert any(p.grad is not None for p in model.blocks.parameters())


def test_frozen_t5_language_encoder():
    config = _tiny(language_encoder="t5", train_language_encoder=False, task_embed_dim=64)
    model = DiTPolicy(config)
    assert model.language_encoder is not None
    assert all(not p.requires_grad for p in model.language_encoder.parameters())
    model(make_fake_batch(config, batch_size=2)).backward()
    assert all(p.grad is None for p in model.language_encoder.parameters())
    model.train()
    assert not model.language_encoder.training


def test_siglip_tiny_forward_and_backward():
    config = _tiny(vision_encoder="siglip", train_vision_encoder=True)
    model = DiTPolicy(config)
    loss = model(make_fake_batch(config, batch_size=2))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.img_backbone.parameters())


def test_build_adamw_skips_frozen_and_scales_vision_lr():
    config = _tiny(train_vision_encoder=True)
    model = DiTPolicy(config)
    optim = build_adamw(model, OptimConfig(learning_rate=1e-4, vision_lr_scale=0.1))
    assert len(optim.param_groups) == 2
    assert abs(optim.param_groups[0]["lr"] - 1e-5) < 1e-12

    frozen = DiTPolicy(_tiny(train_vision_encoder=False))
    n_train, n_total = count_trainable(frozen)
    assert n_train < n_total
    assert len(build_adamw(frozen, OptimConfig()).param_groups) == 1
