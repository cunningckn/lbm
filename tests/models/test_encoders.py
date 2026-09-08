import pytest
import torch

from lbm.config import ClipConfig
from lbm.models.clip import EOT_TOKEN, SOT_TOKEN, CLIPLanguageEncoder
from lbm.models.dino import DinoVisionBackbone
from lbm.models.dit import DiTPolicy
from lbm.models.encoders import (
    build_language_encoder,
    build_vision_backbone,
    load_encoder_weights,
)
from lbm.models.siglip import SiglipVisionBackbone
from lbm.models.t5 import T5LanguageEncoder
from lbm.utils.fake_data import make_fake_batch
from tests.helpers import tiny_dit_config


def _tiny(**kwargs):
    return tiny_dit_config(**kwargs)


def test_build_vision_backbone_dino_and_siglip():
    dino = build_vision_backbone(_tiny(vision_encoder="dino"))
    siglip = build_vision_backbone(_tiny(vision_encoder="siglip"))
    assert isinstance(dino, DinoVisionBackbone)
    assert isinstance(siglip, SiglipVisionBackbone)
    images = torch.randn(1, 3, 32, 32)
    dino_tokens = dino.encode_image_tokens(images)
    siglip_tokens = siglip.encode_image_tokens(images)
    patches = (32 // 16) ** 2
    assert dino_tokens.shape[1] == 1 + patches
    assert siglip_tokens.shape[1] == patches


def test_build_vision_backbone_rejects_unknown():
    with pytest.raises(ValueError, match="unknown vision_encoder"):
        build_vision_backbone(_tiny(vision_encoder="resnet"))


def test_build_language_encoder_none_and_t5():
    assert build_language_encoder(_tiny(language_encoder="none")) is None
    enc = build_language_encoder(_tiny(language_encoder="t5"))
    assert isinstance(enc, T5LanguageEncoder)
    assert enc.out_dim == 64


def test_build_language_encoder_rejects_unknown():
    with pytest.raises(ValueError, match="unknown language_encoder"):
        build_language_encoder(_tiny(language_encoder="bert"))


def test_language_pool_proj_mean_pool_is_pad_invariant():
    from lbm.models.dit import LanguagePoolProj

    pool = LanguagePoolProj(in_dim=4, hidden_size=8).eval()
    tokens = torch.tensor([[[1.0, 0, 0, 0], [2.0, 0, 0, 0], [3.0, 0, 0, 0], [9.0, 0, 0, 0]]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    with torch.no_grad():
        a = pool(tokens, attention_mask=mask)
        b = pool(tokens[:, :3], attention_mask=mask[:, :3])
    torch.testing.assert_close(a, b)


def test_encode_task_none_projects_clip_vec():
    model = DiTPolicy(_tiny(language_encoder="none"))
    batch = make_fake_batch(model.config, batch_size=2)
    out = model.encode_task(batch)
    assert out.shape == (2, model.config.hidden_size)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, batch["task_vec_clip"])


def test_encode_task_t5_uses_tokens_not_clip_vec():
    model = DiTPolicy(_tiny(language_encoder="t5")).eval()
    batch = make_fake_batch(model.config, batch_size=2)
    with torch.no_grad():
        out = model.encode_task(batch)
    assert out.shape == (2, model.config.hidden_size)
    assert not torch.allclose(out, batch["task_vec_clip"])


def test_siglip_t5_policy_forward_and_sample():
    config = _tiny(vision_encoder="siglip", language_encoder="t5")
    model = DiTPolicy(config).eval()
    batch = make_fake_batch(config, batch_size=1)
    loss = model(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    actions = model.sample_actions(batch, num_steps=2)
    assert actions.shape == (1, config.chunk_length, config.action_dim)
    assert torch.isfinite(actions).all()


def test_clip_language_encoder_pads_and_normalizes(clip):
    enc = CLIPLanguageEncoder(ClipConfig()).eval()
    sot = clip.tokenizer.encoder[SOT_TOKEN]
    eot = clip.tokenizer.encoder[EOT_TOKEN]
    short = torch.tensor([[sot, 10, 20, eot]])
    with torch.no_grad():
        out = enc(short)
    assert out.shape == (1, 512)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(1), rtol=1e-4, atol=1e-4)

    long = torch.randint(1, 100, (1, 90))
    long[0, 0] = sot
    long[0, 76] = eot
    with torch.no_grad():
        truncated = enc(long)
    assert truncated.shape == (1, 512)
    assert torch.isfinite(truncated).all()


def test_clip_language_encoder_matches_text_tower(clip):
    enc = CLIPLanguageEncoder(ClipConfig()).eval()
    text = "put the bottles in the bin"
    token_ids = [
        clip.tokenizer.encoder[SOT_TOKEN],
        *clip.tokenizer.encode(text),
        clip.tokenizer.encoder[EOT_TOKEN],
    ]
    ids = torch.zeros(1, clip.model.context_length, dtype=torch.long)
    ids[0, : len(token_ids)] = torch.tensor(token_ids)
    with torch.no_grad():
        from_enc = enc(ids)
        from_tower = clip.model(ids.to(clip.device))
        from_tower = from_tower / from_tower.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(from_enc.cpu(), from_tower.cpu(), rtol=1e-4, atol=1e-4)


def test_load_encoder_weights_siglip_missing_without_download(monkeypatch):
    monkeypatch.setattr("lbm.models.encoders.resolve_siglip_path", lambda: None)
    model = DiTPolicy(_tiny(vision_encoder="siglip"))
    with pytest.raises(FileNotFoundError, match="SigLIP"):
        load_encoder_weights(model, download=False)


def test_load_encoder_weights_t5_from_native_ckpt(tmp_path, monkeypatch):
    from lbm.models.t5 import T5LanguageEncoder

    src = T5LanguageEncoder(_tiny(language_encoder="t5"))
    path = tmp_path / "t5.pt"
    torch.save(src.state_dict(), path)
    monkeypatch.setenv("lbm_T5", str(path))
    model = DiTPolicy(_tiny(language_encoder="t5"))
    loaded = load_encoder_weights(model, download=False)
    assert loaded["t5"] == str(path)
    torch.testing.assert_close(model.language_encoder.embed_tokens.weight, src.embed_tokens.weight)


def test_load_encoder_weights_siglip_from_native_ckpt(tmp_path, monkeypatch):
    src = SiglipVisionBackbone(_tiny(vision_encoder="siglip"))
    path = tmp_path / "siglip.pt"
    torch.save(src.siglip_model.state_dict(), path)
    monkeypatch.setenv("lbm_SIGLIP", str(path))
    model = DiTPolicy(_tiny(vision_encoder="siglip"))
    loaded = load_encoder_weights(model, download=False)
    assert loaded["siglip"] == str(path)
    torch.testing.assert_close(model.img_backbone.siglip_model.pos_embed, src.siglip_model.pos_embed)
