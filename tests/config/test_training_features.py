from dataclasses import asdict

import pytest
import torch

from lbm.config import TrainConfig
from lbm.models.dit import DiTPolicy
from lbm.training_features import MODEL_FIELDS, FeatureDataset, backbone_fingerprint, write_feature_cache
from lbm.utils.fake_data import make_fake_batch
from tests.helpers import tiny_dit_config


@pytest.mark.parametrize('history', [1, 2])
@pytest.mark.parametrize('encoder', ['dino', 'siglip'])
def test_cached_features_preserve_loss_gradients_and_sampling(history, encoder):
    config = tiny_dit_config(vision_encoder=encoder)
    model = DiTPolicy(config).train()
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=.03)
    live = make_fake_batch(config, 2, device='cpu', image_size=32)
    if history > 1:
        live['images'] = {cam: value[:, None].expand(-1, history, -1, -1, -1)
                          for cam, value in live['images'].items()}
    live['camera_mask'] = torch.ones(2, len(config.camera_keys), dtype=torch.bool)
    live['camera_mask'][:, -1] = False
    cached = {k: v for k, v in live.items() if k != 'images'}
    cached['vision_features'] = model.encode_vision_features(live['images'])
    noise = torch.randn_like(live['actions'])
    t = torch.full((2, 1, 1), .3)
    loss = model(live, noise=noise, t=t)
    loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    assert any('vision_pool' in name and grad.abs().max() > 0 for name, grad in grads.items())
    model.zero_grad(set_to_none=True)
    other = model(cached, noise=noise, t=t)
    other.backward()
    torch.testing.assert_close(other, loss)
    for name, p in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(p.grad, grads[name])
    torch.testing.assert_close(model.sample_actions(cached, num_steps=2, noise=noise),
                               model.sample_actions(live, num_steps=2, noise=noise))
    config.train_vision_encoder = True
    with pytest.raises(ValueError, match='frozen'):
        model(cached)


def _cache_batch(config, model):
    batch = make_fake_batch(config, 2, device='cpu', image_size=32)
    batch['vision_features'] = model.encode_vision_features(batch.pop('images'))
    batch['action_mask'] = torch.ones_like(batch['actions'], dtype=torch.bool)
    batch['camera_mask'] = torch.ones(2, len(config.camera_keys), dtype=torch.bool)
    return batch


def test_cache_roundtrip_bfloat16_and_partial_failure(tmp_path):
    config = tiny_dit_config()
    model = DiTPolicy(config)
    batch = _cache_batch(config, model)
    batch['state'] = torch.full_like(batch['state'], 400000).bfloat16()
    dest = tmp_path / 'complete'
    write_feature_cache(dest, [batch], rows=2, metadata={})
    loaded = FeatureDataset(dest)
    torch.testing.assert_close(loaded[0]['state'], batch['state'][0])
    assert loaded[0]['state'].dtype == torch.bfloat16
    fresh = FeatureDataset.__new__(FeatureDataset)
    fresh.__dict__.update(loaded.__getstate__())
    torch.testing.assert_close(fresh[1]['actions'], batch['actions'][1])
    with pytest.raises(ValueError, match='expected'):
        write_feature_cache(tmp_path / 'failed', [batch], rows=3, metadata={})
    assert sorted(p.name for p in tmp_path.iterdir()) == ['complete']
    with pytest.raises(FileExistsError):
        write_feature_cache(dest, [batch], rows=2, metadata={})
    filename = next(iter(loaded.manifest['fields'].values()))['file']
    (dest / filename).write_bytes(b'broken')
    with pytest.raises((ValueError, OSError)):
        FeatureDataset(dest)


def test_feature_training_and_resume_use_cached_inputs(tmp_path, monkeypatch):
    from lbm import train_loop

    config = tiny_dit_config()
    torch.manual_seed(123)
    model = DiTPolicy(config)
    batch = _cache_batch(config, model)
    metadata = dict(model={k: asdict(config)[k] for k in MODEL_FIELDS},
                    backbone_sha256=backbone_fingerprint(model.img_backbone))
    dest = tmp_path / 'features'
    write_feature_cache(dest, [batch], rows=2, metadata=metadata)
    def forbid(*args, **kwargs):
        pytest.fail('feature training must not load source images or language encoders')
    monkeypatch.setattr(train_loop, 'CLIPTextEmbedder', forbid)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    cfg = TrainConfig(model=config, feature_cache=str(dest), batch_size=2, num_workers=0,
                      pretrained_encoders=False, train_steps=2, ckpt_every=2, val_every=1,
                      val_batches=1, dump_batch=False, output_dir=str(tmp_path / 'train'))
    train_loop.main(cfg)
    cfg.resume = str(tmp_path / 'train' / 'last.pt')
    cfg.train_steps = 3
    train_loop.main(cfg)
    assert torch.isfinite(torch.load(cfg.resume, weights_only=False)['model']['x_embedder.weight']).all()


def test_feature_training_cli_selects_real_cache_without_dataset(monkeypatch):
    import sys

    from scripts import train
    captured = []
    monkeypatch.setattr(sys, 'argv', ['train.py', '--feature-cache', '/example/cache'])
    monkeypatch.setattr(train, 'train_main', captured.append)
    train.main()
    assert captured[0].feature_cache == '/example/cache'
    assert not captured[0].fake_data


def test_encoder_fingerprint_rejects_changed_weights(tmp_path):
    config = tiny_dit_config()
    model = DiTPolicy(config)
    batch = _cache_batch(config, model)
    dest = tmp_path / 'features'
    write_feature_cache(dest, [batch], rows=2,
                        metadata={'backbone_sha256': backbone_fingerprint(model.img_backbone)})
    dataset = FeatureDataset(dest)
    dataset.check_backbone(model)
    with torch.no_grad():
        next(model.img_backbone.parameters()).add_(1)
    with pytest.raises(ValueError, match='different encoder'):
        dataset.check_backbone(model)
