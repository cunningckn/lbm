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


def test_cache_preserves_action_semantics_and_rejects_validation_scale_mismatch():
    from copy import deepcopy
    train = FeatureDataset.__new__(FeatureDataset)
    train.metadata = dict(model={}, normalization={'demo': {'state': {}, 'actions': {}}},
                          action_spaces={'demo': [{'kind': 'eef', 'rep': 'rel'}]})
    validation = FeatureDataset.__new__(FeatureDataset)
    validation.metadata = deepcopy(train.metadata)
    train.check_validation(validation)
    payload = train.inference_normalization()
    assert payload['action_space'] == train.metadata['action_spaces']['demo']
    assert payload['norm_stats'] == train.metadata['normalization']['demo']
    validation.metadata['normalization']['demo']['state']['mean'] = [1.]
    with pytest.raises(ValueError, match='normalization'):
        train.check_validation(validation)
    validation.metadata = deepcopy(train.metadata)
    validation.metadata['action_spaces']['demo'][0]['rep'] = 'delta'
    with pytest.raises(ValueError, match='action_spaces'):
        train.check_validation(validation)


def test_sharded_cache_resumes_only_completed_rows_and_checks_data(tmp_path):
    import json

    from lbm.feature_shards import write_sharded_cache

    config = tiny_dit_config()
    model = DiTPolicy(config)
    batch = _cache_batch(config, model)
    cache = tmp_path / 'sharded'
    starts = []
    def interrupted(start):
        starts.append(start)
        yield batch
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        write_sharded_cache(cache, interrupted, rows=6, metadata={}, shard_rows=2)
    assert not cache.exists()
    def resumed(start):
        starts.append(start)
        for _ in range(start, 6, 2):
            yield batch
    write_sharded_cache(cache, resumed, rows=6, metadata={}, shard_rows=2)
    assert starts == [0, 2]
    loaded = FeatureDataset(cache)
    for index in range(6):
        torch.testing.assert_close(loaded[index]['state'], batch['state'][index % 2])
    assert len(loaded._shards) <= 2
    manifest = json.loads((cache / '000001' / 'manifest.json').read_text())
    filename = next(iter(manifest['fields'].values()))['file']
    with (cache / '000001' / filename).open('r+b') as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    fresh = FeatureDataset(cache)
    with pytest.raises(ValueError, match='checksum'):
        fresh[2]


def test_sharded_resume_rejects_changed_contract(tmp_path):
    from lbm.feature_shards import write_sharded_cache

    def broken(start):
        raise RuntimeError('stop')
    with pytest.raises(RuntimeError):
        write_sharded_cache(tmp_path / 'cache', broken, rows=2, metadata={'source': 'A'})
    with pytest.raises(ValueError, match='changed'):
        write_sharded_cache(tmp_path / 'cache', broken, rows=2, metadata={'source': 'B'})


def test_feature_validation_rejects_shared_episode_identity():
    train, validation = FeatureDataset.__new__(FeatureDataset), FeatureDataset.__new__(FeatureDataset)
    train.metadata = {'episode_ids': ['shared']}
    validation.metadata = {'episode_ids': ['shared', 'other']}
    with pytest.raises(ValueError, match='episodes overlap'):
        train.check_validation(validation)


def test_sharded_cache_source_views_keep_global_ranges(tmp_path):
    from lbm.feature_shards import write_sharded_cache

    config = tiny_dit_config()
    batch = _cache_batch(config, DiTPolicy(config))
    metadata = {'sources': [dict(name='A', start=0, stop=2, weight=2),
                            dict(name='B', start=2, stop=4, weight=1)]}
    write_sharded_cache(tmp_path / 'cache', lambda start: iter([batch, batch]),
                        rows=4, metadata=metadata, shard_rows=2)
    cached = FeatureDataset(tmp_path / 'cache')
    assert [ds.spec.name for ds in cached.datasets] == ['A', 'B']
    assert cached.weights == [2, 1]
    torch.testing.assert_close(cached.datasets[1][0]['state'], batch['state'][0])


def test_feature_cache_multiworker_loader(tmp_path):
    from lbm.training_loader import make_loader

    config = tiny_dit_config()
    batch = _cache_batch(config, DiTPolicy(config))
    write_feature_cache(tmp_path / 'cache', [batch], rows=2, metadata={})
    loader, _ = make_loader(FeatureDataset(tmp_path / 'cache'),
                            config=TrainConfig(batch_size=2, num_workers=2),
                            distributed=False, train=True, collate_fn=torch.utils.data.default_collate)
    loaded = next(iter(loader))
    assert loaded['state'].shape == batch['state'].shape


def test_killed_feature_writer_recovers_completed_shard(tmp_path):
    import select
    import subprocess
    import sys

    from lbm.feature_shards import write_sharded_cache

    script = """
import sys, time, torch
from lbm.feature_shards import write_sharded_cache
batch = dict(state=torch.ones(2, 1), actions=torch.ones(2, 1, 1), task_vec_clip=torch.ones(2, 1),
             action_mask=torch.ones(2, 1, 1, dtype=torch.bool), camera_mask=torch.ones(2, 1, dtype=torch.bool),
             vision_features={'cam': torch.ones(2, 1, 1)})
def batches(start):
    yield batch
    print('completed-shard', flush=True)
    time.sleep(300)
write_sharded_cache(sys.argv[1], batches, rows=4, metadata={}, shard_rows=2)
"""
    cache = tmp_path / 'cache'
    proc = subprocess.Popen([sys.executable, '-c', script, str(cache)], stdout=subprocess.PIPE, text=True)
    try:
        ready, _, _ = select.select([proc.stdout], [], [], 60)
        assert ready and proc.stdout.readline().strip() == 'completed-shard'
        proc.kill()
        proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=15)
    batch = dict(state=torch.ones(2, 1), actions=torch.ones(2, 1, 1), task_vec_clip=torch.ones(2, 1),
                 action_mask=torch.ones(2, 1, 1, dtype=torch.bool), camera_mask=torch.ones(2, 1, dtype=torch.bool),
                 vision_features={'cam': torch.ones(2, 1, 1)})
    def resume(start):
        assert start == 2
        yield batch
    write_sharded_cache(cache, resume, rows=4, metadata={}, shard_rows=2)
    assert len(FeatureDataset(cache)) == 4
    assert not list(cache.glob('.*'))
