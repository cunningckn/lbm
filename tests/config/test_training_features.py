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
@pytest.mark.parametrize('timed', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
def test_cached_features_preserve_loss_gradients_and_sampling(history, encoder, timed, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA GPU required')
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    config = tiny_dit_config(vision_encoder=encoder)
    config.history_time_encoding = timed
    config.state_history_length = .2 if timed else 0.
    model = DiTPolicy(config).to(device=device, dtype=dtype).train()
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=.03)
    live = make_fake_batch(config, 2, device=device, dtype=dtype, image_size=32)
    if history > 1:
        live['images'] = {cam: value[:, None].expand(-1, history, -1, -1, -1)
                          for cam, value in live['images'].items()}
    if timed:
        live['history_mask'] = torch.ones(2, history, dtype=torch.bool, device=device)
        live['history_offsets'] = torch.arange(1-history, 1, device=device).float().expand(2, -1)/10
        if history > 1:
            live['history_mask'][:, 0] = False
    live['camera_mask'] = torch.ones(2, len(config.camera_keys), dtype=torch.bool, device=device)
    live['camera_mask'][:, -1] = False
    cached = {k: v for k, v in live.items() if k != 'images'}
    cached['vision_features'] = model.encode_vision_features(live['images'])
    noise = torch.randn_like(live['actions'])
    t = torch.full((2, 1, 1), .3, device=device, dtype=dtype)
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


def test_old_feature_cache_defaults_remain_compatible_but_reject_requested_history():
    from lbm.training_features import HISTORY_MODEL_FIELDS

    model = tiny_dit_config()
    dataset = FeatureDataset.__new__(FeatureDataset)
    dataset.metadata = {'model': {key: getattr(model, key) for key in MODEL_FIELDS
                                  if key not in HISTORY_MODEL_FIELDS}}
    cfg = TrainConfig(model=model)
    dataset.apply_config(cfg)
    assert not cfg.model.history_time_encoding and cfg.model.state_history_length == 0
    cfg.model.state_history_length = .3
    with pytest.raises(ValueError, match='rebuild cache'):
        dataset.apply_config(cfg)


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


@pytest.mark.parametrize('timed', [False, True])
def test_feature_training_and_resume_use_cached_inputs(tmp_path, monkeypatch, timed):
    from lbm import train_loop

    config = tiny_dit_config()
    config.history_time_encoding = timed
    config.history_length = .2 if timed else 0.
    config.state_history_length = .2 if timed else 0.
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


@pytest.mark.parametrize('workers', [0, 2])
def test_sharded_bulk_read_preserves_order_duplicates_and_loader_seed(tmp_path, workers, monkeypatch):
    from lbm.feature_shards import write_sharded_cache
    from lbm.training_sampler import SeededDataset

    cfg = tiny_dit_config()
    batch = _cache_batch(cfg, DiTPolicy(cfg))
    dest = tmp_path / 'shards'
    batches = [dict(batch, state=batch['state'] + i * 100) for i in range(3)]
    write_sharded_cache(dest, lambda start: iter(batches), rows=6, metadata={}, shard_rows=2)
    dataset = FeatureDataset(dest)
    indices = [0, 2, 4, 1, 3, 5, -1, 0]
    expected = torch.utils.data.default_collate([dataset[i] for i in indices])
    opened = []
    original_open = FeatureDataset._open
    def record_open(self):
        opened.append(self.root)
        return original_open(self)
    with monkeypatch.context() as patch:
        patch.setattr(FeatureDataset, '_open', record_open)
        actual = torch.utils.data.default_collate(dataset.__getitems__(indices))
    assert len(opened) <= 3
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    loader = torch.utils.data.DataLoader(SeededDataset(dataset, 123), num_workers=workers,
                                        batch_sampler=[[(0, p, i) for p, i in enumerate(indices)]])
    torch.testing.assert_close(next(iter(loader)), expected, rtol=0, atol=0)
    assert dataset.__getitems__([]) == []
    with pytest.raises(IndexError):
        dataset.__getitems__([len(dataset)])


def test_validation_cache_rejects_different_encoder_fingerprint():
    train = FeatureDataset.__new__(FeatureDataset)
    val = FeatureDataset.__new__(FeatureDataset)
    train.metadata = dict(episode_ids=['train'], backbone_sha256='one')
    val.metadata = dict(episode_ids=['val'], backbone_sha256='two')
    with pytest.raises(ValueError, match='backbone_sha256'):
        train.check_validation(val)


def test_feature_cache_rejects_stale_episode_instruction_mode():
    cache = FeatureDataset.__new__(FeatureDataset)
    cache.metadata = {}
    config = TrainConfig()
    config.data.instruction_mode = 'subtask'
    with pytest.raises(ValueError, match='instruction mode'):
        cache.apply_config(config)
    other = FeatureDataset.__new__(FeatureDataset)
    other.metadata = {'instruction_mode': 'subtask', 'episode_ids': ['heldout']}
    cache.metadata['episode_ids'] = ['train']
    with pytest.raises(ValueError, match='instruction modes'):
        cache.check_validation(other)


def test_old_agibot_feature_cache_requires_rebuild_after_instruction_fix():
    cache = FeatureDataset.__new__(FeatureDataset)
    cache.metadata = {'sources': [{'name': 'agibot'}]}
    with pytest.raises(ValueError, match='predates adapter correction'):
        cache.apply_config(TrainConfig())
    fresh = FeatureDataset.__new__(FeatureDataset)
    fresh.metadata = {'sources': [{'name': 'agibot'}], 'source_revisions': {'agibot': 2}}
    with pytest.raises(ValueError, match='predates adapter correction'):
        fresh.check_validation(cache)
