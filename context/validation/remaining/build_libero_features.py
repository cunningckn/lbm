import sys

import numpy as np

sys.path.insert(0, 'scripts')
import prebuild_features as builder  # noqa: E402
from lbm.dataloader.custom.dataset import CustomMixtureDataset, CustomSingleDataset  # noqa: E402
from lbm.utils.preprocess import compute_norm_stats, save_norm_stats  # noqa: E402

original = builder.load_dataset
partition = sys.argv.pop(1)
task = 'pick up the black bowl between the plate and the ramekin and place it on the plate'

def load(config):
    ds = original(config).datasets[0]
    records = [r for r in ds.records if r.lang.strip() == task]
    if len(records) < 10:
        raise RuntimeError(f'Task has only {len(records)} episodes')
    order = np.random.default_rng(123).permutation(len(records))
    cut = max(1, len(records)//5)
    train_records = [records[i] for i in order[cut:]]
    val_records = [records[i] for i in order[:cut]]
    assert not {r.path for r in train_records} & {r.path for r in val_records}
    common = dict(action_mode=ds.action_mode, action_length=ds.action_length,
                  action_freq=ds.action_freq, root=ds.root, use_mmap=False)
    tr = CustomSingleDataset(ds.spec, records=train_records, **common)
    va = CustomSingleDataset(ds.spec, records=val_records, **common)
    payload = compute_norm_stats(tr, progress=False, scratch_dir='/tmp')
    tr.norm_stats = va.norm_stats = payload['norm_stats']
    save_norm_stats('/tmp/lbm-libero-train-norm.json', payload)
    print('LIBERO_SPLIT', len(train_records), len(val_records), len(tr), len(va), flush=True)
    return CustomMixtureDataset([(tr if partition == 'train' else va, 1.0)])
builder.load_dataset = load
builder.main()
