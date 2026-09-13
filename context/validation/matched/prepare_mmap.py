"""Prepare only the reference cache's training episodes for the mmap comparison."""
import argparse

from benchmarks.matched_throughput import matched_config

from lbm.dataloader.mixture import load_dataset
from lbm.training_features import FeatureDataset
from lbm.training_split import dataset_fingerprint, prepare_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--data-root', required=True)
    args = parser.parse_args()
    cache = FeatureDataset(args.cache)
    cfg = matched_config(cache.metadata, batch=128, workers=2, data_root=args.data_root)
    cfg.data.use_mmap = cfg.data.use_mmap_frames = True
    cache.apply_config(cfg)
    dataset, _ = prepare_validation(load_dataset(cfg), (), cfg)
    if dataset_fingerprint(dataset) != cache.metadata['source_fingerprint']:
        raise ValueError('source episodes or normalization differ from reference')
    dataset.prebuild_mmap_caches(workers=2)


if __name__ == '__main__':
    main()
