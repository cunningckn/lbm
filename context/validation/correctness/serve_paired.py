"""Serialize policy requests and restart diffusion RNG independently for each trial."""
import argparse
import random
import threading

import numpy as np
import torch

from lbm.policy import LBMPolicy
from lbm.serve import serve_policy


class PairedPolicy:
    def __init__(self, policy, seed, offset=0):
        self.policy, self.seed, self.trial = policy, seed, offset
        self.lock = threading.Lock()

    def metadata(self):
        return dict(self.policy.metadata(), policy_seed_base=self.seed, next_trial=self.trial,
                    inference_backend='eager', seed_protocol='base + official initial-state index')

    def _reset(self):
        seed = self.seed + self.trial
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.trial += 1
        self.policy.reset()

    def reset(self):
        with self.lock:
            self._reset()

    def infer(self, observation):
        with self.lock:
            observation = dict(observation)
            if observation.pop('reset', False):
                self._reset()
            return self.policy.infer(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--norm-stats', required=True)
    parser.add_argument('--port', type=int, default=8130)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--init-offset', type=int, default=0)
    args = parser.parse_args()
    policy = LBMPolicy.from_checkpoint(args.checkpoint, robot_type='libero', norm_stats_path=args.norm_stats,
                                       device='cuda', diffusion_steps=10)
    # Use the same eager deployment backend for both trained checkpoints.
    policy.model.configure_conditioning(compile=False)
    serve_policy(PairedPolicy(policy, args.seed, args.init_offset), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
