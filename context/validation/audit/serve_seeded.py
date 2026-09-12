"""Reproducible evaluation server: seed before model setup and record the seed."""
import argparse
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=2026)
args, rest = parser.parse_known_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
print(f'POLICY_SEED {args.seed}', flush=True)
root = Path(__file__).resolve().parents[3]
sys.argv = [str(root / 'scripts/serve_policy.py'), *rest]
runpy.run_path(sys.argv[0], run_name='__main__')
