import argparse
import hashlib

import numpy as np
import torch
from tests.helpers import tiny_dit_config

from lbm.models.dit import DiTPolicy
from lbm.utils.fake_data import make_fake_batch

p = argparse.ArgumentParser()
p.add_argument("--output", required=True)
a = p.parse_args()
result = {}
for duration in (0.0, 0.3):
    torch.manual_seed(713)
    cfg = tiny_dit_config(history_length=duration, history_freq=10.0)
    model = DiTPolicy(cfg).eval()
    for parameter in model.parameters():
        if parameter.requires_grad:
            torch.nn.init.normal_(parameter, std=0.03)
    batch = make_fake_batch(cfg, 2, image_size=32)
    if duration:
        batch["images"] = {key: value[:, None].expand(-1, 3, -1, -1, -1) for key, value in batch["images"].items()}
    noise = torch.randn_like(batch["actions"])
    t = torch.full((2, 1, 1), 0.3)
    result[f"{duration}_loss"] = model(batch, noise=noise, t=t).detach().numpy()
    result[f"{duration}_actions"] = model.sample_actions(batch, noise=noise, num_steps=3).detach().numpy()
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    result[f"{duration}_weights"] = np.asarray(h.hexdigest())
np.savez(a.output, **result)
