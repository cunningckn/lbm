import copy

import numpy as np
import torch
from context.validation.correctness.compare_checkpoints import compare_tree


def test_checkpoint_comparison_distinguishes_signed_zero_and_replay_changes():
    checkpoint = dict(model=torch.tensor([float('nan'), 0.], dtype=torch.bfloat16),
                      optimizer={'step': torch.tensor(4.)}, replay=(np.array([1, 2], dtype=np.uint32), 4))
    other = copy.deepcopy(checkpoint)
    assert compare_tree(checkpoint, other) == dict(tensors_compared=2, bitwise_equal=True, differences=[])
    other['model'][1] = -0.
    other['replay'] = (other['replay'][0], 5)
    result = compare_tree(checkpoint, other)
    assert not result['bitwise_equal']
    assert result['differences'] == ['checkpoint.model', 'checkpoint.replay.1']
