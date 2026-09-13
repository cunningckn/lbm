import subprocess
import sys
from pathlib import Path

import torch
from context.validation.correctness.serve_paired import PairedPolicy


class RandomPolicy:
    def reset(self):
        pass

    def infer(self, observation):
        return torch.rand(4)

    def metadata(self):
        return {}


def test_trial_noise_does_not_depend_on_previous_rollout_length():
    first = PairedPolicy(RandomPolicy(), seed=17)
    second = PairedPolicy(RandomPolicy(), seed=17)
    initial = first.infer({'reset': True})
    for _ in range(9):
        first.infer({})
    next_trial = first.infer({'reset': True})
    torch.testing.assert_close(second.infer({'reset': True}), initial, rtol=0, atol=0)
    second.infer({})
    torch.testing.assert_close(second.infer({'reset': True}), next_trial, rtol=0, atol=0)


def test_existing_evaluation_preserves_its_config_before_loading_simulator(tmp_path):
    (tmp_path / 'result.json').write_text('{"complete": true}')
    config = tmp_path / 'config' / 'config.yaml'
    config.parent.mkdir()
    config.write_text('existing configuration')
    script = Path(__file__).resolve().parents[2] / 'context/validation/remaining/eval_libero.py'
    result = subprocess.run([sys.executable, str(script), '--repo', str(tmp_path / 'missing-repository'),
                             '--output', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'FileExistsError' in result.stderr
    assert config.read_text() == 'existing configuration'
