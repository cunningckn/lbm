import pytest
from benchmarks.train_throughput import parse_args


def test_throughput_args_defaults():
    args = parse_args(["--batch-sizes", "1,4", "--json"])
    assert args.batch_sizes == "1,4"
    assert args.json is True
    assert args.warmup == 3


@pytest.mark.parametrize("argv", [["--iters", "0"], ["--warmup", "-1"], ["--batch-sizes", "0"]])
def test_throughput_args_reject_invalid_values(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
