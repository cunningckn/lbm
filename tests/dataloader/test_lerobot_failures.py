import pytest

from lbm.dataloader.custom.common import lerobot


@pytest.mark.parametrize("operation", ["episode", "rows", "camera"])
@pytest.mark.parametrize("error", [MemoryError, OSError, ValueError])
def test_parquet_failure_never_falls_back_to_full_file(monkeypatch, operation, error):
    import pyarrow.parquet as pq

    def fail(*args, **kwargs):
        raise error("read failed")

    def forbid(*args, **kwargs):
        pytest.fail("must not reload the full parquet file")

    monkeypatch.setattr(pq, "read_table", fail)
    monkeypatch.setattr(pq, "ParquetFile", fail)
    monkeypatch.setattr(lerobot.pd, "read_parquet", forbid)
    with pytest.raises(error, match="read failed"):
        if operation == "episode":
            lerobot._read_episode_parquet("unused", 3)
        elif operation == "rows":
            lerobot._parquet_num_rows("unused")
        else:
            lerobot._parquet_cam_column("unused", "front")
