"""Parquet cache reads must never fall back to unbounded payload reads."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lbm.dataloader.mmap import mmap_io


def test_image_only_parquet_projects_no_payload_columns(tmp_path, monkeypatch):
    path = tmp_path / "images.parquet"
    pq.write_table(pa.table({"image": [b"encoded-image"]}), path)
    original = pd.read_parquet

    def projected_read(path, *, columns):
        assert columns == []
        return original(path, columns=columns)

    monkeypatch.setattr(pd, "read_parquet", projected_read)
    assert list(mmap_io._read_parquet_for_mmap(path).columns) == []


def test_corrupt_schema_does_not_fall_back_to_full_read(tmp_path, monkeypatch):
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"invalid parquet")

    def forbidden(*args, **kwargs):
        pytest.fail("schema failure must not cause a fallback read")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(pa.ArrowInvalid):
        mmap_io._read_parquet_for_mmap(path)


@pytest.mark.parametrize("error", [MemoryError("memory exhausted"), OSError("read failed")])
def test_filtered_read_failure_is_preserved(tmp_path, monkeypatch, error):
    path = tmp_path / "shared.parquet"
    pq.write_table(pa.table({"episode_index": [0, 1], "state": [1.0, 2.0]}), path)

    def failed(*args, **kwargs):
        raise error

    def forbidden(*args, **kwargs):
        pytest.fail("failed episode read must not fall back to reading the whole file")

    monkeypatch.setattr(pq, "read_table", failed)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(type(error)) as caught:
        mmap_io._read_parquet_for_mmap(path, episode_index=1)
    assert caught.value is error


def test_shared_episode_reads_only_requested_rows_and_vectors(tmp_path):
    path = tmp_path / "shared.parquet"
    pq.write_table(pa.table({"episode_index": [0, 1, 1], "state": [1., 2., 3.],
                             "image": [b"a", b"b", b"c"]}), path)
    frame = mmap_io._read_parquet_for_mmap(path, episode_index=1)
    assert frame.to_dict("list") == {"episode_index": [1, 1], "state": [2., 3.]}
    assert frame.index.tolist() == [0, 1]
    with pytest.raises(FileNotFoundError, match="episode 9 not found"):
        mmap_io._read_parquet_for_mmap(path, episode_index=9)
