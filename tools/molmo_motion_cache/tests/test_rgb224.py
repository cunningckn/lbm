from __future__ import annotations

import io
import json
import pickle
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from molmo_motion_cache.cli import main
from molmo_motion_cache.rgb224 import (
    Rgb224FrameReader,
    benchmark_rgb224_payload,
    validate_rgb224_payload,
)


def _jpeg(value: int) -> bytes:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[..., 0] = value
    image[32:96, 48:160, 1] = 255 - value
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="JPEG", quality=95, subsampling=0)
    return stream.getvalue()


def _write_payload(root: Path, *, structured: bool = False) -> list[bytes]:
    root.mkdir(parents=True, exist_ok=True)
    payloads = [_jpeg(value) for value in (12, 91, 203)]
    first = payloads[0] + payloads[1]
    (root / "frames-00000.bin").write_bytes(first)
    (root / "frames-00001.bin").write_bytes(payloads[2])
    rows = [(0, 0, len(payloads[0])), (0, len(payloads[0]), len(payloads[1])), (1, 0, len(payloads[2]))]
    if structured:
        index = np.asarray(
            rows,
            dtype=[("shard", "<u4"), ("offset", "<i8"), ("length", "<u4")],
        )
    else:
        index = np.asarray(rows, dtype=np.int64)
    np.save(root / "frames.npy", index)
    return payloads


class Rgb224ReaderTest(unittest.TestCase):
    def test_mmap_and_seek_keep_bytes_and_pillow_rgb_equal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = _write_payload(root)
            with Rgb224FrameReader(root, max_open_shards=1) as reader:
                self.assertEqual(reader.read_jpeg_bytes(0), payloads[0])
                with reader.borrow_jpeg(0) as blob:
                    self.assertIsInstance(blob, memoryview)
                    self.assertEqual(bytes(blob), payloads[0])
                np.testing.assert_array_equal(
                    reader.decode_rgb(2, mode="seek"), reader.decode_rgb(2, mode="mmap")
                )
                batch = reader.decode_many(np.array([2, 0, 2]), mode="mmap")
                self.assertEqual(batch.shape, (3, 224, 224, 3))
                np.testing.assert_array_equal(batch[0], batch[2])
                self.assertEqual(reader.validate()["payload_shards"], 2)

                spawned = pickle.loads(pickle.dumps(reader))
                try:
                    self.assertEqual(spawned.read_jpeg_bytes(1), payloads[1])
                    np.testing.assert_array_equal(
                        spawned.decode_rgb(1, mode="seek"), spawned.decode_rgb(1, mode="mmap")
                    )
                finally:
                    spawned.close()

    def test_active_view_is_released_before_lru_eviction_and_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = _write_payload(root)
            reader = Rgb224FrameReader(root, max_open_shards=1)
            try:
                with reader.borrow_jpeg(0) as first:
                    with reader.borrow_jpeg(2) as second:
                        self.assertEqual(bytes(first), payloads[0])
                        self.assertEqual(bytes(second), payloads[2])
                        self.assertEqual(set(reader.open_shards), {0, 1})
                        with self.assertRaisesRegex(RuntimeError, "release all RGB224 JPEG views"):
                            reader.close()
                    self.assertEqual(bytes(first), payloads[0])
            finally:
                reader.close()

    def test_structured_index_and_validation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = _write_payload(root, structured=True)
            self.assertEqual(validate_rgb224_payload(root)["frames"], 3)
            with Rgb224FrameReader(root) as reader:
                self.assertEqual(reader.read_jpeg_bytes(2), payloads[2])

            invalid = root / "invalid"
            _write_payload(invalid)
            rows = np.load(invalid / "frames.npy")
            rows[1, 2] += 1_000_000
            np.save(invalid / "frames.npy", rows)
            with Rgb224FrameReader(invalid) as reader:
                with self.assertRaises(EOFError):
                    reader.validate()

            malformed = root / "malformed"
            malformed.mkdir()
            np.save(malformed / "frames.npy", np.zeros((1, 2), dtype=np.int64))
            with self.assertRaisesRegex(ValueError, r"must be an \(N, 3\) integer matrix"):
                Rgb224FrameReader(malformed)

    def test_benchmark_compares_identical_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_payload(root)
            result = benchmark_rgb224_payload(
                root,
                samples=3,
                warmup=1,
                max_open_shards=1,
                seed=20260916,
            )
            self.assertTrue(result["checksums_equal"])
            self.assertEqual(
                result["seek_read_pillow"]["checksum"],
                result["mmap_memoryview_pillow"]["checksum"],
            )

    def test_cli_exposes_validation_and_benchmark_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_payload(root)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["verify-rgb224", "--payload-root", str(root)]), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "passed")

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "benchmark-rgb224",
                            "--payload-root",
                            str(root),
                            "--samples",
                            "2",
                            "--warmup",
                            "1",
                            "--max-open-shards",
                            "1",
                        ]
                    ),
                    0,
                )
            self.assertTrue(json.loads(output.getvalue())["checksums_equal"])
