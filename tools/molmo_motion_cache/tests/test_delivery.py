import tempfile
import unittest
from pathlib import Path

from molmo_motion_cache.common import checksum_entries, write_checksum_manifest, write_json
from molmo_motion_cache.delivery import audit_component, export_component, export_release
from molmo_motion_cache.release import SOURCE_DATASETS, _build_or_resume, verify_release


class DeliveryTest(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        (root / "payload").write_bytes(b"original")
        write_json(
            root / "READY.json",
            {"status": "ready", "format_version": 1, "sha256sums_sha256": write_checksum_manifest(root)},
        )

    def test_corruption_and_coverage(self):
        for mutation in ("corrupt", "missing", "extra", "no-ready", "version", "duplicate", "escape", "digest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "data"
                self.fixture(root)
                if mutation == "corrupt":
                    (root / "payload").write_bytes(b"modified")
                if mutation == "missing":
                    (root / "payload").unlink()
                if mutation == "extra":
                    (root / "extra").write_bytes(b"extra")
                if mutation == "no-ready":
                    (root / "READY.json").unlink()
                if mutation == "version":
                    write_json(root / "READY.json", {"format_version": 99})
                if mutation in {"duplicate", "escape", "digest"}:
                    original = (root / "SHA256SUMS").read_text()
                    line = (
                        original
                        if mutation == "duplicate"
                        else ("0" * 64 + "  ../outside\n" if mutation == "escape" else "z" * 64 + "  payload\n")
                    )
                    (root / "SHA256SUMS").write_text(original + line)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        checksum_entries(root)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    audit_component(root, verify_files=True)

    def test_export_and_legacy_resume_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "copy"
            self.fixture(source)
            export_component(source, target)
            (source / "payload").write_bytes(b"changed")
            audit_component(target, verify_files=True)
            with self.assertRaises(FileExistsError):
                export_component(target, target)
            with self.assertRaises(ValueError):
                _build_or_resume(target, pilot=False, build=lambda: None, contract={"source": "new"})

    def test_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            self.fixture(root)
            try:
                (root / "linked").symlink_to(root / "payload")
            except OSError:
                self.skipTest("symlink unsupported")
            with self.assertRaises(ValueError):
                audit_component(root)

    def test_resume_binds_parameters_and_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "component"
            contract = {"source": "digest", "schema": 1, "code": "revision", "shard_size": 256}

            def build(destination):
                self.fixture(destination)
                return {"output": str(destination)}

            _build_or_resume(root, pilot=False, build=build, contract=contract)
            self.assertEqual(
                _build_or_resume(root, pilot=False, build=build, contract=contract)["status"], "resumed-existing-ready"
            )
            for key in contract:
                changed = {**contract, key: "different"}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    _build_or_resume(root, pilot=False, build=build, contract=changed)

    def test_release_export_excludes_audit_and_binds_root(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "release", Path(temp) / "export"
            (source / "subsets").mkdir(parents=True)
            for relative in [*(f"subsets/{s}" for s in SOURCE_DATASETS), "assets"]:
                self.fixture(source / relative)
            write_json(source / "READY.json", {"format_version": 1, "status": "ready"})
            (source / "_jobs").mkdir()
            (source / "_jobs" / "log").write_text("not published")
            export_release(source, target)
            self.assertFalse((target / "_jobs").exists())
            self.assertEqual(verify_release(target, verify_files=True)["root_integrity"], "verified")
            (target / "unexpected").write_text("extra")
            with self.assertRaises(ValueError):
                verify_release(target, verify_files=False)
