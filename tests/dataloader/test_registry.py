import argparse
from dataclasses import replace
from types import ModuleType

import pytest

from lbm.dataloader.custom import dataset_module, dataset_names, dataset_spec
from lbm.dataloader.custom.datasets import CUSTOM_SPECS, MODULES, dataset_mixes


def test_registry_exposes_one_source_of_dataset_definitions():
    assert dataset_names() == tuple(sorted(MODULES))
    for name in dataset_names():
        assert dataset_module(name).SPEC is dataset_spec(name)
        assert CUSTOM_SPECS[name] is dataset_spec(name)


def test_registry_returns_copy_of_mixes():
    mixes = dataset_mixes()
    mixes["all"].clear()
    assert dataset_mixes()["all"]


def _adapter():
    module = ModuleType("test_adapter")
    module.NAME = "new_dataset"
    module.SPEC = replace(next(iter(CUSTOM_SPECS.values())), name=module.NAME)
    module.scan = module.read_vectors = module.read_frames = lambda *args: None
    return module


def test_new_adapter_enters_all_mixes_and_default_operations(monkeypatch):
    from lbm.dataloader.custom.datasets import _register_module
    from lbm.dataloader.custom.datasets.mixes import named_mixes
    from lbm.train_cli import dumps_from_args

    registry = dict(MODULES)
    adapter = _adapter()
    _register_module(registry, adapter)
    names = tuple(sorted(registry))
    assert (adapter.NAME, 1.0, adapter.NAME) in named_mixes(names)["all"]
    monkeypatch.setattr("lbm.dataloader.custom.datasets.MODULES", registry)
    assert dumps_from_args(argparse.Namespace(dataset="")) == names


@pytest.mark.parametrize("broken", ["name", "spec", "scan", "read_vectors", "read_frames", "duplicate"])
def test_bad_adapter_fails_registration(broken):
    from lbm.dataloader.custom.datasets import _register_module

    adapter = _adapter()
    registry = {}
    if broken == "name":
        adapter.NAME = " "
    elif broken == "spec":
        adapter.SPEC = replace(adapter.SPEC, name="wrong")
    elif broken == "duplicate":
        _register_module(registry, adapter)
    else:
        setattr(adapter, broken, None)
    with pytest.raises(ValueError):
        _register_module(registry, adapter)


def test_invalid_named_recipe_fails_early(monkeypatch):
    from lbm.dataloader.custom.datasets import mixes

    monkeypatch.setattr(mixes, "EXTRA_MIXES", {"broken": (("folder", 1.0, "missing"),)})
    with pytest.raises(ValueError, match="invalid mixture"):
        mixes.named_mixes(dataset_names())
