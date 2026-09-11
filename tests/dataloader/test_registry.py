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
