"""Mixture recipes. All-dataset aliases follow the adapter registry automatically."""

# Add deliberately selected/weighted recipes here. No entry is needed for a
# dataset to join the all-dataset aliases or the preprocessing tools.
EXTRA_MIXES: dict[str, tuple[tuple[str, float, str], ...]] = {}


def named_mixes(names: tuple[str, ...]) -> dict[str, tuple[tuple[str, float, str], ...]]:
    rows = tuple((name, 1.0, name) for name in names)
    out = {alias: rows for alias in ("all", "all_custom", "all_data", "custom_all")}
    for alias, recipe in EXTRA_MIXES.items():
        if alias in out or alias in names:
            raise ValueError(f"mixture name conflicts with an existing dataset or alias: {alias}")
        for folder, weight, spec in recipe:
            if spec not in names or not folder or not 0 < weight < float("inf"):
                raise ValueError(f"invalid mixture row in {alias!r}: {(folder, weight, spec)}")
        out[alias] = recipe
    return out


def __getattr__(name: str):
    # Compatibility for callers that imported ALL / NAMED_MIXES here.
    if name not in {"ALL", "NAMED_MIXES"}:
        raise AttributeError(name)
    from . import NAMED_MIXES, dataset_names

    if name == "ALL":
        return dataset_names()
    if name == "NAMED_MIXES":
        return NAMED_MIXES
    raise AttributeError(name)
