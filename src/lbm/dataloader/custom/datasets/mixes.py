"""Named mixes. Each name is a folder under ``datasets/`` and a spec.

``DATA_MIX=all`` concatenates these embodiment dumps (nested LeRobot repos
inside a folder are still discovered by scan). Edit this file to change the mix.
"""

ALL: tuple[str, ...] = (
    "abc",
    "agibot",
    "das_gripper",
    "droid",
    "egoverse",
    "galaxea",
    "hifi_umi",
    "hy_lance",
    "kai0",
    "libero",
    "rmbench",
    "robotwin",
)


def _rows(names: tuple[str, ...]) -> tuple[tuple[str, float, str], ...]:
    return tuple((name, 1.0, name) for name in names)


NAMED_MIXES: dict[str, tuple[tuple[str, float, str], ...]] = {
    "all": _rows(ALL),
    "all_custom": _rows(ALL),
    "all_data": _rows(ALL),
    "custom_all": _rows(ALL),
}
