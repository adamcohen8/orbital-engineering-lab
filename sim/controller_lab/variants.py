# ruff: noqa: F401,F403,F405,I001
from .bench_common import *
from .config import *

def _set_controller_pointer(root: dict[str, Any], target: ControllerBenchTarget, pointer: dict[str, Any]) -> None:
    object_id = str(target.object_id)
    slot = str(target.slot)
    objects = root.get("objects")
    if isinstance(objects, dict) and object_id in objects:
        object_section = objects.setdefault(object_id, {})
        object_section[slot] = deepcopy(pointer)
        if object_id in {"rocket", "chaser", "target"} and isinstance(root.get(object_id), dict):
            root[object_id][slot] = deepcopy(pointer)
        return

    section = root.setdefault(object_id, {})
    section[slot] = deepcopy(pointer)


def _apply_variant_overrides(root: dict[str, Any], variant: ControllerVariant) -> None:
    for override in variant.overrides:
        path = str(override.path).strip()
        if not path:
            continue
        paths = object_synced_parameter_paths(root, path)
        if not paths:
            paths = [path]
        for resolved_path in paths:
            set_parameter_path_value(root, resolved_path, deepcopy(override.value))


def _apply_controller_variant(
    root: dict[str, Any],
    suite_target: ControllerBenchTarget,
    variant: ControllerVariant,
) -> None:
    if variant.pointer:
        _set_controller_pointer(root, suite_target, variant.pointer)
    for pointer in variant.pointers:
        target = ControllerBenchTarget(object_id=pointer.object_id, slot=pointer.slot)
        _set_controller_pointer(root, target, pointer.pointer)
    _apply_variant_overrides(root, variant)

__all__ = [name for name in globals() if not name.startswith("__")]
