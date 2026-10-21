"""Canonical serialization and schema dispatch for flight-software records."""

from __future__ import annotations

import base64
import json
import keyword
import types
from dataclasses import Field, fields, is_dataclass
from enum import Enum
from math import isfinite
from threading import RLock
from typing import Any, Callable, Literal, Mapping, Union, get_args, get_origin, get_type_hints

from .contracts import BOUNDARY_RECORD_TYPES

_SCHEMA_TYPES: dict[str, type[Any]] = {}
_FORBIDDEN_TYPE_NAMES = {
    "sim.core.models.StateTruth",
    "sim.core.models.StateBelief",
}
_FORBIDDEN_MODULE_PREFIXES = (
    "sim.dynamics",
    "sim.runtime",
)
_FORBIDDEN_FIELD_NAMES = {
    "simulator_truth",
    "state_truth",
    "truth_state",
    "world_truth",
}
_TYPE_GUARD_CACHE: dict[type[object], tuple[str, bool, tuple[Field[Any], ...] | None]] = {}
_BOUNDARY_SCALAR_TYPES = (str, bool, int, float, bytes, Enum)
_SPECIALIZED_BOUNDARY_TYPES = frozenset(BOUNDARY_RECORD_TYPES)
_TRUTH_DATACLASS_VALIDATORS: dict[
    type[object], Callable[[object, list[str], set[int]], None]
] = {}
_TRUSTED_DATACLASS_ENCODERS: dict[
    type[object],
    Callable[[object], dict[str, object]],
] = {}
_SCHEMA_CACHE_LOCK = RLock()


def _type_guard_info(value_type: type[object]) -> tuple[str, bool, tuple[Field[Any], ...] | None]:
    """Cache the reflection needed by both boundary traversals."""

    with _SCHEMA_CACHE_LOCK:
        cached = _TYPE_GUARD_CACHE.get(value_type)
    if cached is not None:
        return cached
    qualified = f"{value_type.__module__}.{value_type.__qualname__}"
    forbidden = qualified in _FORBIDDEN_TYPE_NAMES or value_type.__module__.startswith(
        _FORBIDDEN_MODULE_PREFIXES
    )
    dataclass_fields = tuple(fields(value_type)) if is_dataclass(value_type) else None
    result = (qualified, forbidden, dataclass_fields)
    with _SCHEMA_CACHE_LOCK:
        return _TYPE_GUARD_CACHE.setdefault(value_type, result)


def _path_text(path: list[str]) -> str:
    return "".join(path)


def register_record_types(record_types: tuple[type[Any], ...]) -> None:
    """Register dataclass types that carry a unique literal ``schema`` field."""

    for record_type in record_types:
        schema = _literal_default(record_type, "schema")
        if schema is None:
            continue
        with _SCHEMA_CACHE_LOCK:
            previous = _SCHEMA_TYPES.get(schema)
            if previous is not None and previous is not record_type:
                raise ValueError(f"schema {schema!r} is already registered to {previous.__qualname__}")
            _SCHEMA_TYPES[schema] = record_type


def registered_schemas() -> Mapping[str, type[Any]]:
    """Return a read-only copy of the active schema registry."""

    with _SCHEMA_CACHE_LOCK:
        return dict(_SCHEMA_TYPES)


def assert_truth_free(value: object) -> None:
    """Reject known simulator-truth/runtime values at the FSW boundary."""

    _assert_truth_free(value, path=["$"], seen=set())


def to_primitive(value: object) -> object:
    """Convert a boundary value to deterministic JSON-compatible primitives."""

    # Conversion is itself a boundary traversal.  Enforce the truth firewall
    # during that traversal instead of walking the same object graph once to
    # validate it and a second time to serialize it.
    return _to_primitive(value, path=["$"])


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a boundary value to UTF-8 canonical JSON bytes."""

    primitive = to_primitive(value)
    return _canonical_primitive_json_bytes(primitive)


def _canonical_json_bytes_trusted(value: object) -> bytes:
    """Serialize a value already accepted by the runtime boundary adapter."""

    primitive = _to_primitive_trusted(value)
    return _canonical_primitive_json_bytes(primitive)


def _canonical_primitive_json_bytes(primitive: object) -> bytes:
    return json.dumps(
        primitive,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_text(value: object) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def from_primitive(record_type: type[Any], value: object) -> Any:
    """Decode a primitive representation as one declared boundary type."""

    decoded = _decode(record_type, value, path="$", registry=_SCHEMA_TYPES)
    assert_truth_free(decoded)
    return decoded


def canonical_loads(data: bytes | bytearray | str, record_type: type[Any] | None = None) -> Any:
    """Decode canonical JSON, dispatching by schema when no type is supplied."""

    if isinstance(data, (bytes, bytearray)):
        text = bytes(data).decode("utf-8")
    elif isinstance(data, str):
        text = data
    else:
        raise TypeError("canonical JSON input must be bytes, bytearray, or str")
    primitive = json.loads(text, parse_constant=_reject_json_constant)
    if record_type is None:
        if not isinstance(primitive, dict) or not isinstance(primitive.get("schema"), str):
            raise ValueError("schema dispatch requires a top-level schema identifier")
        try:
            record_type = _SCHEMA_TYPES[primitive["schema"]]
        except KeyError as exc:
            raise ValueError(f"unsupported schema {primitive['schema']!r}") from exc
    return from_primitive(record_type, primitive)


def _literal_default(record_type: type[Any], field_name: str) -> str | None:
    if not is_dataclass(record_type):
        return None
    for item in fields(record_type):
        if item.name == field_name and isinstance(item.default, str):
            return item.default
    return None


def _to_primitive(value: object, *, path: list[str]) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("boundary floating-point values must be finite")
        return value
    if isinstance(value, bytes):
        return {"$bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return _to_primitive(value.value, path=path)
    value_type = type(value)
    qualified, forbidden, dataclass_fields = _type_guard_info(value_type)
    if forbidden:
        raise TypeError(f"{_path_text(path)} contains forbidden simulator-owned value {qualified}")
    if dataclass_fields is not None and not isinstance(value, type):
        result: dict[str, object] = {}
        for item in dataclass_fields:
            name = item.name
            path.append(f".{name}")
            result[name] = _to_primitive(getattr(value, name), path=path)
            path.pop()
        return result
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("boundary mapping keys must be strings")
            if key.lower() in _FORBIDDEN_FIELD_NAMES:
                raise TypeError(f"{_path_text(path)}.{key} is a forbidden simulator-truth field")
            path.append(f".{key}")
            result[key] = _to_primitive(item, path=path)
            path.pop()
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for index, item in enumerate(value):
            path.append(f"[{index}]")
            result.append(_to_primitive(item, path=path))
            path.pop()
        return result
    raise TypeError(f"{_path_text(path)} contains unsupported boundary wrapper {qualified}")


def _to_primitive_trusted(value: object) -> object:
    """Convert an already firewalled internal evidence value without rechecking it."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("boundary floating-point values must be finite")
        return value
    if isinstance(value, bytes):
        return {"$bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Enum):
        return _to_primitive_trusted(value.value)
    value_type = type(value)
    qualified, _forbidden, dataclass_fields = _type_guard_info(value_type)
    if dataclass_fields is not None and not isinstance(value, type):
        with _SCHEMA_CACHE_LOCK:
            encoder = _TRUSTED_DATACLASS_ENCODERS.get(value_type)
        if encoder is None:
            compiled = _compile_trusted_dataclass_encoder(value_type, dataclass_fields)
            with _SCHEMA_CACHE_LOCK:
                encoder = _TRUSTED_DATACLASS_ENCODERS.setdefault(value_type, compiled)
        return encoder(value)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("boundary mapping keys must be strings")
            result[key] = _to_primitive_trusted(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_to_primitive_trusted(item) for item in value]
    raise TypeError(f"unsupported boundary value type {qualified}")


def _compile_trusted_dataclass_encoder(
    value_type: type[object],
    dataclass_fields: tuple[Field[Any], ...],
) -> Callable[[object], dict[str, object]]:
    """Compile direct field access for an already validated dataclass type."""

    field_names = tuple(item.name for item in dataclass_fields)
    entries = ", ".join(
        f"{name!r}: _convert(value.{name})" for name in field_names
    )
    namespace: dict[str, object] = {"_convert": _to_primitive_trusted}
    source = f"def encode(value):\n    return {{{entries}}}\n"
    filename = f"<trusted-dataclass-encoder {value_type.__module__}.{value_type.__qualname__}>"
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - field names are Python identifiers
    encoder = namespace["encode"]
    if not callable(encoder):  # pragma: no cover - compile contract guard
        raise RuntimeError("trusted dataclass encoder compilation failed")
    return encoder  # type: ignore[return-value]


def _decode(annotation: object, value: object, *, path: str, registry: Mapping[str, type[Any]]) -> Any:
    if annotation is Any:
        return _decode_untyped(value, path=path, registry=registry)
    if annotation is object:
        return _decode_untyped(value, path=path, registry=registry)
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        if value not in arguments:
            raise ValueError(f"{path} must equal one of {arguments!r}")
        return value
    if origin in (types.UnionType, Union):
        if value is None and type(None) in arguments:
            return None
        if isinstance(value, dict) and isinstance(value.get("schema"), str):
            schema_type = registry.get(value["schema"])
            if schema_type is not None and schema_type in arguments:
                return _decode(schema_type, value, path=path, registry=registry)
        failures: list[str] = []
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return _decode(candidate, value, path=path, registry=registry)
            except (TypeError, ValueError, KeyError) as exc:
                failures.append(str(exc))
        raise ValueError(f"{path} does not match its declared union: {'; '.join(failures)}")
    if origin in (tuple, list):
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a JSON array")
        if origin is tuple and len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode(arguments[0], item, path=f"{path}[{index}]", registry=registry)
                for index, item in enumerate(value)
            )
        if origin is tuple and arguments:
            if len(value) != len(arguments):
                raise ValueError(f"{path} must contain exactly {len(arguments)} values")
            return tuple(
                _decode(item_type, item, path=f"{path}[{index}]", registry=registry)
                for index, (item_type, item) in enumerate(zip(arguments, value))
            )
        item_type = arguments[0] if arguments else object
        result = [
            _decode(item_type, item, path=f"{path}[{index}]", registry=registry) for index, item in enumerate(value)
        ]
        return tuple(result) if origin is tuple else result
    if origin in (dict, Mapping):
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be a JSON object")
        key_type, value_type = arguments or (str, object)
        return {
            _decode(key_type, key, path=f"{path}.<key>", registry=registry): _decode(
                value_type, item, path=f"{path}.{key}", registry=registry
            )
            for key, item in value.items()
        }
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError as exc:
            raise ValueError(f"{path} has unsupported {annotation.__name__} value {value!r}") from exc
    if annotation is bytes:
        if not isinstance(value, dict) or set(value) != {"$bytes_base64"}:
            raise TypeError(f"{path} must be a canonical byte-string object")
        try:
            return base64.b64decode(value["$bytes_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path} contains invalid base64") from exc
    if annotation in (str, int, float, bool):
        if annotation is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise TypeError(f"{path} must be an integer")
        if annotation is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise TypeError(f"{path} must be a number")
        if annotation is bool and not isinstance(value, bool):
            raise TypeError(f"{path} must be boolean")
        if annotation is str and not isinstance(value, str):
            raise TypeError(f"{path} must be a string")
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be a JSON object")
        hints = get_type_hints(annotation)
        field_names = {item.name for item in fields(annotation)}
        unknown = sorted(set(value) - field_names)
        if unknown:
            raise ValueError(f"{path} contains unknown fields: {', '.join(unknown)}")
        keyword: dict[str, object] = {}
        for item in fields(annotation):
            if item.name in value:
                keyword[item.name] = _decode(
                    hints.get(item.name, object),
                    value[item.name],
                    path=f"{path}.{item.name}",
                    registry=registry,
                )
        return annotation(**keyword)
    raise TypeError(f"{path} uses unsupported annotation {annotation!r}")


def _decode_untyped(value: object, *, path: str, registry: Mapping[str, type[Any]]) -> object:
    if isinstance(value, dict):
        schema = value.get("schema")
        if isinstance(schema, str) and schema in registry:
            return _decode(registry[schema], value, path=path, registry=registry)
        if set(value) == {"$bytes_base64"}:
            return _decode(bytes, value, path=path, registry=registry)
        return {str(key): _decode_untyped(item, path=f"{path}.{key}", registry=registry) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(
            _decode_untyped(item, path=f"{path}[{index}]", registry=registry) for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    raise TypeError(f"{path} contains unsupported JSON value")


def _compile_truth_dataclass_validator(
    value_type: type[object],
    dataclass_fields: tuple[Field[Any], ...],
) -> Callable[[object, list[str], set[int]], None]:
    """Unroll field access while recursively inspecting every actual child value."""

    lines = ["def validate(value, path, seen):"]
    for item in dataclass_fields:
        name = item.name
        # ASCII identifiers retain exact getattr semantics. Other field names
        # can occur on dynamically constructed dataclasses; never interpolate
        # them as executable source or normalize their spelling.
        if name.isascii() and name.isidentifier() and not keyword.iskeyword(name):
            access = f"value.{name}"
        else:
            access = f"getattr(value, {name!r})"
        lines.extend((
            f"    child = {access}",
            "    if child is not None and not isinstance(child, _leaf_types):",
            f"        path.append({('.' + name)!r})",
            "        _walk(child, path=path, seen=seen)",
            "        path.pop()",
        ))
    if not dataclass_fields:
        lines.append("    pass")
    namespace: dict[str, object] = {
        "_leaf_types": _BOUNDARY_SCALAR_TYPES,
        "_walk": _assert_truth_free,
    }
    filename = f"<truth-dataclass-validator {value_type.__module__}.{value_type.__qualname__}>"
    exec(compile("\n".join(lines) + "\n", filename, "exec"), namespace)  # noqa: S102 - field access is escaped above
    validator = namespace["validate"]
    if not callable(validator):  # pragma: no cover - compile contract guard
        raise RuntimeError("truth dataclass validator compilation failed")
    return validator  # type: ignore[return-value]


def _assert_truth_free(value: object, *, path: list[str], seen: set[int]) -> None:
    if value is None or isinstance(value, _BOUNDARY_SCALAR_TYPES):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    value_type = type(value)
    qualified, forbidden, dataclass_fields = _type_guard_info(value_type)
    if forbidden:
        raise TypeError(f"{_path_text(path)} contains forbidden simulator-owned value {qualified}")
    if dataclass_fields is not None and not isinstance(value, type):
        if value_type in _SPECIALIZED_BOUNDARY_TYPES:
            with _SCHEMA_CACHE_LOCK:
                validator = _TRUTH_DATACLASS_VALIDATORS.get(value_type)
            if validator is None:
                compiled = _compile_truth_dataclass_validator(value_type, dataclass_fields)
                with _SCHEMA_CACHE_LOCK:
                    validator = _TRUTH_DATACLASS_VALIDATORS.setdefault(value_type, compiled)
            validator(value, path, seen)
        else:
            # External dataclasses retain reflective traversal, including
            # mutable field metadata; annotations never establish trust.
            for item in dataclass_fields:
                child = getattr(value, item.name)
                if child is None or isinstance(child, _BOUNDARY_SCALAR_TYPES):
                    continue
                path.append(f".{item.name}")
                _assert_truth_free(child, path=path, seen=seen)
                path.pop()
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_FIELD_NAMES:
                raise TypeError(f"{_path_text(path)}.{key} is a forbidden simulator-truth field")
            if item is None or isinstance(item, _BOUNDARY_SCALAR_TYPES):
                continue
            path.append(f".{key}")
            _assert_truth_free(item, path=path, seen=seen)
            path.pop()
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            if item is None or isinstance(item, _BOUNDARY_SCALAR_TYPES):
                continue
            path.append(f"[{index}]")
            _assert_truth_free(item, path=path, seen=seen)
            path.pop()
        return
    # Boundary records are deliberately a closed set.  Treating an arbitrary
    # wrapper as opaque would let a caller hide simulator truth in an ordinary
    # attribute and bypass the firewall.
    raise TypeError(
        f"{_path_text(path)} contains unsupported boundary wrapper "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


register_record_types(BOUNDARY_RECORD_TYPES)
