"""Machine-readable authoring schema and dependency-free structural validation."""

from __future__ import annotations

import math


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1}
NUMBER = {"type": "number"}
VECTOR = {"type": "array", "items": NUMBER, "minItems": 3, "maxItems": 3}
MATRIX = {"type": "array", "items": VECTOR, "minItems": 3, "maxItems": 3}
FRAME = obj(
    {"parent": TEXT, "units": {"enum": ["m", "cm", "mm"]}, "translation": VECTOR, "rotation_to_parent": MATRIX},
    ("parent", "units", "translation", "rotation_to_parent"),
)
COMPONENT = obj(
    {
        "id": TEXT,
        "kind": {"enum": ["thruster", "reaction_wheel", "tracking_sensor"]},
        "frame": TEXT,
        "position": VECTOR,
        "units": {"enum": ["m", "cm", "mm"]},
        "axis": VECTOR,
        "axis_convention": {"enum": ["force", "plume", "positive_spin", "boresight"]},
        "parameters": {"type": "object", "additionalProperties": NUMBER},
        "metadata": {"type": "object"},
    },
    ("id", "kind", "frame", "position", "units", "axis", "axis_convention", "parameters"),
)
SOURCE = obj(
    {"id": TEXT, "path": TEXT, "sha256": TEXT, "revision": TEXT, "extraction": TEXT, "status": TEXT},
    ("id", "path", "sha256", "revision"),
)
REFERENCE = obj(
    {"source_id": TEXT, "locator": TEXT, "basis": {"enum": ["extracted", "derived", "assumed"]}, "explanation": TEXT},
    ("source_id", "locator", "basis"),
)
SCHEMA = obj(
    {
        "schema": {"const": "oel.spacecraft_twin.v1"},
        "id": TEXT,
        "version": TEXT,
        "mass_kg": NUMBER,
        "center_of_mass_body_m": VECTOR,
        "inertia_kg_m2": MATRIX,
        "frames": {"type": "object", "additionalProperties": FRAME},
        "components": {"type": "array", "items": COMPONENT, "maxItems": 256},
        "sources": {"type": "array", "items": SOURCE, "maxItems": 500},
        "evidence": {"type": "object", "additionalProperties": {"type": "array", "items": REFERENCE}},
        "assumptions": {
            "type": "array",
            "items": obj(
                {"field": TEXT, "accepted": {"type": "boolean"}, "reason": {"type": "string"}},
                ("field", "accepted", "reason"),
            ),
        },
        "conflicts": {
            "type": "array",
            "items": obj(
                {
                    "field": TEXT,
                    "resolution": {"type": "string"},
                    "reason": {"type": "string"},
                    "alternatives": {"type": "array"},
                },
                ("field", "resolution", "reason"),
            ),
        },
        "geometry": obj(
            {
                "profile_path": TEXT,
                "sha256": TEXT,
                "configuration": TEXT,
                "qualification_path": TEXT,
                "qualification_sha256": TEXT,
                "cd": NUMBER,
                "cr": NUMBER,
            },
            ("profile_path", "sha256", "configuration", "cd", "cr"),
        ),
        "metadata": {"type": "object"},
    },
    (
        "schema",
        "id",
        "version",
        "mass_kg",
        "center_of_mass_body_m",
        "inertia_kg_m2",
        "frames",
        "components",
        "sources",
        "evidence",
        "assumptions",
        "conflicts",
    ),
)
SCHEMA["$schema"] = "https://json-schema.org/draft/2020-12/schema"


def validate(value, schema=SCHEMA, path="twin"):
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: expected {schema['const']}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: expected one of {schema['enum']}")
    kind = schema.get("type")
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }
    if kind and not valid[kind]:
        raise ValueError(f"{path}: expected {kind}")
    if kind == "number" and not math.isfinite(value):
        raise ValueError(f"{path}: expected finite number")
    if kind == "string" and len(value) < schema.get("minLength", 0):
        raise ValueError(f"{path}: empty value")
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10000):
            raise ValueError(f"{path}: invalid item count")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}/{i}")
    if kind == "object":
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}/{key}: required")
        for key, item in value.items():
            child = schema.get("properties", {}).get(key, schema.get("additionalProperties", {}))
            if child is False:
                raise ValueError(f"{path}/{key}: unknown field")
            validate(item, child, f"{path}/{key}")
