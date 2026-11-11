"""Private local-only agent interfaces for spacecraft twin authoring."""

from integrations.oel_mcp.contracts import ToolContract, object_schema
from integrations.oel_mcp.pro_registry_types import PRO_M5_HANDLING_SCHEMA
from integrations.oel_mcp.public_registry import APPROVAL_SCHEMA
from sim.pro_features import FEATURE_CUSTOMER_PACKAGES
from sim.spacecraft_twin.contracts import MAX_TOTAL_BYTES, MAX_TRAVERSAL_DEPTH, MAX_TRAVERSAL_ENTRIES

PATH = {"type": "string", "minLength": 1}
PURPOSE = {"enum": ["passive", "attitude", "environment"], "default": "attitude"}
OPERATIONS = {
    "capabilities": ({}, (), False),
    "intake": ({"source_dir": PATH, "output_dir": PATH}, ("source_dir", "output_dir"), True),
    "propose": ({"model_path": PATH, "output_dir": PATH, "purpose": PURPOSE}, ("model_path", "output_dir"), True),
    "inspect": ({"model_path": PATH, "purpose": PURPOSE}, ("model_path",), False),
    "geometry": (
        {
            "mesh_path": PATH,
            "output_dir": PATH,
            "sample_count": {"type": "integer", "minimum": 12, "maximum": 2048},
            "resolution": {"type": "integer", "minimum": 16, "maximum": 512},
            "scale_to_m": {"type": "number", "exclusiveMinimum": 0},
            "configuration": PATH,
            "tolerance": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.25},
            "resume": {"type": "boolean"},
            "rotation_to_body": {"type": "array"},
            "translation_body_m": {"type": "array"},
        },
        ("mesh_path", "output_dir", "scale_to_m", "configuration"),
        True,
    ),
    "accept": (
        {"model_path": PATH, "output_dir": PATH, "expected_digest": PATH, "reviewer": PATH, "purpose": PURPOSE},
        ("model_path", "output_dir", "expected_digest", "reviewer"),
        True,
    ),
    "verify": ({"receipt_path": PATH, "expected_receipt_sha256": PATH}, ("receipt_path",), False),
    "diff": ({"left_path": PATH, "right_path": PATH}, ("left_path", "right_path"), False),
    "materialize": (
        {"receipt_path": PATH, "template_path": PATH, "output_dir": PATH, "expected_receipt_sha256": PATH},
        ("receipt_path", "template_path", "output_dir", "expected_receipt_sha256"),
        True,
    ),
    "migrate_v0": ({"manifest_path": PATH, "output_dir": PATH}, ("manifest_path", "output_dir"), True),
}
TWIN_TOOL_IDS = tuple(f"oel.pro.twin.{name}.v1" for name in OPERATIONS)
TWIN_TOOL_CONTRACTS = tuple(
    ToolContract(
        tool_id=f"oel.pro.twin.{name}.v1",
        title=f"Spacecraft twin: {name}",
        description=f"{name.replace('_', ' ').capitalize()} a local source-backed spacecraft twin. Does not execute simulations or contact providers.",
        risk_class="R3_sensitive",
        oel_api=f"sim.spacecraft_twin.{name}",
        maturity="supported",
        install_profile="pro",
        deployment_profiles=("pro_local",),
        data_classes=("spacecraft_twin", "local_only_labeled_evidence"),
        writes=writes,
        executes=False,
        input_schema=object_schema(
            {**properties, "handling": PRO_M5_HANDLING_SCHEMA, **({"approval": APPROVAL_SCHEMA} if writes else {})},
            required=(*required, "handling", *(["approval"] if writes else [])),
        ),
        result_schema={"type": "object"},
        required_entitlement=FEATURE_CUSTOMER_PACKAGES,
        limits={
            "local_only": True,
            "external_communication": False,
            "entitlement_required": True,
            "operator_approval_required": writes,
            "max_source_files": 500,
            "max_source_bytes": MAX_TOTAL_BYTES,
            "max_source_traversal_entries": MAX_TRAVERSAL_ENTRIES,
            "max_source_traversal_depth": MAX_TRAVERSAL_DEPTH,
        },
    )
    for name, (properties, required, writes) in OPERATIONS.items()
)
