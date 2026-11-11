"""MCP path-authorized adapters for the private twin service."""

from pathlib import Path

from integrations.oel_mcp.contracts import legacy_local_tool_id
from integrations.oel_mcp.execution import cancellation_callback, secure_staged_output
from sim.spacecraft_twin import services
from sim.spacecraft_twin.contracts import MAX_FILE_BYTES, contained, load

from .secure_writes import bind_output_writer
from .twin_registry import TWIN_TOOL_IDS

_OUTPUT_OPERATIONS = frozenset({"intake", "propose", "geometry", "accept", "materialize", "migrate_v0"})


def call_twin_tool(handler, contract, arguments, *, cancel_event=None, progress=None):
    legacy_id = legacy_local_tool_id(contract.tool_id)
    if legacy_id not in TWIN_TOOL_IDS:
        return None

    def operation():
        check = cancellation_callback(cancel_event)
        check()
        args = {k: v for k, v in arguments.items() if k not in {"handling", "approval"}}
        name = legacy_id.split(".")[3]
        for key in list(args):
            if key == "output_dir":
                args[key] = handler.path_policy.resolve_write(args[key])
            elif key == "source_dir":
                root = handler.path_policy.resolve_read(args[key], kind="directory")
                args[key] = root
            elif key.endswith("_path"):
                args[key] = handler.path_policy.resolve_read(args[key], kind="file")
                if Path(args[key]).stat().st_size > MAX_FILE_BYTES:
                    raise ValueError("Twin input exceeds size limit.")
        if name == "geometry" and args.get("resume"):
            root = handler.path_policy.resolve_read(args["output_dir"], kind="directory")
            for child in root.rglob("*"):
                if child.is_file():
                    handler.path_policy.resolve_read(child, kind="file")
        # References are checked individually, in addition to the service's
        # root containment rules. A manifest grant never expands the read grant.
        if name in {"propose", "inspect", "accept"}:
            path = Path(args["model_path"])
            model = load(path)
            refs = [s["path"] for s in model.get("sources", [])]
            refs += [s["extraction"] for s in model.get("sources", []) if s.get("extraction")]
            geom = model.get("geometry", {})
            refs += [geom[k] for k in ("profile_path", "qualification_path") if geom.get(k)]
            for ref in refs:
                handler.path_policy.resolve_read(contained(path.parent, ref), kind="file")
        if name in {"verify", "materialize"}:
            path = Path(args["receipt_path"])
            for ref in load(path).get("artifacts", {}):
                handler.path_policy.resolve_read(contained(path.parent, ref), kind="file")
        if name == "migrate_v0":
            path = Path(args["manifest_path"])
            # Grant every local file in the bounded migration root. The service
            # rejects v0 external references before reading them.
            for p in path.parent.rglob("*"):
                if p.is_file():
                    handler.path_policy.resolve_read(p, kind="file")
        if name in {"geometry", "intake"}:
            args["cancellation_check"] = check
        if name == "intake":
            args["source_read_check"] = lambda path: handler.path_policy.resolve_read(path, kind="file")
        if name == "geometry":
            args["progress"] = progress
        if name in _OUTPUT_OPERATIONS:
            output = args["output_dir"]
            if name == "geometry" and args.get("resume"):
                with handler.path_policy.open_secure_writer(output, create=False, require_empty=False) as writer:
                    with bind_output_writer(writer):
                        result = getattr(services, name)(**args)
                check()
            else:
                with secure_staged_output(handler.path_policy, output, require_empty=True) as transaction:
                    with bind_output_writer(transaction.staging):
                        result = getattr(services, name)(**args)
                    check()
                    result = transaction.rewrite(result)
        else:
            result = getattr(services, name)(**args)
        check()
        return {**result, "data_label": arguments["handling"]["data_label"], "external_communication": False}

    return handler._envelope(
        contract=contract,
        arguments=arguments,
        operation=operation,
        outcome_status=lambda result: (
            "partial" if result.get("ready") is False or result.get("status") == "failed" else "completed"
        ),
        evidence=lambda result: {"complete": True, "empty": False, "truncated": False},
    )
