from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from integrations.oel_mcp.acceptance import run_public_workflow_acceptance
from integrations.oel_mcp.conformance import SDKStdioConformanceClient, run_conformance
from integrations.oel_mcp.public_registry import public_contracts_for_profile
from integrations.oel_mcp.resources import PUBLIC_RESOURCE_URIS
from integrations.oel_mcp.sdk_protocol import MCP_SDK_PROTOCOL_VERSION, MCP_SDK_REVIEWED_VERSION

INSPECTOR_NPM_SPEC = "@modelcontextprotocol/inspector@2.0.0"
INSPECTOR_NPM_INTEGRITY = (
    "sha512-uEoeEG7/+ZbrvccPF3EsgbfjcyJ3bWVJXT4pcZtpmDUhA0zdK4T4Tuj2oUphi2Huwl66LqVdo3Mx2PkS2SUHXA=="
)
CAPABILITY_TOOL_ID = "oel.describe_capabilities.v1"
CLAUDE_CAPABILITY_TOOL_ID = "mcp__oel__oel_describe_capabilities_v1"
PLOT_SELECTION_TOOL_ID = "oel.plot_evidence.v1"
CLAUDE_PLOT_SELECTION_TOOL_ID = "mcp__oel__oel_plot_evidence_v1"
PUBLIC_TOOL_IDS = tuple(contract.tool_id for contract in public_contracts_for_profile("public_local"))
HOST_PROMPT = (
    "Call the OEL MCP capability tool exactly once with an empty object. "
    "Do not use shell, web, files, or any other action tool. Then return only a compact JSON object "
    "with host, tool_id, status, transport, and capability_count copied from the tool result."
)


def plot_selection_prompt(output_dir: Path, *, artifact_id: str) -> str:
    arguments = {
        "output_dir": str(output_dir),
        "recipe_id": "relative_position_ric_2d",
        "style": "oel_light",
        "format": "png",
        "artifact_id": artifact_id,
        "approval": {"approval_id": "host-plot-selection", "scope": "write"},
        "handling": {"marking": "PUBLIC", "release_scope": "public"},
    }
    return (
        "A user asked: Plot the completed run's two-dimensional relative trajectory in RIC and return a professional "
        "OEL figure. Use the connected OEL evidence authority and do not use shell, web, files, or host-native "
        "visualization tools. Make exactly one plotting tool call with these arguments: "
        + json.dumps(arguments, sort_keys=True)
        + ". Then report only the tool_id, status, artifact_id, and visual_qa_status from the result."
    )


def run_sdk_stdio(root: Path, python_executable: Path) -> dict[str, Any]:
    client = SDKStdioConformanceClient(
        command=str(python_executable),
        args=("-m", "integrations.oel_mcp"),
        cwd=root,
        env=_server_env(root),
        mode="auto",
    )
    started = time.monotonic()
    conformance = run_conformance(
        client,
        expected_tool_ids=PUBLIC_TOOL_IDS,
        expected_protocol_versions=(MCP_SDK_PROTOCOL_VERSION,),
        check_ping=False,
    )
    error_handling = _run_sdk_error_fixture(client)
    resources = _run_sdk_resource_fixture(client)
    lifecycle = _run_sdk_lifecycle(root, python_executable)
    if not conformance.passed or not error_handling["passed"] or not resources["passed"] or not lifecycle["passed"]:
        raise RuntimeError(
            "Official SDK stdio conformance failed: "
            f"checks={conformance.checks!r}, error={error_handling!r}, "
            f"resources={resources!r}, lifecycle={lifecycle!r}"
        )
    return {
        "status": "passed",
        "sdk_version": MCP_SDK_REVIEWED_VERSION,
        "protocol_revision": MCP_SDK_PROTOCOL_VERSION,
        "tool_ids": list(PUBLIC_TOOL_IDS),
        "conformance_checks": list(conformance.checks),
        "error_handling": error_handling,
        "resources": resources,
        "lifecycle": lifecycle,
        "duration_ms": _duration_ms(started),
    }


def _run_sdk_resource_fixture(client: SDKStdioConformanceClient) -> dict[str, Any]:
    resources = client.list_resources()
    uris = tuple(str(resource.get("uri", "")) for resource in resources)
    reads: list[dict[str, Any]] = []
    for uri in uris:
        result = client.read_resource(uri)
        contents = list(result.get("contents", []) or [])
        text_content = [row for row in contents if row.get("uri") == uri and isinstance(row.get("text"), str)]
        reads.append({"uri": uri, "text_content_count": len(text_content)})
    return {
        "passed": uris == PUBLIC_RESOURCE_URIS and all(row["text_content_count"] == 1 for row in reads),
        "resource_uris": list(uris),
        "reads": reads,
    }


def _run_sdk_error_fixture(client: SDKStdioConformanceClient) -> dict[str, Any]:
    try:
        from mcp import MCPError
    except ImportError as exc:  # pragma: no cover - optional profile diagnostic
        raise RuntimeError('Install the OEL MCP profile with `pip install ".[mcp]"`.') from exc

    try:
        client.call_tool("oel.inspect_run.v1", {})
    except MCPError as exc:
        passed = exc.code == -32602 and exc.message == "Handling metadata is required for this operation."
        return {
            "passed": passed,
            "code": exc.code,
            "message": exc.message,
            "data": exc.data,
        }
    return {
        "passed": False,
        "code": None,
        "message": "The invalid request unexpectedly completed.",
        "data": None,
    }


def run_inspector(root: Path, python_executable: Path, npx_executable: Path) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="oel-mcp-inspector-") as raw:
        config_path = Path(raw) / "mcp.json"
        config_path.write_text(
            json.dumps(_mcp_json_config(root, python_executable), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        base = [
            str(npx_executable),
            "--yes",
            INSPECTOR_NPM_SPEC,
            "--cli",
            "--config",
            str(config_path),
            "--server",
            "oel",
            "--format",
            "json",
        ]
        listed = _run_json_command([*base, "--method", "tools/list"], cwd=root, timeout=120)
        called = _run_json_command(
            [
                *base,
                "--method",
                "tools/call",
                "--tool-name",
                CAPABILITY_TOOL_ID,
                "--tool-args-json",
                "{}",
            ],
            cwd=root,
            timeout=120,
        )
        listed_resources = _run_json_command([*base, "--method", "resources/list"], cwd=root, timeout=120)
        read_resource = _run_json_command(
            [
                *base,
                "--method",
                "resources/read",
                "--uri",
                PUBLIC_RESOURCE_URIS[0],
            ],
            cwd=root,
            timeout=120,
        )

    tools = list(dict(listed.get("result", {}) or {}).get("tools", []) or [])
    names = tuple(str(tool.get("name", "")) for tool in tools)
    payload = dict(dict(called.get("result", {}) or {}).get("structuredContent", {}) or {})
    _validate_capability_payload(payload)
    resources = list(dict(listed_resources.get("result", {}) or {}).get("resources", []) or [])
    resource_uris = tuple(str(resource.get("uri", "")) for resource in resources)
    contents = list(dict(read_resource.get("result", {}) or {}).get("contents", []) or [])
    if names != PUBLIC_TOOL_IDS:
        raise RuntimeError(f"Inspector discovered unexpected tools: {names!r}")
    if resource_uris != PUBLIC_RESOURCE_URIS or not any(
        content.get("uri") == PUBLIC_RESOURCE_URIS[0] and isinstance(content.get("text"), str) for content in contents
    ):
        raise RuntimeError(f"Inspector discovered unexpected resources: {resource_uris!r}")
    return {
        "status": "passed",
        "package": INSPECTOR_NPM_SPEC,
        "registry_integrity": INSPECTOR_NPM_INTEGRITY,
        "protocol_revision": MCP_SDK_PROTOCOL_VERSION,
        "tool_ids": list(names),
        "resource_uris": list(resource_uris),
        "capability_result": _capability_summary("inspector", payload),
        "duration_ms": _duration_ms(started),
    }


def run_codex(root: Path, python_executable: Path, codex_executable: Path) -> dict[str, Any]:
    started = time.monotonic()
    command = [
        str(codex_executable),
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--json",
        "-C",
        str(root),
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "-c",
        "features.shell_tool=false",
        "-c",
        f"mcp_servers.oel.command={json.dumps(str(python_executable))}",
        "-c",
        'mcp_servers.oel.args=["-m","integrations.oel_mcp"]',
        "-c",
        f"mcp_servers.oel.cwd={json.dumps(str(root))}",
        "-c",
        'mcp_servers.oel.env={OEL_MCP_ADAPTER="sdk"}',
        "-c",
        f"mcp_servers.oel.enabled_tools=[{json.dumps(CAPABILITY_TOOL_ID)}]",
        "-c",
        'mcp_servers.oel.default_tools_approval_mode="approve"',
        "-c",
        "mcp_servers.oel.required=true",
        HOST_PROMPT,
    ]
    completed = _run_command(command, cwd=root, timeout=180)
    payload = _parse_codex_payload(completed.stdout)
    version = _version_line([str(codex_executable), "--version"], cwd=root)
    return {
        "status": "passed",
        "host": "codex",
        "host_version": version,
        "protocol_revision": MCP_SDK_PROTOCOL_VERSION,
        "capability_result": _capability_summary("codex", payload),
        "deviations": [
            "The non-interactive fixture explicitly pre-approves the single trusted read-only OEL tool.",
        ],
        "duration_ms": _duration_ms(started),
    }


def run_claude(
    root: Path,
    python_executable: Path,
    claude_executable: Path,
    *,
    model: str = "haiku",
) -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="oel-mcp-claude-") as raw:
        config_path = Path(raw) / "mcp.json"
        config_path.write_text(
            json.dumps(_mcp_json_config(root, python_executable), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [
            str(claude_executable),
            "-p",
            "--model",
            model,
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config",
            str(config_path),
            "--tools",
            "ToolSearch",
            "--allowedTools",
            CLAUDE_CAPABILITY_TOOL_ID,
            "--permission-mode",
            "dontAsk",
            "--output-format",
            "stream-json",
            "--verbose",
            HOST_PROMPT,
        ]
        completed = _run_command(command, cwd=root, timeout=180)
    payload = _parse_claude_payload(completed.stdout)
    version = _version_line([str(claude_executable), "--version"], cwd=root)
    return {
        "status": "passed",
        "host": "claude",
        "host_version": version,
        "model_alias": model,
        "protocol_revision": MCP_SDK_PROTOCOL_VERSION,
        "capability_result": _capability_summary("claude", payload),
        "deviations": [
            "Claude Code normalizes dotted MCP tool IDs to underscores in its host-local tool name.",
            "Claude Code lazily discovers the OEL tool through ToolSearch before the call.",
            "The isolated fixture limits Claude's built-in inventory to ToolSearch.",
        ],
        "duration_ms": _duration_ms(started),
    }


def run_codex_plot_selection(
    root: Path,
    python_executable: Path,
    codex_executable: Path,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_id = "host_codex_relative_position_ric_2d"
    env = _plot_selection_server_env(root)
    env_expression = "{" + ",".join(f"{key}={json.dumps(value)}" for key, value in env.items()) + "}"
    command = [
        str(codex_executable),
        "exec",
        "--ignore-user-config",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--json",
        "-C",
        str(root),
        "-c",
        'approval_policy="never"',
        "-c",
        'web_search="disabled"',
        "-c",
        "features.shell_tool=false",
        "-c",
        f"mcp_servers.oel.command={json.dumps(str(python_executable))}",
        "-c",
        'mcp_servers.oel.args=["-m","integrations.oel_mcp"]',
        "-c",
        f"mcp_servers.oel.cwd={json.dumps(str(root))}",
        "-c",
        f"mcp_servers.oel.env={env_expression}",
        "-c",
        f"mcp_servers.oel.enabled_tools=[{json.dumps(PLOT_SELECTION_TOOL_ID)}]",
        "-c",
        'mcp_servers.oel.default_tools_approval_mode="approve"',
        "-c",
        "mcp_servers.oel.required=true",
        plot_selection_prompt(output_dir, artifact_id=artifact_id),
    ]
    completed = _run_command(command, cwd=root, timeout=180)
    payload = _parse_codex_plot_selection(completed.stdout)
    return {
        "status": "passed",
        "host": "codex",
        "host_version": _version_line([str(codex_executable), "--version"], cwd=root),
        "protocol_revision": MCP_SDK_PROTOCOL_VERSION,
        "tool_id": payload["tool_id"],
        "recipe_id": payload["result"]["recipe_id"],
        "artifact_id": payload["result"]["artifact"]["artifact_id"],
        "duration_ms": _duration_ms(started),
    }


def run_claude_plot_selection(
    root: Path,
    python_executable: Path,
    claude_executable: Path,
    *,
    output_dir: Path,
    model: str = "haiku",
) -> dict[str, Any]:
    started = time.monotonic()
    artifact_id = "host_claude_relative_position_ric_2d"
    with tempfile.TemporaryDirectory(prefix="oel-mcp-claude-plot-") as raw:
        config_path = Path(raw) / "mcp.json"
        config_path.write_text(
            json.dumps(
                _mcp_json_config(root, python_executable, env=_plot_selection_server_env(root)),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        command = [
            str(claude_executable),
            "-p",
            "--model",
            model,
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config",
            str(config_path),
            "--tools",
            "ToolSearch",
            "--allowedTools",
            CLAUDE_PLOT_SELECTION_TOOL_ID,
            "--permission-mode",
            "dontAsk",
            "--output-format",
            "stream-json",
            "--verbose",
            plot_selection_prompt(output_dir, artifact_id=artifact_id),
        ]
        completed = _run_command(command, cwd=root, timeout=180)
    payload = _parse_claude_plot_selection(completed.stdout)
    return {
        "status": "passed",
        "host": "claude",
        "host_version": _version_line([str(claude_executable), "--version"], cwd=root),
        "model_alias": model,
        "protocol_revision": MCP_SDK_PROTOCOL_VERSION,
        "tool_id": payload["tool_id"],
        "recipe_id": payload["result"]["recipe_id"],
        "artifact_id": payload["result"]["artifact"]["artifact_id"],
        "duration_ms": _duration_ms(started),
    }


def _run_sdk_lifecycle(root: Path, python_executable: Path) -> dict[str, Any]:
    try:
        import anyio
    except ImportError as exc:  # pragma: no cover - optional profile diagnostic
        raise RuntimeError('Install the OEL MCP profile with `pip install ".[mcp]"`.') from exc

    async def exercise() -> dict[str, Any]:
        from mcp import Client, StdioServerParameters, stdio_client

        parameters = StdioServerParameters(
            command=str(python_executable),
            args=["-m", "integrations.oel_mcp"],
            cwd=root,
            env=_server_env(root),
        )
        sessions: list[dict[str, Any]] = []
        for index in range(2):
            async with Client(stdio_client(parameters), mode="auto", cache=None) as client:
                cancelled_request = False
                if index == 0:
                    with anyio.move_on_after(0) as scope:
                        await client.list_tools(cache_mode="reload")
                    cancelled_request = bool(scope.cancel_called)
                tools = await client.list_tools(cache_mode="reload")
                listed_resources = await client.list_resources(cache_mode="reload")
                operator_guide = await client.read_resource("oel://docs/operator-guide/v1", cache_mode="reload")
                sessions.append(
                    {
                        "session": index + 1,
                        "protocol_revision": client.protocol_version,
                        "tool_ids": [tool.name for tool in tools.tools],
                        "resource_uris": [resource.uri for resource in listed_resources.resources],
                        "operator_guide_read": bool(operator_guide.contents),
                        "cancelled_request": cancelled_request,
                    }
                )
        passed = (
            sessions[0]["cancelled_request"]
            and all(tuple(row["tool_ids"]) == PUBLIC_TOOL_IDS for row in sessions)
            and all(tuple(row["resource_uris"]) == PUBLIC_RESOURCE_URIS for row in sessions)
            and all(row["operator_guide_read"] for row in sessions)
            and all(row["protocol_revision"] == MCP_SDK_PROTOCOL_VERSION for row in sessions)
        )
        return {
            "passed": passed,
            "sessions": sessions,
            "clean_shutdowns": 2,
            "ping": "not_supported_by_protocol_revision_2026-07-28",
        }

    return anyio.run(exercise)


def _mcp_json_config(
    root: Path,
    python_executable: Path,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "mcpServers": {
            "oel": {
                "type": "stdio",
                "command": str(python_executable),
                "args": ["-m", "integrations.oel_mcp"],
                "cwd": str(root),
                "env": dict(env or {"OEL_MCP_ADAPTER": "sdk"}),
            }
        }
    }


def _server_env(root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "OEL_MCP_ADAPTER": "sdk",
        "OEL_MCP_READ_ROOTS": str(root),
    }


def _plot_selection_server_env(root: Path) -> dict[str, str]:
    return {
        "OEL_MCP_ADAPTER": "sdk",
        "OEL_MCP_READ_ROOTS": str(root),
        "OEL_MCP_WRITE_ROOTS": str(root / "outputs"),
        "OEL_MCP_WRITE_APPROVAL_IDS": "host-plot-selection",
    }


def _run_json_command(command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
    completed = _run_command(command, cwd=cwd, timeout=timeout)
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected one JSON result from {command[0]!r}: {completed.stdout[:500]!r}") from exc
    if not isinstance(value, dict) or value.get("error"):
        raise RuntimeError(f"Command returned an MCP error: {value!r}")
    return value


def _run_command(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {completed.returncode}: {command[0]!r}; "
            f"stdout={completed.stdout[:1000]!r}; stderr={completed.stderr[:1000]!r}"
        )
    return completed


def _parse_codex_payload(stdout: str) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for row in _json_lines(stdout):
        item = dict(row.get("item", {}) or {})
        if row.get("type") != "item.completed" or item.get("type") != "mcp_tool_call":
            continue
        if item.get("server") == "oel" and item.get("tool") == CAPABILITY_TOOL_ID:
            calls.append(item)
    if len(calls) != 1:
        raise RuntimeError(f"Codex made {len(calls)} matching OEL calls; expected exactly one.")
    call = calls[0]
    if call.get("status") != "completed" or call.get("error"):
        raise RuntimeError(f"Codex OEL call failed: {call!r}")
    result = dict(call.get("result", {}) or {})
    payload = dict(result.get("structured_content", result.get("structuredContent", {})) or {})
    _validate_capability_payload(payload)
    return payload


def _parse_claude_payload(stdout: str) -> dict[str, Any]:
    calls: list[str] = []
    payloads: list[dict[str, Any]] = []
    final_result: dict[str, Any] = {}
    for row in _json_lines(stdout):
        if row.get("type") == "assistant":
            message = dict(row.get("message", {}) or {})
            for content in list(message.get("content", []) or []):
                if content.get("type") == "tool_use" and content.get("name") == CLAUDE_CAPABILITY_TOOL_ID:
                    calls.append(str(content.get("id", "")))
        elif row.get("type") == "user":
            tool_result = row.get("tool_use_result")
            if isinstance(tool_result, dict) and isinstance(tool_result.get("structuredContent"), dict):
                payloads.append(dict(tool_result["structuredContent"]))
        elif row.get("type") == "result":
            final_result = row
    if len(calls) != 1 or len(payloads) != 1:
        raise RuntimeError(f"Claude made {len(calls)} matching OEL calls with {len(payloads)} results; expected one.")
    if final_result.get("permission_denials"):
        raise RuntimeError(f"Claude reported permission denials: {final_result['permission_denials']!r}")
    _validate_capability_payload(payloads[0])
    return payloads[0]


def _validate_plot_selection_payload(payload: dict[str, Any]) -> None:
    if payload.get("tool_id") != PLOT_SELECTION_TOOL_ID or payload.get("status") != "completed":
        raise RuntimeError(f"Host did not complete the OEL plot selection contract: {payload!r}")
    result = dict(payload.get("result", {}) or {})
    artifact = dict(result.get("artifact", {}) or {})
    qa = dict(artifact.get("qa", {}) or {})
    if result.get("recipe_id") != "relative_position_ric_2d":
        raise RuntimeError(f"Host selected the wrong OEL plot recipe: {result!r}")
    if not artifact.get("path_exists") or qa.get("visual_qa_status") != "pending_agent_review":
        raise RuntimeError(f"Host plot result is missing artifact or QA evidence: {artifact!r}")


def _parse_codex_plot_selection(stdout: str) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for row in _json_lines(stdout):
        item = dict(row.get("item", {}) or {})
        if (
            row.get("type") == "item.completed"
            and item.get("type") == "mcp_tool_call"
            and item.get("server") == "oel"
            and item.get("tool") == PLOT_SELECTION_TOOL_ID
        ):
            calls.append(item)
    if len(calls) != 1:
        raise RuntimeError(f"Codex made {len(calls)} OEL plot calls; expected exactly one.")
    result = dict(calls[0].get("result", {}) or {})
    payload = dict(result.get("structured_content", result.get("structuredContent", {})) or {})
    _validate_plot_selection_payload(payload)
    return payload


def _parse_claude_plot_selection(stdout: str) -> dict[str, Any]:
    calls: list[str] = []
    payloads: list[dict[str, Any]] = []
    final_result: dict[str, Any] = {}
    for row in _json_lines(stdout):
        if row.get("type") == "assistant":
            message = dict(row.get("message", {}) or {})
            for content in list(message.get("content", []) or []):
                if content.get("type") == "tool_use" and content.get("name") == CLAUDE_PLOT_SELECTION_TOOL_ID:
                    calls.append(str(content.get("id", "")))
        elif row.get("type") == "user":
            tool_result = row.get("tool_use_result")
            if isinstance(tool_result, dict) and isinstance(tool_result.get("structuredContent"), dict):
                payloads.append(dict(tool_result["structuredContent"]))
        elif row.get("type") == "result":
            final_result = row
    if len(calls) != 1 or len(payloads) != 1:
        raise RuntimeError(f"Claude made {len(calls)} OEL plot calls with {len(payloads)} results; expected one.")
    if final_result.get("permission_denials"):
        raise RuntimeError(f"Claude reported plot permission denials: {final_result['permission_denials']!r}")
    _validate_plot_selection_payload(payloads[0])
    return payloads[0]


def _json_lines(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _validate_capability_payload(payload: dict[str, Any]) -> None:
    result = dict(payload.get("result", {}) or {})
    if payload.get("tool_id") != CAPABILITY_TOOL_ID or payload.get("status") != "completed":
        raise RuntimeError(f"Unexpected OEL capability envelope: {payload!r}")
    if result.get("transport") != "stdio":
        raise RuntimeError(f"Unexpected OEL transport: {result.get('transport')!r}")
    names = tuple(str(row.get("tool_id", "")) for row in list(result.get("capabilities", []) or []))
    if names != PUBLIC_TOOL_IDS:
        raise RuntimeError(f"Unexpected OEL capability registry: {names!r}")


def _capability_summary(host: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload["result"])
    return {
        "host": host,
        "tool_id": payload["tool_id"],
        "status": payload["status"],
        "transport": result["transport"],
        "capability_count": len(result["capabilities"]),
        "effects": dict(payload["effects"]),
        "evidence": dict(payload["evidence"]),
    }


def _version_line(command: list[str], *, cwd: Path) -> str:
    return _run_command(command, cwd=cwd, timeout=30).stdout.strip()


def _duration_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1000)


def _git_commit(root: Path) -> str:
    try:
        repository = _run_command(["git", "rev-parse", "--show-toplevel"], cwd=root, timeout=30)
        if Path(repository.stdout.strip()).resolve() != root.resolve():
            return "unavailable_public_export"
        completed = _run_command(["git", "rev-parse", "HEAD"], cwd=root, timeout=30)
    except (OSError, RuntimeError):
        return "unavailable_public_export"
    return completed.stdout.strip() or "unavailable_public_export"


def _git_source_state(root: Path) -> dict[str, Any]:
    commit = _git_commit(root)
    if commit == "unavailable_public_export":
        return {"commit": commit, "clean": None, "release_evidence_eligible": False}
    try:
        completed = _run_command(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            timeout=30,
        )
    except (OSError, RuntimeError):
        return {"commit": commit, "clean": None, "release_evidence_eligible": False}
    clean = not bool(completed.stdout.strip())
    return {"commit": commit, "clean": clean, "release_evidence_eligible": clean}


def _resolve_executable(explicit: str | None, name: str) -> Path:
    value = explicit or shutil.which(name)
    if not value:
        raise RuntimeError(f"Required executable is not available: {name}")
    # Do not resolve symlinks here. Virtual-environment Python launchers are
    # commonly symlinks to a base interpreter, and resolving one would discard
    # the environment whose dependencies the stdio child needs.
    return Path(value).expanduser().absolute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded OEL MCP SDK, Inspector, and host interoperability checks."
    )
    parser.add_argument("--all", action="store_true", help="Run SDK, Inspector, Codex, and Claude checks.")
    parser.add_argument(
        "--release-gate",
        action="store_true",
        help="Run the offline SDK conformance and complete public workflow acceptance gates.",
    )
    parser.add_argument(
        "--with-hosts",
        action="store_true",
        help="Add Inspector, Codex, and Claude checks to --release-gate; may use external host services.",
    )
    parser.add_argument("--sdk", action="store_true", help="Run the official SDK over a real stdio subprocess.")
    parser.add_argument(
        "--acceptance",
        action="store_true",
        help="Run plan/validate/execute/inspect/query/task/compare/plot/report workflows over stdio.",
    )
    parser.add_argument("--inspector", action="store_true", help="Run pinned MCP Inspector CLI checks.")
    parser.add_argument("--codex", action="store_true", help="Run the read-only Codex host fixture.")
    parser.add_argument("--claude", action="store_true", help="Run the read-only Claude host fixture.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", dest="python_executable")
    parser.add_argument("--npx", dest="npx_executable")
    parser.add_argument("--codex-executable")
    parser.add_argument("--claude-executable")
    parser.add_argument("--claude-model", default="haiku")
    parser.add_argument(
        "--plot-selection-output-dir",
        type=Path,
        help=(
            "Opt in to model-backed Codex/Claude natural-language plot-selection checks against this completed run. "
            "The directory must be under <root>/outputs and contain review/run.sqlite."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acceptance-work-root", type=Path)
    args = parser.parse_args(argv)

    selected = {
        "sdk": bool(args.all or args.sdk or args.release_gate),
        "acceptance": bool(args.acceptance or args.release_gate),
        "inspector": bool(args.all or args.inspector or (args.release_gate and args.with_hosts)),
        "codex": bool(args.all or args.codex or (args.release_gate and args.with_hosts)),
        "claude": bool(args.all or args.claude or (args.release_gate and args.with_hosts)),
    }
    if not any(selected.values()):
        parser.error("select at least one check or use --all")

    root = args.root.resolve()
    if args.plot_selection_output_dir:
        plot_output = args.plot_selection_output_dir.resolve()
        try:
            plot_output.relative_to(root / "outputs")
        except ValueError:
            parser.error("--plot-selection-output-dir must remain under <root>/outputs")
        if not (plot_output / "review" / "run.sqlite").is_file():
            parser.error("--plot-selection-output-dir must contain review/run.sqlite")
    python_executable = _resolve_executable(args.python_executable or sys.executable, "python")
    source_state = _git_source_state(root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "oel_commit": source_state["commit"],
        "oel_source_state": source_state,
        "sdk_version": MCP_SDK_REVIEWED_VERSION,
        "protocol_revision": MCP_SDK_PROTOCOL_VERSION,
        "checks": {},
    }
    if selected["sdk"]:
        report["checks"]["official_sdk_stdio"] = run_sdk_stdio(root, python_executable)
    if selected["acceptance"]:
        if args.acceptance_work_root:
            report["checks"]["public_workflow_acceptance"] = run_public_workflow_acceptance(
                root=root,
                python_executable=python_executable,
                work_root=args.acceptance_work_root,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="oel-mcp-acceptance-") as raw:
                report["checks"]["public_workflow_acceptance"] = run_public_workflow_acceptance(
                    root=root,
                    python_executable=python_executable,
                    work_root=Path(raw),
                )
    if selected["inspector"]:
        report["checks"]["inspector"] = run_inspector(
            root,
            python_executable,
            _resolve_executable(args.npx_executable, "npx"),
        )
    if selected["codex"]:
        codex_executable = _resolve_executable(args.codex_executable, "codex")
        report["checks"]["codex"] = run_codex(
            root,
            python_executable,
            codex_executable,
        )
        if args.plot_selection_output_dir:
            report["checks"]["codex_plot_selection"] = run_codex_plot_selection(
                root,
                python_executable,
                codex_executable,
                output_dir=args.plot_selection_output_dir.resolve(),
            )
    if selected["claude"]:
        claude_executable = _resolve_executable(args.claude_executable, "claude")
        report["checks"]["claude"] = run_claude(
            root,
            python_executable,
            claude_executable,
            model=args.claude_model,
        )
        if args.plot_selection_output_dir:
            report["checks"]["claude_plot_selection"] = run_claude_plot_selection(
                root,
                python_executable,
                claude_executable,
                output_dir=args.plot_selection_output_dir.resolve(),
                model=args.claude_model,
            )

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
