from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from integrations.oel_mcp.approval_grants import OperationConflictError, UnknownOperationOutcome
from integrations.oel_mcp.base_handlers import BaseOELMCPHandlers
from integrations.oel_mcp.recovery import reconciliation_error_data

MCP_PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "oel-mcp", "version": "0.2.0-pre-v2"}


class OELMCPServer:
    """Minimal local stdio MCP adapter retained until the official SDK v2 migration."""

    def __init__(self, handlers: BaseOELMCPHandlers) -> None:
        self.handlers = handlers

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = str(request.get("method", ""))
        request_id = request.get("id")
        if request_id is None:
            return None
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                    "instructions": (
                        "Use OEL as the deterministic physics and evidence authority. "
                        "Pre-v2 tools inspect or validate only."
                    ),
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": tool_definitions(self.handlers)}
            elif method == "tools/call":
                params = dict(request.get("params", {}) or {})
                payload = self.handlers.call(str(params.get("name", "")), dict(params.get("arguments", {}) or {}))
                result = {
                    "content": [{"type": "text", "text": json.dumps(payload, sort_keys=True)}],
                    "structuredContent": payload,
                    "isError": payload.get("status") == "failed",
                }
            else:
                return _error(request_id, -32601, f"Method not found: {method}")
        except (TypeError, ValueError, PermissionError) as exc:
            return _error(request_id, -32602, str(exc))
        except (UnknownOperationOutcome, OperationConflictError) as exc:
            return _error(request_id, -32602, str(exc), data=reconciliation_error_data(exc))
        except Exception:
            return _error(request_id, -32603, "Internal server error without local diagnostic details.")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def serve(self, input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> None:
        for line in input_stream:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("JSON-RPC request must be an object.")
                response = self.dispatch(request)
            except (json.JSONDecodeError, ValueError) as exc:
                response = _error(None, -32700, str(exc))
            if response is not None:
                output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
                output_stream.flush()


def tool_definitions(handlers: BaseOELMCPHandlers) -> list[dict[str, Any]]:
    return [contract.mcp_definition() for contract in handlers.contracts.values()]


def _error(request_id: Any, code: int, message: str, *, data: dict[str, Any] | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}
