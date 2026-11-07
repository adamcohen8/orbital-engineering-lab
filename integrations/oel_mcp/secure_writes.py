"""MCP compatibility exports for OEL's shared secure-output writer.

The descriptor implementation lives in :mod:`sim.security.output_writes` so
engine producers can adopt the same logical-output/physical-writer contract
without importing the MCP integration layer.  The MCP names remain stable for
existing handlers and callers.
"""

from sim.security.output_writes import (
    OutputWriter,
    SecureOutputConflict,
    SecureOutputError,
    SecureOutputRoot,
    SecureOutputTransaction,
    SecureOutputUnavailable,
    SecureOutputWriter,
    SecureWriteConflict,
    SecureWriteError,
    SecureWriteRoot,
    SecureWriteUnavailable,
    bind_output_writer,
    copy_output_file,
    current_output_writer,
    mkdir_output,
    open_output,
    read_output_bytes,
    require_output_writer,
    require_secure_write_primitives,
    rewrite_staged_paths,
    secure_write_primitives_available,
    sha256_output,
    staged_output,
    staged_output_file,
    update_output_bytes,
    update_output_text,
    write_output_bytes,
    write_output_text,
)

__all__ = [
    "OutputWriter",
    "SecureOutputConflict",
    "SecureOutputError",
    "SecureOutputRoot",
    "SecureOutputTransaction",
    "SecureOutputUnavailable",
    "SecureOutputWriter",
    "SecureWriteConflict",
    "SecureWriteError",
    "SecureWriteRoot",
    "SecureWriteUnavailable",
    "bind_output_writer",
    "copy_output_file",
    "current_output_writer",
    "mkdir_output",
    "open_output",
    "read_output_bytes",
    "require_secure_write_primitives",
    "require_output_writer",
    "rewrite_staged_paths",
    "sha256_output",
    "secure_write_primitives_available",
    "staged_output",
    "staged_output_file",
    "update_output_bytes",
    "update_output_text",
    "write_output_bytes",
    "write_output_text",
]
