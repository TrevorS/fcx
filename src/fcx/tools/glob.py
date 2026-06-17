"""Glob tool: file discovery by pattern via `rg --files`, sandboxed to root."""

from pathlib import Path
from typing import Any, final, override

from ..ripgrep import resolve_rg, run_rg
from .base import Tool, ToolError, ensure_within

LIMIT = 100

_DESC = (
    "Fast file pattern matching that works with any codebase size. Supports glob patterns like "
    '"**/*.js" or "src/**/*.ts". Returns matching file paths. Use this to find files by name pattern. '
    "Speculatively batch multiple searches in a single response when useful."
)


@final
class GlobTool(Tool):
    name = "Glob"
    description = _DESC
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "The glob pattern to match files."},
            "directory": {
                "type": "string",
                "description": "Absolute directory to search in. Defaults to the workspace root.",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, rg_path: str | None = None, timeout: float = 15.0) -> None:
        self._rg = resolve_rg(rg_path)
        self._timeout = timeout

    @override
    async def run(self, args: dict[str, Any], *, root: Path) -> str:
        pattern = args.get("pattern")
        if not pattern:
            raise ToolError("Glob: `pattern` is required.")
        directory = ensure_within(args.get("directory") or root, root)
        if not directory.is_dir():
            raise ToolError(f"Glob: `{directory}` is not a directory.")

        out = await run_rg(
            [self._rg, "--files", str(directory), "--glob", pattern],
            cwd=str(root),
            timeout=self._timeout,
        )
        matches = [line for line in out.splitlines() if line]
        if not matches:
            return "No files found."
        if len(matches) > LIMIT:
            matches = matches[:LIMIT]
            matches.append(f"... results truncated to first {LIMIT}; use a more specific path or pattern.")
        return "\n".join(matches)
