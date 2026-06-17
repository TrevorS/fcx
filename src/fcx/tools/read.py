"""Read tool: line-numbered file contents, sandboxed to root."""

import asyncio
from pathlib import Path
from typing import Any, final, override

from .base import Tool, ToolError, ensure_within

MAX_LINES = 2000
MAX_LINE_LENGTH = 2000

_DESC = (
    "Reads a file from the local filesystem and returns its line-numbered contents. "
    "Optionally specify a line `offset` (1-indexed; negative counts from the end, -1 is the last line) "
    "and a `limit` for large files; otherwise the whole file is read. "
    "Lines are formatted as LINE_NUMBER|LINE_CONTENT. Long lines and very long files are truncated."
)


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.readlines()


@final
class ReadTool(Tool):
    name = "Read"
    description = _DESC
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "The absolute path of the file to read."},
            "offset": {
                "type": "integer",
                "description": "Line to start from (1-indexed). Negative counts from the end (-1 = last line).",
            },
            "limit": {"type": "integer", "description": "Number of lines to read."},
        },
        "required": ["path"],
    }

    @override
    async def run(self, args: dict[str, Any], *, root: Path) -> str:
        raw_path = args.get("path")
        if not raw_path:
            raise ToolError("Read: `path` is required.")
        path = ensure_within(raw_path, root)
        if not path.exists():
            raise ToolError(f"Read: file `{raw_path}` does not exist.")
        if path.is_dir():
            raise ToolError(f"Read: `{raw_path}` is a directory.")

        lines = await asyncio.to_thread(_read_lines, path)
        n = len(lines)
        if n == 0:
            return "File is empty."

        offset = args.get("offset")
        limit = args.get("limit")
        # Resolve 1-indexed start, supporting negative offsets counting from the end.
        if offset is None:
            start = 1
        elif offset < 0:
            start = max(1, n + offset + 1)
        else:
            start = max(1, offset)
        if start > n:
            return f"Offset {offset} is past end of file ({n} lines)."

        end = n if limit is None else min(n, start + limit - 1)
        truncated = end - start + 1 > MAX_LINES
        if truncated:
            end = start + MAX_LINES - 1

        body = []
        for i in range(start - 1, end):
            line = lines[i]
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "...\n"
            body.append(f"{i + 1}|{line}")
        content = "".join(body)
        suffix = "\n..." if truncated else ""
        return f"```{path}:{start}-{end}\n{content}\n```{suffix}"
