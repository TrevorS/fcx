"""Grep tool: ripgrep-backed content search, sandboxed to root.

Schema mirrors upstream FastContext (Claude-Code-style) but the `count` output mode is wired to
`--count-matches` correctly (upstream checks for a value that never occurs).
"""

from pathlib import Path
from typing import Any, final, override

from ..ripgrep import resolve_rg, run_rg
from .base import Tool, ToolError, ensure_within

LIMIT = 100

_DESC = (
    "A powerful search tool built on ripgrep. Prefer this over shelling out to grep/rg. Supports full "
    "regex syntax. Filter files with `glob` or `type`. Output modes: `content` (matching lines, supports "
    "-A/-B/-C context and -n), `files_with_matches` (file paths, the default), `count` (match counts). "
    "Use `multiline: true` for patterns spanning lines. Results are capped for responsiveness."
)


@final
class GrepTool(Tool):
    name = "Grep"
    description = _DESC
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "The regular expression to search for."},
            "path": {
                "type": "string",
                "description": "File or directory to search. Defaults to the workspace root.",
            },
            "glob": {"type": "string", "description": 'Glob to filter files (e.g. "*.py", "*.{ts,tsx}").'},
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": 'Output mode. Defaults to "files_with_matches".',
            },
            "-B": {"type": "number", "description": "Lines before each match (content mode)."},
            "-A": {"type": "number", "description": "Lines after each match (content mode)."},
            "-C": {"type": "number", "description": "Lines before and after each match (content mode)."},
            "-n": {"type": "boolean", "description": "Show line numbers (content mode). Default true."},
            "-i": {"type": "boolean", "description": "Case insensitive search."},
            "type": {
                "type": "string",
                "description": "File type to search (rg --type), e.g. py, js, rust, go.",
            },
            "head_limit": {
                "type": "number",
                "minimum": 0,
                "description": "Limit output to first N lines/entries.",
            },
            "multiline": {"type": "boolean", "description": "Patterns may span lines. Default false."},
        },
        "required": ["pattern"],
    }

    def __init__(self, rg_path: str | None = None, timeout: float = 15.0) -> None:
        self._rg = resolve_rg(rg_path)
        self._timeout = timeout

    def _build_args(self, args: dict[str, Any], path: Path) -> list[str]:
        cmd = [self._rg, args["pattern"], str(path)]
        if args.get("glob"):
            cmd += ["--glob", args["glob"]]
        if args.get("-i"):
            cmd.append("--ignore-case")
        if args.get("type"):
            cmd += ["--type", args["type"]]
        if args.get("multiline"):
            cmd += ["--multiline", "--multiline-dotall"]

        mode = args.get("output_mode") or "files_with_matches"
        if mode == "content":
            for flag in ("-A", "-B", "-C"):
                if args.get(flag) is not None:
                    cmd += [flag, str(int(args[flag]))]
            if args.get("-n", True):
                cmd.append("-n")
        elif mode == "files_with_matches":
            cmd.append("--files-with-matches")
        elif mode == "count":
            cmd.append("--count-matches")

        cmd += ["--heading", "--color", "never"]
        return cmd

    @override
    async def run(self, args: dict[str, Any], *, root: Path) -> str:
        if not args.get("pattern"):
            raise ToolError("Grep: `pattern` is required.")
        path = ensure_within(args.get("path") or root, root)

        out = await run_rg(self._build_args(args, path), cwd=str(root), timeout=self._timeout)
        if not out.strip():
            return "No matches found."

        limit = LIMIT
        head = args.get("head_limit")
        if head is not None and 0 < int(head) < LIMIT:
            limit = int(head)

        lines = out.splitlines()
        if len(lines) > limit:
            return "\n".join(lines[:limit]) + f"\nResults truncated to first {limit} lines."
        return "\n".join(lines)
