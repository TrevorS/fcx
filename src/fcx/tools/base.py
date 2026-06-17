"""Tool base class, sandbox guard, and a concurrent ToolSet dispatcher."""

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, final

from pydantic import BaseModel

from ..paths import remap_to_real, virtualize_text


class ToolError(Exception):
    """Raised by a tool to return a clean error string to the model."""


def ensure_within(path: str | Path, root: Path) -> Path:
    """Resolve ``path`` and assert it is inside ``root``; raise ToolError otherwise.

    Applied to all three tools (upstream FastContext omits this on Read).
    """
    p = Path(path).resolve()
    root = root.resolve()
    if not (p == root or p.is_relative_to(root)):
        raise ToolError(f"Permission error: `{path}` is outside the workspace `{root}`.")
    return p


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # raw JSON string from the model


class ToolMessage(BaseModel):
    tool_call_id: str
    content: str
    failed: bool = False

    def to_message(self) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def run(self, args: dict[str, Any], *, root: Path) -> str: ...

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@final
class ToolSet:
    def __init__(self, tools: list[Tool], *, root: Path, virtual_root: str, timeout: float) -> None:
        self._tools = {t.name: t for t in tools}
        self.root = root
        self.virtual_root = virtual_root
        self.timeout = timeout

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def _to_virtual(self, text: str) -> str:
        return virtualize_text(text, self.root, self.virtual_root)

    async def _run_one(self, call: ToolCall) -> ToolMessage:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolMessage(tool_call_id=call.id, content=f"Tool `{call.name}` not found.", failed=True)
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            return ToolMessage(
                tool_call_id=call.id, content=f"Tool `{call.name}` arguments are not valid JSON.", failed=True
            )
        # Translate the model's virtual (/workspace) paths to the real filesystem before the tool runs.
        for key in ("path", "directory"):
            if isinstance(args.get(key), str):
                args[key] = str(remap_to_real(args[key], self.root, self.virtual_root))
        try:
            output = await asyncio.wait_for(tool.run(args, root=self.root), timeout=self.timeout)
            return ToolMessage(tool_call_id=call.id, content=self._to_virtual(output) or "(no output)")
        except TimeoutError:
            return ToolMessage(
                tool_call_id=call.id,
                content=f"Tool `{call.name}` timed out after {self.timeout}s.",
                failed=True,
            )
        except ToolError as e:
            return ToolMessage(tool_call_id=call.id, content=self._to_virtual(str(e)), failed=True)
        except Exception as e:  # noqa: BLE001 - surface any tool failure to the model
            return ToolMessage(
                tool_call_id=call.id, content=self._to_virtual(f"{type(e).__name__}: {e}"), failed=True
            )

    async def dispatch(self, calls: list[ToolCall]) -> list[ToolMessage]:
        """Run all of a turn's tool calls concurrently (upstream runs them sequentially)."""
        return list(await asyncio.gather(*(self._run_one(c) for c in calls)))
