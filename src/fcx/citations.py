"""Parsing of the model's ``<final_answer>`` block into structured citations."""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .paths import remap_to_real

_BLOCK_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
# /abs/path/file.py:10-15  or  file.py:10  with an optional trailing reason
_CITE_RE = re.compile(r"^(?P<path>[^\s:]+):(?P<start>\d+)(?:-(?P<end>\d+))?\s*(?P<reason>.*)$")


class Citation(BaseModel):
    path: str
    start: int
    end: int
    reason: str | None = None
    in_root: bool = True


class Usage(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0
    cached: int = 0  # prompt tokens served from the server's prefix cache (prefill we skipped)

    def add(self, raw: dict[str, Any] | None) -> None:
        if not raw:
            return
        self.prompt += raw.get("prompt_tokens", 0)
        self.completion += raw.get("completion_tokens", 0)
        self.total += raw.get("total_tokens", 0)
        details = raw.get("prompt_tokens_details") or {}
        self.cached += details.get("cached_tokens", 0) or 0


class ExploreResult(BaseModel):
    answer: str
    citations: list[Citation] = []
    turns: int = 0
    usage: Usage = Usage()
    exhausted: bool = False


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    m = _BLOCK_RE.search(text)
    return m.group(1).strip() if m else None


def parse_citations(
    block: str | None, root: Path | None = None, virtual_root: str = "/workspace"
) -> list[Citation]:
    if not block:
        return []
    out: list[Citation] = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _CITE_RE.match(line)
        if not m:
            continue
        start = int(m.group("start"))
        end = int(m.group("end")) if m.group("end") else start
        reason = (m.group("reason") or "").strip() or None
        raw = m.group("path")
        # The model cites virtual (/workspace) paths; translate back to the real filesystem.
        path = str(remap_to_real(raw, root, virtual_root)) if root is not None else raw
        in_root = True
        if root is not None:
            try:
                in_root = Path(path).resolve().is_relative_to(root.resolve())
            except (OSError, ValueError):
                in_root = False
        out.append(Citation(path=path, start=start, end=end, reason=reason, in_root=in_root))
    return out
