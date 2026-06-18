"""Parsing, validation, and merging of the model's ``<final_answer>`` citations."""

import re
from collections import defaultdict
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
    # Validation against the real file (only meaningful for in-root citations).
    valid: bool = True
    file_lines: int | None = None
    # Self-consistency support: how many independent samples cited this region.
    votes: int = 1


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

    def absorb(self, other: "Usage") -> None:
        """Fold another Usage in (used when aggregating self-consistency samples)."""
        self.prompt += other.prompt
        self.completion += other.completion
        self.total += other.total
        self.cached += other.cached


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


def validate_range(real_path: str, start: int, end: int, in_root: bool) -> tuple[bool, int | None]:
    """Check a citation against the real file: it must exist and its lines be in bounds.

    Out-of-root citations are not ours to judge (they may point anywhere), so they pass with an
    unknown line count. A missing file or an out-of-bounds range marks the citation invalid.
    """
    if not in_root:
        return True, None
    try:
        p = Path(real_path)
        if not p.is_file():
            return False, None
        with p.open(encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
    except OSError:
        return True, None  # can't read it (e.g. permissions) — don't claim it's wrong
    return (1 <= start <= end <= n), n


def parse_citations(
    block: str | None, root: Path | None = None, virtual_root: str = "/workspace"
) -> list[Citation]:
    if not block:
        return []
    out: list[Citation] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
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
        valid, file_lines = (True, None)
        if root is not None:
            valid, file_lines = validate_range(path, start, end, in_root)
        out.append(
            Citation(
                path=path,
                start=start,
                end=end,
                reason=reason,
                in_root=in_root,
                valid=valid,
                file_lines=file_lines,
            )
        )
    return out


def merge_citations(runs: list[list[Citation]]) -> list[Citation]:
    """Union citations across independent self-consistency samples.

    For each file, overlapping or adjacent line ranges are merged into one interval, and ``votes``
    counts how many of the ``runs`` cited that region — a recall-boosting, confidence-bearing summary.
    The merged interval is re-validated, since the union may extend past the file.
    """
    buckets: dict[str, list[tuple[int, int, str | None, int, bool, int | None]]] = defaultdict(list)
    for idx, run in enumerate(runs):
        for c in run:
            buckets[c.path].append((c.start, c.end, c.reason, idx, c.in_root, c.file_lines))

    out: list[Citation] = []
    for path, items in buckets.items():
        items.sort(key=lambda t: (t[0], t[1]))
        cur_start, cur_end, cur_reason, _, in_root, file_lines = items[0]
        voters: set[int] = {items[0][3]}
        for start, end, reason, run_idx, _, fl in items[1:]:
            if start <= cur_end + 1:  # overlapping or directly adjacent ranges
                cur_end = max(cur_end, end)
                voters.add(run_idx)
                if cur_reason is None and reason:
                    cur_reason = reason
                if file_lines is None and fl is not None:
                    file_lines = fl
            else:
                out.append(_merged(path, cur_start, cur_end, cur_reason, voters, in_root, file_lines))
                cur_start, cur_end, cur_reason, voters, file_lines = start, end, reason, {run_idx}, fl
        out.append(_merged(path, cur_start, cur_end, cur_reason, voters, in_root, file_lines))

    out.sort(key=lambda c: (-c.votes, c.path, c.start))
    return out


def _merged(
    path: str, start: int, end: int, reason: str | None, voters: set[int], in_root: bool, file_lines: int | None
) -> Citation:
    valid, lines = validate_range(path, start, end, in_root)
    return Citation(
        path=path,
        start=start,
        end=end,
        reason=reason,
        in_root=in_root,
        valid=valid,
        file_lines=lines if lines is not None else file_lines,
        votes=len(voters),
    )
