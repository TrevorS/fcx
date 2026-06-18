"""The exploration loop. Transport-agnostic, so it is unit-testable with a mock LLM."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, final

from .citations import Citation, ExploreResult, Usage, extract_final_answer, merge_citations, parse_citations
from .config import Config
from .llm import LLMClient
from .prompts import build_max_turns_prompt, build_query_prompt, build_repair_prompt, build_system_prompt
from .tools import build_toolset
from .tools.base import ToolSet

OnTurn = Callable[[int, str], Awaitable[None]] | None


@final
class Agent:
    def __init__(
        self,
        llm: LLMClient,
        toolset: ToolSet,
        system_prompt: str,
        root: Path,
        virtual_root: str = "/workspace",
        repair: bool = True,
    ) -> None:
        self.llm = llm
        self.toolset = toolset
        self.system_prompt = system_prompt
        self.root = root
        self.virtual_root = virtual_root
        self.repair = repair

    async def explore(self, query: str, max_turns: int, on_turn: OnTurn = None) -> ExploreResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_query_prompt(query)},
        ]
        usage = Usage()
        repaired = False
        turn = 0

        while True:
            turn += 1
            forced_final = turn == max_turns + 1
            if forced_final:
                messages.append({"role": "user", "content": build_max_turns_prompt()})

            step = await self.llm.acall(messages, self.toolset.schemas)
            usage.add(step.usage)
            messages.append(step.as_assistant_message())

            if on_turn is not None:
                calls = ", ".join(tc.name for tc in step.tool_calls) or "final answer"
                await on_turn(turn, calls)

            if step.tool_calls:
                if forced_final:
                    return ExploreResult(answer="", citations=[], turns=max_turns, usage=usage, exhausted=True)
                results = await self.toolset.dispatch(step.tool_calls)
                messages.extend(r.to_message() for r in results)
                continue

            block = extract_final_answer(step.content)
            citations = parse_citations(block, self.root, self.virtual_root)

            # One bounded corrective turn if the model cited missing files or out-of-bounds ranges.
            if self.repair and not repaired and not forced_final:
                invalid = [c for c in citations if not c.valid]
                if invalid:
                    repaired = True
                    messages.append({"role": "user", "content": build_repair_prompt(self._problems(invalid))})
                    continue

            return ExploreResult(
                # Fall back to the full message when the block is missing or empty.
                answer=block if block else (step.content or ""),
                citations=citations,
                turns=turn,
                usage=usage,
                exhausted=False,
            )

    def _problems(self, invalid: list[Citation]) -> list[str]:
        """Describe invalid citations in the model's virtual path space for the repair prompt."""
        vr = self.virtual_root.rstrip("/")
        lines: list[str] = []
        for c in invalid:
            vpath = c.path.replace(str(self.root), vr)
            if c.file_lines is None:
                lines.append(f"- {vpath}:{c.start}-{c.end} — file not found")
            else:
                lines.append(f"- {vpath}:{c.start}-{c.end} — out of range; the file has {c.file_lines} lines")
        return lines


async def explore_consistent(
    agent: Agent, query: str, max_turns: int, samples: int = 1, on_turn: OnTurn = None
) -> ExploreResult:
    """Run ``samples`` independent explorations and merge their citations by agreement (self-consistency).

    ``samples <= 1`` is the plain single-shot path (identical behavior to ``agent.explore``). With more,
    the runs go out concurrently — the resident model server shares its prefix cache across them — and
    citations are unioned with a ``votes`` count so the caller can rank or threshold by agreement.
    """
    if samples <= 1:
        return await agent.explore(query, max_turns, on_turn)

    def sample_on_turn(i: int) -> OnTurn:
        if on_turn is None:
            return None

        async def wrapped(n: int, summary: str) -> None:
            await on_turn(n, f"[sample {i + 1}] {summary}")

        return wrapped

    results = await asyncio.gather(
        *(agent.explore(query, max_turns, sample_on_turn(i)) for i in range(samples))
    )

    usage = Usage()
    for r in results:
        usage.absorb(r.usage)
    merged = merge_citations([r.citations for r in results])
    answer = "\n".join(f"{c.path}:{c.start}-{c.end} {c.reason or ''}".rstrip() for c in merged)
    return ExploreResult(
        answer=answer,
        citations=merged,
        turns=max(r.turns for r in results),
        usage=usage,
        exhausted=all(r.exhausted for r in results),
    )


def make_agent(cfg: Config, llm: LLMClient, root: Path) -> Agent:
    return Agent(
        llm=llm,
        toolset=build_toolset(cfg, root),
        system_prompt=build_system_prompt(root, cfg.virtual_root),
        root=root,
        virtual_root=cfg.virtual_root,
        repair=cfg.repair_invalid_citations,
    )
