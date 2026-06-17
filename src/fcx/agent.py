"""The exploration loop. Transport-agnostic, so it is unit-testable with a mock LLM."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, final

from .citations import ExploreResult, Usage, extract_final_answer, parse_citations
from .config import Config
from .llm import LLMClient
from .prompts import build_max_turns_prompt, build_query_prompt, build_system_prompt
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
    ) -> None:
        self.llm = llm
        self.toolset = toolset
        self.system_prompt = system_prompt
        self.root = root
        self.virtual_root = virtual_root

    async def explore(self, query: str, max_turns: int, on_turn: OnTurn = None) -> ExploreResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_query_prompt(query)},
        ]
        usage = Usage()

        for turn in range(1, max_turns + 2):
            if turn == max_turns + 1:
                messages.append({"role": "user", "content": build_max_turns_prompt()})

            step = await self.llm.acall(messages, self.toolset.schemas)
            usage.add(step.usage)
            messages.append(step.as_assistant_message())

            if on_turn is not None:
                calls = ", ".join(tc.name for tc in step.tool_calls) or "final answer"
                await on_turn(turn, calls)

            if not step.tool_calls:
                block = extract_final_answer(step.content)
                return ExploreResult(
                    # Fall back to the full message when the block is missing or empty.
                    answer=block if block else (step.content or ""),
                    citations=parse_citations(block, self.root, self.virtual_root),
                    turns=turn,
                    usage=usage,
                    exhausted=False,
                )

            results = await self.toolset.dispatch(step.tool_calls)
            messages.extend(r.to_message() for r in results)

        return ExploreResult(answer="", citations=[], turns=max_turns, usage=usage, exhausted=True)


def make_agent(cfg: Config, llm: LLMClient, root: Path) -> Agent:
    return Agent(
        llm=llm,
        toolset=build_toolset(cfg, root),
        system_prompt=build_system_prompt(root, cfg.virtual_root),
        root=root,
        virtual_root=cfg.virtual_root,
    )
