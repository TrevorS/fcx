"""Exploration loop tests with a scripted mock LLM — no model server required."""

from pathlib import Path
from typing import Any

from fcx.agent import Agent, explore_consistent
from fcx.config import Config
from fcx.llm import Step
from fcx.tools import build_toolset
from fcx.tools.base import ToolCall


class MockLLM:
    """Returns a pre-scripted sequence of Steps, recording the messages it was called with."""

    def __init__(self, steps: list[Step]) -> None:
        self._steps = steps
        self.calls: list[list[dict[str, Any]]] = []

    async def acall(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Step:
        self.calls.append(list(messages))
        return self._steps.pop(0)


def _agent(repo: Path, llm) -> Agent:
    cfg = Config()
    return Agent(llm=llm, toolset=build_toolset(cfg, repo), system_prompt="SYS", root=repo)


async def test_returns_final_answer_when_no_tool_calls(tmp_path: Path):
    final = "Found it.\n<final_answer>\n/r/a.py:1-3 (here)\n</final_answer>"
    llm = MockLLM([Step(content=final, tool_calls=[], usage={"total_tokens": 5})])
    res = await _agent(tmp_path, llm).explore("where", max_turns=4)
    assert res.exhausted is False
    assert res.turns == 1
    assert res.citations[0].path == "/r/a.py"
    assert res.usage.total == 5


async def test_runs_tools_then_finishes(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    step1 = Step(
        content="",
        tool_calls=[ToolCall(id="1", name="Glob", arguments='{"pattern": "*.py"}')],
        usage={"total_tokens": 3},
    )
    step2 = Step(content="<final_answer>\n/a.py:1\n</final_answer>", tool_calls=[], usage={"total_tokens": 4})
    llm = MockLLM([step1, step2])
    res = await _agent(tmp_path, llm).explore("q", max_turns=4)
    assert res.turns == 2
    assert res.usage.total == 7
    # second LLM call must include the tool result message
    assert any(m.get("role") == "tool" for m in llm.calls[1])


async def test_exhausts_after_max_turns(tmp_path: Path):
    # always asks for a tool; loop must inject the nudge and then give up
    def tool_step():
        return Step(
            content="", tool_calls=[ToolCall(id="1", name="Glob", arguments='{"pattern": "*"}')], usage=None
        )

    llm = MockLLM([tool_step() for _ in range(10)])
    res = await _agent(tmp_path, llm).explore("q", max_turns=2)
    assert res.exhausted is True
    assert res.turns == 2
    # the max-turns nudge should have been appended as a user message before the final attempt
    assert any(
        m.get("role") == "user" and "Max number of turns" in m.get("content", "")
        for call in llm.calls
        for m in call
    )


async def test_empty_final_block_falls_back_to_prose(tmp_path: Path):
    content = "Here is what I found.\n<final_answer>\n</final_answer>"
    llm = MockLLM([Step(content=content, tool_calls=[], usage=None)])
    res = await _agent(tmp_path, llm).explore("q", max_turns=4)
    assert res.answer == content  # not the empty block
    assert res.citations == []


async def test_progress_callback_fires(tmp_path: Path):
    llm = MockLLM([Step(content="<final_answer>\n/x:1\n</final_answer>", tool_calls=[], usage=None)])
    seen = []

    async def on_turn(n, summary):
        seen.append((n, summary))

    await _agent(tmp_path, llm).explore("q", max_turns=4, on_turn=on_turn)
    assert seen == [(1, "final answer")]


def _agent_with_repair(repo: Path, llm, repair: bool) -> Agent:
    cfg = Config()
    return Agent(
        llm=llm, toolset=build_toolset(cfg, repo), system_prompt="SYS", root=repo, repair=repair
    )


async def test_repairs_invalid_citation(tmp_path: Path):
    (tmp_path / "a.py").write_text("1\n2\n3\n")
    bad = Step(content="<final_answer>\n/workspace/a.py:1-99 (oops)\n</final_answer>", tool_calls=[], usage=None)
    good = Step(content="<final_answer>\n/workspace/a.py:1-2 (fixed)\n</final_answer>", tool_calls=[], usage=None)
    llm = MockLLM([bad, good])
    res = await _agent_with_repair(tmp_path, llm, repair=True).explore("q", max_turns=4)
    assert res.turns == 2  # one extra corrective turn
    assert res.citations[0].valid is True
    assert (res.citations[0].start, res.citations[0].end) == (1, 2)
    # a repair user message must have been injected before the second call
    assert any(
        m.get("role") == "user" and "invalid" in m.get("content", "") for call in llm.calls for m in call
    )


async def test_repair_disabled_returns_invalid_as_is(tmp_path: Path):
    (tmp_path / "a.py").write_text("1\n2\n3\n")
    bad = Step(content="<final_answer>\n/workspace/a.py:1-99 (oops)\n</final_answer>", tool_calls=[], usage=None)
    llm = MockLLM([bad])
    res = await _agent_with_repair(tmp_path, llm, repair=False).explore("q", max_turns=4)
    assert res.turns == 1
    assert res.citations[0].valid is False


async def test_repair_happens_at_most_once(tmp_path: Path):
    (tmp_path / "a.py").write_text("1\n2\n3\n")
    bad = Step(content="<final_answer>\n/workspace/a.py:1-99\n</final_answer>", tool_calls=[], usage=None)
    # both attempts are invalid; repair must fire once then accept the second result
    llm = MockLLM([bad, bad])
    res = await _agent_with_repair(tmp_path, llm, repair=True).explore("q", max_turns=4)
    assert res.turns == 2
    assert res.citations[0].valid is False


async def test_explore_consistent_merges_samples(tmp_path: Path):
    (tmp_path / "a.py").write_text("1\n2\n3\n4\n5\n")

    def final(rng: str):
        return Step(content=f"<final_answer>\n/workspace/a.py:{rng}\n</final_answer>", tool_calls=[], usage=None)

    # three independent samples that overlap on a.py
    llm = MockLLM([final("1-2"), final("1-3"), final("2-3")])
    agent = _agent_with_repair(tmp_path, llm, repair=False)
    res = await explore_consistent(agent, "q", max_turns=4, samples=3)
    assert len(res.citations) == 1
    c = res.citations[0]
    assert (c.start, c.end, c.votes) == (1, 3, 3)
