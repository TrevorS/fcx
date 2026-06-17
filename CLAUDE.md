# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`fcx` is a CLI repository explorer (a reimplementation of Microsoft's FastContext) that uses read-only Read/Glob/Grep tools to search a codebase and return compact `file:line` citations. It runs against a local MLX-served `FastContext-1.0-4B-RL` model or any OpenAI-compatible API. Single `uv` package; entry point is the `fcx` command (`src/fcx/cli.py:main`).

## Commands

```bash
uv sync                    # install deps
uv run pytest              # run tests
uv run pytest tests/test_loop.py::test_returns_final_answer_when_no_tool_calls  # single test
uv run ruff check .        # lint
uv run ty check            # type-check (note: ty, NOT mypy)
uv tool install .          # install the fcx CLI locally
```

**Before committing, all three must pass** (or just run `make check`): `uv run ruff check .`, `uv run ty check`, `uv run pytest`.

## Conventions

- Python >=3.13. Async throughout (asyncio, async subprocess, `AsyncOpenAI`) — no sync wrappers.
- ruff line-length is **110** (not the default 88). ruff is lint-only; there is no separate formatter.
- **Config-flat philosophy:** no provider auto-detection and no backend `if`-branches. Anything that differs between backends (FastContext vs OpenAI) is an explicit `FCX_*` env var. Keep new code this way.
- **`agent.py` stays transport-free:** the exploration loop must remain unit-testable with no CLI/transport dependency. Test it with a scripted mock LLM (see `tests/test_loop.py`), not the live model server.
- **Prompts are jinja templates** in `src/fcx/prompts/`, never Python string literals. Add a new prompt as a `.jinja` file plus a `build_*_prompt` helper in `prompts/__init__.py`.

## Gotchas

- **ripgrep required:** `rg` must be on PATH (or set `FCX_RG_PATH`) or `fcx explore` fails.
- The MLX model server is a **detached singleton** (flock + pidfile + HTTP health check); first run downloads ~4.3GB and stays resident. `fcx stop-model` kills it.
- The model was RL-trained at `/workspace`, so it emits tool calls against that virtual root; `src/fcx/paths.py` transparently translates to/from the real filesystem (`FCX_VIRTUAL_ROOT`).
