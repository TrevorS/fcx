"""Command-line interface for fcx (FastContext explorer), built on typer.

Each invocation is a short-lived process: it finds-or-starts the shared, resident model server
(see model_server.ensure_model_up), runs one exploration, prints the result, and exits. The model
stays resident across invocations; only this CLI process is ephemeral.
"""

import asyncio
import json
import sys

import typer

from .agent import make_agent
from .config import get_config
from .llm import LLM
from .model_server import ensure_model_up, model_status, stop_model
from .ripgrep import resolve_rg

app = typer.Typer(add_completion=False, no_args_is_help=True, help="FastContext repository explorer")


async def _run_explore(
    query: str, path: str | None, max_turns: int | None, json_out: bool, citation: bool, quiet: bool
) -> None:
    cfg = get_config()
    resolve_rg(cfg.rg_path)  # fail loud if ripgrep is missing
    root = cfg.resolved_root(path)

    if cfg.manage_model:
        await ensure_model_up(cfg)

    agent = make_agent(cfg, LLM(cfg), root)

    async def on_turn(n: int, summary: str) -> None:
        if not quiet:
            print(f"turn {n}: {summary}", file=sys.stderr)

    res = await agent.explore(query, max_turns=max_turns or cfg.max_turns, on_turn=on_turn)

    if json_out:
        print(res.model_dump_json(indent=2))
    elif citation:
        print(res.answer)
    else:
        print(res.answer)
        if res.citations:
            print()
            for c in res.citations:
                flag = "" if c.in_root else "  [outside workspace]"
                print(f"{c.path}:{c.start}-{c.end}{flag}  {c.reason or ''}".rstrip())
        print(
            f"\n[{res.turns} turns, {res.usage.total} tokens, {res.usage.cached} cached prompt]",
            file=sys.stderr,
        )


@app.command()
def explore(
    query: str = typer.Argument(..., help="natural-language exploration request"),
    path: str | None = typer.Option(None, "--path", "-p", help="repository root to explore (default: cwd)"),
    max_turns: int | None = typer.Option(None, "--max-turns", "-n", help="max exploration turns"),
    json_out: bool = typer.Option(False, "--json", help="emit the full structured result as JSON"),
    citation: bool = typer.Option(False, "--citation", "-c", help="print only the <final_answer> block"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="suppress per-turn progress on stderr"),
) -> None:
    """Explore a repository and return file:line citations."""
    asyncio.run(_run_explore(query, path, max_turns, json_out, citation, quiet))


@app.command()
def status() -> None:
    """Print the backing model server status."""
    print(json.dumps(asyncio.run(model_status(get_config())), indent=2))


@app.command(name="stop-model")
def stop_model_command() -> None:
    """Stop the locally managed resident model server."""
    print("stopped" if stop_model(get_config()) else "nothing to stop")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
