"""Command-line interface for fcx (FastContext explorer), built on typer.

Each invocation is a short-lived process: it finds-or-starts the shared, resident model server
(see model_server.ensure_model_up), runs one exploration, prints the result, and exits. The model
stays resident across invocations; only this CLI process is ephemeral.
"""

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .agent import explore_consistent, make_agent
from .citations import ExploreResult
from .config import get_config
from .llm import LLM
from .model_server import ensure_model_up, model_status, stop_model
from .ripgrep import resolve_rg

app = typer.Typer(add_completion=False, no_args_is_help=True, help="FastContext repository explorer")


def _rel(path: str, root: Path) -> str:
    """Show citations relative to the explored repo root; fall back to the absolute path."""
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return path


def _print_result(res: ExploreResult, root: Path) -> None:
    """Render the human-facing default view: one table of real-path citations, then a stats line.

    The model speaks in /workspace paths (its training prior); we only ever show the remapped, real
    filesystem paths here. Raw model output is available via --citation; structured data via --json.
    """
    console = Console()
    if res.citations:
        table = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("Location", style="cyan", overflow="fold")
        table.add_column("Why", overflow="fold")
        for c in res.citations:
            loc = Text(f"{_rel(c.path, root)}:{c.start}-{c.end}")
            if not c.valid:
                detail = f"{c.file_lines} lines" if c.file_lines is not None else "missing"
                loc.append(f"  [invalid: {detail}]", style="red")
            if not c.in_root:
                loc.append("  [outside workspace]", style="yellow")
            if c.votes > 1:
                loc.append(f"  [votes: {c.votes}]", style="green")
            table.add_row(loc, c.reason or "")
        console.print(table)
    else:
        console.print(res.answer or "[dim]No citations found.[/dim]")
    Console(stderr=True).print(
        f"[dim]{res.turns} turns · {res.usage.total} tokens · {res.usage.cached} cached · {root}[/dim]"
    )


async def _run_explore(
    query: str,
    path: str | None,
    max_turns: int | None,
    samples: int | None,
    json_out: bool,
    citation: bool,
    quiet: bool,
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

    res = await explore_consistent(
        agent,
        query,
        max_turns=max_turns or cfg.max_turns,
        samples=samples or cfg.samples,
        on_turn=on_turn,
    )

    if json_out:
        print(res.model_dump_json(indent=2))
    elif citation:
        print(res.answer)
    else:
        _print_result(res, root)


@app.command()
def explore(
    query: str = typer.Argument(..., help="natural-language exploration request"),
    path: str | None = typer.Option(None, "--path", "-p", help="repository root to explore (default: cwd)"),
    max_turns: int | None = typer.Option(None, "--max-turns", "-n", help="max exploration turns"),
    samples: int | None = typer.Option(
        None, "--samples", "-k", help="run N independent explorations and merge citations by agreement"
    ),
    json_out: bool = typer.Option(False, "--json", help="emit the full structured result as JSON"),
    citation: bool = typer.Option(False, "--citation", "-c", help="print only the <final_answer> block"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="suppress per-turn progress on stderr"),
) -> None:
    """Explore a repository and return file:line citations."""
    asyncio.run(_run_explore(query, path, max_turns, samples, json_out, citation, quiet))


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
