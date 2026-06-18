"""Command-line interface for fcx (FastContext explorer), built on typer.

Each invocation is a short-lived process: it finds-or-starts the shared, resident model server
(see model_server.ensure_model_up), runs one exploration, prints the result, and exits. The model
stays resident across invocations; only this CLI process is ephemeral.
"""

import asyncio
import json
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .agent import Agent, explore_consistent, make_agent
from .citations import ExploreResult
from .config import Config, get_config
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


def _clean_reason(reason: str | None) -> str:
    """Strip the surrounding parentheses the model wraps reasons in."""
    if not reason:
        return ""
    r = reason.strip()
    if r.startswith("(") and r.endswith(")"):
        r = r[1:-1].strip()
    return r


def _line_span(start: int, end: int) -> Text:
    return Text(str(start) if start == end else f"{start}-{end}", style="magenta")


def _print_result(res: ExploreResult, root: Path) -> None:
    """Render the human-facing default view: one table of real-path citations, then a stats line.

    The model speaks in /workspace paths (its training prior); we only ever show the remapped, real
    filesystem paths here. Raw model output is available via --citation; structured data via --json.
    """
    console = Console()
    if res.citations:
        table = Table(
            box=box.SIMPLE_HEAVY,
            padding=(0, 2),
            expand=True,
            header_style="bold",
        )
        table.add_column("File", style="cyan", overflow="fold", ratio=3)
        table.add_column("Lines", justify="right", no_wrap=True)
        table.add_column("Why", overflow="fold", ratio=4)
        for c in res.citations:
            span = _line_span(c.start, c.end)
            if not c.valid:
                span.append("\ninvalid" if c.file_lines is None else f"\n>{c.file_lines}", style="red")
            if not c.in_root:
                span.append("\nexternal", style="yellow")
            if c.votes > 1:
                span.append(f"\n×{c.votes}", style="green")
            table.add_row(_rel(c.path, root), span, _clean_reason(c.reason))
        console.print()
        console.print(table)
    else:
        console.print("\n[dim]No citations found.[/dim]")

    stats = Text("  ", style="dim")
    stats.append(f"{res.turns} turns · {res.usage.total:,} tokens · {res.usage.cached:,} cached", style="dim")
    Console(stderr=True).print(stats)


async def _explore_with_progress(
    agent: Agent, cfg: Config, query: str, max_turns: int, samples: int
) -> ExploreResult:
    """Run an exploration under a transient spinner that narrates the live turn/tool activity."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=Console(stderr=True),
        transient=True,
    )
    with progress:
        task = progress.add_task("starting model server", total=None)
        if cfg.manage_model:
            await ensure_model_up(cfg)
        progress.update(task, description="exploring")

        async def on_turn(n: int, summary: str) -> None:
            progress.update(task, description=f"exploring  ·  turn {n}  ·  {summary}")

        return await explore_consistent(agent, query, max_turns=max_turns, samples=samples, on_turn=on_turn)


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
    agent = make_agent(cfg, LLM(cfg), root)
    n_turns = max_turns or cfg.max_turns
    n_samples = samples or cfg.samples

    if quiet or json_out:  # machine-facing output: no live spinner
        if cfg.manage_model:
            await ensure_model_up(cfg)
        res = await explore_consistent(agent, query, max_turns=n_turns, samples=n_samples)
    else:
        res = await _explore_with_progress(agent, cfg, query, n_turns, n_samples)

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
