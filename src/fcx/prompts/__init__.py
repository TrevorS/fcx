"""Prompt construction — every prompt the loop sends is a jinja template in this package.

``system.md.jinja`` mirrors upstream FastContext's ``system.md`` (the RL-trained format, including the
``<final_answer>`` contract); ``query.md.jinja`` and ``max_turns.md.jinja`` carry the per-turn user
messages. Keeping all prompt text in templates — not Python string literals — means wording changes
never touch code. The model is told the workspace is ``virtual_root`` (matching its training prior);
tool paths are translated to the real filesystem underneath (see ``paths``).
"""

import os
import platform
from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined

_env = Environment(
    loader=PackageLoader("fcx", "prompts"),
    undefined=StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,
)


def build_system_prompt(work_dir: Path, virtual_root: str) -> str:
    try:
        listing = "\n".join(sorted(os.listdir(work_dir)))
    except OSError as e:
        listing = f"(could not list workspace: {e})"
    return _env.get_template("system.md.jinja").render(
        os_kind=platform.platform(),
        shell_name=os.environ.get("SHELL", "unknown"),
        work_dir=virtual_root,
        work_dir_ls=listing,
    )


def build_query_prompt(query: str) -> str:
    """The user message that opens an exploration."""
    return _env.get_template("query.md.jinja").render(query=query)


def build_max_turns_prompt() -> str:
    """The nudge appended on the final turn, asking the model to answer now."""
    return _env.get_template("max_turns.md.jinja").render()
