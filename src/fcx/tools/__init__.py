"""Read-only exploration tools and the concurrent dispatcher."""

from pathlib import Path

from ..config import Config
from .base import ToolSet
from .glob import GlobTool
from .grep import GrepTool
from .read import ReadTool


def build_toolset(cfg: Config, root: Path) -> ToolSet:
    return ToolSet(
        [
            ReadTool(),
            GlobTool(rg_path=cfg.rg_path, timeout=cfg.tool_timeout),
            GrepTool(rg_path=cfg.rg_path, timeout=cfg.tool_timeout),
        ],
        root=root,
        virtual_root=cfg.virtual_root,
        timeout=cfg.tool_timeout,
    )


__all__ = ["GlobTool", "GrepTool", "ReadTool", "build_toolset"]
