"""Flat, explicit, env-driven configuration.

There is no provider auto-detection or runtime adaptation: every field that differs between
backends is set here. The defaults are the local MLX FastContext values; override via the
``FCX_`` environment prefix (or a ``.env`` file) to point at any OpenAI-compatible API.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EXTRA_BODY: dict[str, Any] = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "fcx"


class Config(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="FCX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- backend identity ---
    base_url: str = "http://localhost:8080/v1"
    model: str = "usermma/FastContext-1.0-4B-RL-mlx-8Bit"
    api_key: SecretStr = SecretStr("EMPTY")
    manage_model: bool = True

    # --- request shape (sent verbatim; None => omitted from the request) ---
    # Defaults follow the base model's official guidance (Qwen3-4B-Instruct-2507:
    # temperature 0.7, top_p 0.8, top_k 20, min_p 0, ~16k output tokens).
    temperature: float | None = 0.7
    top_p: float | None = 0.8
    max_tokens: int | None = 16_384
    token_param: str = "max_completion_tokens"
    extra_body: dict[str, Any] | None = DEFAULT_EXTRA_BODY

    # --- loop / tools ---
    root: str | None = None
    virtual_root: str = "/workspace"  # the path the model expects the repo at (SWE-bench training prior)
    max_turns: int = 8
    tool_timeout: int = 15
    rg_path: str | None = None
    # If the model cites a missing file or out-of-bounds range, spend one extra turn asking it to fix.
    repair_invalid_citations: bool = True
    # Self-consistency: run N independent explorations and merge their citations by agreement.
    samples: int = 1

    # --- local model lifecycle ---
    startup_timeout: int = 600
    lock_path: str | None = None
    model_log: str | None = None
    pid_path: str | None = None

    def resolved_root(self, override: str | None = None) -> Path:
        root = override or self.root or os.getcwd()
        return Path(root).resolve()

    @property
    def lock_file(self) -> Path:
        return Path(self.lock_path) if self.lock_path else _cache_dir() / "model.lock"

    @property
    def log_file(self) -> Path:
        return Path(self.model_log) if self.model_log else _cache_dir() / "model.log"

    @property
    def pid_file(self) -> Path:
        return Path(self.pid_path) if self.pid_path else _cache_dir() / "model.pid"


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
