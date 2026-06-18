"""Lazy-singleton lifecycle for a local MLX model server.

Each `fcx` invocation is a short-lived process, but the model must be a single shared instance that
outlives any one of them. Singleton-ness is enforced at three layers: an HTTP health check, an advisory
``flock``, and the OS port bind. The spawned server is detached (new session, stdio redirected to a log)
so it survives the process that started it.
"""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.parse import ParseResult, urlparse

import httpx
from filelock import FileLock, Timeout

from .config import Config

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _parsed(cfg: Config) -> ParseResult:
    return urlparse(cfg.base_url)


def is_local(cfg: Config) -> bool:
    return (_parsed(cfg).hostname or "") in _LOCAL_HOSTS


def server_port(cfg: Config) -> int:
    return _parsed(cfg).port or 80


def _serves_model(body: Any, cfg: Config) -> bool:
    """Does an OpenAI ``/models`` payload show our model is actually being served?

    A bare 200 is not enough: an unrelated dev server or proxy squatting on the port (OrbStack, Vite,
    …) happily returns HTML with a 200, which would look healthy and then 404 every real request. We
    require a well-formed models list, and for a managed local server we require the configured model
    to be the one loaded.
    """
    if not isinstance(body, dict):
        return False
    served = {m.get("id") for m in body.get("data", []) if isinstance(m, dict)}
    if cfg.manage_model and is_local(cfg):
        return cfg.model in served
    return bool(served)


async def _healthy(cfg: Config) -> bool:
    url = cfg.base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(url, headers={"Authorization": f"Bearer {cfg.api_key.get_secret_value()}"})
        if r.status_code != 200:
            return False
        body = r.json()
    except (httpx.HTTPError, ValueError):  # ValueError = JSON decode failure (e.g. an HTML page)
        return False
    return _serves_model(body, cfg)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _spawn_detached(cfg: Config) -> None:
    """Spawn mlx_lm.server in its own session with stdio redirected to a log file.

    Runs the Python interpreter directly (no shell). Detached via ``start_new_session`` and never
    awaited, so it outlives the short-lived ``fcx`` process that started it; stdio goes to a log so it
    never writes to this process's streams. The child PID is recorded for ``stop_model``.
    """
    log_path = cfg.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", buffering=1)  # noqa: SIM115 - fd handed to the detached child
    cmd = [sys.executable, "-m", "mlx_lm.server", "--model", cfg.model, "--port", str(server_port(cfg))]
    proc = subprocess.Popen(  # noqa: S603 - args are config-controlled, not user input
        cmd,
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    cfg.pid_file.write_text(str(proc.pid))


async def _wait_healthy(cfg: Config, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    delay = 1.0
    while time.monotonic() < deadline:
        if await _healthy(cfg):
            return
        await asyncio.sleep(delay)
        delay = min(delay * 1.5, 5.0)
    raise TimeoutError(f"model server did not become healthy within {timeout}s (see {cfg.log_file})")


async def ensure_model_up(cfg: Config) -> None:
    """Idempotent, race-safe: return once an endpoint at base_url is serving the model."""
    if await _healthy(cfg):
        return
    if not is_local(cfg):
        raise RuntimeError(
            f"MANAGE_MODEL is on but BASE_URL ({cfg.base_url}) is not local; cannot manage it."
        )

    # thread_local=False: we acquire in a worker thread (to_thread) but release on the event-loop
    # thread, so the lock state must not be thread-local or the release would be a no-op.
    lock = FileLock(str(cfg.lock_file), thread_local=False)
    cfg.lock_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(lock.acquire, timeout=cfg.startup_timeout)
    except Timeout as e:
        raise TimeoutError(f"timed out waiting for model-startup lock {cfg.lock_file}") from e
    try:
        if await _healthy(cfg):  # started while we waited
            return
        port = server_port(cfg)
        if _port_in_use(port):
            raise RuntimeError(
                f"port {port} is in use but not serving {cfg.model} (another process is bound to it); "
                f"set FCX_BASE_URL to a free port and retry."
            )
        _spawn_detached(cfg)
        await _wait_healthy(cfg, cfg.startup_timeout)
    finally:
        lock.release()


async def model_status(cfg: Config) -> dict[str, Any]:
    return {
        "base_url": cfg.base_url,
        "model": cfg.model,
        "managed": cfg.manage_model,
        "local": is_local(cfg),
        "port": server_port(cfg),
        "healthy": await _healthy(cfg),
        "log": str(cfg.log_file),
    }


def stop_model(cfg: Config) -> bool:
    """Best-effort teardown of a server fcx spawned, via the recorded PID (no shell commands)."""
    pid_file = cfg.pid_file
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return False
    pid_file.unlink(missing_ok=True)
    return True
