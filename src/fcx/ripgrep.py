"""ripgrep resolution and async subprocess execution.

Both Glob and Grep shell out to ``rg``. Upstream FastContext hardcodes ``/usr/bin/rg`` (broken on
macOS) and uses a blocking ``subprocess.run``; we resolve ``rg`` from config/PATH and run it via
``asyncio.create_subprocess_exec`` so the event loop is never blocked.
"""

import asyncio
import shutil


class RipgrepNotFound(RuntimeError):
    pass


def resolve_rg(rg_path: str | None = None) -> str:
    rg = rg_path or shutil.which("rg")
    if not rg:
        raise RipgrepNotFound("ripgrep (`rg`) not found on PATH; set FCX_RG_PATH or `brew install ripgrep`.")
    return rg


async def run_rg(args: list[str], *, cwd: str, timeout: float) -> str:
    """Run ``rg`` with ``args``; return stdout on success, else stderr. Never raises on rg exit code."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    if proc.returncode in (0, 1):  # 1 == no matches, not an error
        return stdout.decode("utf-8", errors="replace")
    return stderr.decode("utf-8", errors="replace")
