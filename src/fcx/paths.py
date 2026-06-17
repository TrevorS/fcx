"""Virtual-root path translation.

FastContext was RL-trained on SWE-bench, where every repo is mounted at a fixed container path
(``/workspace``). The model therefore issues tool calls against ``/workspace`` regardless of the real
location. Rather than fight that prior, we present the repo to the model *as* the virtual root and
translate at the boundary: model paths (``/workspace/...``) map to the real filesystem on the way in,
and real paths map back to ``/workspace/...`` on the way out (tool output and citations).
"""

from pathlib import Path


def remap_to_real(path: str, real_root: Path, virtual_root: str) -> Path:
    """Translate a model-supplied path into a real filesystem path (not yet sandbox-validated)."""
    p = path.strip()
    vr = virtual_root.rstrip("/")
    if p == vr or p == vr + "/":
        return real_root
    if p.startswith(vr + "/"):
        return real_root / p[len(vr) + 1 :]
    pp = Path(p)
    if pp.is_absolute():
        return pp  # a real absolute path; ensure_within will judge whether it is in-root
    return real_root / p  # relative or bare name


def virtualize_text(text: str, real_root: Path, virtual_root: str) -> str:
    """Replace every occurrence of the real root prefix in free text (tool output) with the virtual one."""
    return text.replace(str(real_root), virtual_root.rstrip("/"))
