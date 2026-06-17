from pathlib import Path

from fcx.paths import remap_to_real, virtualize_text

ROOT = Path("/Users/me/repo")
VR = "/workspace"


def test_remap_to_real_virtual_root_itself():
    assert remap_to_real("/workspace", ROOT, VR) == ROOT
    assert remap_to_real("/workspace/", ROOT, VR) == ROOT


def test_remap_to_real_nested():
    assert remap_to_real("/workspace/src/app.py", ROOT, VR) == ROOT / "src/app.py"


def test_remap_to_real_relative():
    assert remap_to_real("src/app.py", ROOT, VR) == ROOT / "src/app.py"


def test_remap_to_real_foreign_absolute_unchanged():
    # a real absolute path that is not under the virtual root is left alone (sandbox will judge it)
    assert remap_to_real("/etc/passwd", ROOT, VR) == Path("/etc/passwd")


def test_virtualize_text_replaces_prefix():
    text = f"{ROOT}/src/a.py:1\n{ROOT}/b.py:2"
    assert virtualize_text(text, ROOT, VR) == "/workspace/src/a.py:1\n/workspace/b.py:2"
