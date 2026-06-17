import json
from pathlib import Path

import pytest

from fcx.tools import GlobTool, GrepTool, ReadTool, build_toolset
from fcx.tools.base import ToolCall, ToolError, ensure_within
from fcx.config import Config


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    port = 8080\n    return port\n")
    (tmp_path / "src" / "util.js").write_text("export const PORT = 3000;\n")
    (tmp_path / "README.md").write_text("# demo\n")
    return tmp_path


def test_ensure_within_blocks_escape(tmp_path: Path):
    with pytest.raises(ToolError):
        ensure_within(tmp_path / ".." / "etc", tmp_path / "sub")


def test_ensure_within_allows_root(tmp_path: Path):
    assert ensure_within(tmp_path, tmp_path) == tmp_path.resolve()


async def test_read_sandboxed(repo: Path):
    tool = ReadTool()
    with pytest.raises(ToolError):
        await tool.run({"path": "/etc/passwd"}, root=repo)


async def test_read_numbers_lines(repo: Path):
    out = await ReadTool().run({"path": str(repo / "src" / "app.py")}, root=repo)
    assert "2|    port = 8080" in out


async def test_read_negative_offset(repo: Path):
    out = await ReadTool().run({"path": str(repo / "src" / "app.py"), "offset": -1}, root=repo)
    assert "3|    return port" in out
    assert "1|def main" not in out


async def test_glob_finds_python(repo: Path):
    out = await GlobTool().run({"pattern": "**/*.py"}, root=repo)
    assert "app.py" in out


async def test_grep_default_files_with_matches(repo: Path):
    out = await GrepTool().run({"pattern": "port"}, root=repo)
    assert "app.py" in out
    assert "8080" not in out  # files mode: no content lines


async def test_grep_content_mode(repo: Path):
    out = await GrepTool().run({"pattern": "8080", "output_mode": "content"}, root=repo)
    assert "8080" in out


async def test_grep_count_mode(repo: Path):
    # count mode must map to --count-matches (the upstream bug); expect a numeric count, no source text
    out = await GrepTool().run({"pattern": "port", "output_mode": "count"}, root=repo)
    assert any(ch.isdigit() for ch in out)
    assert "def main" not in out


async def test_grep_sandboxed(repo: Path):
    with pytest.raises(ToolError):
        await GrepTool().run({"pattern": "x", "path": "/etc"}, root=repo)


async def test_dispatch_runs_concurrently(repo: Path):
    cfg = Config(rg_path=None)
    ts = build_toolset(cfg, repo)
    calls = [
        ToolCall(id="1", name="Glob", arguments=json.dumps({"pattern": "**/*.py"})),
        ToolCall(id="2", name="Grep", arguments=json.dumps({"pattern": "PORT"})),
        ToolCall(id="3", name="Read", arguments=json.dumps({"path": str(repo / "README.md")})),
        ToolCall(id="4", name="Nope", arguments="{}"),
    ]
    results = await ts.dispatch(calls)
    assert [r.tool_call_id for r in results] == ["1", "2", "3", "4"]
    assert results[3].failed is True  # unknown tool


async def test_dispatch_bad_json(repo: Path):
    ts = build_toolset(Config(), repo)
    [res] = await ts.dispatch([ToolCall(id="1", name="Read", arguments="{not json")])
    assert res.failed is True


async def test_dispatch_translates_virtual_paths(repo: Path):
    # The model issues /workspace paths (its training prior); dispatch must map them to the real
    # repo on input and back to /workspace in the output it returns to the model.
    ts = build_toolset(Config(), repo)  # default virtual_root = /workspace
    [res] = await ts.dispatch([ToolCall(id="1", name="Read", arguments='{"path": "/workspace/README.md"}')])
    assert res.failed is False
    assert "/workspace/README.md" in res.content  # output virtualized
    assert str(repo) not in res.content  # real root never leaks to the model


async def test_dispatch_grep_under_virtual_root(repo: Path):
    ts = build_toolset(Config(), repo)
    [res] = await ts.dispatch(
        [ToolCall(id="1", name="Grep", arguments='{"pattern": "port", "path": "/workspace"}')]
    )
    assert res.failed is False
    assert "app.py" in res.content
    assert str(repo) not in res.content
