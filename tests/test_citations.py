from pathlib import Path

from fcx.citations import Usage, extract_final_answer, parse_citations


def test_extract_block():
    text = "Some prose.\n<final_answer>\n/a/b.py:10-15 (core)\n</final_answer>"
    assert extract_final_answer(text) == "/a/b.py:10-15 (core)"


def test_extract_missing():
    assert extract_final_answer("no block here") is None
    assert extract_final_answer(None) is None


def test_parse_range_and_single():
    block = "/repo/a.py:10-15 (reason here)\n/repo/b.js:42"
    cites = parse_citations(block, root=Path("/repo"))
    assert len(cites) == 2
    assert (cites[0].path, cites[0].start, cites[0].end, cites[0].reason) == (
        "/repo/a.py",
        10,
        15,
        "(reason here)",
    )
    assert (cites[1].start, cites[1].end) == (42, 42)


def test_parse_remaps_virtual_to_real(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x\ny\n")
    cites = parse_citations("/workspace/src/a.py:1-2 (here)", root=tmp_path, virtual_root="/workspace")
    assert cites[0].path == str(tmp_path / "src/a.py")
    assert cites[0].in_root is True


def test_parse_marks_outside_root():
    block = "/repo/in.py:1-2\n/etc/passwd:1"
    cites = parse_citations(block, root=Path("/repo"))
    by_path = {c.path: c.in_root for c in cites}
    assert by_path["/repo/in.py"] is True
    assert by_path["/etc/passwd"] is False


def test_parse_skips_garbage():
    assert parse_citations("not a citation line\n\n", root=Path("/repo")) == []


def test_usage_accumulates():
    u = Usage()
    u.add({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    u.add({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    u.add(None)
    assert (u.prompt, u.completion, u.total) == (11, 7, 18)
