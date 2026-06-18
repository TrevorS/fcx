from pathlib import Path

from fcx.citations import (
    Citation,
    Usage,
    extract_final_answer,
    merge_citations,
    parse_citations,
    validate_range,
)


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


def test_usage_absorb():
    a = Usage(prompt=10, completion=5, total=15, cached=2)
    b = Usage(prompt=1, completion=2, total=3, cached=1)
    a.absorb(b)
    assert (a.prompt, a.completion, a.total, a.cached) == (11, 7, 18, 3)


# --- validation ---


def test_validate_range_in_bounds(tmp_path):
    f = tmp_path / "a.py"
    _ = f.write_text("1\n2\n3\n")
    assert validate_range(str(f), 1, 3, in_root=True) == (True, 3)


def test_validate_range_out_of_bounds(tmp_path):
    f = tmp_path / "a.py"
    _ = f.write_text("1\n2\n3\n")
    ok, n = validate_range(str(f), 1, 99, in_root=True)
    assert ok is False and n == 3


def test_validate_range_missing_file(tmp_path):
    ok, n = validate_range(str(tmp_path / "nope.py"), 1, 2, in_root=True)
    assert ok is False and n is None


def test_validate_range_outside_root_is_unjudged():
    # Out-of-root citations may point anywhere; we don't claim they're wrong.
    assert validate_range("/etc/passwd", 1, 999_999, in_root=False) == (True, None)


def test_parse_validates_citations(tmp_path):
    (tmp_path / "a.py").write_text("x\ny\nz\n")
    block = "/workspace/a.py:1-2 (good)\n/workspace/a.py:1-50 (too long)\n/workspace/missing.py:1 (gone)"
    cites = parse_citations(block, root=tmp_path, virtual_root="/workspace")
    by_lines = {(c.start, c.end): c for c in cites}
    assert by_lines[(1, 2)].valid is True
    assert by_lines[(1, 2)].file_lines == 3
    assert by_lines[(1, 50)].valid is False  # past EOF
    assert by_lines[(1, 1)].valid is False  # missing file
    assert by_lines[(1, 1)].file_lines is None


# --- merge / self-consistency ---


def _c(path: str, start: int, end: int) -> Citation:
    return Citation(path=path, start=start, end=end, in_root=False)


def test_merge_counts_votes_and_unions_overlaps():
    runs = [
        [_c("/r/a.py", 10, 15), _c("/r/b.py", 1, 2)],
        [_c("/r/a.py", 12, 20)],  # overlaps a.py 10-15 -> union 10-20, 2 votes
        [_c("/r/a.py", 12, 18)],  # also overlaps -> 3 votes total
    ]
    merged = merge_citations(runs)
    a = next(c for c in merged if c.path == "/r/a.py")
    assert (a.start, a.end, a.votes) == (10, 20, 3)
    # most-agreed citation sorts first
    assert merged[0].path == "/r/a.py"
    b = next(c for c in merged if c.path == "/r/b.py")
    assert b.votes == 1


def test_merge_keeps_disjoint_ranges_separate():
    runs = [[_c("/r/a.py", 1, 5)], [_c("/r/a.py", 100, 110)]]
    merged = merge_citations(runs)
    assert len(merged) == 2
    assert {(c.start, c.end) for c in merged} == {(1, 5), (100, 110)}
