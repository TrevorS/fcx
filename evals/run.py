"""fcx eval harness.

Runs each case in ``cases.toml`` through the real ``fcx explore`` CLI and scores the returned
citations against ground truth with the FastContext paper's metric: file-F1 (did we name the right
files) and line-F1 (did we land on the right lines). This is the measurement substrate for every
quality change — quantization, self-consistency, prompt edits — so improvements are numbers, not vibes.

    uv run python evals/run.py                  # baseline, greedy (temperature 0) for reproducibility
    uv run python evals/run.py --samples 3      # self-consistency; uses sampling temperature
    uv run python evals/run.py --only-valid     # score only citations that passed validation
    uv run python evals/run.py --min-file-f1 0.7  # exit nonzero if aggregate file-F1 falls below

Greedy decoding is forced for single-sample runs so the baseline is stable; with --samples > 1 the
configured sampling temperature is kept, since diversity across samples is the whole point.
"""

import argparse
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Expected:
    file: str
    start: int
    end: int


@dataclass
class Case:
    id: str
    query: str
    path: str
    expect: list[Expected]


@dataclass
class Score:
    case_id: str
    file_f1: float
    line_f1: float
    n_pred: int
    n_invalid: int


def load_cases(path: Path) -> list[Case]:
    data = tomllib.loads(path.read_text())
    cases: list[Case] = []
    for raw in data.get("case", []):
        expect = [Expected(e["file"], int(e["start"]), int(e["end"])) for e in raw.get("expect", [])]
        cases.append(Case(raw["id"], raw["query"], raw.get("path", "."), expect))
    return cases


def _f1(pred: Set[object], gold: Set[object]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(gold)
    return 2 * prec * rec / (prec + rec)


def _rel(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def run_case(case: Case, *, samples: int, max_turns: int | None, only_valid: bool) -> Score:
    root = (REPO_ROOT / case.path).resolve()
    cmd = ["uv", "run", "fcx", "explore", case.query, "-p", str(root), "--json", "--quiet"]
    if samples > 1:
        cmd += ["--samples", str(samples)]
    if max_turns is not None:
        cmd += ["--max-turns", str(max_turns)]

    env = dict(os.environ)
    if samples <= 1:
        env.setdefault("FCX_TEMPERATURE", "0")  # reproducible greedy baseline

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0:
        print(f"  ! {case.id}: fcx failed\n{proc.stderr}", file=sys.stderr)
        return Score(case.id, 0.0, 0.0, 0, 0)

    result = json.loads(proc.stdout)
    citations = result.get("citations", [])
    n_invalid = sum(1 for c in citations if not c.get("valid", True))
    if only_valid:
        citations = [c for c in citations if c.get("valid", True)]

    pred_files = {_rel(c["path"], root) for c in citations}
    gold_files = {e.file for e in case.expect}

    pred_lines = {(_rel(c["path"], root), n) for c in citations for n in range(c["start"], c["end"] + 1)}
    gold_lines = {(e.file, n) for e in case.expect for n in range(e.start, e.end + 1)}

    return Score(
        case_id=case.id,
        file_f1=_f1(pred_files, gold_files),
        line_f1=_f1(pred_lines, gold_lines),
        n_pred=len(citations),
        n_invalid=n_invalid,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Score fcx citations against ground truth.")
    _ = ap.add_argument("--cases", type=Path, default=Path(__file__).resolve().parent / "cases.toml")
    _ = ap.add_argument("--samples", type=int, default=1)
    _ = ap.add_argument("--max-turns", type=int, default=None)
    _ = ap.add_argument("--only-valid", action="store_true", help="score only validated citations")
    _ = ap.add_argument("--min-file-f1", type=float, default=None, help="exit nonzero below this aggregate")
    _ = ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    scores = [
        run_case(c, samples=args.samples, max_turns=args.max_turns, only_valid=args.only_valid)
        for c in cases
    ]

    mean_file = sum(s.file_f1 for s in scores) / len(scores) if scores else 0.0
    mean_line = sum(s.line_f1 for s in scores) / len(scores) if scores else 0.0

    if args.json:
        print(
            json.dumps(
                {
                    "samples": args.samples,
                    "mean_file_f1": round(mean_file, 4),
                    "mean_line_f1": round(mean_line, 4),
                    "cases": [vars(s) for s in scores],
                },
                indent=2,
            )
        )
    else:
        print(f"\n  {'case':<26} {'file-F1':>8} {'line-F1':>8} {'cites':>6} {'invalid':>8}")
        print(f"  {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 8}")
        for s in scores:
            print(f"  {s.case_id:<26} {s.file_f1:>8.2f} {s.line_f1:>8.2f} {s.n_pred:>6} {s.n_invalid:>8}")
        print(f"  {'-' * 26} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 8}")
        print(f"  {'MEAN':<26} {mean_file:>8.2f} {mean_line:>8.2f}   (n={len(scores)}, k={args.samples})\n")

    if args.min_file_f1 is not None and mean_file < args.min_file_f1:
        print(f"FAIL: mean file-F1 {mean_file:.2f} < {args.min_file_f1}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
