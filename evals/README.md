# fcx evals

Citation accuracy, measured. Each case in `cases.toml` is a natural-language query plus the
ground-truth `file:line` ranges that answer it; the runner drives the real `fcx explore` CLI and
scores the returned citations with the FastContext paper's metric:

- **file-F1** — did we cite the right *files* (set F1 over file paths).
- **line-F1** — did we land on the right *lines* (set F1 over `(file, line)` tuples).

The cases explore *this* repository, so the harness is self-contained — no external checkout needed.

## Run

```bash
make eval                              # baseline: greedy (temperature 0), reproducible
uv run python evals/run.py --samples 3 # self-consistency (uses sampling temperature)
uv run python evals/run.py --only-valid  # score only citations that passed validation
uv run python evals/run.py --json        # machine-readable, for CI / A-B diffing
```

Single-sample runs force `FCX_TEMPERATURE=0` for a stable baseline. `--samples > 1` keeps the
configured sampling temperature, since cross-sample diversity is what makes voting work.

## What each knob measures

| Change | How to measure |
| --- | --- |
| Citation validation (`repair`) | `--only-valid` vs default; and toggle `FCX_REPAIR_INVALID_CITATIONS=false` |
| Self-consistency | `--samples 1` vs `--samples 3/5`; watch line-F1 (recall) |
| Quantization (8-bit vs bf16) | A/B below |

## Quantization A/B (8-bit vs bf16)

The model is selected entirely by `FCX_MODEL`, so comparing quants is config-only:

```bash
# baseline (current default, 8-bit)
make eval

# higher precision — point FCX_MODEL at a bf16/6-bit MLX build of FastContext-1.0-4B-RL
FCX_MODEL=<org>/FastContext-1.0-4B-RL-mlx-bf16 uv run python evals/run.py --json
```

If no bf16 MLX build exists on the Hub, convert one locally:

```bash
uv run python -m mlx_lm.convert --hf-path microsoft/FastContext-1.0-4B-RL --mlx-path ./fc-bf16
# then: FCX_MODEL=./fc-bf16 FCX_MANAGE_MODEL=... evaluate
```

Keep whichever scores higher; if bf16 wins, change the default `model` in `src/fcx/config.py`.
Restart the resident server between models: `fcx stop-model`.

## Adding cases

Append a `[[case]]` with `id`, `query`, and `expect = [{ file, start, end }]`. Ranges may be slightly
generous — line-F1 rewards overlap, so a citation landing inside the range earns partial credit.
