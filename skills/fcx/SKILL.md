---
name: fcx
description: Locate code in a repository — returns compact file:line citations for "where is X", "what calls Y", "where is Z defined" questions, best on large or unfamiliar codebases. Reach for it instead of a multi-file grep/glob/read chain when finding the code is the work. Write a specific query that names the behavior, subsystem, error, or file you are after — vague queries are its main failure mode. It is a locator, not a reasoner: treat every returned citation as a lead and read/verify the cited range before acting on it.
allowed-tools: Bash(fcx explore *)
---

# fcx

Fast, autonomous repository explorer (FastContext, RL-trained). Given a natural-language question it
searches the codebase with read-only Read/Glob/Grep and returns compact `file:line` citations as focused
evidence — without spending the main agent's context on the exploratory reads. It is a **locator**: it
finds where code lives, it does **not** judge whether that code is correct.

## When to use

- **Find code** before editing, reviewing, debugging, or explaining it — "where is Y defined", "what calls Z"
- **Trace a path** across files or layers (request → handler → service → DB) when you know the endpoints
- **Map dependencies** — what a symbol depends on, or what depends on it
- A repo that is **large or unfamiliar**, where a manual grep/read chain would burn many turns

## When NOT to use

- You already read the exact file this session
- A single obvious grep in one known file (just grep)
- A vague or open-ended ask ("explain everything about auth") — it is tuned for *targeted* location
- A judgment call ("is this correct", "is this a bug") — that is reasoning, not location

## How to query well — this is the main lever

The model is RL-trained to reward precise, targeted citations and to penalize vague, sprawling output.
Query quality, not model size, drives accuracy. So:

- **Name the target.** Reference the behavior, subsystem, symbol, error string, or file. Don't ask "how
  does this work" — ask "where is the PreToolUse hook registered in settings.json".
- **One target per call.** Ask for one thing, not a tour. Split compound questions into separate calls.
- **Concrete > abstract.** Use the repo's own nouns (function names, config keys, error text).

Bad → Good:

- `how do hooks get wired and which scripts run` → `where is the jj guard PreToolUse hook registered in settings.json`
- `explain the auth flow` → `where does session validation happen in the auth middleware`

If a query comes back vague or wrong, **rephrase it more specifically and re-run** — that fixes most misses.

## Verify before you act

fcx returns leads, not ground truth (a 4B model can cite confidently and be wrong). After a call:

1. `Read` only the cited ranges, with narrow line windows.
2. Confirm the citation actually answers the question before editing or concluding.
3. Don't re-run broad repo-wide searches for the same thing — that throws away the token savings.

## Usage

```bash
# Precise answer with file:line citations (run from the repo, or pass --path)
fcx explore "where is request validation handled in the API router" --citation

# Machine-readable output for programmatic use
fcx explore "where are DB migrations defined" --json
```

Leave `--max-turns` at its default (8) — that is the model's RL training horizon, so it explores best
within it; raising it pushes the model out of distribution and rarely helps. Only nudge it up for a
genuinely deep cross-layer trace, and don't expect linear gains.

The first invocation starts the shared local model server (it stays resident across calls); later calls
attach in milliseconds. `fcx status` shows the model server; `fcx stop-model` tears it down.
