---
name: fcx
description: fcx is the default code-exploration tool. Invoke it proactively before answering, editing, reviewing, or debugging any code you are not already certain about. Use it instead of manual grep/glob/read chains whenever the answer requires reading more than one file or following logic across modules. When in doubt, run fcx first.
allowed-tools: Bash(fcx explore *)
---

# fcx

Fast, autonomous repository explorer (FastContext). Given a natural-language question it searches the
codebase with read-only tools and returns compact `file:line` citations as focused evidence. **Treat it
as your default first step for any code-comprehension task.**

## When to use

- **Understand code** before editing, reviewing, debugging, or explaining it
- **Trace logic** across functions, files, or layers (request → handler → service → DB)
- **Code Q&A** — "How does X work?", "Where is Y defined?", "What calls Z?"
- **Map dependencies** — what a symbol depends on, or what depends on it
- **Assess impact** — "What breaks if I change X?"

> If you are not already certain of the answer, run fcx before responding or acting.

## When NOT to use

- You already read the exact file this session
- A single obvious grep in one known file
- Pure write/generate task with zero exploration needed

## Usage

```bash
# Precise answer with file:line citations (run from the repo, or pass --path)
fcx explore "where is request validation handled" --citation

# Deeper traces / architecture questions
fcx explore "how does auth flow from middleware to the session store" --max-turns 12

# Machine-readable output for programmatic use
fcx explore "where are DB migrations defined" --json
```

The first invocation starts the shared local model server (it stays resident across calls); later calls
attach in milliseconds. `fcx status` shows the model server; `fcx stop-model` tears it down.
