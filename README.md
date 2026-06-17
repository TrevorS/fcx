<p align="center">
  <img src="assets/fcx.jpeg" alt="fcx" width="440">
</p>

A command-line **repository explorer**. `fcx explore "<query>"` searches a codebase with read-only
`Read`/`Glob`/`Grep` tools and returns compact `file:line` citations - so a coding agent can delegate
exploration instead of burning its own context on broad reads.

A reimplementation of Microsoft's [FastContext](https://github.com/microsoft/fastcontext) agent, served
locally via MLX (`FastContext-1.0-4B-RL`) or any OpenAI-compatible API.

## Install

```sh
make install   # uv tool install .
```

Requires Python 3.13+ and [ripgrep](https://github.com/BurntSushi/ripgrep) on `PATH`.

## Use

```sh
fcx explore "how is the local model server kept a single shared instance across processes" --citation
```

```text
/workspace/src/fcx/model_server.py:1-144 (Core: lazy-singleton pattern using FileLock + port binding + detached subprocess)
/workspace/src/fcx/model_server.py:97-116 (Key: FileLock with thread_local=False ensuring cross-process coordination)
/workspace/src/fcx/model_server.py:109-112 (OS port-in-use check as second layer of singleton enforcement)
/workspace/src/fcx/model_server.py:55-74 (Detached subprocess spawning with start_new_session=True, stdout/stderr redirected to log file)
```

```sh
fcx explore "where are the model's read-only tool calls executed concurrently" --citation
```

```text
/workspace/src/fcx/agent.py:63-64 (Core dispatch point: collects tool calls and passes them to toolset.dispatch())
/workspace/src/fcx/tools/base.py:105-107 (Concurrency entry point: dispatch() uses asyncio.gather() to run all tool calls concurrently)
/workspace/src/fcx/tools/base.py:75-103 (Individual tool call execution: _run_one() executes each tool call with timeout/error handling)
/workspace/src/fcx/tools/read.py:43-83 (ReadTool.run(): uses asyncio.to_thread() for file I/O, running synchronously in threads)
```

The first run boots a resident local model server; later calls attach instantly. Paths are shown at the
model's `/workspace` root. Other flags: `--json` (structured result), `--path <dir>` (explore another
repo), plus `fcx status` and `fcx stop-model`.

## Configure

Set `FCX_`-prefixed env vars (or a `.env`; see [`.env.example`](.env.example)). Defaults target fcx's
managed local MLX server. To use a local OpenAI-compatible server you already run - e.g.
[Ollama](https://ollama.com) or [LM Studio](https://lmstudio.ai) serving the FastContext model - set
`FCX_MANAGE_MODEL=false` and point at it:

```sh
FCX_BASE_URL=http://localhost:11434/v1   # LM Studio: http://localhost:1234/v1
FCX_MODEL=FastContext-1.0-4B-RL
FCX_API_KEY=ollama                       # LM Studio: any value
FCX_MANAGE_MODEL=false
FCX_EXTRA_BODY={}
```

## Develop

Dev tasks run through the [Makefile](Makefile):

```sh
make          # list targets
make check    # lint + type-check + tests
```
