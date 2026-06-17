"""OpenAI-compatible client wrapper.

A pure pass-through payload builder: the request is constructed directly from config fields with no
provider detection or runtime adaptation. A field set to ``None`` is omitted. The only non-trivial
behavior is restarting a managed local model and retrying once on a connection error.
"""

from typing import Any, Protocol, final

from openai import APIConnectionError, AsyncOpenAI

from .config import Config
from .tools.base import ToolCall


@final
class Step:
    """One assistant turn: free-text content plus any tool calls and token usage."""

    def __init__(self, content: str | None, tool_calls: list[ToolCall], usage: dict[str, Any] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.usage = usage

    def as_assistant_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return msg


class LLMClient(Protocol):
    """Structural interface the loop depends on; satisfied by ``LLM`` and test mocks alike."""

    async def acall(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Step: ...


@final
class LLM:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = AsyncOpenAI(api_key=cfg.api_key.get_secret_value(), base_url=cfg.base_url)

    def _payload(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        c = self.cfg
        payload: dict[str, Any] = {"model": c.model, "messages": messages, c.token_param: c.max_tokens}
        if c.temperature is not None:
            payload["temperature"] = c.temperature
        if c.top_p is not None:
            payload["top_p"] = c.top_p
        if c.extra_body:
            payload["extra_body"] = c.extra_body
        if tools:
            payload["tools"] = tools
        return {k: v for k, v in payload.items() if v is not None}

    async def acall(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Step:
        payload = self._payload(messages, tools)
        try:
            resp = await self.client.chat.completions.create(**payload)
        except APIConnectionError:
            if not self.cfg.manage_model:
                raise
            from .model_server import ensure_model_up

            await ensure_model_up(self.cfg)
            resp = await self.client.chat.completions.create(**payload)

        choice = resp.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
        ]
        usage = resp.usage.model_dump() if resp.usage else None
        return Step(content=msg.content, tool_calls=tool_calls, usage=usage)
