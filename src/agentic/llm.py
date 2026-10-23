"""One call to Claude through the official SDK."""
from __future__ import annotations

from dataclasses import dataclass

import anthropic

MAX_TOKENS = 16000  # within the SDK's limit for a request that is not streamed
MAX_RETRIES = 60    # the SDK waits up to 8 s between tries, so a lost network is waited out for about ten minutes
_client: anthropic.Anthropic | None = None


@dataclass
class Reply:
    content: list      # response.content; append it unchanged as the assistant turn
    text: str          # the text blocks, joined
    stop_reason: str   # "end_turn" | "max_tokens" | "refusal" | ...
    usage: dict


def chat(system: str, messages: list, *, model: str, effort: str) -> Reply:
    """Send the conversation and return the reply.

    The system prompt is cached, and so is the growing conversation. Thinking is always on for these models, so only
    ``effort`` sets its depth. The request is not streamed, so the SDK retries the whole exchange, reading the reply
    included, when the connection drops, times out or the server is overloaded: a search runs for an hour on a
    network that drops now and then. Server-side refusal fallbacks are not enabled: a fallback would switch the model
    under test in the middle of a run.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=MAX_RETRIES)
    r = _client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
        cache_control={"type": "ephemeral"},
        output_config={"effort": effort},
    )
    u = r.usage
    return Reply(
        content=r.content,
        text="\n".join(b.text for b in r.content if b.type == "text"),
        stop_reason=r.stop_reason,
        usage={"input": u.input_tokens, "output": u.output_tokens,
               "cache_write": u.cache_creation_input_tokens or 0, "cache_read": u.cache_read_input_tokens or 0},
    )
