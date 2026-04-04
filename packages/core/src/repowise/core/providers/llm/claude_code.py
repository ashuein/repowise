"""Claude Code provider for repowise.

Routes LLM calls through the Claude Agent SDK, which uses the same
authentication as your Claude Code installation. If you have a Claude Max
subscription, this means wiki generation costs nothing beyond your subscription.

The provider invokes the Agent SDK's ``query()`` with no tools enabled and
``permission_mode="dontAsk"``, so Claude simply generates text — no file reads,
no bash commands, no agentic loops. It behaves like a plain completion API.

Requirements:
    pip install claude-agent-sdk

Usage:
    provider = ClaudeCodeProvider(model="claude-sonnet-4-6")
    response = await provider.generate(system_prompt="...", user_prompt="...")
"""

from __future__ import annotations

import asyncio

import structlog

from repowise.core.providers.llm.base import (
    BaseProvider,
    GeneratedResponse,
    ProviderError,
)
from repowise.core.rate_limiter import RateLimiter

log = structlog.get_logger(__name__)

# Rough estimate: ~4 characters per token for English text.
_CHARS_PER_TOKEN = 4

_MAX_RETRIES = 3


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length (conservative)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


class ClaudeCodeProvider(BaseProvider):
    """Claude Code provider using the Claude Agent SDK.

    Routes LLM calls through the locally installed Claude Code binary,
    using whatever authentication it's configured with (API key or
    Claude Max subscription).

    No tools are enabled — Claude generates text only, making this
    behave like a standard completion API.

    Args:
        model:        Model to use. Defaults to claude-sonnet-4-6.
                      Passed as ``model`` in ClaudeAgentOptions.
        max_turns:    Maximum agent turns. Defaults to 1 since we only
                      need a single text response (no tool loops).
        rate_limiter: Optional RateLimiter for concurrency control.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_turns: int = 2,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._model = model
        self._max_turns = max_turns
        self._rate_limiter = rate_limiter

        # Validate that the SDK is importable at construction time
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ClaudeCodeProvider requires the 'claude-agent-sdk' package. "
                "Install it with: pip install claude-agent-sdk"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "claude_code"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        request_id: str | None = None,
    ) -> GeneratedResponse:
        if self._rate_limiter:
            await self._rate_limiter.acquire(estimated_tokens=max_tokens)

        log.debug(
            "claude_code.generate.start",
            model=self._model,
            max_tokens=max_tokens,
            request_id=request_id,
        )

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = await self._do_generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                )
                log.debug(
                    "claude_code.generate.done",
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    request_id=request_id,
                )
                return result
            except ProviderError:
                raise
            except Exception as exc:
                last_error = exc
                log.warning(
                    "claude_code.generate.retry",
                    attempt=attempt,
                    error=str(exc),
                    request_id=request_id,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(2**attempt)

        raise ProviderError(
            "claude_code",
            f"All {_MAX_RETRIES} retries exhausted: {last_error}",
        )

    async def _do_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> GeneratedResponse:
        """Execute a single generation call via the Agent SDK."""
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            query,
        )

        # Build options: no tools, no file access — pure text generation.
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=[],
            permission_mode="dontAsk",
            model=self._model,
            max_turns=self._max_turns,
        )

        # Collect text from the agent's response stream.
        text_parts: list[str] = []

        try:
            async for message in query(prompt=user_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text") and block.text:
                            text_parts.append(block.text)
                elif isinstance(message, ResultMessage) and hasattr(message, "result") and message.result:
                    text_parts.append(message.result)
        except Exception as exc:
            error_msg = str(exc)
            if "API key" in error_msg or "authentication" in error_msg.lower():
                raise ProviderError(
                    "claude_code",
                    "Authentication failed. Ensure Claude Code is logged in "
                    "(run 'claude' in terminal) or set ANTHROPIC_API_KEY.",
                ) from exc
            raise ProviderError("claude_code", f"Agent SDK error: {error_msg}") from exc

        content = "\n".join(text_parts).strip()

        if not content:
            raise ProviderError(
                "claude_code",
                "Agent SDK returned empty response. Ensure Claude Code is "
                "authenticated and the model is available.",
            )

        # Estimate token usage (Agent SDK does not expose usage stats).
        input_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        output_tokens = _estimate_tokens(content)

        return GeneratedResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=0,
            usage={
                "estimated": True,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
