"""Claude Code provider for repowise.

Routes LLM calls through the ``claude`` CLI in print mode (``-p``), which
uses whatever authentication your Claude Code installation is configured with.
If you have a Claude Max subscription, wiki generation costs nothing beyond
your subscription — no API key needed.

The provider invokes ``claude -p`` with ``--tools ""`` (no tools) so Claude
simply generates text. No file reads, no bash commands, no agentic loops.

Requirements:
    - Claude Code CLI installed and authenticated (``claude --version``)

Usage:
    provider = ClaudeCodeProvider(model="sonnet")
    response = await provider.generate(system_prompt="...", user_prompt="...")
"""

from __future__ import annotations

import asyncio
import json
import shutil

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
    """Claude Code provider using the ``claude`` CLI.

    Spawns ``claude -p <prompt>`` as a subprocess for each generation call,
    using the CLI's own authentication (OAuth subscription or API key).

    No tools are enabled — Claude generates text only, making this
    behave like a standard completion API.

    Args:
        model:        Model alias or full name. Defaults to "sonnet".
                      Passed as ``--model`` to the CLI.
        rate_limiter: Optional RateLimiter for concurrency control.
    """

    def __init__(
        self,
        model: str = "sonnet",
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._model = model
        self._rate_limiter = rate_limiter

        # Validate that the claude CLI is available at construction time
        if not shutil.which("claude"):
            raise ProviderError(
                "claude_code",
                "Claude Code CLI not found on PATH. "
                "Install it from: https://claude.ai/download",
            )

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
        """Execute a single generation call via the claude CLI."""
        import os

        cmd = [
            "claude",
            "-p",                               # print mode (non-interactive)
            "--output-format", "json",           # structured output with usage
            "--model", self._model,
            "--system-prompt", system_prompt,
            "--tools", "",                       # disable all tools
            "--no-session-persistence",          # don't save session to disk
            user_prompt,
        ]

        # Build a clean env that strips ANTHROPIC_API_KEY so the CLI
        # falls back to subscription / OAuth auth instead of using a
        # potentially empty or low-credit API key.
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError as exc:
            raise ProviderError(
                "claude_code",
                "Claude Code CLI not found. Install from: https://claude.ai/download",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                "claude_code",
                f"Failed to spawn claude CLI: {exc}",
            ) from exc

        # The CLI outputs JSON to stdout even on errors (exit code 1),
        # so always try to parse stdout first.
        raw_output = stdout.decode("utf-8", errors="replace").strip()
        err_text = stderr.decode("utf-8", errors="replace").strip()

        if not raw_output:
            if proc.returncode != 0:
                raise ProviderError(
                    "claude_code",
                    f"claude CLI exited with code {proc.returncode}: "
                    f"{err_text or 'no output'}",
                )
            raise ProviderError(
                "claude_code",
                "claude CLI returned empty output.",
            )

        # Parse JSON output — the CLI returns:
        # {"type":"result", "result":"...", "is_error":false,
        #  "usage":{"input_tokens":N, "output_tokens":N, ...}, ...}
        content = raw_output
        input_tokens = 0
        output_tokens = 0

        try:
            data = json.loads(raw_output)

            # Check for errors reported in the JSON response
            if data.get("is_error"):
                error_result = data.get("result", "Unknown error")
                if "auth" in error_result.lower():
                    raise ProviderError(
                        "claude_code",
                        "Authentication failed. Run 'claude' interactively to log in, "
                        "or set ANTHROPIC_API_KEY.",
                    )
                raise ProviderError("claude_code", f"claude CLI error: {error_result}")

            # Extract the text content
            content = data.get("result", "")

            # Extract real usage from JSON (the CLI provides actual token counts)
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        except ProviderError:
            raise
        except (json.JSONDecodeError, AttributeError) as exc:
            # If not valid JSON, treat raw output as the content (text mode fallback).
            # This can happen if --output-format json isn't supported.
            if proc.returncode != 0:
                raise ProviderError(
                    "claude_code",
                    f"claude CLI exited with code {proc.returncode}: "
                    f"{err_text or raw_output}",
                ) from exc

        if not content:
            raise ProviderError(
                "claude_code",
                "claude CLI returned empty response.",
            )

        # Fall back to estimation if usage not available from JSON
        if input_tokens == 0:
            input_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        if output_tokens == 0:
            output_tokens = _estimate_tokens(content)

        return GeneratedResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=0,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated": input_tokens == 0,
            },
        )
