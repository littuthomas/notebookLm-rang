import asyncio
import logging
import random
import re
import time
from typing import AsyncIterator, Generator, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Returned verbatim (without calling the LLM) when retrieval finds no usable context,
# and also instructed to the LLM when the context does not contain the answer.
NO_ANSWER_MESSAGE = (
    "The answer was not found in the documents in this notebook."
)

SYSTEM_INSTRUCTION = (
    "You are a retrieval-augmented assistant. Answer the user's question using ONLY "
    "the provided context from the currently selected notebook.\n"
    "Rules:\n"
    "1. Base every statement strictly on the provided context.\n"
    "2. Do not use outside or internal knowledge, and never invent facts, names or numbers.\n"
    "3. Answer ONLY the specific question asked. Do not summarize the notebook, do not "
    "answer other questions, and do not include unrelated facts or topics that were not asked about.\n"
    "4. Cite the sources you rely on inline using their labels, e.g. [Source 1]. "
    "Cite a source only if it actually supports the statement.\n"
    "5. If the context does not contain the answer, reply exactly with: "
    f"\"{NO_ANSWER_MESSAGE}\"\n"
    "6. Be thorough and complete: explain your reasoning step by step, cover all "
    "relevant details from the context, and include specific facts, names and numbers "
    "from the retrieved sources. Do not truncate a complete answer just to be brief.\n"
)

# Generic, user-safe error message. Raw provider errors, API keys, and stack
# traces are only logged server-side, never surfaced through this message.
GENERATION_UNAVAILABLE_MESSAGE = (
    "The answer generation service is temporarily unavailable. Please try again later."
)

# Nemotron-class fallback models may spend reasoning tokens from the same
# max_tokens budget as the visible answer on some providers. Pad the fallback
# budget modestly so reasoning cannot consume the whole answer allocation.
FALLBACK_MAX_TOKENS_MULTIPLIER = 2

# Number of characters of the PRIMARY stream that are held back before any
# token is flushed to the client. If Gemini dies while the buffer is still
# holding (e.g. a transport error mid-request), the partial output is discarded
# and the fallback streams a clean, single-model answer — no stitched output.
# Once flushed, mid-stream failures keep their previous behaviour (an error):
# switching models after real tokens reached the user would splice two
# different models' output together.
STREAM_FLUSH_THRESHOLD_CHARS = 40

# Lowercase signatures of TRANSIENT Gemini API errors worth retrying with
# exponential backoff before giving up and using the fallback LLM. The most
# common one in practice is 503 UNAVAILABLE ("model is currently experiencing
# high demand"), which is usually gone after a short delay.
_TRANSIENT_ERROR_SIGNATURES = (
    "503",
    "unavailable",
    "overloaded",
    "429",
    "resource_exhausted",
    "rate limit",
    "500",
    "internal error",
    "504",
    "deadline exceeded",
    "timeout",
    "connection reset",
    "connection aborted",
)


def _is_transient_generation_error(exc: Exception) -> bool:
    """True when ``exc`` looks like a transient provider error worth retrying.

    Auth/validation errors, empty/blocked responses and anything else not in
    the signature list are treated as permanent and fail fast.
    """
    message = str(exc).lower()
    return any(sig in message for sig in _TRANSIENT_ERROR_SIGNATURES)


def _generation_retry_delay(attempt: int) -> float:
    """Exponential backoff delay (seconds) for retry ``attempt`` (1-based)."""
    delay = settings.GEMINI_LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
    # Full jitter to avoid thundering-herd retries across concurrent requests.
    return min(random.uniform(0, delay), settings.GEMINI_LLM_RETRY_MAX_DELAY)

# Matches a <think>...</think> reasoning block at the start of a response.
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def build_rag_prompt(question: str, context: str) -> str:
    """Build the grounded prompt sent to the LLM from the retrieved context."""
    return (
        "Use the context below to answer the question.\n\n"
        "--- CONTEXT START ---\n"
        f"{context}\n"
        "--- CONTEXT END ---\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def extract_answer_text(text: str) -> str:
    """Strip reasoning artifacts from a model response.

    Nemotron models may emit a ``<think>...</think>`` reasoning trace before the
    final answer (visible thinking). Only the final answer content is returned;
    reasoning traces are never exposed to users.
    """
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    return cleaned.strip()


class GenerationService:
    """Service for generating grounded answers with a primary + fallback LLM.

    Primary: Gemini (non-streaming and streaming), unchanged.
    Fallback: NVIDIA Nemotron 3 Ultra, routed through OpenRouter's
    OpenAI-compatible API, used ONLY when the Gemini call fails before producing
    a complete answer. Embeddings, retrieval and the RAG prompt are untouched by
    this fallback.

    Kept separate from :class:`~app.services.embedding.EmbeddingService` so the
    embedding model (gemini-embedding-2) and the generation model can evolve
    independently. Reuses the existing GEMINI_API_KEY configuration.

    Supports both non-streaming (used by POST /api/chat) and streaming (used by
    the WebSocket RAG flow) generation.
    """

    def __init__(self) -> None:
        self._model: str = settings.GEMINI_LLM_MODEL
        self._temperature: float = settings.GEMINI_LLM_TEMPERATURE
        self._max_output_tokens: int = settings.GEMINI_LLM_MAX_OUTPUT_TOKENS
        # Name of the model that produced the most recent answer, so callers
        # (e.g. POST /api/chat) can report the actual model used.
        self.last_model_used: str = self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer to ``question`` grounded in ``context``.

        Tries Gemini first (primary). If the Gemini call fails, falls back to
        NVIDIA Nemotron 3 Ultra via OpenRouter (same question, same context,
        same prompt). If both fail, raises a sanitized RuntimeError — raw
        provider errors never reach the caller/user.

        Args:
            question: The user's question.
            context: A pre-built context string of retrieved chunks.

        Returns:
            The generated answer text.

        Raises:
            ValueError: If the question or context is empty.
            RuntimeError: If both the primary and fallback LLM calls fail.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")

        prompt = build_rag_prompt(question=question, context=context)

        try:
            answer = self._generate_gemini(prompt)
            self.last_model_used = self._model
            return answer
        except ValueError:
            raise
        except Exception as e:
            # Gemini failed at call time (e.g. 503 high demand) — nothing was
            # returned to the user yet, so the fallback is safe here.
            logger.error(f"Primary LLM (Gemini) generation failed, attempting fallback: {e}")

        if not settings.LLM_FALLBACK_ENABLED:
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE)

        if not settings.OPENROUTER_API_KEY:
            logger.error("Fallback unavailable: OPENROUTER_API_KEY is not configured")
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE)

        try:
            answer = self._generate_nemotron(prompt)
            self.last_model_used = settings.OPENROUTER_LLM_MODEL
            return answer
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Fallback LLM (Nemotron) generation also failed: {e}")
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE) from e

    async def stream_answer(self, question: str, context: str) -> AsyncIterator[str]:
        """
        Stream an answer to ``question`` grounded in ``context``.

        Tries Gemini streaming first (primary). The first tokens are buffered
        until STREAM_FLUSH_THRESHOLD_CHARS characters accumulate; if Gemini
        fails while nothing has been flushed yet (request-time 503, mid-request
        transport error, …), the buffered partial output is discarded and the
        fallback NVIDIA Nemotron (OpenRouter) streams the answer instead (same
        question, same context, same prompt — never a stitched, two-model
        answer). If Gemini fails after tokens were already flushed to the
        client, the error is raised as before — the fallback is never started
        mid-answer.

        Args:
            question: The user's question.
            context: A pre-built context string of retrieved chunks.

        Yields:
            Answer text tokens progressively as they arrive from the LLM.

        Raises:
            ValueError: If the question or context is empty.
            RuntimeError: If streaming fails from both providers (before any
                token was flushed to the client), or mid-stream after flushed
                tokens were yielded.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")

        prompt = build_rag_prompt(question=question, context=context)

        # Buffered prefix of the primary stream: nothing reaches the client
        # until STREAM_FLUSH_THRESHOLD_CHARS characters have accumulated (or
        # the primary stream completes). This widens the safe fallback window
        # from "before the first token" to "before the first flushed token",
        # covering mid-request Gemini failures without ever splicing two
        # different models' output into one answer.
        yielded_any = False
        pending: list[str] = []
        pending_len = 0
        try:
            async for token in self._stream_gemini(prompt):
                if not yielded_any:
                    pending.append(token)
                    pending_len += len(token)
                    if pending_len < STREAM_FLUSH_THRESHOLD_CHARS:
                        continue
                    # Threshold reached — flush the buffer as real output.
                    for buffered in pending:
                        yield buffered
                    pending.clear()
                    pending_len = 0
                    yielded_any = True
                    continue
                yield token
            # Primary stream completed: flush whatever was still buffered.
            for buffered in pending:
                yield buffered
            self.last_model_used = self._model
            return
        except ValueError:
            raise
        except Exception as e:
            if yielded_any:
                # Tokens already went out to the client — switching models
                # halfway through would produce a stitched, inconsistent answer.
                logger.error(f"Primary LLM (Gemini) streaming failed mid-stream: {e}")
                raise RuntimeError(
                    GENERATION_UNAVAILABLE_MESSAGE
                ) from e
            # Gemini failed before ANY token was flushed (buffer discarded) —
            # e.g. a 503 at request time or a mid-request transport error.
            logger.error(
                f"Primary LLM (Gemini) streaming failed before any flushed "
                f"token, attempting fallback: {e}"
            )

        if not settings.LLM_FALLBACK_ENABLED:
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE)

        if not settings.OPENROUTER_API_KEY:
            logger.error("Fallback unavailable: OPENROUTER_API_KEY is not configured")
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE)

        try:
            async for token in self._stream_nemotron(prompt):
                yield token
            self.last_model_used = settings.OPENROUTER_LLM_MODEL
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Fallback LLM (Nemotron) streaming failed: {e}")
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE) from e

    @property
    def model_name(self) -> str:
        """Name of the configured primary (Gemini) generation model."""
        return self._model

    # ------------------------------------------------------------------
    # Primary: Gemini (unchanged behavior)
    # ------------------------------------------------------------------

    def _generate_gemini(self, prompt: str) -> str:
        """Generate a completion using the Gemini API (non-streaming).

        Uses the synchronous google-genai client. This runs inside
        ``asyncio.to_thread`` in the chat endpoint, so blocking here only
        consumes one worker-thread slot — it does not stall the event loop.
        """
        from google import genai
        from google.genai import types

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured")

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Typed config with AFC disabled: this pipeline never passes tools, so
        # automatic function calling is dead weight. With it disabled the SDK
        # takes the direct request path and skips its AFC wrapper, which logs
        # a "Direct use of automatic function calling" warning.
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        # Retry transient provider errors (e.g. 503 "high demand") with
        # exponential backoff before giving up and using the fallback LLM.
        attempt = 0
        while True:
            try:
                response = client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                break
            except Exception as e:
                attempt += 1
                if not _is_transient_generation_error(e) or attempt > settings.GEMINI_LLM_MAX_RETRIES:
                    if attempt > settings.GEMINI_LLM_MAX_RETRIES:
                        logger.error(
                            "Gemini still failing after %d retr%s: %s",
                            settings.GEMINI_LLM_MAX_RETRIES,
                            "y" if settings.GEMINI_LLM_MAX_RETRIES == 1 else "ies",
                            e,
                        )
                    raise
                delay = _generation_retry_delay(attempt)
                logger.warning(
                    "Transient Gemini error (attempt %d), retrying in %.1fs: %s",
                    attempt,
                    delay,
                    e,
                )
                time.sleep(delay)  # blocks the worker thread, not the event loop

        try:
            answer = response.text
        except Exception as e:
            # Happens when the response was blocked or contains no text parts
            raise RuntimeError(f"Gemini returned no usable text: {e}") from e

        if not answer or not answer.strip():
            raise RuntimeError("Gemini returned an empty answer")

        return answer.strip()

    async def _stream_gemini(self, prompt: str) -> AsyncIterator[str]:
        """Generate a completion using the Gemini API (streaming)."""
        from google import genai
        from google.genai import types

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured")

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Typed config with AFC disabled (see _generate_gemini above).
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        # Retry transient errors while opening the stream: the HTTP request is
        # only actually sent when the first chunk is pulled, so a retry header
        # never leaks tokens to the client and the fallback logic upstream
        # (which refuses to switch models mid-answer) is never jeopardized.
        #
        # The google-genai SDK iterator is synchronous: opening the stream and
        # pulling chunks are blocking network round-trips. Everything that
        # touches the network runs on worker threads (asyncio.to_thread / a
        # pump thread) so this async generator never blocks the event loop —
        # other WebSocket streams and HTTP requests stay responsive while a
        # Gemini chunk is in flight.
        attempt = 0
        while True:
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content_stream,
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                chunk_iter = iter(response)
                first_chunk = await asyncio.to_thread(
                    lambda: next(chunk_iter, None)
                )  # sends the request
                break
            except Exception as e:
                attempt += 1
                if not _is_transient_generation_error(e) or attempt > settings.GEMINI_LLM_MAX_RETRIES:
                    if attempt > settings.GEMINI_LLM_MAX_RETRIES:
                        logger.error(
                            "Gemini streaming still failing after %d retr%s: %s",
                            settings.GEMINI_LLM_MAX_RETRIES,
                            "y" if settings.GEMINI_LLM_MAX_RETRIES == 1 else "ies",
                            e,
                        )
                    raise
                delay = _generation_retry_delay(attempt)
                logger.warning(
                    "Transient Gemini streaming error (attempt %d), retrying in %.1fs: %s",
                    attempt,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

        # Pump the remaining chunks on a dedicated daemon thread and consume
        # them via the default executor, so no chunk wait ever stalls the loop.
        # First chunk is queued BEFORE the pump starts to preserve stream order.
        import queue as _queue
        import threading

        chunk_queue: _queue.Queue = _queue.Queue()
        _SENTINEL = object()

        def _pump():
            try:
                for chunk in chunk_iter:
                    chunk_queue.put(chunk)
            except Exception as e:  # transport/HTTP error mid-stream
                chunk_queue.put(e)
            finally:
                chunk_queue.put(_SENTINEL)

        yielded_any = False
        try:
            if first_chunk is not None:
                chunk_queue.put(first_chunk)
            threading.Thread(target=_pump, daemon=True).start()
            while True:
                chunk = await asyncio.to_thread(chunk_queue.get)
                if chunk is _SENTINEL:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise RuntimeError(f"Gemini streaming failed: {e}") from e

        if not yielded_any:
            raise RuntimeError("Gemini streaming returned no usable text")

    # ------------------------------------------------------------------
    # Fallback: NVIDIA Nemotron 3 Ultra via OpenRouter (OpenAI-compatible API)
    # ------------------------------------------------------------------

    def _fallback_max_tokens(self) -> int:
        """max_tokens for the OpenRouter fallback, with reasoning headroom.

        Nemotron 3 Ultra's reasoning tokens count against max_tokens on
        OpenRouter. With the plain answer budget the model could spend it all
        reasoning and return empty content (finish_reason="length"), so the
        budget is padded and reasoning effort is set to "low".
        """
        return self._max_output_tokens * FALLBACK_MAX_TOKENS_MULTIPLIER

    def _get_nemotron_client(self):
        """Create an OpenAI client pointed at the OpenRouter API.

        The API key is passed via the SDK and is never logged.
        """
        from openai import OpenAI

        return OpenAI(
            base_url=settings.OPENROUTER_LLM_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )

    def _build_nemotron_messages(self, prompt: str) -> list[dict]:
        """Build chat messages for Nemotron from the shared RAG prompt.

        Uses the exact same system instruction and user prompt that Gemini
        receives, so the fallback is grounded in the same retrieved context.
        """
        return [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]

    def _extract_nemotron_content(self, message) -> str:
        """Extract only the final answer content from a Nemotron message.

        Ignores any ``reasoning_content`` field (internal reasoning trace) and
        strips visible ``<think>`` blocks, so only clean answer text is used.
        """
        content = getattr(message, "content", None) or ""
        # Reasoning traces, when present as a separate field, are ignored:
        # getattr(message, "reasoning_content", None) is intentionally not used.
        return extract_answer_text(content)

    def _generate_nemotron(self, prompt: str) -> str:
        """Generate a completion using Nemotron 3 Ultra via OpenRouter (non-streaming fallback)."""
        client = self._get_nemotron_client()

        # OpenRouter's normalized reasoning parameter: exclude reasoning tokens
        # from the response ("low" effort keeps the reasoning spend small).
        # Any <think> blocks that still slip through are stripped later by
        # extract_answer_text().
        response = client.chat.completions.create(
            model=settings.OPENROUTER_LLM_MODEL,
            messages=self._build_nemotron_messages(prompt),
            temperature=self._temperature,
            max_tokens=self._fallback_max_tokens(),
            extra_body={"reasoning": {"effort": "low", "exclude": True}},
        )

        if not response.choices:
            raise RuntimeError("Nemotron returned no choices")

        answer = self._extract_nemotron_content(response.choices[0].message)

        if not answer or not answer.strip():
            raise RuntimeError("Nemotron returned an empty answer")

        return answer

    async def _stream_nemotron(self, prompt: str) -> AsyncIterator[str]:
        """Stream a completion from NVIDIA Nemotron (streaming fallback).

        The OpenAI SDK's SSE parsing handles the stream; each delta's content
        is yielded as a plain token, matching what the existing WebSocket
        flow expects from Gemini. ``<think>`` blocks, if any are still
        emitted, are buffered and filtered out instead of being forwarded.
        """
        client = self._get_nemotron_client()
        messages = self._build_nemotron_messages(prompt)
        model = settings.OPENROUTER_LLM_MODEL
        temperature = self._temperature
        max_tokens = self._fallback_max_tokens()

        def _iter_stream():
            # OpenRouter's normalized reasoning parameter: exclude reasoning
            # tokens from the stream (see _generate_nemotron). Any <think>
            # blocks that still slip through are filtered below.
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                extra_body={"reasoning": {"effort": "low", "exclude": True}},
            )

        # Run the blocking SDK stream setup on a worker thread so the event
        # loop (WebSocket) is not blocked, then iterate chunks off-thread too.
        # Uses the default executor; the chat REST endpoint pins its RAG work to
        # a dedicated executor via RAG_EXECUTOR in chat.py.
        stream = await asyncio.to_thread(_iter_stream)

        think_buffer: str = ""
        inside_think: bool = False
        emitted_any = False

        def _process(chunk) -> Optional[str]:
            """Turn one SSE chunk into user-facing text, filtering thinking."""
            nonlocal think_buffer, inside_think, emitted_any
            choices = getattr(chunk, "choices", None)
            if not choices:
                # Some providers send a trailing chunk with empty choices
                # (e.g. usage-only) — nothing to forward.
                return None
            delta = getattr(choices[0], "delta", None)
            text = getattr(delta, "content", None) if delta is not None else None
            if not text:
                return None

            # Some deployments still emit <think>...</think> in content even
            # with enable_thinking disabled; filter defensively.
            if inside_think:
                end_idx = text.find("</think>")
                if end_idx == -1:
                    think_buffer += text
                    return None
                think_buffer = ""
                inside_think = False
                text = text[end_idx + len("</think>"):]
                if not text:
                    return None

            combined = think_buffer + text
            think_buffer = ""

            # If a <think> block starts in this piece, hold back everything
            # from "<think>" onwards (we may only have a partial tag).
            start_idx = combined.find("<think>")
            if start_idx != -1:
                kept = combined[:start_idx]
                tail = combined[start_idx:]
                end_idx = tail.find("</think>")
                if end_idx == -1:
                    inside_think = True
                    # Keep the tail in case "</think>" arrives split across
                    # chunks; a partial closing tag is harmless to hold back.
                    think_buffer = tail
                    if kept:
                        emitted_any = True
                        return kept
                    return None
                inside_think = False
                combined = kept + tail[end_idx + len("</think>"):]

            if combined:
                emitted_any = True
                return combined
            return None

        try:
            import queue as _queue
            import threading

            q: _queue.Queue = _queue.Queue()
            SENTINEL = object()

            def _pump():
                try:
                    for chunk in stream:
                        q.put(chunk)
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(SENTINEL)

            pump_thread = threading.Thread(target=_pump, daemon=True)
            pump_thread.start()

            while True:
                item = await asyncio.to_thread(q.get)
                if item is SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                text = _process(item)
                if text:
                    yield text

            if not emitted_any:
                raise RuntimeError("Nemotron streaming returned no usable text")
        except Exception as e:
            logger.error(f"Nemotron streaming error: {e}")
            raise RuntimeError(f"Nemotron streaming failed: {e}") from e

        if think_buffer or inside_think:
            logger.warning("Nemotron stream ended inside an unterminated <think> block")


# Global instance
_generation_service: Optional[GenerationService] = None


def get_generation_service() -> GenerationService:
    """Get or create the global generation service instance."""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service
