import asyncio
import logging
import re
from typing import AsyncIterator, Generator, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Returned verbatim (without calling the LLM) when retrieval finds no usable context,
# and also instructed to the LLM when the context does not contain the answer.
NO_ANSWER_MESSAGE = (
    "I could not find the answer to your question in the available documents."
)

SYSTEM_INSTRUCTION = (
    "You are a retrieval-augmented assistant. Answer the user's question using ONLY "
    "the provided context.\n"
    "Rules:\n"
    "1. Base every statement strictly on the provided context.\n"
    "2. Do not use outside or internal knowledge, and never invent facts, names or numbers.\n"
    "3. Cite the sources you rely on inline using their labels, e.g. [Source 1].\n"
    "4. If the context does not contain the answer, reply exactly with: "
    f"\"{NO_ANSWER_MESSAGE}\"\n"
    "5. Be concise and answer directly."
)

# Generic, user-safe error message. Raw provider errors, API keys, and stack
# traces are only logged server-side, never surfaced through this message.
GENERATION_UNAVAILABLE_MESSAGE = (
    "The answer generation service is temporarily unavailable. Please try again later."
)

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
    Fallback: NVIDIA Nemotron (OpenAI-compatible API), used ONLY when the Gemini
    call fails before producing a complete answer. Embeddings, retrieval and the
    RAG prompt are untouched by this fallback.

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
        NVIDIA Nemotron (same question, same context, same prompt). If both
        fail, raises a sanitized RuntimeError — raw provider errors never reach
        the caller/user.

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

        if not settings.NVIDIA_API_KEY:
            logger.error("Fallback unavailable: NVIDIA_API_KEY is not configured")
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE)

        try:
            answer = self._generate_nemotron(prompt)
            self.last_model_used = settings.NVIDIA_LLM_MODEL
            return answer
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Fallback LLM (Nemotron) generation also failed: {e}")
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE) from e

    async def stream_answer(self, question: str, context: str) -> AsyncIterator[str]:
        """
        Stream an answer to ``question`` grounded in ``context``.

        Tries Gemini streaming first (primary). If Gemini fails BEFORE the
        first token is yielded, streams the fallback NVIDIA Nemotron response
        instead (same question, same context, same prompt). If Gemini fails
        after it has already yielded tokens, the error is raised as before —
        the fallback is never started mid-answer.

        Args:
            question: The user's question.
            context: A pre-built context string of retrieved chunks.

        Yields:
            Answer text tokens progressively as they arrive from the LLM.

        Raises:
            ValueError: If the question or context is empty.
            RuntimeError: If streaming fails from both providers (before any
                token was yielded), or mid-stream after tokens were yielded.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        if not context or not context.strip():
            raise ValueError("Context cannot be empty")

        prompt = build_rag_prompt(question=question, context=context)

        yielded_any = False
        try:
            async for token in self._stream_gemini(prompt):
                yielded_any = True
                yield token
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
            # Gemini failed before the first token (e.g. 503 at request time).
            logger.error(
                f"Primary LLM (Gemini) streaming failed before any token, "
                f"attempting fallback: {e}"
            )

        if not settings.LLM_FALLBACK_ENABLED:
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE)

        if not settings.NVIDIA_API_KEY:
            logger.error("Fallback unavailable: NVIDIA_API_KEY is not configured")
            raise RuntimeError(GENERATION_UNAVAILABLE_MESSAGE)

        try:
            async for token in self._stream_nemotron(prompt):
                yield token
            self.last_model_used = settings.NVIDIA_LLM_MODEL
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
        """Generate a completion using the Gemini API (non-streaming)."""
        from google import genai
        from google.genai import types

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured")

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        # Typed config with AFC disabled: this pipeline never passes tools, so
        # automatic function calling is dead weight. With it disabled the SDK
        # takes the direct request path and skips its AFC wrapper, which logs
        # a "Direct use of automatic function calling" warning.
        response = client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

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
        response = client.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

        yielded_any = False
        try:
            for chunk in response:
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise RuntimeError(f"Gemini streaming failed: {e}") from e

        if not yielded_any:
            raise RuntimeError("Gemini streaming returned no usable text")

    # ------------------------------------------------------------------
    # Fallback: NVIDIA Nemotron (OpenAI-compatible API)
    # ------------------------------------------------------------------

    def _get_nemotron_client(self):
        """Create an OpenAI client pointed at the NVIDIA API.

        The API key is passed via the SDK and is never logged.
        """
        from openai import OpenAI

        return OpenAI(
            base_url=settings.NVIDIA_LLM_BASE_URL,
            api_key=settings.NVIDIA_API_KEY,
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
        """Generate a completion using NVIDIA Nemotron (non-streaming fallback)."""
        client = self._get_nemotron_client()

        # Nemotron 3 Ultra supports disabling visible thinking via the chat
        # template (documented option) so we receive a clean final answer.
        response = client.chat.completions.create(
            model=settings.NVIDIA_LLM_MODEL,
            messages=self._build_nemotron_messages(prompt),
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
        model = settings.NVIDIA_LLM_MODEL
        temperature = self._temperature
        max_tokens = self._max_output_tokens

        def _iter_stream():
            # Nemotron 3 Ultra: disable visible thinking via the documented
            # chat-template option so the stream contains only answer content.
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        # Run the blocking SDK stream setup on a worker thread so the event
        # loop (WebSocket) is not blocked, then iterate chunks off-thread too.
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
