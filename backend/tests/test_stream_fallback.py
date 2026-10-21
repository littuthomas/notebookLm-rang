"""
Tests for the STREAMING fallback in GenerationService.stream_answer.

Behaviour under test (no network, no DB — provider streams are faked):

  * Primary fails BEFORE the flush threshold  -> partial output is discarded,
    the fallback streams a clean answer, and no stitched output reaches the
    client (fallback model reported via last_model_used).
  * Primary fails AFTER the flush threshold   -> fallback is NOT started
    (tokens already reached the client); a sanitized RuntimeError is raised.
  * Primary succeeds (even with a short answer) -> its buffered tokens are
    flushed at completion, fallback is never called, model = primary.
  * Fallback disabled or missing key          -> sanitized RuntimeError, no leak.
"""
import pytest

from app.services.generation import (
    GENERATION_UNAVAILABLE_MESSAGE,
    STREAM_FLUSH_THRESHOLD_CHARS,
    GenerationService,
)


# ------------------------------------------------------------------ fakes

def async_gen(items, raise_after=None, exc=None):
    """Async generator yielding ``items`` then optionally raising."""

    async def _gen():
        for item in items:
            yield item
        if raise_after is not None:
            raise exc or RuntimeError("simulated provider crash")

    return _gen()


class Fakes:
    """Per-instance stream fakes assigned onto a GenerationService."""

    def __init__(self, primary, fallback=None):
        self.primary = primary  # async generator instance factory
        self.fallback = fallback
        self.fallback_called = False

    async def _stream_gemini(self, prompt: str):
        async for token in self.primary:
            yield token
        # (raise_after handled by pre-raising generators below)

    async def _stream_nemotron(self, prompt: str):
        self.fallback_called = True
        for token in self.fallback or []:
            yield token


def make_service(primary, fallback=None):
    svc = GenerationService()
    fakes = Fakes(primary, fallback)
    svc._stream_gemini = fakes._stream_gemini
    svc._stream_nemotron = fakes._stream_nemotron
    return svc, fakes


async def collect(agen):
    out = []
    async for tok in agen:
        out.append(tok)
    return "".join(out)


# ---------------------------------------------------- fallback before flush

def test_primary_crash_before_first_token_falls_back():
    async def crash():
        raise RuntimeError("503 UNAVAILABLE - high demand")
        yield  # pragma: no cover - makes this an async generator

    svc, fakes = make_service(
        primary=crash(),
        fallback=["Nem", "otron ", "answer [Source 1]"],
    )
    out = asyncio_run(collect(svc.stream_answer("q", "context")))
    assert out == "Nemotron answer [Source 1]"  # clean, single-model output
    assert fakes.fallback_called is True
    assert svc.last_model_used == "nvidia/nemotron-3-ultra-550b-a55b"


def test_partial_primary_below_threshold_is_discarded_then_fallback():
    # 10 chars < 40-char threshold: nothing was flushed, so the partial
    # primary text must NOT appear in the client-visible output (no stitch).
    async def partial_crash():
        yield "Paris is "  # 9 chars
        yield "th"         # 11 chars total — still buffered
        raise RuntimeError("connection reset by peer")

    svc, fakes = make_service(
        primary=partial_crash(),
        fallback=["Clean ", "fallback ", "answer."],
    )
    out = asyncio_run(collect(svc.stream_answer("q", "context")))
    assert out == "Clean fallback answer."
    assert "Paris" not in out  # no stitched output from two models
    assert fakes.fallback_called is True
    assert svc.last_model_used == "nvidia/nemotron-3-ultra-550b-a55b"


def test_partial_primary_crossing_threshold_then_crash_no_fallback():
    # Enough text to flush (> threshold): tokens already reached the client,
    # so the fallback must NOT start — an error is raised instead.
    big = "x" * (STREAM_FLUSH_THRESHOLD_CHARS + 10)

    async def crash_after_flush():
        yield big
        raise RuntimeError("connection reset by peer")

    svc, fakes = make_service(primary=crash_after_flush(), fallback=["never"])
    with pytest.raises(RuntimeError) as ei:
        asyncio_run(collect(svc.stream_answer("q", "context")))
    assert str(ei.value) == GENERATION_UNAVAILABLE_MESSAGE
    assert fakes.fallback_called is False  # no stitched model output


def test_short_successful_primary_is_flushed_and_fallback_untouched():
    svc, fakes = make_service(primary=async_gen(["Paris", " [Source 1]"]), fallback=["nope"])
    out = asyncio_run(collect(svc.stream_answer("q", "context")))
    assert out == "Paris [Source 1]"  # buffered tokens flushed at completion
    assert fakes.fallback_called is False
    assert svc.last_model_used == svc._model  # primary model reported


def test_long_successful_primary_flushes_incrementally():
    tokens = ["y" * 15] * 6  # 90 chars total, crosses the 40-char threshold
    svc, fakes = make_service(primary=async_gen(tokens), fallback=["nope"])
    out = asyncio_run(collect(svc.stream_answer("q", "context")))
    assert out == "y" * 90
    assert fakes.fallback_called is False


# ------------------------------------------------- config / sanitization

def test_fallback_disabled_raises_sanitized(monkeypatch):
    from app.core.config import get_settings

    async def crash():
        raise RuntimeError("503 unavailable")
        yield

    svc, fakes = make_service(primary=crash(), fallback=["x"])
    monkeypatch.setattr(get_settings(), "LLM_FALLBACK_ENABLED", False)
    with pytest.raises(RuntimeError) as ei:
        asyncio_run(collect(svc.stream_answer("q", "context")))
    assert str(ei.value) == GENERATION_UNAVAILABLE_MESSAGE
    assert fakes.fallback_called is False


def test_empty_question_or_context_raises_value_error_without_providers():
    svc, fakes = make_service(primary=async_gen(["a"]), fallback=["b"])
    with pytest.raises(ValueError):
        asyncio_run(collect(svc.stream_answer("", "context")))
    with pytest.raises(ValueError):
        asyncio_run(collect(svc.stream_answer("q", "   ")))
    assert fakes.fallback_called is False


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
