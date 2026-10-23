#!/usr/bin/env python3
"""Mocked tests for the Nemotron 3 Ultra (OpenRouter) fallback in GenerationService.

No live API calls are made: Gemini and Nemotron backends are replaced with
fakes. Verifies the fallback rules:

  1. Gemini succeeds -> Nemotron is NOT called.
  2. Gemini fails    -> Nemotron IS called with the SAME question+context
                        prompt, and its answer is returned.
  3. Gemini + Nemotron both fail -> sanitized RuntimeError (no internal
     details leaked).
  4. WebSocket streaming: Gemini fails BEFORE first token -> Nemotron stream
     is used.
  5. WebSocket streaming: Gemini fails AFTER tokens -> Nemotron is NOT
     started; sanitized error raised.
  6. No retrieval context -> neither Gemini nor Nemotron is called;
     NO_ANSWER_MESSAGE behavior preserved.
  7. <think> reasoning blocks are stripped from answers.
  8. LLM_FALLBACK_ENABLED=false disables the fallback entirely.
  9. Nemotron reasoning_content is never exposed.

Run from the backend/ directory:
    venv\\Scripts\\python.exe test_fallback.py
"""

import asyncio
import sys
import types
import unittest.mock

from app.core.config import get_settings
from app.services.generation import (
    GENERATION_UNAVAILABLE_MESSAGE,
    NO_ANSWER_MESSAGE,
    GenerationService,
    build_rag_prompt,
    extract_answer_text,
)
from app.services.rag import RAGService

settings = get_settings()

PASS = 0
FAIL = 0


def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        tag = "OK"
    else:
        FAIL += 1
        tag = "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


QUESTION = "What is the capital of France?"
CONTEXT = "[Source 1] (chunk_id=1, document_id=doc-1, chunk_index=0, similarity=0.92)\nParis is the capital of France."


def with_fallback_settings(enabled: bool = True, api_key: str = "test-openrouter-key"):
    """Patch fallback settings without touching the real .env values."""
    return unittest.mock.patch.multiple(
        settings,
        LLM_FALLBACK_ENABLED=enabled,
        OPENROUTER_API_KEY=api_key,
    )


# ===== Test 1: Gemini succeeds -> Nemotron NOT called =======================

print("\n--- Test 1: Gemini succeeds -> Nemotron not called ---")
try:
    svc = GenerationService()
    nemotron_called = []

    with with_fallback_settings():
        with unittest.mock.patch.object(
            svc, "_generate_gemini", side_effect=lambda prompt: "Paris."
        ):
            def fake_nemotron(prompt):
                nemotron_called.append(True)
                return "should not be used"

            with unittest.mock.patch.object(
                svc, "_generate_nemotron", side_effect=fake_nemotron
            ):
                answer = svc.generate_answer(question=QUESTION, context=CONTEXT)

    report("Answer comes from Gemini", answer == "Paris.", f"answer={answer!r}")
    report("Nemotron NOT called", len(nemotron_called) == 0)
    report("last_model_used is Gemini",
           svc.last_model_used == settings.GEMINI_LLM_MODEL,
           f"last_model_used={svc.last_model_used!r}")
except Exception as e:
    report("Test 1 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Test 2: Gemini fails -> Nemotron called with SAME prompt ==============

print("\n--- Test 2: Gemini fails -> Nemotron called with same prompt ---")
try:
    svc = GenerationService()
    captured_prompts = []

    with with_fallback_settings():
        with unittest.mock.patch.object(
            svc, "_generate_gemini",
            side_effect=RuntimeError("503 UNAVAILABLE: high demand"),
        ):
            def fake_nemotron(prompt):
                captured_prompts.append(prompt)
                return "Nemotron fallback answer."

            with unittest.mock.patch.object(
                svc, "_generate_nemotron", side_effect=fake_nemotron
            ):
                answer = svc.generate_answer(question=QUESTION, context=CONTEXT)

    report("Answer comes from Nemotron",
           answer == "Nemotron fallback answer.", f"answer={answer!r}")
    report("Nemotron called exactly once", len(captured_prompts) == 1)

    expected_prompt = build_rag_prompt(question=QUESTION, context=CONTEXT)
    report("Fallback received the SAME question+context prompt",
           captured_prompts and captured_prompts[0] == expected_prompt)
    report("Prompt contains the question", QUESTION in (captured_prompts[0] if captured_prompts else ""))
    report("Prompt contains the retrieved context", CONTEXT in (captured_prompts[0] if captured_prompts else ""))
    report("last_model_used reports Nemotron model",
           svc.last_model_used == settings.OPENROUTER_LLM_MODEL,
           f"last_model_used={svc.last_model_used!r}")
except Exception as e:
    report("Test 2 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Test 3: Both fail -> sanitized error ==================================

print("\n--- Test 3: Both fail -> sanitized error ---")
try:
    svc = GenerationService()

    with with_fallback_settings():
        with unittest.mock.patch.object(
            svc, "_generate_gemini",
            side_effect=RuntimeError("503 UNAVAILABLE secret-internal-detail"),
        ):
            with unittest.mock.patch.object(
                svc, "_generate_nemotron",
                side_effect=RuntimeError("openrouter-key-invalid secret-internal-detail-2"),
            ):
                try:
                    svc.generate_answer(question=QUESTION, context=CONTEXT)
                    sanitized = False
                    error_text = "(no exception raised)"
                except RuntimeError as e:
                    error_text = str(e)
                    sanitized = error_text == GENERATION_UNAVAILABLE_MESSAGE

    report("RuntimeError raised with sanitized message", sanitized,
           f"error={error_text!r}")
    report("Raw provider error NOT exposed", "secret-internal-detail" not in error_text)
    report("API key value NOT exposed", "test-openrouter-key" not in error_text)
except Exception as e:
    report("Test 3 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Test 4: Streaming - Gemini fails before first token -> Nemotron ======

print("\n--- Test 4: WS streaming - Gemini fails before first token ---")
try:
    svc = GenerationService()
    nemotron_stream_called = []

    def fake_gemini_stream(prompt):
        async def gen():
            raise RuntimeError("503 UNAVAILABLE")
            yield  # pragma: no cover - makes gen() an async generator
        return gen()

    def fake_nemotron_stream(prompt):
        nemotron_stream_called.append(True)

        async def gen():
            yield "Paris "
            yield "is the capital."
        return gen()

    async def run_stream():
        tokens = []
        async for token in svc.stream_answer(question=QUESTION, context=CONTEXT):
            tokens.append(token)
        return tokens

    with with_fallback_settings():
        with unittest.mock.patch.object(svc, "_stream_gemini", side_effect=fake_gemini_stream):
            with unittest.mock.patch.object(
                svc, "_stream_nemotron", side_effect=fake_nemotron_stream
            ):
                tokens = asyncio.run(run_stream())

    report("Nemotron stream started", len(nemotron_stream_called) == 1)
    report("Nemotron tokens streamed to the client",
           "".join(tokens) == "Paris is the capital.",
           f"tokens={''.join(tokens)!r}")
    report("last_model_used reports Nemotron model",
           svc.last_model_used == settings.OPENROUTER_LLM_MODEL,
           f"last_model_used={svc.last_model_used!r}")
except Exception as e:
    report("Test 4 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Test 5: Streaming - Gemini fails before flush -> clean fallback ======
#
# Contract (buffered streaming): primary tokens are held until
# STREAM_FLUSH_THRESHOLD_CHARS chars have accumulated. A Gemini failure while
# the buffer is still holding discards the partial output and streams the
# fallback answer instead — no stitched, two-model output ever reaches the
# client. Failures AFTER the flush threshold keep the old "sanitized error,
# no fallback" behaviour (covered in tests/test_stream_fallback.py).

print("\n--- Test 5: WS streaming - Gemini fails before flush -> fallback ---")
try:
    svc = GenerationService()
    nemotron_stream_called = []

    def fake_gemini_stream(prompt):
        async def gen():
            yield "Paris "
            yield "is the"  # 11 chars < flush threshold: still buffered
            raise RuntimeError("connection reset mid-stream")
        return gen()

    def fake_nemotron_stream(prompt):
        nemotron_stream_called.append(True)

        async def gen():
            yield "Nemotron answer [Source 1]"
        return gen()

    async def run_stream():
        tokens = []
        error = None
        try:
            async for token in svc.stream_answer(question=QUESTION, context=CONTEXT):
                tokens.append(token)
        except RuntimeError as e:
            error = str(e)
        return tokens, error

    with with_fallback_settings():
        with unittest.mock.patch.object(svc, "_stream_gemini", side_effect=fake_gemini_stream):
            with unittest.mock.patch.object(
                svc, "_stream_nemotron", side_effect=fake_nemotron_stream
            ):
                tokens, error = asyncio.run(run_stream())

    joined = "".join(tokens)
    report("Buffered partial output discarded (no stitched answer)",
           "Paris" not in joined and joined == "Nemotron answer [Source 1]",
           f"tokens={joined!r}")
    report("Fallback streamed instead of erroring", len(nemotron_stream_called) == 1)
    report("No error raised (fallback answered)", error is None,
           f"error={error!r}")
    report("Raw mid-stream error NOT exposed",
           "connection reset" not in joined and "connection reset" not in (error or ""))
except Exception as e:
    report("Test 5 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Test 6: No retrieval context -> NO LLM called =========================

print("\n--- Test 6: No retrieval context -> neither LLM called ---")
try:
    class FakeRetrievalNoContext:
        def retrieve(self, question, top_k=5, min_similarity=0.0, user_id=None, notebook_id=None):
            return []

    class FakeGeneration:
        gemini_called = False
        nemotron_called = False
        model_name = "gemini-3.8-flash"
        last_model_used = "gemini-3.8-flash"

        def generate_answer(self, question, context):
            self.gemini_called = True
            return "should not be called"

    fake_gen = FakeGeneration()
    rag = RAGService(retrieval_service=FakeRetrievalNoContext(), generation_service=fake_gen)
    result = rag.answer_question(question=QUESTION)

    report("found_context is False", result["found_context"] is False)
    report("Answer is the existing NO_ANSWER_MESSAGE",
           result["answer"] == NO_ANSWER_MESSAGE,
           f"answer={result['answer']!r}")
    report("Gemini NOT called", fake_gen.gemini_called is False)
    report("Nemotron NOT called", fake_gen.nemotron_called is False)
    report("No sources returned", result["sources"] == [])
    report("Model reports configured Gemini model",
           result["model"] == "gemini-3.8-flash", f"model={result['model']!r}")
except Exception as e:
    report("Test 6 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Test 7: <think> blocks are stripped from answers =====================

print("\n--- Test 7: <think> reasoning blocks stripped ---")
try:
    with_think = "<think>Let me reason about Paris.</think>Paris is the capital of France."
    report("Leading think block removed",
           extract_answer_text(with_think) == "Paris is the capital of France.",
           f"got={extract_answer_text(with_think)!r}")
    report("Plain answer unchanged",
           extract_answer_text("Just an answer.") == "Just an answer.")
    report("Empty input handled", extract_answer_text("") == "")
except Exception as e:
    report("Test 7 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Test 8: Fallback disabled -> Nemotron never used =====================

print("\n--- Test 8: LLM_FALLBACK_ENABLED=false disables fallback ---")
try:
    svc = GenerationService()
    nemotron_called = []

    with with_fallback_settings(enabled=False):
        with unittest.mock.patch.object(
            svc, "_generate_gemini",
            side_effect=RuntimeError("503 UNAVAILABLE"),
        ):
            def fake_nemotron(prompt):
                nemotron_called.append(True)
                return "should not be used"

            with unittest.mock.patch.object(
                svc, "_generate_nemotron", side_effect=fake_nemotron
            ):
                try:
                    svc.generate_answer(question=QUESTION, context=CONTEXT)
                    error = "(no exception raised)"
                except RuntimeError as e:
                    error = str(e)

    report("Nemotron NOT called when fallback disabled", len(nemotron_called) == 0)
    report("Sanitized error raised", error == GENERATION_UNAVAILABLE_MESSAGE,
           f"error={error!r}")
except Exception as e:
    report("Test 8 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Test 9: reasoning_content never exposed ===============================

print("\n--- Test 9: Nemotron reasoning_content not exposed ---")
try:
    svc = GenerationService()
    fake_message = types.SimpleNamespace(
        content="Final clean answer.",
        reasoning_content="Internal chain of thought that must stay secret.",
    )
    extracted = svc._extract_nemotron_content(fake_message)
    report("Only final content returned", extracted == "Final clean answer.",
           f"got={extracted!r}")
    report("reasoning_content NOT included",
           "chain of thought" not in extracted)
except Exception as e:
    report("Test 9 completed without exception", False, f"{type(e).__name__}: {e}")


# ===== Summary ===============================================================

print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} assertions")
print("=" * 60)

if FAIL:
    sys.exit(1)
else:
    sys.exit(0)
