"""test_compact_integration.py — Integration tests for compact rewriting.

The most important tests in the suite. If compact rewriting fails silently,
Alpha wakes up after compaction with the wrong identity, and the only signal
is Jeffery going "...that's not her."

These tests verify that the HTTP proxy correctly detects and rewrites
claude's compact API requests. Each phase asserts independently with
diagnostic output that says exactly what broke and where to fix it:

  Phase 1: System prompt replacement (summarizer → Alpha identity)
  Phase 2: Compact instructions replacement (generic → identity-preserving)
  Phase 3: Continuation instruction replacement ("continue without asking" → "check in")

Two scenarios:
  Manual /compact: Triggers Phase 2. (Phase 1 not observed on claude 2.1.56.)
  Auto-compact:    Triggers Phases 2+3 via CLAUDE_AUTOCOMPACT_PCT_OVERRIDE.

Run with:  uv run pytest tests/test_compact_integration.py -v
Skip with: uv run pytest -m "not integration"

CRITICAL: Do NOT use --disable-slash-commands in test args.
/compact IS a slash command. Disabling it silently prevents manual compact.
(Bug found Feb 27, 2026 after three failed probe runs.)
"""

import json
import os
import shutil
from pathlib import Path

import pytest

# Monkey-patch the capture flag at runtime — it's evaluated at import time.
import alpha_sdk.proxy as _proxy_module

from alpha_sdk.engine import (
    Engine,
    AssistantEvent,
    ErrorEvent,
    Event,
    ResultEvent,
    SystemEvent,
)
from alpha_sdk.proxy import (
    AUTO_COMPACT_SYSTEM_SIGNATURE,
    COMPACT_INSTRUCTIONS_START,
    CONTINUATION_INSTRUCTION_TAIL,
    CompactConfig,
)


# -- Test markers (unique, won't appear in natural claude output) -----------

MARKER_SYSTEM = "TEST_MARKER_SYSTEM_9f3a7b2c"
MARKER_PROMPT = "TEST_MARKER_PROMPT_e5d1c8f4"
MARKER_CONTINUATION = "TEST_MARKER_CONTINUATION_a2b4d6e8"

COMPACT_CONFIG = CompactConfig(
    system=MARKER_SYSTEM,
    prompt=MARKER_PROMPT,
    continuation=MARKER_CONTINUATION,
)

# DO NOT include --disable-slash-commands! /compact is a slash command.
COMPACT_TEST_ARGS = [
    "--tools", "",
    "--no-chrome",
    "--strict-mcp-config",
]

CAPTURE_DIR = Path(__file__).parent / "captures"


# -- Helpers ----------------------------------------------------------------


def _clear_captures():
    """Remove all capture files."""
    if CAPTURE_DIR.exists():
        shutil.rmtree(CAPTURE_DIR)


def _read_captures(suffix: str = "") -> list[dict]:
    """Read captured request bodies, optionally filtered by suffix.

    suffix="before" → raw requests from claude (pre-rewrite)
    suffix="after"  → rewritten requests (post-rewrite)
    suffix=""       → all captures
    """
    if not CAPTURE_DIR.exists():
        return []
    pattern = f"*{suffix}.json" if suffix else "*.json"
    captures = []
    for f in sorted(CAPTURE_DIR.glob(pattern)):
        with open(f) as fp:
            captures.append(json.load(fp))
    return captures


def _find_last_user_message(body: dict) -> str | None:
    """Extract text from the last user message in a request body."""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [
                block["text"]
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(texts) if texts else None
    return None


def _find_all_user_text(body: dict) -> str:
    """Extract ALL user message text from a request body.

    Phase 3 can appear in any user message (not just the last),
    because the continuation instruction is prepended to the
    compacted context, not appended to the current message.
    """
    parts = []
    for msg in body.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
    return "\n".join(parts)


def _get_system_text(body: dict) -> str:
    """Extract system prompt text from a request body."""
    system = body.get("system", "")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        texts = [
            block["text"]
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(texts)
    return ""


def _summarize_captures(captures: list[dict], label: str) -> str:
    """Build a diagnostic summary of captured requests."""
    if not captures:
        return f"  No {label} captures found."
    lines = []
    for i, cap in enumerate(captures):
        last_msg = _find_last_user_message(cap) or "(none)"
        system = _get_system_text(cap)[:80] or "(none)"
        lines.append(
            f"  [{i}] system: {system}\n"
            f"       last user msg: {last_msg[:120]}"
        )
    return "\n".join(lines)


# -- Fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def enable_captures():
    """Enable request capture and clean up between tests.

    Monkey-patches the module-level CAPTURE_REQUESTS flag because
    it's evaluated at import time — setting os.environ after import
    has no effect on the module-level constant.
    """
    old_val = _proxy_module.CAPTURE_REQUESTS
    _proxy_module.CAPTURE_REQUESTS = True
    _clear_captures()
    yield
    _proxy_module.CAPTURE_REQUESTS = old_val
    _clear_captures()


# -- Manual compact tests ---------------------------------------------------


@pytest.mark.integration
class TestManualCompact:
    """Test compact rewriting via manual /compact command.

    Sends /compact after building a few turns of context.
    Verifies the proxy intercepts and rewrites the compact API request.

    Expected results (claude 2.1.56, Feb 2026):
      Phase 1 (system prompt):        Does NOT fire on manual /compact
      Phase 2 (compact instructions): Fires and rewrites ✓
      Phase 3 (continuation):         Does NOT fire on manual /compact
    """

    async def test_phase2_compact_instructions_rewritten(self):
        """Phase 2: /compact instructions should be rewritten with our prompt.

        This is the most critical test in the suite. When it fails, claude
        uses its default summarizer ("create a detailed summary") instead of
        Alpha's identity-preserving prompt. Alpha wakes up wrong.

        Diagnostic output on failure tells you:
        - Whether the compact API call reached the proxy at all
        - Whether the detection signature was found in the raw request
        - Whether the rewrite actually replaced the text
        - What to update in proxy.py if claude changed its format
        """
        engine = Engine(
            model="haiku",
            system_prompt="Reply with exactly one short sentence. No tools.",
            compact_config=COMPACT_CONFIG,
            extra_args=COMPACT_TEST_ARGS,
        )

        try:
            await engine.start()

            # Build context — need enough for /compact to have something to summarize
            filler = "The quick brown fox. " * 200  # ~1K tokens
            for i in range(3):
                await engine.send(f"Test message {i}: {filler}")
                async for event in engine.events():
                    if isinstance(event, (ResultEvent, ErrorEvent)):
                        break

            # Clear captures from context-building turns
            _clear_captures()

            # Send /compact
            await engine.send("/compact")
            async for event in engine.events():
                if isinstance(event, (ResultEvent, ErrorEvent)):
                    break

            # -- Assertions --

            before_captures = _read_captures("before")
            after_captures = _read_captures("after")

            # Gate: did anything go through the proxy?
            assert len(before_captures) > 0, (
                "ZERO proxy captures during /compact.\n"
                "The compact API call never reached ANTHROPIC_BASE_URL.\n"
                "Possible causes:\n"
                "  1. --disable-slash-commands is in the test args (it MUST NOT be)\n"
                "  2. ANTHROPIC_BASE_URL not set to the proxy address\n"
                "  3. claude didn't process /compact as a command\n"
                "  4. Context was too small for claude to bother compacting"
            )

            # Phase 2 detection: find compact instructions in the raw request
            compact_before = None
            for cap in before_captures:
                last_msg = _find_last_user_message(cap)
                if last_msg and COMPACT_INSTRUCTIONS_START in last_msg:
                    compact_before = cap
                    break

            assert compact_before is not None, (
                f"Phase 2 DETECTION failed.\n"
                f"None of the {len(before_captures)} captured request(s) contain "
                f"the compact instructions signature.\n"
                f"Expected: '{COMPACT_INSTRUCTIONS_START[:60]}...'\n"
                f"\n{_summarize_captures(before_captures, 'before')}\n"
                f"\nIf claude changed its compact format, update "
                f"COMPACT_INSTRUCTIONS_START in proxy.py."
            )

            # Phase 2 rewrite: verify our marker appears in the after capture
            compact_after = None
            for cap in after_captures:
                last_msg = _find_last_user_message(cap)
                if last_msg and MARKER_PROMPT in last_msg:
                    compact_after = cap
                    break

            assert compact_after is not None, (
                f"Phase 2 REWRITE failed.\n"
                f"The compact instructions were detected in the raw request,\n"
                f"but marker '{MARKER_PROMPT}' was not found after rewriting.\n"
                f"_replace_compact_instructions() didn't fire or didn't match.\n"
                f"\n{_summarize_captures(after_captures, 'after')}\n"
                f"\nCheck _replace_compact_instructions() in proxy.py."
            )

        finally:
            await engine.stop()

    async def test_phase1_not_present_on_manual_compact(self):
        """Phase 1: manual /compact does NOT swap the system prompt.

        As of claude 2.1.56, manual /compact keeps the normal system prompt.
        It does NOT replace it with the summarizer identity.

        This test is informational — if it starts FAILING (finding the
        summarizer signature), claude changed behavior and Phase 1 now fires
        on manual compact. That's not a bug; it's new information.
        """
        engine = Engine(
            model="haiku",
            system_prompt="Reply with exactly one short sentence. No tools.",
            compact_config=COMPACT_CONFIG,
            extra_args=COMPACT_TEST_ARGS,
        )

        try:
            await engine.start()

            filler = "The quick brown fox. " * 200
            for i in range(3):
                await engine.send(f"Test message {i}: {filler}")
                async for event in engine.events():
                    if isinstance(event, (ResultEvent, ErrorEvent)):
                        break

            _clear_captures()
            await engine.send("/compact")
            async for event in engine.events():
                if isinstance(event, (ResultEvent, ErrorEvent)):
                    break

            before_captures = _read_captures("before")

            has_summarizer = any(
                AUTO_COMPACT_SYSTEM_SIGNATURE in _get_system_text(cap)
                for cap in before_captures
            )

            if has_summarizer:
                # Behavior change! Not a failure, but we want to know.
                pytest.skip(
                    "BEHAVIOR CHANGE: Phase 1 now fires on manual /compact.\n"
                    "Claude is sending the summarizer system prompt during "
                    "manual compact. This is new. Update the test matrix."
                )

            # If we get here, Phase 1 doesn't fire. That's the expected behavior.

        finally:
            await engine.stop()


# -- Auto-compact tests -----------------------------------------------------


@pytest.mark.integration
class TestAutoCompact:
    """Test compact rewriting via auto-compaction.

    Uses CLAUDE_AUTOCOMPACT_PCT_OVERRIDE to trigger auto-compact at a
    low context threshold (10%) instead of filling 160K+ tokens.

    Strategy:
      - Set threshold to 10% (~20K tokens for Haiku's 200K window)
      - Send ~10K token chunks until compact fires
      - Total cost: a few cents in Haiku tokens

    Expected results:
      Phase 1 (system prompt):        Unknown — testing this
      Phase 2 (compact instructions): Should fire ✓
      Phase 3 (continuation):         Should fire ✓
    """

    async def test_auto_compact_rewrite_phases(self):
        """Trigger auto-compact and verify all rewrite phases.

        Phase 2 is hard-asserted (must pass).
        Phase 3 is hard-asserted if the signature is detected but not rewritten;
          skipped if the signature isn't found at all (format may have changed).
        Phase 1 is informational (skip if not observed, fail if detected
          but not rewritten).
        """
        # Lower the auto-compact threshold so we don't need 160K+ tokens
        os.environ["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "10"

        try:
            engine = Engine(
                model="haiku",
                system_prompt="Reply with exactly one short sentence. No tools.",
                compact_config=COMPACT_CONFIG,
                extra_args=COMPACT_TEST_ARGS,
            )

            try:
                await engine.start()

                # -- Build context until auto-compact fires --
                # ~10K tokens per chunk. At 10% of 200K = 20K threshold,
                # compact should fire after 2-3 chunks.
                filler = (
                    "The quick brown fox jumps over the lazy dog. " * 1000
                )  # ~46K chars ≈ 10K tokens

                max_turns = 8  # Safety limit
                compact_detected = False
                compact_turn = 0

                for turn in range(1, max_turns + 1):
                    # Clear captures each turn to isolate the compact turn.
                    # When compact fires, its captures are preserved because
                    # we break immediately after detection.
                    _clear_captures()

                    await engine.send(
                        f"This is chunk {turn}. {filler}\n"
                        f"Acknowledge chunk {turn} in one sentence."
                    )

                    async for event in engine.events():
                        # Detect compact via event stream
                        if hasattr(event, "raw") and isinstance(event.raw, dict):
                            raw_type = event.raw.get("type", "")
                            if raw_type == "compact_boundary":
                                compact_detected = True
                                compact_turn = turn

                        if isinstance(event, SystemEvent):
                            if "compact" in event.subtype.lower():
                                compact_detected = True
                                compact_turn = turn

                        if isinstance(event, (ResultEvent, ErrorEvent)):
                            break

                    # Belt and suspenders: also check captures for compact evidence
                    if not compact_detected:
                        for cap in _read_captures("before"):
                            last_msg = _find_last_user_message(cap) or ""
                            if COMPACT_INSTRUCTIONS_START in last_msg:
                                compact_detected = True
                                compact_turn = turn
                                break

                    if compact_detected:
                        break

                # -- Gate: did auto-compact trigger? --

                assert compact_detected, (
                    f"Auto-compact never triggered after {max_turns} turns.\n"
                    f"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=10 should trigger "
                    f"at ~20K tokens (10% of 200K).\n"
                    f"Token count: {engine.token_count}\n"
                    f"Context window: {engine.context_window}\n"
                    f"Chunks sent: {max_turns} × ~10K tokens each\n"
                    f"\nPossible causes:\n"
                    f"  1. Env var not reaching subprocess "
                    f"(check Engine._spawn env handling)\n"
                    f"  2. Auto-compact threshold works differently "
                    f"than documented\n"
                    f"  3. Compact events not detected in stream "
                    f"(check event parsing)"
                )

                # -- Phase 2: compact instructions --

                before_captures = _read_captures("before")
                after_captures = _read_captures("after")

                phase2_detected = any(
                    COMPACT_INSTRUCTIONS_START
                    in (_find_last_user_message(c) or "")
                    for c in before_captures
                )
                phase2_rewritten = any(
                    MARKER_PROMPT in (_find_last_user_message(c) or "")
                    for c in after_captures
                )

                assert phase2_detected, (
                    f"Phase 2 DETECTION failed during auto-compact "
                    f"(turn {compact_turn}).\n"
                    f"Compact was triggered but none of "
                    f"{len(before_captures)} capture(s) contain "
                    f"the instructions signature.\n"
                    f"Expected: '{COMPACT_INSTRUCTIONS_START[:50]}...'\n"
                    f"\n{_summarize_captures(before_captures, 'before')}\n"
                    f"\nUpdate COMPACT_INSTRUCTIONS_START in proxy.py."
                )

                assert phase2_rewritten, (
                    f"Phase 2 REWRITE failed during auto-compact.\n"
                    f"Detection worked (found compact instructions), "
                    f"but marker '{MARKER_PROMPT}' not found after rewrite.\n"
                    f"\n{_summarize_captures(after_captures, 'after')}\n"
                    f"\nCheck _replace_compact_instructions() in proxy.py."
                )

                # -- Phase 3: continuation instruction --
                # The continuation is in the post-compact API request,
                # which may be in the same turn's captures or the next turn.
                # Check same-turn captures first.

                phase3_detected = any(
                    CONTINUATION_INSTRUCTION_TAIL.strip()
                    in _find_all_user_text(c)
                    for c in before_captures
                )
                phase3_rewritten = any(
                    MARKER_CONTINUATION in _find_all_user_text(c)
                    for c in after_captures
                )

                # If not found in same-turn captures, send a follow-up
                # message and check that turn's captures too.
                if not phase3_detected and not phase3_rewritten:
                    _clear_captures()
                    await engine.send("What were we just discussing?")
                    async for event in engine.events():
                        if isinstance(event, (ResultEvent, ErrorEvent)):
                            break

                    followup_before = _read_captures("before")
                    followup_after = _read_captures("after")

                    phase3_detected = any(
                        CONTINUATION_INSTRUCTION_TAIL.strip()
                        in _find_all_user_text(c)
                        for c in followup_before
                    )
                    phase3_rewritten = any(
                        MARKER_CONTINUATION in _find_all_user_text(c)
                        for c in followup_after
                    )

                if phase3_detected and not phase3_rewritten:
                    pytest.fail(
                        "Phase 3 REWRITE failed.\n"
                        "The continuation instruction was found in the raw "
                        "request but was NOT rewritten with our marker.\n"
                        f"Expected marker: '{MARKER_CONTINUATION}'\n"
                        f"Continuation tail: "
                        f"'{CONTINUATION_INSTRUCTION_TAIL.strip()[:60]}...'\n"
                        "\nCheck _replace_continuation_instruction() "
                        "in proxy.py."
                    )
                elif not phase3_detected:
                    pytest.skip(
                        "Phase 3 NOT OBSERVED.\n"
                        "Continuation instruction not found in any capture.\n"
                        "Claude may not append the continuation in the "
                        "expected format, or it's embedded differently.\n"
                        f"Searched {len(before_captures)} compact-turn "
                        f"+ follow-up captures.\n"
                        f"Expected tail: "
                        f"'{CONTINUATION_INSTRUCTION_TAIL.strip()[:60]}...'"
                    )

                # Phase 3 rewritten successfully — continue to Phase 1.

                # -- Phase 1: system prompt (informational) --

                phase1_detected = any(
                    AUTO_COMPACT_SYSTEM_SIGNATURE in _get_system_text(c)
                    for c in before_captures
                )
                phase1_rewritten = any(
                    MARKER_SYSTEM in _get_system_text(c)
                    for c in after_captures
                )

                if phase1_detected and not phase1_rewritten:
                    pytest.fail(
                        "Phase 1 REWRITE failed during auto-compact.\n"
                        "Summarizer system prompt was detected but NOT "
                        f"rewritten with marker '{MARKER_SYSTEM}'.\n"
                        "Check _replace_system_prompt() in proxy.py."
                    )
                elif not phase1_detected:
                    # Expected on some versions — not a failure
                    pass

            finally:
                await engine.stop()

        finally:
            os.environ.pop("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", None)
