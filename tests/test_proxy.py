"""test_proxy.py — Unit tests for compact prompt rewriting.

Tests the pure rewrite functions that detect and replace claude's
default compact ceremony. No HTTP server, no subprocess — just
dict manipulation.
"""

import copy

import pytest

from alpha_sdk.proxy import (
    CompactConfig,
    rewrite_compact,
    _has_summarizer_system,
    _extract_additional_instructions,
    _replace_system_prompt,
    _replace_compact_instructions,
    _replace_continuation_instruction,
    AUTO_COMPACT_SYSTEM_SIGNATURE,
    COMPACT_INSTRUCTIONS_START,
    CONTINUATION_INSTRUCTION_TAIL,
)


# -- Test fixtures -----------------------------------------------------------


@pytest.fixture
def compact_config():
    """A test CompactConfig with recognizable replacement strings."""
    return CompactConfig(
        system="ALPHA COMPACT SYSTEM",
        prompt="ALPHA COMPACT PROMPT",
        continuation="ALPHA CONTINUATION",
    )


@pytest.fixture
def summarizer_system_string():
    """System prompt as a plain string (claude's format)."""
    return f"{AUTO_COMPACT_SYSTEM_SIGNATURE}. Please create a summary."


@pytest.fixture
def summarizer_system_blocks():
    """System prompt as content blocks (claude's format)."""
    return [
        {"type": "text", "text": "SDK preamble text here."},
        {"type": "text", "text": f"{AUTO_COMPACT_SYSTEM_SIGNATURE}. Please summarize."},
    ]


@pytest.fixture
def compact_instructions_body():
    """A request body with compact instructions in the last user message."""
    return {
        "messages": [
            {"role": "assistant", "content": "I was helping with something."},
            {
                "role": "user",
                "content": f"Previous context here.\n\n{COMPACT_INSTRUCTIONS_START} of this conversation.",
            },
        ]
    }


@pytest.fixture
def continuation_body():
    """A request body with the continuation instruction."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Summary of what happened.{CONTINUATION_INSTRUCTION_TAIL}",
            },
        ]
    }


# -- Phase 1: Summarizer system prompt detection ----------------------------


class TestHasSummarizerSystem:
    """Detect the auto-compact summarizer system prompt."""

    def test_string_with_signature(self, summarizer_system_string):
        assert _has_summarizer_system(summarizer_system_string) is True

    def test_string_without_signature(self):
        assert _has_summarizer_system("You are Alpha.") is False

    def test_blocks_with_signature(self, summarizer_system_blocks):
        assert _has_summarizer_system(summarizer_system_blocks) is True

    def test_blocks_without_signature(self):
        blocks = [{"type": "text", "text": "You are Alpha."}]
        assert _has_summarizer_system(blocks) is False

    def test_empty_list(self):
        assert _has_summarizer_system([]) is False

    def test_empty_string(self):
        assert _has_summarizer_system("") is False


class TestReplaceSystemPrompt:
    """Phase 1: Replace the summarizer system prompt."""

    def test_string_system(self, summarizer_system_string, compact_config):
        body = {"system": summarizer_system_string}
        result = _replace_system_prompt(body, compact_config.system)
        assert result is True
        assert body["system"] == [{"type": "text", "text": "ALPHA COMPACT SYSTEM"}]

    def test_blocks_system_preserves_preamble(self, summarizer_system_blocks, compact_config):
        body = {"system": summarizer_system_blocks}
        result = _replace_system_prompt(body, compact_config.system)
        assert result is True
        # Preamble preserved
        assert body["system"][0]["text"] == "SDK preamble text here."
        # Summarizer replaced
        assert body["system"][1]["text"] == "ALPHA COMPACT SYSTEM"

    def test_no_match_returns_false(self, compact_config):
        body = {"system": "You are Alpha."}
        result = _replace_system_prompt(body, compact_config.system)
        assert result is False
        assert body["system"] == "You are Alpha."

    def test_missing_system_key(self, compact_config):
        body = {"messages": []}
        result = _replace_system_prompt(body, compact_config.system)
        assert result is False

    def test_only_first_match_replaced(self, compact_config):
        """If somehow there are two summarizer blocks, only the first is replaced."""
        body = {
            "system": [
                {"type": "text", "text": f"{AUTO_COMPACT_SYSTEM_SIGNATURE} first"},
                {"type": "text", "text": f"{AUTO_COMPACT_SYSTEM_SIGNATURE} second"},
            ]
        }
        _replace_system_prompt(body, compact_config.system)
        assert body["system"][0]["text"] == "ALPHA COMPACT SYSTEM"
        assert AUTO_COMPACT_SYSTEM_SIGNATURE in body["system"][1]["text"]


# -- Phase 2: Compact instructions ------------------------------------------


class TestExtractAdditionalInstructions:
    """Extract /compact arguments from the compact text."""

    def test_with_instructions(self):
        text = (
            f"{COMPACT_INSTRUCTIONS_START} ...\n\n"
            "Additional Instructions:\n"
            "Focus on the SDK work\n\n"
            "IMPORTANT: Do NOT use any tools"
        )
        result = _extract_additional_instructions(text)
        assert result == "Focus on the SDK work"

    def test_without_instructions(self):
        text = f"{COMPACT_INSTRUCTIONS_START} of the conversation."
        result = _extract_additional_instructions(text)
        assert result is None

    def test_empty_instructions(self):
        text = (
            f"{COMPACT_INSTRUCTIONS_START} ...\n\n"
            "Additional Instructions:\n"
            "\n"
            "IMPORTANT: Do NOT use any tools"
        )
        result = _extract_additional_instructions(text)
        assert result is None

    def test_instructions_without_important_footer(self):
        text = (
            f"{COMPACT_INSTRUCTIONS_START} ...\n\n"
            "Additional Instructions:\n"
            "Focus on the SDK work"
        )
        result = _extract_additional_instructions(text)
        assert result == "Focus on the SDK work"


class TestReplaceCompactInstructions:
    """Phase 2: Replace compact instructions in the last user message."""

    def test_string_content(self, compact_instructions_body, compact_config):
        result = _replace_compact_instructions(
            compact_instructions_body, compact_config.prompt
        )
        assert result is True
        content = compact_instructions_body["messages"][-1]["content"]
        assert "ALPHA COMPACT PROMPT" in content
        assert COMPACT_INSTRUCTIONS_START not in content
        # Original context preserved
        assert "Previous context here." in content

    def test_block_content(self, compact_config):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Some preamble."},
                        {
                            "type": "text",
                            "text": f"Context.\n\n{COMPACT_INSTRUCTIONS_START} of this convo.",
                        },
                    ],
                }
            ]
        }
        result = _replace_compact_instructions(body, compact_config.prompt)
        assert result is True
        text_block = body["messages"][0]["content"][1]["text"]
        assert "ALPHA COMPACT PROMPT" in text_block
        assert COMPACT_INSTRUCTIONS_START not in text_block

    def test_preserves_additional_instructions(self, compact_config):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Context.\n\n{COMPACT_INSTRUCTIONS_START} ...\n\n"
                        "Additional Instructions:\n"
                        "Focus on the SDK\n\n"
                        "IMPORTANT: Do NOT use any tools"
                    ),
                }
            ]
        }
        _replace_compact_instructions(body, compact_config.prompt)
        content = body["messages"][0]["content"]
        assert "ALPHA COMPACT PROMPT" in content
        assert "Focus on the SDK" in content

    def test_no_match_returns_false(self, compact_config):
        body = {"messages": [{"role": "user", "content": "Normal message."}]}
        result = _replace_compact_instructions(body, compact_config.prompt)
        assert result is False

    def test_scans_from_last_user_message(self, compact_config):
        """Only the LAST user message is checked."""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Old message with {COMPACT_INSTRUCTIONS_START}",
                },
                {"role": "assistant", "content": "Response."},
                {"role": "user", "content": "Final message without compact."},
            ]
        }
        result = _replace_compact_instructions(body, compact_config.prompt)
        assert result is False  # Only checks last user message

    def test_empty_messages(self, compact_config):
        body = {"messages": []}
        result = _replace_compact_instructions(body, compact_config.prompt)
        assert result is False

    def test_no_messages_key(self, compact_config):
        body = {}
        result = _replace_compact_instructions(body, compact_config.prompt)
        assert result is False


# -- Phase 3: Continuation instruction --------------------------------------


class TestReplaceContinuationInstruction:
    """Phase 3: Replace the post-compact continuation instruction."""

    def test_string_content(self, continuation_body, compact_config):
        result = _replace_continuation_instruction(
            continuation_body, compact_config.continuation
        )
        assert result is True
        content = continuation_body["messages"][0]["content"]
        assert "ALPHA CONTINUATION" in content
        assert CONTINUATION_INSTRUCTION_TAIL not in content
        # Original text preserved
        assert "Summary of what happened." in content

    def test_block_content(self, compact_config):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Summary.{CONTINUATION_INSTRUCTION_TAIL}",
                        }
                    ],
                }
            ]
        }
        result = _replace_continuation_instruction(body, compact_config.continuation)
        assert result is True
        text = body["messages"][0]["content"][0]["text"]
        assert "ALPHA CONTINUATION" in text

    def test_no_match(self, compact_config):
        body = {"messages": [{"role": "user", "content": "Normal message."}]}
        result = _replace_continuation_instruction(body, compact_config.continuation)
        assert result is False

    def test_skips_assistant_messages(self, compact_config):
        """Continuation is only in user messages."""
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Text.{CONTINUATION_INSTRUCTION_TAIL}",
                }
            ]
        }
        result = _replace_continuation_instruction(body, compact_config.continuation)
        assert result is False

    def test_multiple_user_messages(self, compact_config):
        """Continuation can appear in any user message (not just last)."""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": f"First.{CONTINUATION_INSTRUCTION_TAIL}",
                },
                {"role": "assistant", "content": "Response."},
                {"role": "user", "content": "Second message. No continuation."},
            ]
        }
        result = _replace_continuation_instruction(body, compact_config.continuation)
        assert result is True
        assert "ALPHA CONTINUATION" in body["messages"][0]["content"]


# -- Full rewrite (all three phases) ----------------------------------------


class TestRewriteCompact:
    """Test the combined three-phase rewrite."""

    def test_compact_request_phases_1_and_2(self, compact_config):
        """The compact API call has phases 1 (system) and 2 (instructions)."""
        body = {
            "system": f"{AUTO_COMPACT_SYSTEM_SIGNATURE}. Summarize this.",
            "messages": [
                {"role": "assistant", "content": "I was helping."},
                {
                    "role": "user",
                    "content": f"Context.\n\n{COMPACT_INSTRUCTIONS_START} of the conversation.",
                },
            ],
        }

        result = rewrite_compact(body, compact_config)
        assert result is True

        # Phase 1: system prompt replaced
        assert body["system"][0]["text"] == "ALPHA COMPACT SYSTEM"

        # Phase 2: compact instructions replaced
        assert "ALPHA COMPACT PROMPT" in body["messages"][1]["content"]

    def test_post_compact_request_phase_3(self, compact_config):
        """The post-compact turn has phase 3 (continuation instruction)."""
        body = {
            "system": "You are Alpha.",
            "messages": [
                {
                    "role": "user",
                    "content": f"Compact summary.{CONTINUATION_INSTRUCTION_TAIL}",
                }
            ],
        }

        result = rewrite_compact(body, compact_config)
        assert result is True

        # Phase 1 did NOT trigger (normal system prompt)
        assert body["system"] == "You are Alpha."

        # Phase 3: continuation replaced
        assert "ALPHA CONTINUATION" in body["messages"][0]["content"]

    def test_normal_request_untouched(self, compact_config):
        """A normal (non-compact) request is not modified."""
        body = {
            "system": "You are Alpha.",
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }
        original = copy.deepcopy(body)
        result = rewrite_compact(body, compact_config)
        assert result is False
        assert body == original

    def test_partial_match(self, compact_config):
        """Only matching phases trigger."""
        body = {
            "system": "Normal system prompt.",
            "messages": [
                {
                    "role": "user",
                    "content": f"Summary.{CONTINUATION_INSTRUCTION_TAIL}",
                }
            ],
        }
        result = rewrite_compact(body, compact_config)
        assert result is True  # Phase 3 triggered
        assert body["system"] == "Normal system prompt."  # Phase 1 didn't trigger
        assert "ALPHA CONTINUATION" in body["messages"][0]["content"]


# -- CompactConfig -----------------------------------------------------------


class TestCompactConfig:
    """Test the configuration dataclass."""

    def test_creation(self):
        config = CompactConfig(
            system="sys", prompt="prompt", continuation="cont"
        )
        assert config.system == "sys"
        assert config.prompt == "prompt"
        assert config.continuation == "cont"

    def test_equality(self):
        a = CompactConfig(system="s", prompt="p", continuation="c")
        b = CompactConfig(system="s", prompt="p", continuation="c")
        assert a == b
