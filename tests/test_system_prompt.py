"""Tests for system_prompt.py — golden reference comparisons."""

from pathlib import Path

import pytest

from alpha_sdk.system_prompt import assemble_system_prompt

FIXTURES = Path(__file__).parent / "fixtures" / "jnsq"


async def test_full_assembly():
    """Soul + bill of rights + here, byte-for-byte against golden reference."""
    result = await assemble_system_prompt(
        identity_dir=FIXTURES,
        here="You are in a test.",
    )
    expected = (FIXTURES / "expected_full.txt").read_text()
    assert result == expected


async def test_no_bill_of_rights(tmp_path):
    """Soul + here, no bill of rights. Proves optional pieces are skipped."""
    # Build a minimal JNSQ with just a soul doc
    prompts = tmp_path / "prompts" / "system"
    prompts.mkdir(parents=True)
    soul = (FIXTURES / "prompts" / "system" / "soul.md").read_text()
    (prompts / "soul.md").write_text(soul)

    result = await assemble_system_prompt(
        identity_dir=tmp_path,
        here="You are in a test.",
    )
    expected = (FIXTURES / "expected_no_bill.txt").read_text()
    assert result == expected


async def test_no_here():
    """Full fixture but no here string. Soul + bill of rights only."""
    result = await assemble_system_prompt(identity_dir=FIXTURES)
    expected = (FIXTURES / "expected_no_here.txt").read_text()
    assert result == expected


async def test_no_soul_raises(tmp_path):
    """No soul doc at all. Must fail loud."""
    prompts = tmp_path / "prompts" / "system"
    prompts.mkdir(parents=True)
    # No soul.md — just an empty directory

    with pytest.raises(FileNotFoundError, match="Soul not found"):
        await assemble_system_prompt(identity_dir=tmp_path)


async def test_no_identity_dir_raises(monkeypatch):
    """No identity_dir and no env var. Must fail loud."""
    monkeypatch.delenv("JE_NE_SAIS_QUOI", raising=False)

    with pytest.raises(RuntimeError, match="No identity directory configured"):
        await assemble_system_prompt()
