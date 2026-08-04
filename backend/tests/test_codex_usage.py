from __future__ import annotations

from backend.codex_usage import parse_codex_jsonl_usage


def test_codex_jsonl_usage_ignores_noise_and_keeps_input_output_separate() -> None:
    usage = parse_codex_jsonl_usage(
        """warning: unrelated stderr line
{"type":"turn.completed","usage":{"input_tokens":17542,"cached_input_tokens":12000,"output_tokens":505,"reasoning_output_tokens":300}}
{"type":"item.completed","item":{"type":"agent_message","text":"done"}}
"""
    )

    assert usage.input_tokens == 17_542
    assert usage.cached_input_tokens == 12_000
    assert usage.output_tokens == 505
    assert usage.reasoning_output_tokens == 300
    assert usage.total_tokens == 18_047


def test_codex_jsonl_usage_sums_completed_turns_without_counting_cached_twice() -> None:
    usage = parse_codex_jsonl_usage(
        """{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":70,"output_tokens":10,"reasoning_output_tokens":4}}
{"type":"turn.completed","usage":{"input_tokens":50,"cached_input_tokens":0,"output_tokens":5,"reasoning_output_tokens":0}}
"""
    )

    assert usage.input_tokens == 150
    assert usage.cached_input_tokens == 70
    assert usage.output_tokens == 15
    assert usage.total_tokens == 165
