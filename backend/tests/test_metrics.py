from __future__ import annotations

import pytest

from backend.adapters.autoresearch import (
    KarpathyAutoresearchAdapter,
    is_better,
    live_tokens_from_output,
    parse_evaluation_summary,
    parse_peak_vram_mb,
    parse_val_bpb,
)

VALID_SUMMARY = """step 00012
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     22010.2
mfu_percent:      18.50
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
"""


def test_full_terminal_summary_parser_accepts_canonical_upstream_output() -> None:
    summary = parse_evaluation_summary(VALID_SUMMARY, active_duration_seconds=325.8)
    assert summary.val_bpb == pytest.approx(0.9979)
    assert summary.training_seconds == pytest.approx(300.1)
    assert summary.num_steps == 953
    assert parse_val_bpb(VALID_SUMMARY) == pytest.approx(0.9979)
    assert parse_peak_vram_mb(VALID_SUMMARY) == pytest.approx(22010.2)
    with pytest.raises(ValueError, match="status 1"):
        parse_val_bpb(VALID_SUMMARY, returncode=1)


def test_live_token_counter_uses_only_completed_step_throughput() -> None:
    output = """step 00000 (0.0%) | loss: 9.0 | dt: 2,000ms | tok/sec: 100,000 | remaining: 300s
step 00001 (1.0%) | loss: 8.0 | dt: 2500ms | tok/sec: 200,000 | remaining: 297s
partial line without a completed throughput sample
"""
    assert live_tokens_from_output(output) == 700_000


def test_metric_only_duplicate_missing_and_non_finite_summaries_are_rejected() -> None:
    with pytest.raises(ValueError, match="full terminal summary"):
        parse_val_bpb("---\nval_bpb: 0.000001\n")
    with pytest.raises(ValueError, match="exactly one"):
        parse_val_bpb(VALID_SUMMARY + VALID_SUMMARY)
    with pytest.raises(ValueError, match="full terminal summary"):
        parse_val_bpb(VALID_SUMMARY.replace("num_params_M:     50.3\n", ""))
    with pytest.raises(ValueError, match="finite"):
        parse_val_bpb(VALID_SUMMARY.replace("0.997900", "1e999"))


def test_short_or_inconsistent_complete_looking_summaries_are_rejected() -> None:
    two_second_summary = VALID_SUMMARY.replace("300.1", "2.0").replace("325.9", "2.1")
    with pytest.raises(ValueError, match="training_seconds"):
        parse_evaluation_summary(two_second_summary, active_duration_seconds=2.1)
    with pytest.raises(ValueError, match="active execution duration"):
        parse_evaluation_summary(VALID_SUMMARY, active_duration_seconds=2.0)
    with pytest.raises(ValueError, match="greater than or equal"):
        parse_evaluation_summary(
            VALID_SUMMARY.replace("total_seconds:    325.9", "total_seconds:    299.0"),
            active_duration_seconds=325.8,
        )
    with pytest.raises(ValueError, match="at least 12"):
        parse_evaluation_summary(
            VALID_SUMMARY.replace("num_steps:        953", "num_steps:        11"),
            active_duration_seconds=325.8,
        )


def test_metric_comparison_is_strict_and_directional() -> None:
    assert is_better(0.9, 1.0, "lower_is_better")
    assert not is_better(1.0, 1.0, "lower_is_better")
    assert not is_better(1.1, 1.0, "lower_is_better")
    assert is_better(2.0, 1.0, "higher_is_better")
    assert not is_better(1.0, 1.0, "higher_is_better")
    assert is_better(1.0, None, "lower_is_better")
    with pytest.raises(ValueError):
        is_better(1.0, 2.0, "sideways")


def test_oom_retry_order_preserves_fixed_validation_sequence(tmp_path) -> None:
    adapter = object.__new__(KarpathyAutoresearchAdapter)
    adapter.workspace = tmp_path
    train = tmp_path / "train.py"
    train.write_text(
        """from prepare import MAX_SEQ_LEN
TOTAL_BATCH_SIZE = 2**19
DEPTH = 8
DEVICE_BATCH_SIZE = 32
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
""",
        encoding="utf-8",
    )
    changes = [adapter._reduce_for_oom() for _ in range(7)]
    assert "32 to 16" in changes[0]
    assert "16 to 8" in changes[1]
    assert "2048 to 1024" in changes[2]
    assert "1024 to 512" in changes[3]
    assert "DEPTH from 8 to 6" in changes[4]
    assert "DEPTH from 6 to 4" in changes[5]
    assert changes[6] is None
    source = train.read_text(encoding="utf-8")
    assert "TRAIN_SEQ_LEN = 512" in source
    assert "tokens_per_fwdbwd = DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN" in source
    assert (
        'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN, "train")'
        in source
    )
    assert "MAX_SEQ_LEN" in source
