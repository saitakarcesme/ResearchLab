from __future__ import annotations

from backend.db import Database


def test_database_persists_research_experiments_logs_and_articles(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("NVIDIA GeForce RTX 3090")
    research = database.create_research(
        {
            "title": "Test Harness",
            "original_prompt": "Improve the harness",
            "objective": "Minimize val_bpb using persisted experiments.",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 60,
        }
    )
    experiment = database.create_experiment(
        research["id"], "Lower batch noise", "Changed one setting", None
    )
    database.update_experiment_token_count(experiment["id"], 1400)
    database.finish_experiment(
        experiment["id"],
        metric_value=1.25,
        accepted=True,
        git_commit="abc1234",
        token_count=1234,
    )
    database.record_codex_token_usage(
        research["id"],
        "candidate",
        call_id="candidate-1",
        input_tokens=1000,
        cached_input_tokens=200,
        output_tokens=234,
        reasoning_output_tokens=40,
    )
    database.update_research(
        research["id"],
        {
            "baseline_value": 1.25,
            "best_value": 1.25,
            "best_git_commit": "abc1234",
        },
    )
    log = database.add_log(
        research["id"],
        "experiment_accepted",
        "Experiment accepted.",
        experiment_id=experiment["id"],
        data={"metric_value": 1.25},
    )
    article = database.upsert_article(
        research["id"], "Test Harness", "# Test Harness\n"
    )

    reopened = Database(settings.database_path)
    reopened.initialize()
    detail = reopened.get_research(research["id"], detail=True)
    assert detail is not None
    assert detail["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert detail["experiment_count"] == 1
    assert detail["total_tokens"] == 1234
    assert detail["input_tokens"] == 1000
    assert detail["cached_input_tokens"] == 200
    assert detail["output_tokens"] == 234
    assert detail["reasoning_output_tokens"] == 40
    assert detail["training_tokens"] == 1400
    assert detail["best_git_commit"] == "abc1234"
    assert detail["experiments"][0]["accepted"] is True
    assert detail["experiments"][0]["token_count"] == 1400
    assert detail["logs"][0]["id"] == log["id"]
    assert detail["logs"][0]["data"] == {"metric_value": 1.25}
    assert reopened.get_article(article["id"])["markdown"] == "# Test Harness\n"


def test_startup_backfills_tokens_from_persisted_benchmark_logs(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = database.create_research(
        {
            "title": "Token Backfill",
            "original_prompt": "measure token usage",
            "objective": "Keep historical token totals",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
        }
    )
    experiment = database.create_experiment(
        research["id"], "Measure a profile", "No code change", None
    )
    database.finish_experiment(
        experiment["id"], metric_value=5.0, accepted=True, git_commit=None
    )
    database.add_log(
        research["id"],
        "experiment_accepted",
        "Profile measured.",
        experiment_id=experiment["id"],
        data={
            "batch_size": 2,
            "warmup": {
                "input_tokens_per_request": 10,
                "total_output_tokens": 4,
            },
            "repeats": [
                {
                    "input_tokens_per_request": 10,
                    "total_output_tokens": 8,
                }
            ],
        },
    )

    reopened = Database(settings.database_path)
    reopened.initialize()
    detail = reopened.get_research(research["id"], detail=True)
    assert detail is not None
    assert detail["total_tokens"] == 0
    assert detail["training_tokens"] == 52
    assert detail["experiments"][0]["token_count"] == 52


def test_startup_creates_only_real_local_source_and_recovers_running(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("NVIDIA GeForce RTX 3090")
    database.ensure_local_gpu_source("NVIDIA GeForce RTX 3090")
    assert database.list_gpu_sources() == [source]
    assert database.list_researches() == []

    research = database.create_research(
        {
            "title": "Recovery",
            "original_prompt": "recovery test",
            "objective": "Persist state",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
        }
    )
    incomplete = database.create_experiment(
        research["id"], "Interrupted candidate", "Unvalidated change", None
    )
    assert database.transition_research(research["id"], ["stopped"], "running")
    assert database.recover_interrupted_researches() == 1
    recovered = database.get_research(research["id"], detail=True)
    assert recovered["status"] == "paused"
    closed = database.get_experiment(incomplete["id"])
    assert closed["accepted"] is False
    assert closed["completed_at"] is not None
    assert "before the experiment completed" in closed["error"]
    assert {item["event_type"] for item in recovered["logs"]} == {
        "service_restarted",
        "experiment_recovered",
    }


def test_detail_returns_the_newest_200_logs_in_chronological_order(settings) -> None:
    database = Database(settings.database_path)
    database.initialize()
    source = database.ensure_local_gpu_source("Local GPU")
    research = database.create_research(
        {
            "title": "Logs",
            "original_prompt": "log window",
            "objective": "Keep the latest persisted log window",
            "gpu_source_id": source["id"],
            "target_gpu_allocation": 100,
        }
    )
    for index in range(205):
        database.add_log(research["id"], "test", f"log-{index}")
    logs = database.get_research(research["id"], detail=True)["logs"]
    assert len(logs) == 200
    assert logs[0]["message"] == "log-5"
    assert logs[-1]["message"] == "log-204"
