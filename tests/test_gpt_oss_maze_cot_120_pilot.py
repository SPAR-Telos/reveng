import json
from pathlib import Path

from reveng.experiments import gpt_oss_maze_cot_120_pilot as pilot


def test_schedule_is_m11_for_each_model_and_grid():
    grids = pilot.generate_grid_rows_flexible(pilot.PILOT_CONFIG)
    schedule = pilot._expand_schedule_for_m11(
        pilot.build_schedule(pilot.PILOT_CONFIG, grids), pilot.PILOT_CONFIG
    )
    assert len(grids) == 120
    assert len(schedule) == 120 * 11 * 2
    for model in ("GPT-OSS-20B", "Gemma-4-31B-IT"):
        model_rows = [row for row in schedule if row["model"] == model]
        assert len(model_rows) == 1320
        for grid in grids:
            rows = [row for row in model_rows if row["grid_id"] == grid["grid_id"]]
            sampled = [row for row in rows if row["sampling_condition"] == "sampled"]
            greedy = [row for row in rows if row["sampling_condition"] != "sampled"]
            assert len(sampled) == 10
            assert len(greedy) == 1
            assert {row["temperature"] for row in sampled} == {0.7}
            assert {row["temperature"] for row in greedy} == {0.0}
            assert len({row["trajectory_id"] for row in rows}) == 11


def test_retry_only_config_lock_change_is_migrated(tmp_path: Path):
    locked = json.loads(json.dumps(pilot.PILOT_CONFIG))
    locked["max_api_attempts"] = 3
    (tmp_path / "config.lock.json").write_text(json.dumps(locked))
    pilot._validate_or_migrate_config_lock(tmp_path, pilot.PILOT_CONFIG)
    migrated = json.loads((tmp_path / "config.lock.json").read_text())
    assert migrated["max_api_attempts"] == 8


def test_collection_design_config_change_is_rejected(tmp_path: Path):
    import pytest

    locked = json.loads(json.dumps(pilot.PILOT_CONFIG))
    locked["max_output_tokens"] = 1280
    (tmp_path / "config.lock.json").write_text(json.dumps(locked))
    with pytest.raises(ValueError, match="frozen collection design"):
        pilot._validate_or_migrate_config_lock(tmp_path, pilot.PILOT_CONFIG)


def test_query_stage_retries_token_contract_failure_unattended(
    monkeypatch, tmp_path: Path
):
    calls = {"count": 0}
    (tmp_path / "config.lock.json").write_text(json.dumps(pilot.PILOT_CONFIG))
    monkeypatch.setattr(pilot, "load_dotenv", lambda *args, **kwargs: True)
    monkeypatch.setattr(pilot, "make_together_query", lambda api_key=None: object())
    monkeypatch.setattr(pilot.time, "sleep", lambda seconds: None)

    def flaky_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise pilot.TokenSequenceUnavailable("temporary malformed response")
        return [{"ok": True}]

    monkeypatch.setattr(pilot, "run_api_schedule", flaky_run)
    result = pilot.query_stage(pilot.PILOT_CONFIG, [], tmp_path)
    assert result == [{"ok": True}]
    assert calls["count"] == 2


def test_query_stage_forwards_explicit_api_key(monkeypatch, tmp_path: Path):
    sentinel_query = object()
    captured = {}

    def fake_load_dotenv(path, override):
        captured.update(env_path=path, env_override=override)
        return True

    def fake_make_together_query(api_key=None):
        captured["api_key"] = api_key
        return sentinel_query

    def fake_run_api_schedule(config, schedule, raw_path, query_fn, progress_fn):
        captured.update(
            config=config,
            schedule=schedule,
            raw_path=raw_path,
            query_fn=query_fn,
        )
        return []

    monkeypatch.setattr(pilot, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(pilot, "make_together_query", fake_make_together_query)
    monkeypatch.setattr(pilot, "run_api_schedule", fake_run_api_schedule)

    (tmp_path / "config.lock.json").write_text(json.dumps(pilot.PILOT_CONFIG))
    schedule = [{"trajectory_id": "test"}]
    result = pilot.query_stage(
        pilot.PILOT_CONFIG,
        schedule,
        tmp_path,
        api_key="explicit-test-key",
    )

    assert result == []
    assert captured["api_key"] == "explicit-test-key"
    assert captured["query_fn"] is sentinel_query
    assert captured["env_path"] == Path(pilot.__file__).resolve().parents[3] / ".env"
    assert captured["env_override"] is False
    assert captured["schedule"] == schedule
    assert captured["raw_path"] == tmp_path / "raw_api_calls.jsonl"
