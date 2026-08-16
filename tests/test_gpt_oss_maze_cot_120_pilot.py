from pathlib import Path

from reveng.experiments import gpt_oss_maze_cot_120_pilot as pilot


def test_query_stage_forwards_explicit_api_key(monkeypatch, tmp_path: Path):
    sentinel_query = object()
    captured = {}

    def fake_load_dotenv(path, override):
        captured.update(env_path=path, env_override=override)
        return True

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
