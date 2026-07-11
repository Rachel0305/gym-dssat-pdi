"""Lightweight framework tests that do not launch DSSAT training."""

from __future__ import annotations

import copy
from pathlib import Path

from benchmark.checkpoint_manager import CheckpointManager
from benchmark.config_loader import canonical_station_code, load_config
from benchmark.experiment_registry import ExperimentRegistry, ManifestStatus, expand_experiments
from benchmark.provenance import compute_config_hash
from benchmark.result_registry import ResultRegistry
from benchmark.validators import validate_config


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "experiments" / "021_00_framework_smoke.yaml"


def test_station_aliases_use_result_schema_codes() -> None:
    assert canonical_station_code("YCA") == "YC"
    assert canonical_station_code("FQA") == "FQ"
    assert canonical_station_code("LCA") == "LC"
    assert canonical_station_code("SYA") == "SY"
    assert canonical_station_code("HL") == "HLA"


def test_smoke_config_loads_validates_and_expands() -> None:
    config = load_config(SMOKE, ROOT)
    assert config["site"]["station_code"] == "HLA"
    assert config["algorithm"]["n_steps"] == 5
    assert validate_config(config, ROOT) == []
    specs = expand_experiments(config, ROOT)
    assert [(spec.station_code, spec.year, spec.seed) for spec in specs] == [("HLA", 2007, 0)]


def test_invalid_action_is_intercepted() -> None:
    config = load_config(SMOKE, ROOT)
    broken = copy.deepcopy(config)
    broken["action_space"]["irrigation_levels"] = [0, 60]
    errors = validate_config(broken, ROOT)
    assert any("exceeds daily_irrigation_cap" in error for error in errors)


def test_config_hash_is_stable_and_science_sensitive() -> None:
    config = load_config(SMOKE, ROOT)
    first = compute_config_hash(config, ROOT)
    second = compute_config_hash(copy.deepcopy(config), ROOT)
    changed = copy.deepcopy(config)
    changed["reward"]["nitrogen_cost"] = 5.5
    assert first == second
    assert first != compute_config_hash(changed, ROOT)


def test_manifest_state_and_hash_lookup(tmp_path: Path) -> None:
    config = load_config(SMOKE, ROOT)
    spec = expand_experiments(config, ROOT)[0]
    registry = ExperimentRegistry(tmp_path)
    manifest = registry.register(spec)
    assert manifest["status"] == "pending"
    registry.update_status(spec.experiment_id, ManifestStatus.RUNNING, message="test")
    registry.update_status(spec.experiment_id, ManifestStatus.PARTIAL, message="interrupt")
    assert registry.find_by_hash(spec.config_hash)[0]["status"] == "partial"


def test_result_registry_atomic_upsert(tmp_path: Path) -> None:
    registry = ResultRegistry(tmp_path / "result_registry.csv")
    row = {
        "experiment_id": "case",
        "config_hash": "abc",
        "station_code": "HLA",
        "year": 2007,
        "seed": 0,
        "scenario": "dqn",
        "status": "partial",
    }
    registry.upsert(row)
    registry.upsert({**row, "status": "completed"})
    assert len(registry.list()) == 1
    assert registry.find(config_hash="abc")[0]["status"] == "completed"


def test_checkpoint_resume_plan_detects_nested_model_and_replay(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "checkpoint_100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.zip").write_bytes(b"model")
    (checkpoint / "replay_buffer.pkl").write_bytes(b"buffer")
    (checkpoint / "rng_state.pt").write_bytes(b"rng")
    plan = CheckpointManager(tmp_path).resume_plan(total_timesteps=200)
    assert plan.completed_timesteps == 100
    assert plan.resume_kind == "checkpoint_replay_resume"
    assert plan.replay_buffer == (checkpoint / "replay_buffer.pkl").resolve()

