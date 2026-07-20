from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import stable_baselines3
import sb3_contrib
from sb3_contrib import MaskablePPO


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_hla2010_stage_maskable_ppo_seed0_027_02 as hla
from run_sy2014_stage_maskable_ppo_seed1_curve_026_02 import evaluate as evaluate_sy
from smoke_sy2014_stage_maskable_ppo_train_026_01 import AuditedStageEnv


OUT = ROOT / "benchmark_results" / "027_05_ppo_frozen_daily_completion"
SY_MODEL = ROOT / "benchmark_results" / "026_03" / "checkpoint_000120.zip"
HLA_MODEL = ROOT / "benchmark_results" / "027_02_attempt2" / "checkpoint_000180.zip"
SY_EXPECTED_HASH = "d88938a7d939cc772f5da185360298f59af86ea0b95cf411d3ccb0722ba541ab"
HLA_EXPECTED_HASH = "6715fffb4bbe251cf0cdad13121331d9e6f875a3e6629c26423500e502e042da"
SY_EXPECTED = ROOT / "benchmark_results" / "026_03" / "026_03_seed0_checkpoint_summary.csv"
HLA_EXPECTED = ROOT / "benchmark_results" / "027_02_attempt2" / "027_02_hla2010_seed0_checkpoint_summary.csv"
SY_SCALER = ROOT / "benchmark_results" / "021_24" / "021_24_observation_scaler.csv"
SY_THRESHOLDS = ROOT / "benchmark_results" / "022_01" / "022_01_thresholds_and_provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_snapshot(source: Path, destination: Path) -> None:
    required = ["Summary.OUT", "Weather.OUT", "SoilWat.OUT", "PlantGro.OUT", "MgmtEvent.OUT"]
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Snapshot source {source} is missing {missing}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot {destination}")
    shutil.copytree(source, destination)
    copied_missing = [name for name in required if not (destination / name).exists()]
    if copied_missing:
        raise FileNotFoundError(f"Copied snapshot {destination} is missing {copied_missing}")


def expected_row(path: Path, checkpoint: int) -> pd.Series:
    frame = pd.read_csv(path)
    rows = frame[pd.to_numeric(frame["checkpoint"], errors="coerce").eq(checkpoint)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one checkpoint {checkpoint} row in {path}, got {len(rows)}")
    return rows.iloc[0]


def close_enough(actual: Any, expected: Any, atol: float = 1e-6) -> bool:
    return abs(float(actual) - float(expected)) <= atol


def run_sy() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    expected = expected_row(SY_EXPECTED, 120)
    model_hash_before = sha256(SY_MODEL)
    if model_hash_before != SY_EXPECTED_HASH:
        raise RuntimeError(f"SY model hash mismatch: {model_hash_before}")
    env = AuditedStageEnv(OUT / "SY2014" / "runtime_eval", SY_SCALER, seed=1000, phase="027_05_frozen_eval")
    try:
        model = MaskablePPO.load(SY_MODEL, device="cpu")
        thresholds = json.loads(SY_THRESHOLDS.read_text(encoding="utf-8"))
        result, stages = evaluate_sy(model, env, 120, thresholds)
        tmp = Path(getattr(env.raw_env.unwrapped, "_tmp_folder"))
        copy_snapshot(tmp, OUT / "SY2014" / "snapshot")
        checks = {
            "model_hash_unchanged": sha256(SY_MODEL) == model_hash_before,
            "model_num_timesteps_120": int(model.num_timesteps) == 120,
            "action_sequence_exact": result["action_sequence"] == str(expected["action_sequence"]),
            "yield_exact": close_enough(result["final_yield"], expected["final_yield"]),
            "biomass_exact": close_enough(result["final_biomass"], expected["final_biomass"]),
            "irrigation_exact": close_enough(result["irrigation_total"], expected["irrigation_total"]),
            "nitrogen_exact": close_enough(result["nitrogen_total"], expected["nitrogen_total"]),
            "wp_et_exact": close_enough(result["WP_ET_kg_m3"], expected["WP_ET_kg_m3"]),
            "pfp_n_exact": close_enough(result["PFP_N_kg_kg"], expected["PFP_N_kg_kg"]),
            "six_stage_rows": len(stages) == 6,
        }
        if not all(checks.values()):
            raise RuntimeError(f"SY frozen reevaluation mismatch: {checks}")
        return {"site": "SY", "year": 2014, "seed": 0, "checkpoint": 120, "model_sha256": model_hash_before, **result}, stages, checks
    finally:
        env.close()


def run_hla() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    expected = expected_row(HLA_EXPECTED, 180)
    model_hash_before = sha256(HLA_MODEL)
    if model_hash_before != HLA_EXPECTED_HASH:
        raise RuntimeError(f"HLA model hash mismatch: {model_hash_before}")
    scaler = pd.read_csv(hla.SCALER_PATH).sort_values("observation_index")
    reward = json.loads(hla.REWARD_PATH.read_text(encoding="utf-8"))
    env = hla.make_env(OUT / "HLA2010" / "runtime_eval", scaler, reward, seed=1000, phase="027_05_frozen_eval")
    try:
        model = MaskablePPO.load(HLA_MODEL, device="cpu")
        result, stages = hla.evaluate(model, env, 180)
        copy_snapshot(hla.snapshot_path(env), OUT / "HLA2010" / "snapshot")
        checks = {
            "model_hash_unchanged": sha256(HLA_MODEL) == model_hash_before,
            "model_num_timesteps_180": int(model.num_timesteps) == 180,
            "action_sequence_exact": result["action_sequence"] == str(expected["action_sequence"]),
            "yield_exact": close_enough(result["final_yield"], expected["final_yield"]),
            "biomass_exact": close_enough(result["final_biomass"], expected["final_biomass"]),
            "irrigation_exact": close_enough(result["irrigation_total"], expected["irrigation_total"]),
            "nitrogen_exact": close_enough(result["nitrogen_total"], expected["nitrogen_total"]),
            "wp_et_exact": close_enough(result["WP_ET_kg_m3"], expected["WP_ET_kg_m3"]),
            "pfp_n_exact": close_enough(result["PFP_N_kg_kg"], expected["PFP_N_kg_kg"]),
            "six_stage_rows": len(stages) == 6,
        }
        if not all(checks.values()):
            raise RuntimeError(f"HLA frozen reevaluation mismatch: {checks}")
        return {"site": "HLA", "year": 2010, "seed": 0, "checkpoint": 180, "model_sha256": model_hash_before, **result}, stages, checks
    finally:
        env.close()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    required = [SY_MODEL, HLA_MODEL, SY_EXPECTED, HLA_EXPECTED, SY_SCALER, SY_THRESHOLDS, hla.SCALER_PATH, hla.REWARD_PATH]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"027_05 frozen reevaluation precheck missing: {missing}")
    OUT.mkdir(parents=True)
    payload: dict[str, Any] = {
        "status": "running",
        "training_steps": 0,
        "learn_called": False,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "site_order": ["SY2014", "HLA2010"],
    }
    try:
        sy_result, sy_stages, sy_checks = run_sy()
        hla_result, hla_stages, hla_checks = run_hla()
        results = [sy_result, hla_result]
        stages = [
            *[{"site": "SY", "year": 2014, "seed": 0, **row} for row in sy_stages],
            *[{"site": "HLA", "year": 2010, "seed": 0, **row} for row in hla_stages],
        ]
        pd.DataFrame(results).to_csv(OUT / "027_05_ppo_frozen_reevaluation_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(stages).to_csv(OUT / "027_05_ppo_frozen_reevaluation_stage_actions.csv", index=False, encoding="utf-8-sig")
        payload.update({
            "status": "completed",
            "dssat_seasons": 2,
            "sites_completed": 2,
            "checks": {"SY2014": sy_checks, "HLA2010": hla_checks},
            "snapshots": {
                "SY2014": "benchmark_results/027_05_ppo_frozen_daily_completion/SY2014/snapshot",
                "HLA2010": "benchmark_results/027_05_ppo_frozen_daily_completion/HLA2010/snapshot",
            },
        })
        (OUT / "027_05_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as error:
        payload.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        (OUT / "027_05_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
