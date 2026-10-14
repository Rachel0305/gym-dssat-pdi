"""071: single-season SY replay provenance audit.

No training and no writes into frozen or 070 roots.  It compares the existing
070 direct evaluator path with the 046_06 snapshot/replay path for one year
and checkpoint only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_ppo_configured_046_02 as base04602

PLAN = ROOT / "configs" / "071_sy_replay_provenance_audit.json"
OUT = ROOT / "benchmark_results" / "071_00_sya_seed1_replay_provenance_audit"
BASE_YAML = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plan() -> dict[str, Any]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def existing_summary(cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / cfg["expected_summary"]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, keep_default_na=False)
    return df[(pd.to_numeric(df.checkpoint_step, errors="coerce").eq(cfg["checkpoint"])) & (pd.to_numeric(df.year, errors="coerce").eq(cfg["year"]))].copy()


def effective_yaml(cfg: dict[str, Any]) -> Path:
    import yaml

    data = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    data["seed"] = int(cfg["seed"])
    data["total_timesteps"] = 100000
    data["paths"]["output_root"] = rel(OUT)
    data["ppo"]["learning_rate"] = 0.0003
    data["ppo"]["gamma"] = 1.0
    data["ppo"]["gae_lambda"] = 1.0
    data["ppo"]["n_steps"] = 144
    data["ppo"]["batch_size"] = 144
    data["ppo"]["n_epochs"] = 5
    data["ppo"]["ent_coef"] = 0.01
    data["ppo"]["clip_range"] = 0.2
    data["ppo"]["net_arch"] = [64, 64]
    path = OUT / "configs" / "071_00_effective.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def setup_runtime(cfg: dict[str, Any], out: Path, ep: Path):
    old_engine = {k: getattr(engine, k) for k in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_batch = (batch.CONFIG, batch.OUT, batch.DOC, batch.SEED, dict(batch.SITE_NAMES))
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    engine.TASK_ID = "071_00"
    engine.TASK_NAME = cfg["task_name"]
    engine.BASE_OUT = out
    engine.BASE_DOC = ROOT / "docs" / "2026-08-13_sy_replay_provenance_audit.md"
    engine.PROMPT = ROOT / cfg["prompt"]
    engine.LOWIC_INPUT_ROOT = base04602.INPUT_PROFILES[cfg["input_profile"]]
    engine.STATION, engine.SITES = "SYA", ["SYA"]
    engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
    engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
    batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = ep, out, engine.BASE_DOC, int(cfg["seed"])
    batch.SITE_NAMES["SYA"] = "SY"
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
    return old_engine, old_batch, old_root


def restore(state) -> None:
    old_engine, old_batch, old_root = state
    for key, value in old_engine.items():
        setattr(engine, key, value)
    batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = old_batch[:4]
    batch.SITE_NAMES.clear()
    batch.SITE_NAMES.update(old_batch[4])
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def input_inventory() -> dict[str, Any]:
    root = base04602.INPUT_PROFILES["originIC"]
    entries = []
    for name in ["CNSY1401.MZX", "CNSY1401.WTH", "MZCER048.CUL", "SY99001200.SOL"]:
        candidates = list(root.rglob(name))
        if candidates:
            p = candidates[0]
            entries.append({"name": name, "path": rel(p), "sha256": sha256(p), "size": p.stat().st_size})
        else:
            entries.append({"name": name, "path": "", "missing": True})
    return {"root": rel(root), "files": entries}


def direct_eval(cfg: dict[str, Any], ep: Path) -> dict[str, Any]:
    out = OUT / "direct_eval"
    out.mkdir(parents=True, exist_ok=True)
    state = setup_runtime(cfg, out, ep)
    try:
        config = batch.load_config()
        selection = batch.build_selection(batch.load_split())
        envcfg = batch.direct_ppo.build_env_config(config, selection)
        inventory = pd.read_csv(ROOT / cfg["ppo_run_dir"] / "evaluation" / "070_00_training_checkpoint_inventory.csv", keep_default_na=False)
        row = inventory[pd.to_numeric(inventory.checkpoint_step, errors="coerce").eq(cfg["checkpoint"])].iloc[0]
        result = batch.evaluate_checkpoint(config, envcfg, row, int(cfg["year"]))
        result_path = out / "direct_eval_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {"result": result, "result_path": rel(result_path), "daily_path": str(result.get("daily_csv_path", ""))}
    finally:
        restore(state)


def replay_eval(cfg: dict[str, Any], ep: Path) -> dict[str, Any]:
    out = OUT / "replay"
    out.mkdir(parents=True, exist_ok=True)
    state = setup_runtime(cfg, out, ep)
    try:
        import build_sya_configured_five_scenario_daily_plots_046_06 as replay

        replay_cfg = {
            "task_id": "070_00",
            "task_name": "sya_originIC_seed1_maskableppo_100k",
            "reference_run": "046_02_sya_originIC_binary_timing_ppo",
            "station_code": "SYA",
            "input_profile": "originIC",
            "seed": 1,
            "training": {"total_timesteps": 100000, "checkpoint_steps": [25000, 50000, 75000, 100000]},
            "scope": {"train_years": [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013], "validation_years": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]},
            "actions": {"irrigation_levels_mm": [0.0, 15.0, 30.0, 45.0], "nitrogen_levels_kg_ha": [0.0, 40.0, 80.0, 120.0]},
            "observation_contract": {"base": "046_02_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False},
            "report_checkpoint": 25000,
        }
        try:
            target = replay.replay_ppo_snapshot(replay_cfg, cfg["checkpoint"], cfg["year"], out, ROOT / cfg["ppo_run_dir"])
            status = "completed"
            error = ""
        except Exception as exc:
            target = out / "snapshots" / "SYA" / str(cfg["year"]) / "rl_candidate"
            status = "failed_endpoint_or_replay"
            error = f"{type(exc).__name__}: {exc}"
        payload = {"status": status, "error": error, "snapshot": rel(target) if target.exists() else ""}
        if target.exists():
            for name in ["Summary.OUT", "PlantGro.OUT", "MgmtEvent.OUT", "MgmtOps.OUT", "dssat-pdi.yml", "CNSY1401.WTH"]:
                p = target / name
                if p.exists():
                    payload.setdefault("files", []).append({"name": name, "path": rel(p), "sha256": sha256(p), "size": p.stat().st_size})
            try:
                daily = replay.daily_from_snapshot(target, int(cfg["year"]), "rl_candidate")
                payload["replay_endpoint_grain_yield_kg_ha"] = float(pd.to_numeric(daily.grain_yield_kg_ha, errors="coerce").iloc[-1])
            except Exception as exc:
                payload["snapshot_parse_error"] = f"{type(exc).__name__}: {exc}"
        (out / "replay_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return payload
    finally:
        restore(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["dry-run", "audit"], required=True)
    args = parser.parse_args()
    cfg = plan()
    if args.phase == "dry-run":
        summary = existing_summary(cfg)
        print(json.dumps({"task": cfg["task_id"] + "_" + cfg["task_name"], "summary_rows": len(summary), "input_inventory": input_inventory(), "output": rel(OUT), "next_step_allowed": len(summary) == 1 and not OUT.exists()}, ensure_ascii=False, indent=2))
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    ep = effective_yaml(cfg)
    summary = existing_summary(cfg)
    if len(summary) != 1:
        raise RuntimeError(f"Expected one saved summary row, got {len(summary)}")
    direct = direct_eval(cfg, ep)
    replay = replay_eval(cfg, ep)
    saved = float(pd.to_numeric(summary.iloc[0]["final_grnwt"], errors="coerce"))
    direct_yield = float(direct["result"]["final_grnwt"])
    replay_yield = replay.get("replay_endpoint_grain_yield_kg_ha")
    result = {
        "task": cfg["task_id"] + "_" + cfg["task_name"],
        "checkpoint": cfg["checkpoint"],
        "year": cfg["year"],
        "saved_training_endpoint_kg_ha": saved,
        "direct_evaluator_endpoint_kg_ha": direct_yield,
        "replay_endpoint_kg_ha": replay_yield,
        "saved_to_direct_difference_kg_ha": direct_yield - saved,
        "direct_to_replay_difference_kg_ha": (replay_yield - direct_yield) if replay_yield is not None else None,
        "saved_to_direct_within_0p5": bool(np.isclose(saved, direct_yield, rtol=0.0, atol=0.5)),
        "direct_to_replay_within_0p5": bool(replay_yield is not None and np.isclose(direct_yield, replay_yield, rtol=0.0, atol=0.5)),
        "input_inventory": input_inventory(),
        "direct": direct,
        "replay": replay,
        "next_step_allowed": bool(replay_yield is not None and np.isclose(saved, direct_yield, rtol=0.0, atol=0.5) and np.isclose(direct_yield, replay_yield, rtol=0.0, atol=0.5)),
    }
    (OUT / "071_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
