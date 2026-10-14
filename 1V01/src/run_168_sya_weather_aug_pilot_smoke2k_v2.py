"""168: corrected six-scenario SYA weather-only MaskablePPO smoke.

The six training cases all simulate source year 2005.  A scenario-specific
input root is selected before constructing each DSSAT environment, so no
pseudo-year is used and no existing training/result directory is overwritten.
The physical gate from task 167 must pass before training starts.
"""

from __future__ import annotations

import argparse
import copy
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
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


TASK_ID = "168"
TASK_NAME = "sya_weather_aug_pilot_smoke2k_v2"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
SOURCE_PILOT_CONFIG = ROOT / "configs" / "161_sya_weather_aug_pilot_smoke2k.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
STATION = "SYA"
SITE = "SY"
SEED = 0
CHECKPOINT_STEPS = [1000, 2000]
VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
SCENARIO_KEYS = {2005: "original", 2006: "early_dry", 2007: "mid_dry", 2008: "late_dry", 2009: "early_wet", 2010: "mid_wet"}
IRRIGATION_GRID = {0.0, 15.0, 30.0, 45.0}
NITROGEN_GRID = {0.0, 40.0, 80.0, 120.0}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    path = ROOT / str(cfg["scenario_manifest"])
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path, keep_default_na=False)
    manifest = manifest[manifest["station_code"].eq(STATION)].copy()
    if manifest["variant"].tolist() != VARIANTS:
        manifest = manifest.set_index("variant").loc[VARIANTS].reset_index()
    if manifest["simulation_year"].astype(int).nunique() != 1 or int(manifest["simulation_year"].iloc[0]) != 2005:
        raise ValueError("All 168 scenarios must simulate source year 2005")
    return manifest.reset_index(drop=True)


def load_split(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, variant in SCENARIO_KEYS.items():
        row = manifest[manifest["variant"].eq(variant)].iloc[0]
        rows.append(
            {
                "station_code": STATION,
                "site": SITE,
                "station_name": SITE,
                "source_year": 2005,
                "year": int(year),
                "split": "train",
                "weather_variant": variant,
                "scenario_type": f"{TASK_ID}_{variant}",
                "weather_file": "CNSY0501.WTH",
                "target_weather": str(row["target_weather"]),
                "scenario_input_root": str(row["scenario_input_root"]),
                "selected_for_train": True,
                "selected_for_eval": False,
                "selection_reason": "corrected_2005_weather_only_augmented_smoke",
                "rain_multiplier": float(row["rain_multiplier"]),
                "perturbation_window": str(row["window"]),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def verify_gate(cfg: dict[str, Any]) -> dict[str, Any]:
    gate_path = ROOT / "benchmark_results" / str(cfg["physical_gate_task"]) / "164_summary.json"
    if not gate_path.exists():
        raise FileNotFoundError(gate_path)
    gate = load_json(gate_path)
    if not bool(gate.get("physical_response_passed")):
        raise RuntimeError("Physical weather-response gate is not passed; refuse PPO training")
    return gate


def patch_runtime(manifest: pd.DataFrame) -> dict[str, Any]:
    old_engine = {
        key: copy.deepcopy(getattr(engine, key))
        for key in [
            "TASK_ID",
            "TASK_NAME",
            "BASE_OUT",
            "BASE_DOC",
            "PROMPT",
            "LOWIC_INPUT_ROOT",
            "STATION",
            "SITES",
            "BINARY_IRRIGATION_LEVELS",
            "BINARY_NITROGEN_LEVELS",
        ]
    }
    old_base = {
        "SITES": list(base03222.SITES),
        "SITE_NAMES": dict(base03222.SITE_NAMES),
        "OUT": base03222.OUT,
        "DOC": base03222.DOC,
        "PROMPT": base03222.PROMPT,
        "TOTAL_TIMESTEPS": base03222.TOTAL_TIMESTEPS,
        "CHECKPOINT_STEPS": list(base03222.CHECKPOINT_STEPS),
        "load_split": base03222.load_split,
        "build_selection": base03222.build_selection,
        "base_make_env": base03222.base.make_env,
    }
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    variant_roots = {
        str(row.variant): ROOT / str(row.scenario_input_root)
        for row in manifest.itertuples(index=False)
    }

    def scenario_make_env(config, env_config, station, year, seed, run_tag, evaluation=False):
        key = int(year)
        if key not in SCENARIO_KEYS:
            raise KeyError(f"Unknown corrected weather scenario key: {key}")
        variant = SCENARIO_KEYS[key]
        old = ppo_safe_rendering.MULTISITE_INPUT_ROOT
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = variant_roots[variant]
        try:
            # The DSSAT simulation year and planting date are always the real
            # 2005 case; ``key`` is only an internal scenario selector.
            return old_base["base_make_env"](
                config,
                env_config,
                station,
                2005,
                seed,
                f"{STATION}_2005_{variant}_{TASK_ID}_{'eval' if evaluation else 'train'}",
                evaluation=evaluation,
            )
        finally:
            ppo_safe_rendering.MULTISITE_INPUT_ROOT = old

    engine.TASK_ID = TASK_ID
    engine.TASK_NAME = TASK_NAME
    engine.BASE_OUT = OUT
    engine.BASE_DOC = DOC
    engine.PROMPT = ROOT / "prompts" / "2026-08-18_sya_weather_aug_pilot_smoke2k.md"
    engine.LOWIC_INPUT_ROOT = variant_roots["original"]
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
    engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
    engine.patch_base_module(OUT, DOC, 2000, CHECKPOINT_STEPS)
    base03222.SITES = [STATION]
    base03222.SITE_NAMES.clear()
    base03222.SITE_NAMES.update({STATION: SITE})
    base03222.RandomYearEnv = base03222.RandomYearEnv
    base03222.load_split = lambda: load_split(manifest).copy()
    base03222.build_selection = lambda split: split.copy()
    base03222.base.make_env = scenario_make_env
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = variant_roots["original"]
    return {"engine": old_engine, "base": old_base, "input_root": old_input_root}


def restore_runtime(old: dict[str, Any]) -> None:
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["input_root"]
    for key, value in old["base"].items():
        if key == "SITE_NAMES":
            base03222.SITE_NAMES.clear()
            base03222.SITE_NAMES.update(value)
        elif key == "base_make_env":
            base03222.base.make_env = value
        else:
            setattr(base03222, key, value)
    for key, value in old["engine"].items():
        setattr(engine, key, value)


def evaluate_and_audit(manifest: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_path = OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv"
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    ok = inventory[inventory["run_status"].astype(str).str.startswith("ok")].copy()
    runtime_config = engine.load_config()
    env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
    metric_rows: list[dict[str, Any]] = []
    for _, model_row in ok.iterrows():
        for scenario_key in split["year"].astype(int).tolist():
            result = base03222.evaluate_checkpoint(runtime_config, env_config, model_row, scenario_key)
            result["weather_variant"] = SCENARIO_KEYS[scenario_key]
            result["scenario_key"] = scenario_key
            result["source_year"] = 2005
            metric_rows.append(result)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "evaluation" / "168_scenario_episode_metrics.csv", index=False, encoding="utf-8-sig")

    action_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for checkpoint in sorted(metrics["checkpoint_step"].astype(int).unique().tolist()):
        ckpt = metrics[metrics["checkpoint_step"].astype(int).eq(checkpoint)]
        base_metric = ckpt[ckpt["weather_variant"].eq("original")].iloc[0]
        base_daily = pd.read_csv(ROOT / str(base_metric["daily_csv_path"]), keep_default_na=False)
        base_i, base_n = "safe_action_amir", "safe_action_anfer"
        for variant in VARIANTS:
            metric = ckpt[ckpt["weather_variant"].eq(variant)].iloc[0]
            daily = pd.read_csv(ROOT / str(metric["daily_csv_path"]), keep_default_na=False)
            pair = base_daily[["dap", base_i, base_n]].rename(columns={base_i: "baseline_irrigation", base_n: "baseline_n"}).merge(
                daily[["dap", base_i, base_n]].rename(columns={base_i: "variant_irrigation", base_n: "variant_n"}),
                on="dap",
                how="inner",
            )
            pair["checkpoint_step"] = checkpoint
            pair["weather_variant"] = variant
            pair["changed"] = (
                ~np.isclose(pair["baseline_irrigation"], pair["variant_irrigation"])
                | ~np.isclose(pair["baseline_n"], pair["variant_n"])
            )
            action_rows.extend(pair.to_dict(orient="records"))
            legal = bool(
                daily[base_i].round(6).isin(IRRIGATION_GRID).all()
                and daily[base_n].round(6).isin(NITROGEN_GRID).all()
            )
            after = pair[pair["dap"].gt(1)]
            summary_rows.append(
                {
                    "checkpoint_step": checkpoint,
                    "weather_variant": variant,
                    "scenario_key": int(metric["scenario_key"]),
                    "matched_decision_states": int(len(pair)),
                    "action_changed_rows": int(pair["changed"].sum()),
                    "action_change_rate": float(pair["changed"].mean()) if len(pair) else np.nan,
                    "action_change_rate_after_dap1": float(after["changed"].mean()) if len(after) else np.nan,
                    "all_actions_on_grid": legal,
                    "final_grnwt": float(metric.get("final_grnwt", np.nan)),
                    "total_irrigation": float(metric.get("total_irrigation", np.nan)),
                    "total_n": float(metric.get("total_n", np.nan)),
                }
            )
    action_detail = pd.DataFrame(action_rows)
    action_summary = pd.DataFrame(summary_rows)
    action_detail.to_csv(OUT / "evaluation" / "168_action_pair_detail.csv", index=False, encoding="utf-8-sig")
    action_summary.to_csv(OUT / "evaluation" / "168_action_change_summary.csv", index=False, encoding="utf-8-sig")
    return metrics, action_summary


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    gate = verify_gate(cfg)
    manifest = read_manifest(cfg)
    split = load_split(manifest)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "configs").mkdir()
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "168_training_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    old = patch_runtime(manifest)
    status = "completed"
    error = ""
    try:
        base03222.main()
        metrics, action_summary = evaluate_and_audit(manifest, split)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        metrics = pd.DataFrame()
        action_summary = pd.DataFrame()
    finally:
        restore_runtime(old)
    non_original = action_summary[action_summary["weather_variant"].ne("original")] if not action_summary.empty else pd.DataFrame()
    response_passed = bool(
        status == "completed"
        and not non_original.empty
        and bool(non_original["all_actions_on_grid"].all())
        and float(pd.to_numeric(non_original["action_change_rate_after_dap1"], errors="coerce").fillna(0.0).max()) > 0.01
    )
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "training_executed": True,
        "total_timesteps": 2000,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "station": STATION,
        "scenario_count": len(VARIANTS),
        "simulation_years": [2005],
        "physical_gate_passed": True,
        "validation_executed": False,
        "action_response_passed": response_passed,
        "outputs": {
            "training_inventory": rel(OUT / "evaluation" / "032_22_training_checkpoint_inventory.csv"),
            "scenario_metrics": rel(OUT / "evaluation" / "168_scenario_episode_metrics.csv"),
            "action_summary": rel(OUT / "evaluation" / "168_action_change_summary.csv"),
        },
        "error": error[-5000:] if error else "",
    }
    (OUT / "168_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {TASK_ID} SYA corrected weather augmentation PPO smoke",
        "",
        "- Scope: SYA only; one cultivar/soil input set; six RAIN-only scenarios.",
        "- Simulation year: 2005 for every scenario; scenario key is not a DSSAT year.",
        "- Training: MaskablePPO, seed 0, 2000 timesteps; checkpoints 1000 and 2000.",
        "- Physical gate passed before training: `True`.",
        "- Validation executed: `False`.",
        f"- Training status: `{status}`.",
        f"- Action response passed: `{response_passed}`.",
        f"- Training inventory: `{rel(OUT / 'evaluation' / '032_22_training_checkpoint_inventory.csv')}`.",
        f"- Action summary: `{rel(OUT / 'evaluation' / '168_action_change_summary.csv')}`.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if status != "completed":
        raise RuntimeError(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
