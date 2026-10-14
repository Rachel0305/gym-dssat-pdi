"""103/104 isolated stage-aware framework smoke runner for HL and YC."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_078_079_hl_yc_stability_tuning as audit_base

BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
LEVELS_I = [0.0, 15.0, 30.0, 45.0]
LEVELS_N = [0.0, 40.0, 80.0, 120.0]
STATION_TO_SITE = {"HLA": "HL", "YCA": "YC"}


class HLStageRewardWrapper(engine.base04036.LateIrrigationReserveMaskWrapper):
    """Adds bounded water-relief reward only in the registered critical stage."""

    def step(self, action):
        dap = self._dap()
        obs, reward, terminated, truncated, info = super().step(action)
        cfg = self.config.get("reward", {})
        start = int(cfg.get("critical_stage_start_dap", 45))
        end = int(cfg.get("critical_stage_end_dap", 90))
        coef = float(cfg.get("critical_stage_water_stress_relief_coef", 0.0))
        safe_i = float(self.last_action_info.get("safe_action_amir", 0.0) or 0.0)
        relief = float(self.last_action_info.get("water_stress_relief", 0.0) or 0.0)
        scale = float(cfg.get("reward_scale", 1.0))
        extra = coef * safe_i * relief * scale if start <= dap <= end else 0.0
        self.last_action_info.update({
            "critical_stage_reward_enabled": True,
            "critical_stage_start_dap": start,
            "critical_stage_end_dap": end,
            "critical_stage_reward_extra": extra,
        })
        return obs, float(reward + extra), terminated, truncated, info


class YCStageMaskWrapper(engine.base04036.LateIrrigationReserveMaskWrapper):
    """Reserves cumulative irrigation capacity for YC mid/late growth stages."""

    def _action_is_legal_without_clipping(self, raw: dict[str, float], dap: int) -> bool:
        if not super()._action_is_legal_without_clipping(raw, dap):
            return False
        irrigation = float(raw.get("amir", 0.0) or 0.0)
        if irrigation <= 1e-9:
            return True
        cfg = self.config.get("stage_mask", {})
        after_i = float(self.safety_state.cumulative_irrigation) + irrigation
        if dap <= int(cfg.get("early_dap_end", 30)) and after_i > float(cfg.get("early_cumulative_irrigation_cap_mm", 90.0)) + 1e-9:
            return False
        if dap <= int(cfg.get("mid_dap_end", 60)) and after_i > float(cfg.get("mid_cumulative_irrigation_cap_mm", 150.0)) + 1e-9:
            return False
        return True

    def step(self, action):
        dap = self._dap()
        before = float(self.safety_state.cumulative_irrigation)
        obs, reward, terminated, truncated, info = super().step(action)
        self.last_action_info.update({
            "yc_stage_mask_enabled": True,
            "yc_stage_mask_dap_before_action": dap,
            "yc_stage_mask_irrigation_before_action_mm": before,
            "yc_stage_mask_irrigation_after_action_mm": float(self.safety_state.cumulative_irrigation),
        })
        return obs, reward, terminated, truncated, info


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("station_code") not in STATION_TO_SITE:
        raise ValueError("only HLA/YCA are supported")
    if cfg.get("site") != STATION_TO_SITE[cfg["station_code"]]:
        raise ValueError("station/site mismatch")
    if cfg.get("training", {}).get("total_timesteps") != 2000:
        raise ValueError("framework smoke is fixed at 2K")
    return cfg


def input_root() -> Path:
    return ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual"


def source_mzx(cfg: dict[str, Any]) -> Path:
    return input_root() / ("HL/CNHL0701_corrected_IC123.MZX" if cfg["station_code"] == "HLA" else "YC/CNYC0801.MZX")


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    split = batch.load_split().copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    split = split[split["station_code"].astype(str).eq(cfg["station_code"])].sort_values("year")
    train = split.loc[split["split"].eq("train"), "year"].astype(int).tolist()
    valid = split.loc[split["split"].eq("validation"), "year"].astype(int).tolist()
    issues: list[str] = []
    if not input_root().exists() or not source_mzx(cfg).exists():
        issues.append(f"missing input/MZX: {rel(source_mzx(cfg))}")
    if train != list(map(int, cfg["scope"]["train_years"])):
        issues.append(f"train split mismatch: {train}")
    if valid != list(map(int, cfg["scope"]["validation_years"])):
        issues.append(f"validation split mismatch: {valid}")
    if cfg["actions"]["irrigation_levels_mm"] != LEVELS_I or cfg["actions"]["nitrogen_levels_kg_ha"] != LEVELS_N:
        issues.append("action grid mismatch")
    if cfg["station_code"] == "HLA":
        expected = {"critical_stage_start_dap", "critical_stage_end_dap", "critical_stage_water_stress_relief_coef"}
        if set(cfg.get("reward_override", {})) != expected:
            issues.append("HL stage reward keys mismatch")
    if cfg["station_code"] == "YCA":
        expected = {"early_dap_end", "early_cumulative_irrigation_cap_mm", "mid_dap_end", "mid_cumulative_irrigation_cap_mm", "reason"}
        if set(cfg.get("stage_mask", {})) != expected:
            issues.append("YC stage mask keys mismatch")
    return {
        "task_id": cfg["task_id"],
        "station_code": cfg["station_code"],
        "site": cfg["site"],
        "seed": int(cfg["seed"]),
        "input_root": rel(input_root()),
        "source_mzx": rel(source_mzx(cfg)),
        "train_years": train,
        "validation_years": valid,
        "issues": issues,
        "next_step_allowed": not issues,
    }


def run(cfg_path: Path, dry_run: bool) -> dict[str, Any]:
    cfg = load_config(cfg_path)
    pf = preflight(cfg)
    out = ROOT / cfg["output_root"]
    if dry_run:
        return {**pf, "mode": "dry_run", "smoke_only": True, "output_root": rel(out)}
    if not pf["next_step_allowed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    (out / "configs").mkdir()
    prompt = ROOT / cfg["prompt"]
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    shutil.copy2(prompt, out / "configs" / prompt.name)
    effective_yaml = ROOT / "configs" / f"{cfg['task_id']}_effective_framework_ppo_config.yaml"
    yaml_cfg = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))
    yaml_cfg["seed"] = int(cfg["seed"])
    yaml_cfg["ppo"].update(cfg["fixed_ppo"])
    yaml_cfg["reward"].update(cfg.get("reward_override", {}))
    if cfg.get("stage_mask"):
        yaml_cfg["stage_mask"] = cfg["stage_mask"]
    effective_yaml.write_text(yaml.safe_dump(yaml_cfg, sort_keys=False), encoding="utf-8")

    old_engine = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS"]}
    old_batch = {key: getattr(batch, key) for key in ["CONFIG", "OUT", "DOC", "SEED"]}
    old_names = dict(batch.SITE_NAMES)
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    original_wrapper = engine.base04036.LateIrrigationReserveMaskWrapper
    start = time.perf_counter()
    rss0 = psutil.Process().memory_info().rss
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    try:
        engine.TASK_ID, engine.TASK_NAME, engine.BASE_OUT = cfg["task_id"], cfg["task_name"], out
        engine.BASE_DOC = doc
        engine.PROMPT = prompt
        engine.LOWIC_INPUT_ROOT = input_root()
        engine.STATION, engine.SITES = cfg["station_code"], [cfg["station_code"]]
        engine.BINARY_IRRIGATION_LEVELS, engine.BINARY_NITROGEN_LEVELS = LEVELS_I, LEVELS_N
        batch.CONFIG, batch.OUT, batch.DOC, batch.SEED = effective_yaml, out, doc, int(cfg["seed"])
        batch.SITE_NAMES[cfg["station_code"]] = cfg["site"]
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root()
        engine.base04036.LateIrrigationReserveMaskWrapper = HLStageRewardWrapper if cfg["station_code"] == "HLA" else YCStageMaskWrapper
        engine.run_training(2000, [1000, 2000], suffix="")
    finally:
        engine.base04036.LateIrrigationReserveMaskWrapper = original_wrapper
        for key, value in old_engine.items():
            setattr(engine, key, value)
        for key, value in old_batch.items():
            setattr(batch, key, value)
        batch.SITE_NAMES.clear()
        batch.SITE_NAMES.update(old_names)
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root

    summary_path = out / "evaluation/042_10_checkpoint_validation_summary.csv"
    inventory_path = out / "evaluation/042_10_training_checkpoint_inventory.csv"
    summary = pd.read_csv(summary_path, keep_default_na=False)
    final = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
    audit, gate = audit_base.audit_rows(final, require_crossyear_signature=True)
    (out / "audits").mkdir(exist_ok=True)
    audit_path = out / "audits" / f"{cfg['task_id']}_action_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    row = inventory[pd.to_numeric(inventory["checkpoint_step"], errors="coerce").eq(2000)].iloc[0]
    observed = audit_base.observed_kwargs(ROOT / str(row["model_path"]))
    metrics = final[[c for c in ["final_grnwt", "total_irrigation", "total_n", "PFP_N"] if c in final.columns]].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True).to_dict()
    result = {
        **pf,
        "mode": "completed_framework_smoke",
        "output_root": rel(out),
        "effective_yaml": rel(effective_yaml),
        "effective_kwargs_from_loaded_model": observed,
        "validation_metrics_mean": metrics,
        "WP_ET_status": "not_available_in_2K_training_validation_summary; not inferred",
        "action_gate": gate,
        "elapsed_s": round(time.perf_counter() - start, 3),
        "rss_delta_bytes": psutil.Process().memory_info().rss - rss0,
        "next_step_allowed": bool(gate.get("technical_pass")),
        "next_step_note": "Only if true may a separate native-auto five-scenario replay be proposed; no automatic long training.",
    }
    (out / f"{cfg['task_id']}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / f"{cfg['task_id']}_manifest.json").write_text(json.dumps({"config": cfg, "preflight": pf, "result": result}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines = [f"# {cfg['task_id']} {cfg['site']} framework 2K smoke", "", "- external-N: false; weather forecast: false; full replay: false", "", "## Gate", ""]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    lines.extend(["", "- WP_ET 未从 2K training summary 推断。", "- 仅 gate 通过才允许另行设计 five-scenario replay。", ""])
    doc.write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(run(path, args.dry_run), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
