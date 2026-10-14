"""140_00 paired SY forecast/no-forecast training gate.

Run one branch per process to keep memory bounded.  The forecast branch reuses
the validated rolling-7-day observation contract; the control branch reuses
the 046_10 raw-observation runner.  This file never starts 100K training.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import run_111_112_sy_lc_rolling7d_perfect_forecast_smoke2k as rolling
import run_sya_originIC_expanded_action_maskableppo_046_10 as nof


CONFIGS = {
    "forecast": ROOT / "configs/140F_sya_originIC_paired_forecast.json",
    "no_forecast": ROOT / "configs/140N_sya_originIC_paired_noforecast.json",
}
PROMPT = ROOT / "prompts/2026-08-16_sya_paired_forecast_noforecast_5k_response_gate.md"
BRANCH_IDS = {"forecast": "140F", "no_forecast": "140N"}
TRAIN_YEARS = list(range(2005, 2014))
VALIDATION_YEARS = list(range(2014, 2024))
IRRIGATION = [0.0, 15.0, 30.0, 45.0]
NITROGEN = [0.0, 40.0, 80.0, 120.0]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_cfg(branch: str) -> dict[str, Any]:
    return json.loads(CONFIGS[branch].read_text(encoding="utf-8"))


def phase_config(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    planned = copy.deepcopy(cfg)
    if smoke:
        planned["task_name"] = f"{cfg['task_name']}_paired2k"
        planned["training"] = {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}
    else:
        planned["task_name"] = f"{cfg['task_name']}_paired5k"
        planned["training"] = {"total_timesteps": 5000, "checkpoint_steps": [2000, 5000]}
    return planned


def preflight(branch: str) -> dict[str, Any]:
    cfg = read_cfg(branch)
    issues: list[str] = []
    if not PROMPT.exists():
        issues.append("prompt_missing")
    if cfg.get("input_profile") != "originIC":
        issues.append("input_profile_must_be_originIC")
    if list(map(float, cfg["actions"]["irrigation_levels_mm"])) != IRRIGATION:
        issues.append("irrigation_grid_mismatch")
    if list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"])) != NITROGEN:
        issues.append("nitrogen_grid_mismatch")
    if cfg["scope"].get("external_n") or cfg["scope"].get("native_dssat_automatic_irrigation"):
        issues.append("automatic_or_external_n_must_be_false")
    obs = cfg["observation_contract"]
    if branch == "forecast":
        if not obs.get("weather_forecast_enabled") or len(cfg.get("forecast_features", {}).get("feature_names", [])) != 28:
            issues.append("forecast_contract_mismatch")
    else:
        if obs.get("weather_forecast_enabled"):
            issues.append("no_forecast_contract_mismatch")
    return {
        "branch": branch,
        "task_id": cfg["task_id"],
        "task_name": cfg["task_name"],
        "train_years": TRAIN_YEARS,
        "validation_years": VALIDATION_YEARS,
        "seed": cfg["seed"],
        "actions": cfg["actions"],
        "observation_contract": obs,
        "issues": issues,
        "next_step_allowed": not issues,
    }


def smoke_result_path(branch: str) -> Path:
    cfg = read_cfg(branch)
    out = ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}_paired2k"
    return out / f"{cfg['task_id']}_smoke_result.json"


def verify_both_smokes() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for branch in ["forecast", "no_forecast"]:
        path = smoke_result_path(branch)
        checks[f"{branch}_smoke_result_exists"] = path.exists()
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            gate = result.get("smoke_gate") or result.get("action_audit") or {}
            checks[f"{branch}_smoke_gate"] = bool(gate.get("next_step_allowed", result.get("next_step_allowed", False)))
        else:
            checks[f"{branch}_smoke_gate"] = False
    checks["both_smokes_passed"] = bool(checks.get("forecast_smoke_gate") and checks.get("no_forecast_smoke_gate"))
    return checks


def run_branch(branch: str, phase: str) -> dict[str, Any]:
    cfg_path = CONFIGS[branch]
    pf = preflight(branch)
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight failed: " + "; ".join(pf["issues"]))
    if phase == "smoke2k":
        if branch == "forecast":
            rolling.patch_contract()
            forecast.prepare_config = phase_config
            result = forecast.run_forecast_experiment(cfg_path, PROMPT, BRANCH_IDS[branch], False, True, False)
        else:
            nof.prepare_config = phase_config
            nof.PROMPT = PROMPT
            result = nof.run(cfg_path, False, True, False)
        return result
    if phase != "train5k":
        raise ValueError("phase must be smoke2k, train5k, or dry-run")
    smoke_checks = verify_both_smokes()
    if not smoke_checks["both_smokes_passed"]:
        raise RuntimeError("5K is blocked until both 2K smoke gates pass: " + json.dumps(smoke_checks, ensure_ascii=False))
    if branch == "forecast":
        rolling.patch_contract()
        forecast.prepare_config = phase_config
        result = forecast.run_forecast_experiment(cfg_path, PROMPT, BRANCH_IDS[branch], False, False, True)
    else:
        nof.prepare_config = phase_config
        nof.PROMPT = PROMPT
        result = nof.run(cfg_path, False, False, True)
    result["paired_smoke_checks"] = smoke_checks
    return result


def find_output(branch: str, phase: str) -> Path:
    cfg = read_cfg(branch)
    suffix = "paired2k" if phase == "smoke2k" else "paired5k"
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}_{suffix}"


def load_branch_summary(branch: str, phase: str) -> pd.DataFrame:
    cfg = read_cfg(branch)
    out = find_output(branch, phase)
    path = out / "evaluation" / f"{cfg['task_id']}_checkpoint_validation_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype(int)
    frame["checkpoint_step"] = pd.to_numeric(frame["checkpoint_step"], errors="coerce").astype(int)
    frame["branch"] = branch
    frame["phase"] = phase
    return frame


def load_daily(row: pd.Series) -> pd.DataFrame:
    daily = pd.read_csv(ROOT / str(row["daily_csv_path"]), keep_default_na=False)
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    for col in ["safe_action_amir", "safe_action_anfer"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
    return daily


def aggregate() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    phase_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for phase in ["smoke2k", "train5k"]:
        phase_frames[phase] = {}
        for branch in ["forecast", "no_forecast"]:
            phase_frames[phase][branch] = load_branch_summary(branch, phase)
        f, n = phase_frames[phase]["forecast"], phase_frames[phase]["no_forecast"]
        checkpoints = sorted(set(f["checkpoint_step"]) & set(n["checkpoint_step"]))
        for checkpoint in checkpoints:
            for year in VALIDATION_YEARS:
                fr = f[(f["checkpoint_step"] == checkpoint) & (f["year"] == year)]
                nr = n[(n["checkpoint_step"] == checkpoint) & (n["year"] == year)]
                if fr.empty or nr.empty:
                    continue
                fr, nr = fr.iloc[0], nr.iloc[0]
                fd, nd = load_daily(fr), load_daily(nr)
                aligned = fd[["dap", "safe_action_amir", "safe_action_anfer"]].merge(nd[["dap", "safe_action_amir", "safe_action_anfer"]], on="dap", suffixes=("_forecast", "_no_forecast"))
                changed = (aligned["safe_action_amir_forecast"] != aligned["safe_action_amir_no_forecast"]) | (aligned["safe_action_anfer_forecast"] != aligned["safe_action_anfer_no_forecast"])
                rows.append({
                    "phase": phase,
                    "checkpoint_step": int(checkpoint),
                    "year": int(year),
                    "forecast_yield": float(fr.get("final_grnwt", np.nan)),
                    "no_forecast_yield": float(nr.get("final_grnwt", np.nan)),
                    "yield_delta_forecast_minus_no_forecast": float(fr.get("final_grnwt", np.nan) - nr.get("final_grnwt", np.nan)),
                    "forecast_irrigation": float(fr.get("total_irrigation", np.nan)),
                    "no_forecast_irrigation": float(nr.get("total_irrigation", np.nan)),
                    "forecast_pfp_n": float(fr.get("PFP_N", np.nan)),
                    "no_forecast_pfp_n": float(nr.get("PFP_N", np.nan)),
                    "action_change_rate": float(changed.mean()) if len(aligned) else 0.0,
                    "action_changed_rows": int(changed.sum()),
                    "aligned_rows": int(len(aligned)),
                    "forecast_signature_count": int((fd["safe_action_amir"].round(6).astype(str) + "/" + fd["safe_action_anfer"].round(6).astype(str)).nunique()),
                    "no_forecast_signature_count": int((nd["safe_action_amir"].round(6).astype(str) + "/" + nd["safe_action_anfer"].round(6).astype(str)).nunique()),
                })
    comparison = pd.DataFrame(rows)
    out = ROOT / "benchmark_results/140_00_sya_paired_forecast_noforecast_response_gate"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    comparison.to_csv(out / "140_00_paired_year_checkpoint_comparison.csv", index=False, encoding="utf-8-sig")
    means = comparison.groupby(["phase", "checkpoint_step"], as_index=False).agg(
        mean_forecast_yield=("forecast_yield", "mean"),
        mean_no_forecast_yield=("no_forecast_yield", "mean"),
        mean_yield_delta=("yield_delta_forecast_minus_no_forecast", "mean"),
        mean_action_change_rate=("action_change_rate", "mean"),
        mean_forecast_signatures=("forecast_signature_count", "mean"),
        mean_no_forecast_signatures=("no_forecast_signature_count", "mean"),
    )
    means.to_csv(out / "140_00_paired_checkpoint_means.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "completed_paired_readout",
        "comparison_csv": rel(out / "140_00_paired_year_checkpoint_comparison.csv"),
        "means_csv": rel(out / "140_00_paired_checkpoint_means.csv"),
        "rows": int(len(comparison)),
        "means": means.to_dict(orient="records"),
        "wp_et": "not_inferred_without_summary_out_etcp",
        "interpretation_rule": "forecast branch must be judged by response plus yield/PFP_N, not response alone",
    }
    (out / "140_00_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    (ROOT / "docs/2026-08-16_sya_paired_forecast_noforecast_response_gate_execution.md").write_text(
        "\n".join([
            "# 140_00 SY paired forecast/no-forecast response gate",
            "",
            "本轮将同一 seed、动作空间、reward、安全层和年份下的 rolling-forecast PPO 与 raw-observation PPO 配对比较。未启用自动灌溉或 `dssat_auto_external_n`，未运行 25K/100K。",
            "",
            "## 汇总",
            "",
            means.to_markdown(index=False) if not means.empty else "无汇总记录。",
            "",
            "WP_ET 未从缺少有效 ETCP 的逐日 CSV 推断。只有 forecast 分支既产生天气响应又不牺牲产量/PFP_N，才允许继续独立 seed 验证。",
        ]), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=["forecast", "no_forecast"], default="forecast")
    parser.add_argument("--phase", choices=["dry-run", "smoke2k", "train5k", "aggregate"], default="dry-run")
    args = parser.parse_args()
    if args.phase == "dry-run":
        result = preflight(args.branch)
    elif args.phase == "aggregate":
        result = aggregate()
    else:
        result = run_branch(args.branch, args.phase)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=True))
    return 0 if result.get("next_step_allowed", True) or args.phase == "aggregate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
