"""Read-only comparison of existing SY forecast and no-forecast 2K runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/125_00_sya_forecast_vs_noforecast_2k_comparison.json"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summary_path(run: Path, task_id: str) -> Path:
    return run / "evaluation" / f"{task_id}_checkpoint_validation_summary.csv"


def load_summary(run: Path, task_id: str, years: list[int], checkpoint: int) -> pd.DataFrame:
    path = summary_path(run, task_id)
    frame = pd.read_csv(path, keep_default_na=False)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype(int)
    frame["checkpoint_step"] = pd.to_numeric(frame["checkpoint_step"], errors="coerce").astype(int)
    frame = frame[frame["checkpoint_step"].eq(checkpoint) & frame["year"].isin(years)].copy()
    frame["variant"] = "forecast" if task_id == "111_00" else "no_forecast"
    return frame


def load_daily(row: pd.Series) -> pd.DataFrame:
    path = ROOT / str(row["daily_csv_path"])
    daily = pd.read_csv(path, keep_default_na=False)
    daily["dap"] = pd.to_numeric(daily["dap"], errors="coerce")
    daily["safe_action_amir"] = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    daily["safe_action_anfer"] = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
    daily["action_signature"] = daily["safe_action_amir"].round(6).astype(str) + "/" + daily["safe_action_anfer"].round(6).astype(str)
    return daily


def main() -> int:
    cfg = load_json(CONFIG)
    out = ROOT / cfg["output_root"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    years = list(map(int, cfg["validation_years"]))
    issues: list[str] = []
    forecast_run = ROOT / cfg["forecast_run"]
    no_forecast_run = ROOT / cfg["no_forecast_run"]
    for path in [forecast_run, no_forecast_run]:
        if not path.exists():
            issues.append(f"run_missing:{rel(path)}")
    f = load_summary(forecast_run, "111_00", years, int(cfg["checkpoint_step"])) if not issues else pd.DataFrame()
    n = load_summary(no_forecast_run, "046_10", years, int(cfg["checkpoint_step"])) if not issues else pd.DataFrame()
    if len(f) != len(years):
        issues.append(f"forecast_summary_rows:{len(f)}")
    if len(n) != len(years):
        issues.append(f"no_forecast_summary_rows:{len(n)}")
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    forecast_required = ["forecast_raw_" + name for name in ["rain_d1_mm", "srad_d1_mj_m2", "tmax_d1_c", "tmin_d1_c"]]
    for year in years:
        fr = f[f["year"].eq(year)].iloc[0]
        nr = n[n["year"].eq(year)].iloc[0]
        fd = load_daily(fr)
        nd = load_daily(nr)
        aligned = fd[["dap", "safe_action_amir", "safe_action_anfer"]].merge(nd[["dap", "safe_action_amir", "safe_action_anfer"]], on="dap", suffixes=("_forecast", "_no_forecast"))
        action_changed = (aligned["safe_action_amir_forecast"] != aligned["safe_action_amir_no_forecast"]) | (aligned["safe_action_anfer_forecast"] != aligned["safe_action_anfer_no_forecast"])
        forecast_cols = [c for c in fd.columns if c.startswith("forecast_raw_")]
        varying = int(sum(float(pd.to_numeric(fd[c], errors="coerce").max() - pd.to_numeric(fd[c], errors="coerce").min()) > 1e-9 for c in forecast_cols)) if forecast_cols else 0
        invalid_f = int((~fd["safe_action_amir"].isin(cfg["actions"]["irrigation_levels_mm"])).sum() + (~fd["safe_action_anfer"].isin(cfg["actions"]["nitrogen_levels_kg_ha"])).sum())
        invalid_n = int((~nd["safe_action_amir"].isin(cfg["actions"]["irrigation_levels_mm"])).sum() + (~nd["safe_action_anfer"].isin(cfg["actions"]["nitrogen_levels_kg_ha"])).sum())
        row = {"year": year, "forecast_yield": float(fr["final_grnwt"]), "no_forecast_yield": float(nr["final_grnwt"]), "yield_delta_forecast_minus_no_forecast": float(fr["final_grnwt"] - nr["final_grnwt"]), "forecast_irrigation": float(fr["total_irrigation"]), "no_forecast_irrigation": float(nr["total_irrigation"]), "irrigation_delta": float(fr["total_irrigation"] - nr["total_irrigation"]), "forecast_total_n": float(fr["total_n"]), "no_forecast_total_n": float(nr["total_n"]), "forecast_pfp_n": float(fr["PFP_N"]), "no_forecast_pfp_n": float(nr["PFP_N"]), "pfp_n_delta": float(fr["PFP_N"] - nr["PFP_N"]), "action_change_rate": float(action_changed.mean()) if len(aligned) else 0.0, "action_changed_rows": int(action_changed.sum()), "aligned_rows": int(len(aligned)), "forecast_action_signature_count": int(fd["action_signature"].nunique()), "no_forecast_action_signature_count": int(nd["action_signature"].nunique()), "forecast_columns_present": int(len(forecast_cols)), "forecast_columns_varying": varying, "forecast_invalid_action_rows": invalid_f, "no_forecast_invalid_action_rows": invalid_n, "wp_et_available": False}
        rows.append(row)
        if year == 2017:
            for rec in aligned.itertuples(index=False):
                action_rows.append({"year": year, "dap": rec.dap, "forecast_irrigation": rec.safe_action_amir_forecast, "forecast_nitrogen": rec.safe_action_anfer_forecast, "no_forecast_irrigation": rec.safe_action_amir_no_forecast, "no_forecast_nitrogen": rec.safe_action_anfer_no_forecast, "changed": bool((rec.safe_action_amir_forecast != rec.safe_action_amir_no_forecast) or (rec.safe_action_anfer_forecast != rec.safe_action_anfer_no_forecast))})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "125_00_year_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(action_rows).to_csv(out / "125_00_2017_action_comparison.csv", index=False, encoding="utf-8-sig")
    means = {"forecast_yield_mean": float(comparison["forecast_yield"].mean()), "no_forecast_yield_mean": float(comparison["no_forecast_yield"].mean()), "yield_delta_mean": float(comparison["yield_delta_forecast_minus_no_forecast"].mean()), "forecast_irrigation_mean": float(comparison["forecast_irrigation"].mean()), "no_forecast_irrigation_mean": float(comparison["no_forecast_irrigation"].mean()), "forecast_pfp_n_mean": float(comparison["forecast_pfp_n"].mean()), "no_forecast_pfp_n_mean": float(comparison["no_forecast_pfp_n"].mean()), "pfp_n_delta_mean": float(comparison["pfp_n_delta"].mean()), "action_change_rate_mean": float(comparison["action_change_rate"].mean())}
    sy2017 = comparison[comparison["year"].eq(2017)].iloc[0].to_dict() if len(comparison[comparison["year"].eq(2017)]) else {}
    result = {"task": f"{cfg['task_id']}_{cfg['task_name']}", "status": "completed_read_only", "issues": issues, "checkpoint_step": cfg["checkpoint_step"], "n_years": int(len(comparison)), "means": means, "sy2017": sy2017, "wp_et": "unavailable_in_source_summary_and_daily_csv", "forecast_columns_gate": bool((comparison["forecast_columns_present"] >= 28).all() and (comparison["forecast_columns_varying"] >= 4).all()), "action_grid_gate": bool((comparison["forecast_invalid_action_rows"] == 0).all() and (comparison["no_forecast_invalid_action_rows"] == 0).all()), "next_step_allowed": bool(not issues and means["action_change_rate_mean"] > 0.0)}
    (out / "125_00_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "125_00_config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
