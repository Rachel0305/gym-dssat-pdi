"""Read-only action/reward/forecast diagnostic for SY experiment 126."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/127_00_sya_forecast_checkpoint_collapse_diagnostic.json"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def entropy(values: pd.Series) -> float:
    counts = values.astype(str).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum()) if len(counts) else 0.0


def numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(index=frame.index, dtype=float)), errors="coerce")


def bool_count(frame: pd.DataFrame, col: str) -> int:
    if col not in frame.columns:
        return 0
    values = frame[col].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    return int(values.sum())


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output_root"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {rel(out)}")
    out.mkdir(parents=True)
    source = ROOT / cfg["source_run"]
    years = list(map(int, cfg["validation_years"]))
    checkpoints = list(map(int, cfg["checkpoints"]))
    issues: list[str] = []
    action_rows: list[dict] = []
    reward_rows: list[dict] = []
    corr_rows: list[dict] = []
    valid_rows = 0
    expected_i = set(map(float, cfg["actions"]["irrigation_levels_mm"]))
    expected_n = set(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
    for checkpoint in checkpoints:
        checkpoint_frame: list[pd.DataFrame] = []
        for year in years:
            path = source / "daily_outputs" / "SYA" / f"SYA_{year}_seed0_ckpt{checkpoint}_daily.csv"
            if not path.exists():
                issues.append(f"daily_missing:{year}:{checkpoint}")
                continue
            daily = pd.read_csv(path, keep_default_na=False)
            valid_rows += 1
            safe_i = numeric(daily, "safe_action_amir").fillna(0.0)
            safe_n = numeric(daily, "safe_action_anfer").fillna(0.0)
            req_i = numeric(daily, "requested_action_amir")
            req_n = numeric(daily, "requested_action_anfer")
            if not set(np.round(safe_i.dropna().unique(), 6)).issubset(set(np.round(list(expected_i), 6))):
                issues.append(f"irrigation_off_grid:{year}:{checkpoint}")
            if not set(np.round(safe_n.dropna().unique(), 6)).issubset(set(np.round(list(expected_n), 6))):
                issues.append(f"nitrogen_off_grid:{year}:{checkpoint}")
            pair = safe_i.round(6).astype(str) + "/" + safe_n.round(6).astype(str)
            post = numeric(daily, "dap").gt(1)
            positive = (safe_i > 1e-9) | (safe_n > 1e-9)
            checkpoint_frame.append(daily.assign(_safe_i=safe_i, _safe_n=safe_n, _pair=pair))
            action_rows.append({
                "checkpoint_step": checkpoint, "year": year, "rows": int(len(daily)), "distinct_action_pairs": int(pair.nunique()), "pair_entropy_bits": entropy(pair), "irrigation_entropy_bits": entropy(safe_i), "nitrogen_entropy_bits": entropy(safe_n), "positive_action_rate": float(positive.mean()), "positive_action_rate_after_dap1": float(positive[post].mean()) if post.any() else 0.0, "requested_safe_irrigation_mismatch": int((numeric(daily, "raw_action_amir") - safe_i).abs().gt(1e-9).sum()), "requested_safe_nitrogen_mismatch": int((numeric(daily, "raw_action_anfer") - safe_n).abs().gt(1e-9).sum()), "mask_forced_noop_count": bool_count(daily, "mask_forced_noop"), "safety_trigger_count": bool_count(daily, "action_safety_triggered"), "mean_valid_action_count": float(numeric(daily, "mask_valid_action_count").mean()), "forecast_columns": int(sum(c.startswith("forecast_raw_") for c in daily.columns)), "forecast_columns_varying": int(sum(numeric(daily, c).max() - numeric(daily, c).min() > 1e-9 for c in daily.columns if c.startswith("forecast_raw_"))), "wp_et_available": False
            })
            reward_cols = ["reward", "reward_unscaled", "reward_stress_aware", "yield_component", "resource_cost", "water_stress_relief", "nitrogen_stress_relief", "stress_relief_bonus", "swfac_guardrail_penalty_scaled", "reward_after_swfac_guardrail"]
            rec = {"checkpoint_step": checkpoint, "year": year}
            for col in reward_cols:
                rec[f"mean_{col}"] = float(numeric(daily, col).mean()) if col in daily.columns else np.nan
                rec[f"sum_{col}"] = float(numeric(daily, col).sum()) if col in daily.columns else np.nan
            reward_rows.append(rec)
            for col in [c for c in daily.columns if c.startswith("forecast_raw_")]:
                x = numeric(daily, col)
                ok = x.notna() & safe_i.notna()
                corr = float(x[ok].corr(safe_i[ok])) if ok.sum() >= 3 and x[ok].nunique() > 1 and safe_i[ok].nunique() > 1 else np.nan
                corr_rows.append({"checkpoint_step": checkpoint, "year": year, "feature": col, "pearson_corr_with_safe_irrigation": corr, "feature_std": float(x[ok].std()) if ok.any() else np.nan, "action_distinct": int(safe_i[ok].nunique()) if ok.any() else 0})
        if checkpoint_frame:
            combined = pd.concat(checkpoint_frame, ignore_index=True)
            pair = combined["_pair"]
            action_rows.append({"checkpoint_step": checkpoint, "year": "ALL", "rows": int(len(combined)), "distinct_action_pairs": int(pair.nunique()), "pair_entropy_bits": entropy(pair), "irrigation_entropy_bits": entropy(combined["_safe_i"]), "nitrogen_entropy_bits": entropy(combined["_safe_n"]), "positive_action_rate": float(((combined["_safe_i"] > 1e-9) | (combined["_safe_n"] > 1e-9)).mean()), "positive_action_rate_after_dap1": np.nan, "requested_safe_irrigation_mismatch": np.nan, "requested_safe_nitrogen_mismatch": np.nan, "mask_forced_noop_count": np.nan, "safety_trigger_count": np.nan, "mean_valid_action_count": np.nan, "forecast_columns": np.nan, "forecast_columns_varying": np.nan, "wp_et_available": False})
    action_df = pd.DataFrame(action_rows)
    reward_df = pd.DataFrame(reward_rows)
    corr_df = pd.DataFrame(corr_rows)
    action_df.to_csv(out / "127_00_action_entropy_by_year.csv", index=False, encoding="utf-8-sig")
    reward_df.to_csv(out / "127_00_reward_components_by_year.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(out / "127_00_forecast_action_correlations.csv", index=False, encoding="utf-8-sig")
    aggregate = action_df[action_df["year"].eq("ALL")].copy()
    summary = {"task": "127_00_sya_forecast_checkpoint_collapse_diagnostic", "status": "completed_read_only", "issues": issues, "n_checkpoint_year_files": valid_rows, "action_entropy": aggregate.to_dict(orient="records"), "reward_component_file": rel(out / "127_00_reward_components_by_year.csv"), "forecast_correlation_file": rel(out / "127_00_forecast_action_correlations.csv"), "wp_et_available": False, "next_step_allowed": bool(not issues)}
    (out / "127_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
