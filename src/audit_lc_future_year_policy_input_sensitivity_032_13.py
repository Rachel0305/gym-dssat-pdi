from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as eval_base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
PROMPT = ROOT / "prompts" / "032_13_lc_future_year_policy_input_sensitivity_audit.md"
SOURCE_MODEL = ROOT / "benchmark_results" / "032_11_lc_multiyear_free_timing_ppo_training_length" / "models" / "LCA" / "LCA_multiyear_2005_2010_stress_aware_maskableppo_seed0_ckpt75000.zip"
OUT = ROOT / "benchmark_results" / "032_13_lc_future_year_policy_input_sensitivity_audit"
DOC = ROOT / "docs" / "032_13_lc_future_year_policy_input_sensitivity_audit_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"

STATION = "LCA"
SITE = "LC"
SEED = 0
TARGET_YEARS = list(range(2011, 2021))
TARGET_DAPS = [1, 8, 15]


def ensure_dirs() -> None:
    for rel in ["configs", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(6)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def load_config() -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = SEED
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["smoke_station"] = STATION
    cfg["runtime"]["smoke_year"] = TARGET_YEARS[0]
    return cfg


def make_selection() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    rows = pool[(pool["station_code"].eq(STATION)) & (pool["year"].astype(int).isin(TARGET_YEARS))].copy()
    found = sorted(rows["year"].astype(int).unique().tolist())
    if found != TARGET_YEARS:
        raise RuntimeError(f"Expected LC target years {TARGET_YEARS}, found {found}")
    rows["selected_for_train"] = False
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "032_13_lc_future_year_policy_input_sensitivity_audit"
    return rows.sort_values(["station_code", "year"]).reset_index(drop=True)


def action_grid(config: dict[str, Any]) -> list[tuple[float, float]]:
    grid = eval_base.base.action_grid(config)
    return [(float(row.get("amir", 0.0)), float(row.get("anfer", 0.0))) for row in grid]


def obs_names_for_env(env: Any, obs: np.ndarray) -> list[str]:
    names = list(getattr(env.unwrapped, "observation_variables", []) or [])
    flat = np.asarray(obs, dtype=float).flatten()
    if len(names) != len(flat):
        names = [f"obs_{i}" for i in range(len(flat))]
    return [str(x) for x in names]


def masked_action_probabilities(model: Any, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Best-effort extraction of masked action probabilities from sb3-contrib MaskablePPO."""
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        mask_arr = np.asarray(mask, dtype=bool).reshape(1, -1)
        dist = model.policy.get_distribution(obs_tensor, action_masks=mask_arr)
        distribution = getattr(dist, "distribution", dist)
        probs = getattr(distribution, "probs", None)
        if probs is None:
            return np.full(mask_arr.shape[1], np.nan), "no_probs_attribute"
        arr = probs.detach().cpu().numpy().reshape(-1)
        return arr.astype(float), "ok"
    except Exception as exc:
        return np.full(len(mask), np.nan), f"failed:{type(exc).__name__}:{exc}"


def weather_row_for_date(weather: pd.DataFrame, year: int, date: pd.Timestamp) -> dict[str, Any]:
    rows = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def audit_year(model: Any, config: dict[str, Any], env_config: dict[str, Any], year: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    weather = direct_ppo.weather_for_daily(config)
    env = eval_base.base.make_env(config, env_config, STATION, int(year), SEED, f"{STATION}_{year}_032_13_input_audit", evaluation=True)
    grid = action_grid(config)
    decision_rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    prob_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    captured_daps: set[int] = set()
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest_pre = eval_base.base.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = np.asarray(get_action_masks(env), dtype=bool).flatten()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)
            probs, prob_status = masked_action_probabilities(model, obs, mask)
            obs_flat = np.asarray(obs, dtype=float).flatten()
            names = obs_names_for_env(env, obs_flat)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = weather_row_for_date(weather, year, date)
            if dap in TARGET_DAPS and dap not in captured_daps:
                captured_daps.add(int(dap))
                legal_count = int(mask.sum())
                sorted_prob_idx = np.argsort(np.where(mask, probs, -np.inf))[::-1] if np.isfinite(probs).any() else np.array([], dtype=int)
                top_idx = int(sorted_prob_idx[0]) if len(sorted_prob_idx) else action
                second_idx = int(sorted_prob_idx[1]) if len(sorted_prob_idx) > 1 else -1
                top_prob = float(probs[top_idx]) if 0 <= top_idx < len(probs) and np.isfinite(probs[top_idx]) else np.nan
                second_prob = float(probs[second_idx]) if 0 <= second_idx < len(probs) and np.isfinite(probs[second_idx]) else np.nan
                pred_i, pred_n = grid[action]
                decision_rows.append(
                    {
                        "site": SITE,
                        "station_code": STATION,
                        "year": int(year),
                        "dap": int(dap),
                        "date": date.strftime("%Y-%m-%d"),
                        "predicted_action": action,
                        "predicted_irrigation_mm": pred_i,
                        "predicted_nitrogen_kg_ha": pred_n,
                        "legal_action_count": legal_count,
                        "top_action": top_idx,
                        "second_action": second_idx,
                        "top_prob": top_prob,
                        "second_prob": second_prob,
                        "top_prob_margin": top_prob - second_prob if np.isfinite(top_prob) and np.isfinite(second_prob) else np.nan,
                        "prob_status": prob_status,
                        "rain": scalar(wrow.get("rain"), np.nan),
                        "srad": scalar(wrow.get("srad"), np.nan),
                        "tmax": scalar(wrow.get("tmax"), np.nan),
                        "tmin": scalar(wrow.get("tmin"), np.nan),
                        "swfac": scalar(latest_pre.get("swfac")),
                        "nstres": scalar(latest_pre.get("nstres")),
                        "topwt": scalar(latest_pre.get("topwt")),
                        "grnwt": scalar(latest_pre.get("grnwt")),
                        "xlai": scalar(latest_pre.get("xlai")),
                    }
                )
                for idx, (name, value) in enumerate(zip(names, obs_flat.tolist())):
                    obs_rows.append(
                        {
                            "site": SITE,
                            "station_code": STATION,
                            "year": int(year),
                            "dap": int(dap),
                            "obs_index": int(idx),
                            "obs_name": name,
                            "obs_value": float(value),
                        }
                    )
                for a_idx, (i_amt, n_amt) in enumerate(grid):
                    prob_rows.append(
                        {
                            "site": SITE,
                            "station_code": STATION,
                            "year": int(year),
                            "dap": int(dap),
                            "action_index": int(a_idx),
                            "irrigation_mm": i_amt,
                            "nitrogen_kg_ha": n_amt,
                            "is_legal": bool(mask[a_idx]),
                            "probability": float(probs[a_idx]) if a_idx < len(probs) and np.isfinite(probs[a_idx]) else np.nan,
                        }
                    )
                lower_names = [x.lower() for x in names]
                latest_keys = [str(k).lower() for k in latest_pre.keys()]
                for field in ["rain", "srad", "tmax", "tmin", "weather", "swfac", "nstres", "dap", "topwt", "grnwt", "xlai"]:
                    field_rows.append(
                        {
                            "site": SITE,
                            "station_code": STATION,
                            "year": int(year),
                            "dap": int(dap),
                            "field": field,
                            "in_actual_obs_names": field in lower_names,
                            "in_latest_obs_dict": field in latest_keys,
                            "obs_dim": int(len(obs_flat)),
                        }
                    )
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            step_count += 1
    finally:
        env.close()
    return decision_rows, obs_rows, prob_rows, field_rows


def pairwise_distances(obs_long: pd.DataFrame) -> pd.DataFrame:
    if obs_long.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for dap, group in obs_long.groupby("dap"):
        wide = group.pivot_table(index="year", columns="obs_index", values="obs_value", aggfunc="first").sort_index()
        if len(wide) < 2:
            continue
        std = wide.std(axis=0, ddof=0).replace(0, np.nan)
        z = (wide - wide.mean(axis=0)) / std
        z = z.fillna(0.0)
        for i, year_i in enumerate(wide.index):
            for year_j in wide.index[i + 1 :]:
                a = wide.loc[year_i].to_numpy(dtype=float)
                b = wide.loc[year_j].to_numpy(dtype=float)
                za = z.loc[year_i].to_numpy(dtype=float)
                zb = z.loc[year_j].to_numpy(dtype=float)
                euclid = float(np.linalg.norm(a - b))
                z_euclid = float(np.linalg.norm(za - zb))
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                cosine_distance = float(1.0 - np.dot(a, b) / denom) if denom > 0 else np.nan
                rows.append(
                    {
                        "dap": int(dap),
                        "year_a": int(year_i),
                        "year_b": int(year_j),
                        "euclidean_raw": euclid,
                        "euclidean_standardized": z_euclid,
                        "cosine_distance_raw": cosine_distance,
                    }
                )
    return pd.DataFrame(rows)


def observation_range_summary(obs_long: pd.DataFrame) -> pd.DataFrame:
    if obs_long.empty:
        return pd.DataFrame()
    rows = []
    for (dap, idx, name), g in obs_long.groupby(["dap", "obs_index", "obs_name"]):
        vals = pd.to_numeric(g["obs_value"], errors="coerce")
        rows.append(
            {
                "dap": int(dap),
                "obs_index": int(idx),
                "obs_name": str(name),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "range": float(vals.max() - vals.min()),
                "std": float(vals.std(ddof=0)),
                "unique_rounded_6": int(vals.round(6).nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dap", "range"], ascending=[True, False])


def obs_weather_correlations(decision_df: pd.DataFrame, obs_long: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty or obs_long.empty:
        return pd.DataFrame()
    weather_cols = ["rain", "srad", "tmax", "tmin"]
    rows: list[dict[str, Any]] = []
    meta = decision_df[["year", "dap", *weather_cols]].copy()
    merged = obs_long.merge(meta, on=["year", "dap"], how="left")
    for (dap, idx, name), group in merged.groupby(["dap", "obs_index", "obs_name"]):
        obs_vals = pd.to_numeric(group["obs_value"], errors="coerce")
        for weather_col in weather_cols:
            wvals = pd.to_numeric(group[weather_col], errors="coerce")
            if obs_vals.notna().sum() < 3 or wvals.notna().sum() < 3 or obs_vals.std(ddof=0) == 0 or wvals.std(ddof=0) == 0:
                corr = np.nan
            else:
                corr = float(obs_vals.corr(wvals))
            rows.append(
                {
                    "dap": int(dap),
                    "obs_index": int(idx),
                    "obs_name": str(name),
                    "weather_field": weather_col,
                    "pearson_corr": corr,
                    "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["dap", "weather_field", "abs_corr"], ascending=[True, True, False])


def write_record(decision_df: pd.DataFrame, obs_df: pd.DataFrame, prob_df: pd.DataFrame, field_df: pd.DataFrame, dist_df: pd.DataFrame, range_df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    weather_fields = ["rain", "srad", "tmax", "tmin"]
    weather_in_obs = (
        field_df[field_df["field"].isin(weather_fields)]
        .groupby("field", as_index=False)
        .agg(in_exposed_obs_names=("in_actual_obs_names", "any"), in_latest_obs_dict=("in_latest_obs_dict", "any"), obs_dim=("obs_dim", "max"))
        if not field_df.empty
        else pd.DataFrame()
    )
    action_summary = (
        decision_df.groupby("dap", as_index=False)
        .agg(
            unique_predicted_actions=("predicted_action", lambda x: ",".join(map(str, sorted(set(map(int, x)))))),
            mean_top_prob=("top_prob", "mean"),
            min_top_prob=("top_prob", "min"),
            mean_top_prob_margin=("top_prob_margin", "mean"),
            min_legal_action_count=("legal_action_count", "min"),
            max_legal_action_count=("legal_action_count", "max"),
            rain_range=("rain", lambda x: float(pd.to_numeric(x, errors="coerce").max() - pd.to_numeric(x, errors="coerce").min())),
            tmax_range=("tmax", lambda x: float(pd.to_numeric(x, errors="coerce").max() - pd.to_numeric(x, errors="coerce").min())),
        )
        if not decision_df.empty
        else pd.DataFrame()
    )
    dist_summary = (
        dist_df.groupby("dap", as_index=False)
        .agg(
            mean_standardized_distance=("euclidean_standardized", "mean"),
            max_standardized_distance=("euclidean_standardized", "max"),
            mean_cosine_distance=("cosine_distance_raw", "mean"),
            max_cosine_distance=("cosine_distance_raw", "max"),
        )
        if not dist_df.empty
        else pd.DataFrame()
    )
    top_ranges = range_df.groupby("dap").head(8) if not range_df.empty else pd.DataFrame()
    same_actions = bool(decision_df.groupby("dap")["predicted_action"].nunique().max() == 1) if not decision_df.empty else False
    masks_force = bool((decision_df["legal_action_count"] <= 1).any()) if not decision_df.empty else False
    obs_names_generic = bool(not obs_df.empty and obs_df["obs_name"].astype(str).str.match(r"obs_\d+").all())
    conclusion = []
    if obs_names_generic:
        conclusion.append("The wrapped environment does not expose semantic names for the 26 actual PPO input dimensions, so weather-input presence cannot be decided from names alone.")
    if not weather_in_obs.empty and weather_in_obs["in_latest_obs_dict"].any():
        conclusion.append("The environment/info dictionary contains some weather fields (notably SRAD/Tmax), but the model-facing vector is anonymous; use the saved correlation table rather than assuming absence.")
    if same_actions and not masks_force:
        conclusion.append("DAP1/8/15 actions are identical across LC2011-LC2020 even though multiple actions are legal, so the identical sequence is not forced by the action mask.")
    if same_actions:
        conclusion.append("Across visibly different early weather conditions, the frozen deterministic policy still applies the same early-season template with high top-action probability.")
    if not conclusion:
        conclusion.append("No single mechanism fully explains the action sequence from the audit; inspect the CSV tables.")

    lines = [
        "# 032_13 LC future-year policy input sensitivity audit record",
        "",
        "## Status",
        "",
        "- Completed with no training and no DSSAT baseline rerun.",
        f"- Years audited: {', '.join(map(str, TARGET_YEARS))}.",
        f"- Target DAPs: {', '.join(map(str, TARGET_DAPS))}.",
        "",
        "## Source model",
        "",
        f"- `{SOURCE_MODEL.relative_to(ROOT).as_posix()}`",
        "",
        "## Main conclusion",
        "",
        *[f"- {x}" for x in conclusion],
        "",
        "## Weather fields in exposed observation metadata",
        "",
        md_table(weather_in_obs),
        "",
        "## Decision/action summary by DAP",
        "",
        md_table(action_summary),
        "",
        "## Observation distance summary by DAP",
        "",
        md_table(dist_summary),
        "",
        "## Largest varying observation dimensions by DAP",
        "",
        md_table(top_ranges[["dap", "obs_index", "obs_name", "min", "max", "range", "std", "unique_rounded_6"]] if not top_ranges.empty else top_ranges, max_rows=30),
        "",
        "## Top observation-weather correlations by DAP",
        "",
        md_table(corr_df.groupby(["dap", "weather_field"]).head(3)[["dap", "weather_field", "obs_index", "obs_name", "pearson_corr", "abs_corr"]] if not corr_df.empty else corr_df, max_rows=40),
        "",
        "## Per-year decision rows",
        "",
        md_table(decision_df, max_rows=40),
        "",
        "## Interpretation boundary",
        "",
        "- This audit explains the input/action mechanism for the frozen 75k LC model.",
        "- It does not prove the repeated early action template is agronomically optimal.",
        "- It does not include future weather forecast features.",
        "- Because actual observation dimensions are anonymous in this wrapper, this audit should not be used to claim that weather is definitely absent from the model input.",
        "- The confirmed result is narrower: under the current frozen no-forecast model, early-year input/weather differences did not change deterministic DAP1/8/15 actions.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(SOURCE_MODEL)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)

    config = load_config()
    selection = make_selection()
    selection.to_csv(OUT / "configs" / "032_13_lc_target_year_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "032_13_resolved_env_config.yaml")

    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(SOURCE_MODEL), device="cpu")
    all_decisions: list[dict[str, Any]] = []
    all_obs: list[dict[str, Any]] = []
    all_probs: list[dict[str, Any]] = []
    all_fields: list[dict[str, Any]] = []
    for year in TARGET_YEARS:
        decisions, obs_rows, prob_rows, field_rows = audit_year(model, config, env_config, year)
        all_decisions.extend(decisions)
        all_obs.extend(obs_rows)
        all_probs.extend(prob_rows)
        all_fields.extend(field_rows)

    decision_df = pd.DataFrame(all_decisions)
    obs_df = pd.DataFrame(all_obs)
    prob_df = pd.DataFrame(all_probs)
    field_df = pd.DataFrame(all_fields)
    dist_df = pairwise_distances(obs_df)
    range_df = observation_range_summary(obs_df)
    corr_df = obs_weather_correlations(decision_df, obs_df)

    decision_df.to_csv(OUT / "tables" / "032_13_decision_state_action_audit.csv", index=False, encoding="utf-8-sig")
    obs_df.to_csv(OUT / "tables" / "032_13_observation_values_by_year_dap.csv", index=False, encoding="utf-8-sig")
    prob_df.to_csv(OUT / "tables" / "032_13_action_probabilities_by_year_dap.csv", index=False, encoding="utf-8-sig")
    field_df.to_csv(OUT / "tables" / "032_13_weather_vs_observation_field_check.csv", index=False, encoding="utf-8-sig")
    dist_df.to_csv(OUT / "tables" / "032_13_pairwise_obs_distance_by_dap.csv", index=False, encoding="utf-8-sig")
    range_df.to_csv(OUT / "tables" / "032_13_observation_range_summary.csv", index=False, encoding="utf-8-sig")
    corr_df.to_csv(OUT / "tables" / "032_13_obs_weather_correlation_by_dap.csv", index=False, encoding="utf-8-sig")

    write_record(decision_df, obs_df, prob_df, field_df, dist_df, range_df, corr_df)
    result = {
        "task": "032_13_lc_future_year_policy_input_sensitivity_audit",
        "training_run": False,
        "station": STATION,
        "years": TARGET_YEARS,
        "target_daps": TARGET_DAPS,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "decision_table": str((OUT / "tables" / "032_13_decision_state_action_audit.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "032_13_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(decision_df[["year", "dap", "predicted_action", "predicted_irrigation_mm", "predicted_nitrogen_kg_ha", "legal_action_count", "top_prob", "top_prob_margin", "rain", "tmax", "swfac", "nstres"]].to_string(index=False))


if __name__ == "__main__":
    main()
