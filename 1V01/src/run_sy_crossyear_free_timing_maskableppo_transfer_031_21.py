from __future__ import annotations

import hashlib
import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_17 as ppo17
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_21_sy_crossyear_free_timing_maskableppo_transfer.yaml"
OUT = ROOT / "benchmark_results" / "031_21_sy_crossyear_free_timing_maskableppo_transfer"
DOC = ROOT / "docs" / "031_21_sy_crossyear_free_timing_maskableppo_transfer_record.md"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/SYA", "runs", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_selection(config: dict) -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    years = [int(y) for y in config["target_years"]]
    rows = pool[(pool["station_code"].eq(config["target_site"])) & (pool["year"].astype(int).isin(years))].copy()
    if sorted(rows["year"].astype(int).tolist()) != sorted(years):
        raise RuntimeError(f"Expected target years {years}, got {rows['year'].tolist()}")
    rows["selected_for_train"] = False
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "031_21_sy_crossyear_frozen_transfer_eval_only"
    return rows


def clean_scenario(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value)
    if text.strip() == "" or text.lower() == "nan":
        return "null"
    return text


def load_baselines(config: dict) -> pd.DataFrame:
    path = ROOT / config["paths"]["four_baseline_summary_csv"]
    df = pd.read_csv(path)
    df["scenario"] = df["scenario"].map(clean_scenario)
    years = [int(y) for y in config["target_years"]]
    scenarios = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert"]
    rows = df[(df["site"].eq("SY")) & (df["year"].astype(int).isin(years)) & (df["scenario"].isin(scenarios))].copy()
    rows = rows.drop_duplicates(["site", "year", "scenario"], keep="first")
    expected = len(years) * len(scenarios)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} four-baseline rows, found {len(rows)}")
    rows["algorithm"] = "four_baseline_reference_028_05"
    rows["seed"] = np.nan
    rows["checkpoint"] = np.nan
    return rows


def action_sequence(daily: pd.DataFrame) -> str:
    if daily.empty:
        return ""
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    nonzero = daily[(irr > 0) | (n > 0)]
    return "; ".join(
        f"DAP{int(row.dap)} I{float(row.safe_action_amir):g}/N{float(row.safe_action_anfer):g}"
        for row in nonzero.itertuples(index=False)
    )


def evaluate_one(
    config: dict,
    env_config: dict,
    model,
    model_path: Path,
    model_seed: int,
    year: int,
) -> dict[str, Any]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    station = "SYA"
    run_tag = f"SYA_{year}_031_21_seed{model_seed}_frozen_eval"
    env = ppo17.make_env(config, env_config, station, year, model_seed, run_tag, evaluation=True)
    weather = direct_ppo.weather_for_daily(config)
    records: list[dict[str, Any]] = []
    status = "ok"
    notes = ""
    snapshot = OUT / "runs" / str(year) / f"seed{model_seed}" / "snapshot"
    final_y = math.nan
    total_i = math.nan
    total_n = math.nan
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest_pre = ppo17.latest_observation_dict(env, obs, info)
            dap_raw = ppo17.scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = ppo17.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "site": "SY",
                    "year": int(year),
                    "seed": int(model_seed),
                    "source_train_site": config["source_train_site"],
                    "source_train_year": int(config["source_train_year"]),
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": ppo17.scalar(wrow.get("rain"), np.nan),
                    "srad": ppo17.scalar(wrow.get("srad"), np.nan),
                    "tmax": ppo17.scalar(wrow.get("tmax"), np.nan),
                    "tmin": ppo17.scalar(wrow.get("tmin"), np.nan),
                    "swfac": ppo17.scalar(latest.get("swfac")),
                    "nstres": ppo17.scalar(latest.get("nstres")),
                    "topwt": ppo17.scalar(latest.get("topwt")),
                    "grnwt": ppo17.scalar(latest.get("grnwt")),
                    "xlai": ppo17.scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"Evaluation did not finish for SYA {year} seed {model_seed}")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0).sum())
        total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0).sum())
        tmp_snapshot = siteppo.snapshot_from_env(env)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_snapshot, snapshot, dirs_exist_ok=True)
        daily_path = OUT / "daily_outputs" / station / f"{year}_seed{model_seed}_free_timing_frozen_transfer_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        metrics = siteppo.strict_metrics_from_snapshot(snapshot, final_y, total_i, total_n)
        swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
        nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
        return {
            "site": "SY",
            "station": "Shenyang",
            "year": int(year),
            "algorithm": "free_timing_discrete_MaskablePPO",
            "seed": int(model_seed),
            "checkpoint": "031_17_031_18_final_20k",
            "scenario": "rl_candidate_031_21",
            "source_train_site": config["source_train_site"],
            "source_train_year": int(config["source_train_year"]),
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256_file(model_path),
            "run_status": status,
            "notes": notes,
            "final_grain_kg_ha": final_y,
            "final_biomass_kg_ha": float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1]),
            "rain_total_mm": float(pd.to_numeric(daily["rain"], errors="coerce").fillna(0).sum()),
            "irrigation_event_total_mm": total_i,
            "nitrogen_event_total_kg_ha": total_n,
            "max_water_stress_wspd": float(swfac.max()) if len(swfac) else np.nan,
            "max_nitrogen_stress_nstd": float(nstres.max()) if len(nstres) else np.nan,
            "common_reward_total": final_y - total_i - 5.0 * total_n if np.isfinite(final_y) else np.nan,
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            "action_sequence": action_sequence(daily),
            "irrigation_event_count": int((pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0) > 0).sum()),
            "n_event_count": int((pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0) > 0).sum()),
            "first_irrigation_dap": int(daily.loc[pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0) > 0, "dap"].iloc[0])
            if (pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0) > 0).any()
            else np.nan,
            "first_n_dap": int(daily.loc[pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0) > 0, "dap"].iloc[0])
            if (pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0) > 0).any()
            else np.nan,
            "etcp_mm": metrics["etcp_mm"],
            "wp_et_kg_m3": metrics["WP_ET_kg_m3"],
            "pfp_n_kg_kg": metrics["PFP_N_kg_kg"],
            "n_uptake_kg_ha": np.nan,
            "n_leaching_kg_ha": np.nan,
            "summary_match_score": metrics["summary_match_score"],
        }
    except Exception:
        status = "failed"
        notes = traceback.format_exc()
        return {
            "site": "SY",
            "station": "Shenyang",
            "year": int(year),
            "algorithm": "free_timing_discrete_MaskablePPO",
            "seed": int(model_seed),
            "scenario": "rl_candidate_031_21",
            "source_train_site": config["source_train_site"],
            "source_train_year": int(config["source_train_year"]),
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": sha256_file(model_path) if model_path.exists() else "",
            "run_status": status,
            "notes": notes[-2500:],
        }
    finally:
        env.close()


def add_gap_columns(candidates: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in candidates.itertuples(index=False):
        y = int(row.year)
        base = baselines[baselines["year"].astype(int).eq(y)].copy()
        max_y = float(pd.to_numeric(base["final_grain_kg_ha"], errors="coerce").max())
        max_wp = float(pd.to_numeric(base["wp_et_kg_m3"], errors="coerce").max())
        pfp_series = pd.to_numeric(base["pfp_n_kg_kg"], errors="coerce").dropna()
        max_pfp = float(pfp_series.max()) if not pfp_series.empty else math.nan
        d = row._asdict()
        ry = float(d.get("final_grain_kg_ha", math.nan))
        rwp = float(d.get("wp_et_kg_m3", math.nan))
        rpfp = float(d.get("pfp_n_kg_kg", math.nan))
        d.update(
            {
                "baseline_max_yield": max_y,
                "baseline_max_wp_et": max_wp,
                "baseline_max_pfp_n_positive_n": max_pfp,
                "gap_yield": ry - max_y if math.isfinite(ry) else math.nan,
                "gap_wp_et": rwp - max_wp if math.isfinite(rwp) else math.nan,
                "gap_pfp_n": rpfp - max_pfp if math.isfinite(rpfp) and math.isfinite(max_pfp) else math.nan,
                "gap_pct_yield": (ry / max_y - 1.0) * 100.0 if math.isfinite(ry) and max_y else math.nan,
                "gap_pct_wp_et": (rwp / max_wp - 1.0) * 100.0 if math.isfinite(rwp) and max_wp else math.nan,
                "gap_pct_pfp_n": (rpfp / max_pfp - 1.0) * 100.0 if math.isfinite(rpfp) and math.isfinite(max_pfp) and max_pfp else math.nan,
                "yield_strict_win": bool(math.isfinite(ry) and ry > max_y),
                "wp_et_strict_win": bool(math.isfinite(rwp) and rwp > max_wp),
                "pfp_n_strict_win": bool(math.isfinite(rpfp) and math.isfinite(max_pfp) and rpfp > max_pfp),
            }
        )
        d["advisor_any_metric_strict_winner"] = bool(d["yield_strict_win"] or d["wp_et_strict_win"] or d["pfp_n_strict_win"])
        wins = [name for name, flag in [("yield", d["yield_strict_win"]), ("WP_ET", d["wp_et_strict_win"]), ("PFP_N", d["pfp_n_strict_win"])] if flag]
        d["winning_metrics"] = ";".join(wins)
        rows.append(d)
    return pd.DataFrame(rows)


def write_record(config: dict, candidates: pd.DataFrame, gap: pd.DataFrame, baselines: pd.DataFrame) -> None:
    ok = candidates[candidates["run_status"].eq("ok")].copy()
    lines = [
        "# 031_21 SY cross-year frozen transfer of free-timing MaskablePPO",
        "",
        "## Scope",
        "",
        "- Corrective task after 031_20: this is SY same-station cross-year frozen transfer, not cross-site transfer.",
        "- No new training; load SYA2014 seed0/1/2 models from 031_17/031_18.",
        "- Target years: SYA2012 and SYA2015, chosen because four baseline Summary.OUT metrics already exist from 028_05.",
        "- Reuse four baseline rows only; old 028_05 RL candidate rows are not reused.",
        "- Candidate WP_ET/PFP_N are parsed from DSSAT Summary.OUT snapshot for each new frozen evaluation.",
        "",
        "## Model and constraint",
        "",
        "- Algorithm: free-timing discrete MaskablePPO.",
        "- Source train site-year: SYA2014.",
        "- Action grid: irrigation [0, 6, 12, 18, 24] mm x nitrogen [0, 40, 80, 120, 160] kg/ha.",
        "- Safety: I cap 160 mm, N cap 250 kg/ha, 7-day min interval, irrigation DAP 1-120, fertilization DAP 1-90.",
        "- Reward used during original training: 0.001 * (0.158*final_yield - 1.1*I - 1.58*N).",
        "",
        "## Candidate summary",
        "",
        ok[
            [
                "year",
                "seed",
                "final_grain_kg_ha",
                "wp_et_kg_m3",
                "pfp_n_kg_kg",
                "irrigation_event_total_mm",
                "nitrogen_event_total_kg_ha",
                "max_water_stress_wspd",
                "max_nitrogen_stress_nstd",
                "action_sequence",
            ]
        ].to_string(index=False)
        if not ok.empty
        else "No successful candidate rows.",
        "",
        "## Gap versus four baselines",
        "",
        gap[
            [
                "year",
                "seed",
                "gap_yield",
                "gap_wp_et",
                "gap_pfp_n",
                "yield_strict_win",
                "wp_et_strict_win",
                "pfp_n_strict_win",
                "advisor_any_metric_strict_winner",
                "winning_metrics",
            ]
        ].to_string(index=False)
        if not gap.empty
        else "No gap rows.",
        "",
        "## Four-baseline source",
        "",
        baselines[
            [
                "year",
                "scenario",
                "final_grain_kg_ha",
                "wp_et_kg_m3",
                "pfp_n_kg_kg",
                "irrigation_event_total_mm",
                "nitrogen_event_total_kg_ha",
            ]
        ].to_string(index=False),
        "",
        "## Interpretation boundary",
        "",
        "- This task answers only whether SYA2014 free-timing PPO frozen models transfer to SYA2012/SYA2015 under the same station.",
        "- It does not answer cross-site generalization.",
        "- It does not retrain or tune hyperparameters.",
        "- If results are poor, the result is recorded as transfer failure rather than fixed in-place.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    config = direct_ppo.load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    selection = make_selection(config)
    selection.to_csv(OUT / "configs" / "031_21_sy_target_year_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "031_21_resolved_env_config.yaml")
    baselines = load_baselines(config)
    baselines.to_csv(OUT / "evaluation" / "031_21_four_baselines_reused_from_028_05.csv", index=False, encoding="utf-8-sig")

    from sb3_contrib import MaskablePPO

    candidate_rows = []
    for model_spec in config["source_models"]:
        model_seed = int(model_spec["seed"])
        model_path = ROOT / model_spec["model_path"]
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = MaskablePPO.load(str(model_path), device="cpu")
        for year in [int(y) for y in config["target_years"]]:
            row = evaluate_one(config, env_config, model, model_path, model_seed, year)
            candidate_rows.append(row)
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(OUT / "evaluation" / "031_21_candidate_summary.csv", index=False, encoding="utf-8-sig")

    full_cols = list(baselines.columns)
    for col in candidates.columns:
        if col not in full_cols:
            full_cols.append(col)
    five = pd.concat([baselines.reindex(columns=full_cols), candidates.reindex(columns=full_cols)], ignore_index=True, sort=False)
    five.to_csv(OUT / "evaluation" / "031_21_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    ok_candidates = candidates[candidates["run_status"].eq("ok")].copy()
    gap = add_gap_columns(ok_candidates, baselines) if not ok_candidates.empty else pd.DataFrame()
    gap.to_csv(OUT / "evaluation" / "031_21_advisor_any_metric_gap_summary.csv", index=False, encoding="utf-8-sig")
    write_record(config, candidates, gap, baselines)

    result = {
        "task": "031_21_sy_crossyear_free_timing_maskableppo_transfer",
        "training_run": False,
        "source_train_site_year": "SYA2014",
        "target_years": config["target_years"],
        "record_md": str(DOC.relative_to(ROOT)),
        "candidate_summary": str((OUT / "evaluation" / "031_21_candidate_summary.csv").relative_to(ROOT)),
        "gap_summary": str((OUT / "evaluation" / "031_21_advisor_any_metric_gap_summary.csv").relative_to(ROOT)),
    }
    (OUT / "031_21_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not gap.empty:
        print(gap[["year", "seed", "gap_yield", "gap_wp_et", "gap_pfp_n", "advisor_any_metric_strict_winner", "winning_metrics"]].to_string(index=False))
    else:
        print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()

