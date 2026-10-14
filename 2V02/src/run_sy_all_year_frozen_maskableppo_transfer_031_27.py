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
import yaml

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_18 as ppo18
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_27_sy_all_year_frozen_maskableppo_transfer.yaml"
OUT = ROOT / "benchmark_results" / "031_27_sy_all_year_frozen_maskableppo_transfer"
DOC = ROOT / "docs" / "031_27_sy_all_year_frozen_maskableppo_transfer_record.md"
SOURCE_OUT = ROOT / "benchmark_results" / "031_26_maskableppo_checkpoint_selection_sy_crossyear"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/SYA", "runs", "evaluation", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sy_years(cfg: dict[str, Any]) -> list[int]:
    pool = pd.read_csv(ROOT / cfg["scenario_pool_csv"])
    rows = pool[
        pool["station_code"].eq(cfg["station_code"])
        & (pd.to_numeric(pool["year"], errors="coerce") >= int(cfg.get("min_year", 2000)))
    ].copy()
    years = sorted(rows["year"].astype(int).unique().tolist())
    if not years:
        raise RuntimeError("No SYA years found in scenario pool.")
    return years


def build_env_config_for_years(base_config: dict[str, Any], years: list[int]) -> dict[str, Any]:
    pool = pd.read_csv(ROOT / base_config["paths"]["scenario_pool_csv"])
    selection = pool[
        pool["station_code"].eq("SYA") & (pool["year"].astype(int).isin([int(y) for y in years]))
    ].copy()
    selection["selected_for_train"] = selection["year"].astype(int).eq(2014)
    selection["selected_for_eval"] = True
    selection["selection_reason"] = "031_27_sy_all_year_frozen_transfer"
    env_config = direct_ppo.build_env_config(base_config, selection)
    return env_config


def model_path_for(seed: int, checkpoint_step: int) -> Path:
    return SOURCE_OUT / "models" / "SYA" / f"free_timing_discrete_maskableppo_seed{seed}_ckpt{checkpoint_step}.zip"


def action_sequence(daily: pd.DataFrame) -> str:
    irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0)
    nonzero = daily[(irr > 0) | (n > 0)]
    return "; ".join(
        f"DAP{int(row.dap)} I{float(row.safe_action_amir):g}/N{float(row.safe_action_anfer):g}"
        for row in nonzero.itertuples(index=False)
    )


def evaluate_model_year(
    run_config: dict[str, Any],
    env_config: dict[str, Any],
    seed: int,
    checkpoint_step: int,
    year: int,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    station = "SYA"
    model_path = model_path_for(seed, checkpoint_step)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = MaskablePPO.load(str(model_path), device="cpu")
    env = ppo18.make_env(
        run_config,
        env_config,
        station,
        int(year),
        int(seed),
        f"SYA_{year}_031_27_seed{seed}_ckpt{checkpoint_step}_eval",
        evaluation=True,
    )
    weather = direct_ppo.weather_for_daily(run_config)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, int(year))["planting_date"])
        while not done and step_count < int(run_config["runtime"]["max_steps"]):
            latest_pre = ppo18.latest_observation_dict(env, obs, info)
            dap_raw = ppo18.scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = ppo18.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "site": "SY",
                    "year": int(year),
                    "seed": int(seed),
                    "checkpoint_step": int(checkpoint_step),
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": ppo18.scalar(wrow.get("rain"), np.nan),
                    "srad": ppo18.scalar(wrow.get("srad"), np.nan),
                    "tmax": ppo18.scalar(wrow.get("tmax"), np.nan),
                    "tmin": ppo18.scalar(wrow.get("tmin"), np.nan),
                    "swfac": ppo18.scalar(latest.get("swfac")),
                    "nstres": ppo18.scalar(latest.get("nstres")),
                    "topwt": ppo18.scalar(latest.get("topwt")),
                    "grnwt": ppo18.scalar(latest.get("grnwt")),
                    "xlai": ppo18.scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"Evaluation did not finish for year={year}, seed={seed}, ckpt={checkpoint_step}")
        daily_path = OUT / "daily_outputs" / station / f"{year}_seed{seed}_ckpt{checkpoint_step}_maskableppo_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

        irr = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0)
        n = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0)
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        total_i = float(irr.sum())
        total_n = float(n.sum())
        snapshot = OUT / "runs" / str(year) / f"seed{seed}_ckpt{checkpoint_step}" / "snapshot"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(siteppo.snapshot_from_env(env), snapshot, dirs_exist_ok=True)
        metrics = siteppo.strict_metrics_from_snapshot(snapshot, final_y, total_i, total_n)
        swfac = pd.to_numeric(daily["swfac"], errors="coerce")
        nstres = pd.to_numeric(daily["nstres"], errors="coerce")
        dap = pd.to_numeric(daily["dap"], errors="coerce")
        return {
            "algorithm": "free_timing_discrete_MaskablePPO",
            "station_code": station,
            "site": "SY",
            "year": int(year),
            "seed": int(seed),
            "checkpoint_step": int(checkpoint_step),
            "source_train_year": 2014,
            "scenario": f"rl_candidate_031_27_from031_26_seed{seed}_ckpt{checkpoint_step}",
            "run_status": "ok",
            "model_path": str(model_path.relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": sha256_file(model_path),
            "final_grain_kg_ha": final_y,
            "final_biomass_kg_ha": final_b,
            "total_irrigation": total_i,
            "total_n": total_n,
            "profit_simple": final_y - total_i - 5.0 * total_n,
            "PFP_N": final_y / total_n if total_n > 0 else math.nan,
            "irrigation_event_total_mm": total_i,
            "nitrogen_event_total_kg_ha": total_n,
            "etcp_mm": metrics["etcp_mm"],
            "wp_et_kg_m3": metrics["WP_ET_kg_m3"],
            "pfp_n_kg_kg": metrics["PFP_N_kg_kg"],
            "max_water_stress_wspd": float(swfac.max()),
            "max_nitrogen_stress_nstd": float(nstres.max()),
            "swfac_stress_days_gt_0p05": int((swfac > 0.05).sum()),
            "nstres_days_gt_0p05": int((nstres > 0.05).sum()),
            "early_dap1_10_irrigation": float(irr[dap <= 10].sum()),
            "early_dap1_10_n": float(n[dap <= 10].sum()),
            "irrigation_event_count": int((irr > 0).sum()),
            "n_event_count": int((n > 0).sum()),
            "first_irrigation_dap": int(daily.loc[irr > 0, "dap"].iloc[0]) if (irr > 0).any() else np.nan,
            "first_n_dap": int(daily.loc[n > 0, "dap"].iloc[0]) if (n > 0).any() else np.nan,
            "daily_csv_path": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            "action_sequence": action_sequence(daily),
            "summary_match_score": metrics["summary_match_score"],
        }
    finally:
        env.close()


def clean_scenario(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.strip() == "" or text.lower() == "nan":
        return "null"
    return text


def load_sy_baselines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["site"].eq("SY")].copy()
    df["scenario"] = df["scenario"].map(clean_scenario)
    return df


def add_baseline_comparison(eval_df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_years = set(baselines["year"].astype(int).tolist())
    for row in eval_df.itertuples(index=False):
        d = row._asdict()
        year = int(d["year"])
        if year not in baseline_years:
            d.update(
                {
                    "baseline_comparison_status": "baseline_missing_not_compared",
                    "advisor_any_metric_strict_winner": np.nan,
                    "winning_metrics": "",
                }
            )
            rows.append(d)
            continue
        base = baselines[baselines["year"].astype(int).eq(year)].copy()
        max_y = float(pd.to_numeric(base["grain_yield_kg_ha"], errors="coerce").max())
        max_wp = float(pd.to_numeric(base["WP_ET_recomputed_kg_m3"], errors="coerce").max())
        pfp_vals = pd.to_numeric(base["PFP_N_recomputed_kg_kg"], errors="coerce").dropna()
        max_pfp = float(pfp_vals.max()) if not pfp_vals.empty else math.nan
        y = float(d["final_grain_kg_ha"])
        wp = float(d["wp_et_kg_m3"])
        pfp = float(d["pfp_n_kg_kg"]) if pd.notna(d["pfp_n_kg_kg"]) else math.nan
        gap_y = y - max_y
        gap_wp = wp - max_wp
        gap_pfp = pfp - max_pfp if math.isfinite(pfp) and math.isfinite(max_pfp) else math.nan
        flags = {
            "yield": bool(gap_y > 0),
            "WP_ET": bool(gap_wp > 0),
            "PFP_N": bool(math.isfinite(gap_pfp) and gap_pfp > 0),
        }
        d.update(
            {
                "baseline_comparison_status": "compared_existing_four_baselines",
                "baseline_max_yield": max_y,
                "baseline_max_wp_et": max_wp,
                "baseline_max_pfp_n_positive_n": max_pfp,
                "gap_yield": gap_y,
                "gap_wp_et": gap_wp,
                "gap_pfp_n": gap_pfp,
                "yield_strict_win": flags["yield"],
                "wp_et_strict_win": flags["WP_ET"],
                "pfp_n_strict_win": flags["PFP_N"],
                "advisor_any_metric_strict_winner": any(flags.values()),
                "winning_metrics": ";".join([k for k, v in flags.items() if v]),
            }
        )
        rows.append(d)
    return pd.DataFrame(rows)


def write_record(years: list[int], eval_df: pd.DataFrame, comp_df: pd.DataFrame, baseline_years: list[int]) -> None:
    ok = eval_df[eval_df["run_status"].eq("ok")].copy()
    matrix = (
        comp_df.pivot_table(
            index="year",
            columns="seed",
            values="advisor_any_metric_strict_winner",
            aggfunc="first",
        )
        if not comp_df.empty
        else pd.DataFrame()
    )
    lines = [
        "# 031_27 SY all-year frozen MaskablePPO transfer audit",
        "",
        "## Scope",
        "",
        "- No new training.",
        "- Source models are the three validation-selected 031_26 MaskablePPO checkpoints.",
        "- Site: SY/SYA.",
        f"- Years from scenario pool: {min(years)}-{max(years)} ({len(years)} years): {', '.join(map(str, years))}.",
        "- Four-baseline comparison is only computed where an existing SY four-baseline envelope is available.",
        "",
        "## Source selected models",
        "",
        "- seed0: 031_26 checkpoint 20k",
        "- seed1: 031_26 checkpoint 100k",
        "- seed2: 031_26 checkpoint 50k",
        "",
        "## Completion summary",
        "",
        f"- Expected evaluations: {len(years) * 3}",
        f"- Successful evaluations: {len(ok)}",
        f"- Baseline-comparable years: {', '.join(map(str, baseline_years)) if baseline_years else 'none'}",
        "",
        "## Candidate endpoint summary",
        "",
        ok[
            [
                "year",
                "seed",
                "checkpoint_step",
                "final_grain_kg_ha",
                "wp_et_kg_m3",
                "pfp_n_kg_kg",
                "total_irrigation",
                "total_n",
                "max_water_stress_wspd",
                "max_nitrogen_stress_nstd",
                "first_irrigation_dap",
                "first_n_dap",
            ]
        ].to_string(index=False)
        if not ok.empty
        else "No successful evaluations.",
        "",
        "## Existing-baseline comparison matrix",
        "",
        matrix.to_string() if not matrix.empty else "No comparable years.",
        "",
        "## Boundary",
        "",
        "Years without existing four-baseline envelopes are not labelled as successes or failures against the four baselines in this task.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cfg = load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    source_config = load_yaml(ROOT / cfg["source_base_config"])
    base_config = load_yaml(ROOT / source_config["base_config"])
    run_config = json.loads(json.dumps(base_config))
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    years = sy_years(cfg)
    env_config = build_env_config_for_years(run_config, years)
    with (OUT / "configs" / "031_27_resolved_env_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(env_config, f, allow_unicode=True, sort_keys=False)

    selected = {int(k): int(v) for k, v in cfg["selected_checkpoints"].items()}
    rows: list[dict[str, Any]] = []
    for seed, ckpt in selected.items():
        for year in years:
            try:
                rows.append(evaluate_model_year(run_config, env_config, seed, ckpt, int(year)))
            except Exception:
                rows.append(
                    {
                        "algorithm": "free_timing_discrete_MaskablePPO",
                        "station_code": "SYA",
                        "site": "SY",
                        "year": int(year),
                        "seed": int(seed),
                        "checkpoint_step": int(ckpt),
                        "source_train_year": 2014,
                        "run_status": "failed",
                        "notes": traceback.format_exc()[-2500:],
                    }
                )
    eval_df = pd.DataFrame(rows)
    eval_path = OUT / "evaluation" / "031_27_sy_all_year_candidate_summary.csv"
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")

    baselines = load_sy_baselines(ROOT / cfg["four_baseline_metrics_csv"])
    comp_df = add_baseline_comparison(eval_df, baselines)
    comp_path = OUT / "evaluation" / "031_27_sy_available_baseline_comparison.csv"
    comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")

    year_seed_matrix = comp_df.pivot_table(
        index="year",
        columns="seed",
        values="advisor_any_metric_strict_winner",
        aggfunc="first",
    )
    year_seed_matrix.to_csv(OUT / "evaluation" / "031_27_sy_year_seed_matrix.csv", encoding="utf-8-sig")
    write_record(years, eval_df, comp_df, sorted(set(baselines["year"].astype(int).tolist())))
    result = {
        "status": "completed" if eval_df["run_status"].eq("ok").all() else "completed_with_failures",
        "years": years,
        "expected_evaluations": len(years) * len(selected),
        "successful_evaluations": int(eval_df["run_status"].eq("ok").sum()),
        "baseline_comparable_years": sorted(set(baselines["year"].astype(int).tolist())),
        "outputs": {
            "candidate_summary": str(eval_path.relative_to(ROOT)).replace("\\", "/"),
            "available_baseline_comparison": str(comp_path.relative_to(ROOT)).replace("\\", "/"),
            "record": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (OUT / "031_27_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
