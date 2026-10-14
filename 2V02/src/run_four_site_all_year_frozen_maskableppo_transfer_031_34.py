from __future__ import annotations

import argparse
import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_17 as base
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_34_four_site_all_year_frozen_maskableppo_transfer.yaml"
OUT = ROOT / "benchmark_results" / "031_34_four_site_all_year_frozen_maskableppo_transfer"
DOC = ROOT / "docs" / "031_34_four_site_all_year_frozen_maskableppo_transfer_record.md"


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs", "runs", "evaluation", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_candidates(meta: dict[str, Any], mode: str) -> pd.DataFrame:
    path = ROOT / meta["candidate_selection_csv"]
    df = pd.read_csv(path, keep_default_na=False)
    df = df[np.isclose(pd.to_numeric(df["yield_guardrail_fraction"], errors="coerce"), float(meta["candidate_yield_guardrail_fraction"]))].copy()
    df = df[df["station_code"].isin(meta["target_stations"])].copy()
    eval_path = ROOT / meta["checkpoint_eval_summary_csv"]
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    required = ["station_code", "seed", "checkpoint_step", "year", "model_path"]
    missing = [c for c in required if c not in eval_df.columns]
    if missing:
        raise RuntimeError(f"Checkpoint eval summary is missing required columns: {missing}")
    eval_keyed = eval_df[required].drop_duplicates(["station_code", "seed", "checkpoint_step"])
    df = df.merge(eval_keyed, on=["station_code", "seed", "checkpoint_step"], how="left", validate="one_to_one")
    if df[["year", "model_path"]].isna().any().any() or (df["model_path"].astype(str).str.len() == 0).any():
        bad = df[df[["year", "model_path"]].isna().any(axis=1) | df["model_path"].astype(str).str.len().eq(0)]
        raise RuntimeError(
            "Failed to map all 031_34a candidates back to 031_33 checkpoint metadata:\n"
            + bad[["station_code", "seed", "checkpoint_step"]].to_string(index=False)
        )
    if mode == "smoke":
        sm = meta["smoke"]
        df = df[
            df["station_code"].eq(str(sm["station_code"]))
            & (pd.to_numeric(df["seed"], errors="coerce").astype(int).eq(int(sm["seed"])))
        ].copy()
    if df.empty:
        raise RuntimeError(f"No candidates found for mode={mode}")
    return df


def available_years(meta: dict[str, Any], stations: list[str], mode: str) -> pd.DataFrame:
    pool = pd.read_csv(ROOT / meta["scenario_pool_csv"])
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    rows = pool[pool["station_code"].isin(stations) & (pool["year"] >= int(meta.get("min_year", 2000)))].copy()
    if mode == "smoke":
        sm = meta["smoke"]
        rows = rows[rows["station_code"].eq(str(sm["station_code"])) & rows["year"].eq(int(sm["year"]))].copy()
    rows["selected_for_train"] = False
    rows["selected_for_eval"] = True
    rows["selection_reason"] = "031_34_four_site_all_year_frozen_transfer"
    return rows


def build_env_config(base_config: dict[str, Any], selection: pd.DataFrame) -> dict[str, Any]:
    return direct_ppo.build_env_config(base_config, selection)


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
    candidate: pd.Series,
    eval_year: int,
    site_map: dict[str, str],
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    station = str(candidate["station_code"])
    site = str(site_map[station])
    seed = int(candidate["seed"])
    checkpoint_step = int(candidate["checkpoint_step"])
    source_train_year = int(candidate["year"])
    model_path = ROOT / str(candidate["model_path"])
    if not model_path.exists():
        return {
            "station_code": station,
            "site": site,
            "year": int(eval_year),
            "seed": seed,
            "checkpoint_step": checkpoint_step,
            "source_train_year": source_train_year,
            "run_status": "missing_model",
            "model_path": str(model_path),
        }
    model = MaskablePPO.load(str(model_path), device="cpu")
    env = base.make_env(
        run_config,
        env_config,
        station,
        int(eval_year),
        seed,
        f"{station}_{eval_year}_031_34_seed{seed}_ckpt{checkpoint_step}_eval",
        evaluation=True,
    )
    weather = direct_ppo.weather_for_daily(run_config)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, int(eval_year))["planting_date"])
        while not done and step_count < int(run_config["runtime"]["max_steps"]):
            latest_pre = base.latest_observation_dict(env, obs, info)
            dap_raw = base.scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append(
                {
                    "station_code": station,
                    "site": site,
                    "year": int(eval_year),
                    "seed": seed,
                    "checkpoint_step": checkpoint_step,
                    "source_train_year": source_train_year,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": base.scalar(wrow.get("rain"), np.nan),
                    "srad": base.scalar(wrow.get("srad"), np.nan),
                    "tmax": base.scalar(wrow.get("tmax"), np.nan),
                    "tmin": base.scalar(wrow.get("tmin"), np.nan),
                    "swfac": base.scalar(latest.get("swfac")),
                    "nstres": base.scalar(latest.get("nstres")),
                    "topwt": base.scalar(latest.get("topwt")),
                    "grnwt": base.scalar(latest.get("grnwt")),
                    "xlai": base.scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **dict(env.last_action_info),
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"Evaluation did not finish for {station}{eval_year}, seed={seed}, ckpt={checkpoint_step}")
        daily_dir = OUT / "daily_outputs" / station
        daily_dir.mkdir(parents=True, exist_ok=True)
        daily_path = daily_dir / f"{station}_{eval_year}_seed{seed}_ckpt{checkpoint_step}_daily.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

        irr = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0)
        n = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0)
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        total_i = float(irr.sum())
        total_n = float(n.sum())
        snapshot = OUT / "runs" / station / str(eval_year) / f"seed{seed}_ckpt{checkpoint_step}" / "snapshot"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(siteppo.snapshot_from_env(env), snapshot, dirs_exist_ok=True)
        metrics = siteppo.strict_metrics_from_snapshot(snapshot, final_y, total_i, total_n)
        swfac = pd.to_numeric(daily["swfac"], errors="coerce")
        nstres = pd.to_numeric(daily["nstres"], errors="coerce")
        dap = pd.to_numeric(daily["dap"], errors="coerce")
        return {
            "algorithm": "free_timing_discrete_MaskablePPO",
            "station_code": station,
            "site": site,
            "year": int(eval_year),
            "seed": seed,
            "checkpoint_step": checkpoint_step,
            "source_train_year": source_train_year,
            "source_train_year_known_win_95guardrail": bool(candidate.get("known_win_with_yield_guardrail", False)),
            "scenario": f"rl_candidate_031_34_from031_33_seed{seed}_ckpt{checkpoint_step}",
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
    except Exception:
        return {
            "algorithm": "free_timing_discrete_MaskablePPO",
            "station_code": station,
            "site": site,
            "year": int(eval_year),
            "seed": seed,
            "checkpoint_step": checkpoint_step,
            "source_train_year": source_train_year,
            "run_status": "failed",
            "notes": traceback.format_exc()[-4000:],
        }
    finally:
        env.close()


def clean_scenario(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.strip() == "" or text.lower() == "nan":
        return "null"
    return text


def load_baselines(meta: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source in meta.get("baseline_sources", []):
        path = ROOT / source
        if not path.exists():
            continue
        df = pd.read_csv(path, keep_default_na=False)
        if not {"site", "year", "scenario"}.issubset(df.columns):
            continue
        df["baseline_source"] = str(path.relative_to(ROOT)).replace("\\", "/")
        df["scenario"] = df["scenario"].map(clean_scenario)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    base = pd.concat(frames, ignore_index=True, sort=False)
    # Harmonize common column names.
    rename = {
        "grain_yield_kg_ha": "final_grain_kg_ha",
        "WP_ET_recomputed_kg_m3": "wp_et_kg_m3",
        "PFP_N_recomputed_kg_kg": "pfp_n_kg_kg",
    }
    for old, new in rename.items():
        if old in base.columns and new not in base.columns:
            base[new] = base[old]
    metrics = ["final_grain_kg_ha", "irrigation_event_total_mm", "nitrogen_event_total_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg"]
    for col in ["year", *metrics]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")
    keep = ["site", "year", "scenario", "baseline_source", *[c for c in metrics if c in base.columns]]
    base = base[[c for c in keep if c in base.columns]].copy()
    # Prefer later all-year SY completion, then representative package, then older 027 source by source order.
    base["_source_order"] = base["baseline_source"].map({str((ROOT / s).relative_to(ROOT)).replace("\\", "/"): i for i, s in enumerate(meta.get("baseline_sources", []))})
    base = base.sort_values(["site", "year", "scenario", "_source_order"]).drop_duplicates(["site", "year", "scenario"], keep="first")
    return base.drop(columns=["_source_order"], errors="ignore")


def add_baseline_comparison(eval_df: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if baselines.empty:
        out = eval_df.copy()
        out["baseline_comparison_status"] = "baseline_missing_not_compared"
        return out
    for row in eval_df.itertuples(index=False):
        d = row._asdict()
        if str(d.get("run_status")) != "ok":
            d["baseline_comparison_status"] = "candidate_failed_not_compared"
            rows.append(d)
            continue
        site = str(d["site"])
        year = int(d["year"])
        base_rows = baselines[
            baselines["site"].astype(str).eq(site)
            & pd.to_numeric(baselines["year"], errors="coerce").astype("Int64").eq(year)
            & baselines["scenario"].isin(["null", "recorded_farmer", "dssat_auto", "official_extension_expert"])
        ].copy()
        if len(base_rows) < 4:
            d.update({"baseline_comparison_status": f"baseline_incomplete_{len(base_rows)}rows", "advisor_any_metric_strict_winner": np.nan, "winning_metrics": ""})
            rows.append(d)
            continue
        max_y = float(pd.to_numeric(base_rows["final_grain_kg_ha"], errors="coerce").max())
        max_wp = float(pd.to_numeric(base_rows["wp_et_kg_m3"], errors="coerce").max())
        pfp_vals = pd.to_numeric(base_rows["pfp_n_kg_kg"], errors="coerce").dropna()
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
                "baseline_comparison_status": "ok",
                "baseline_row_count": int(len(base_rows)),
                "baseline_max_yield": max_y,
                "baseline_max_wp_et": max_wp,
                "baseline_max_pfp_n": max_pfp,
                "gap_yield": gap_y,
                "gap_wp_et": gap_wp,
                "gap_pfp_n": gap_pfp,
                "yield_strict_win": flags["yield"],
                "wp_et_strict_win": flags["WP_ET"],
                "pfp_n_strict_win": flags["PFP_N"],
                "advisor_any_metric_strict_winner": bool(any(flags.values())),
                "winning_metrics": ";".join([name for name, flag in flags.items() if flag]),
            }
        )
        rows.append(d)
    return pd.DataFrame(rows)


def write_record(mode: str, eval_df: pd.DataFrame, compared: pd.DataFrame) -> None:
    cols = [
        "station_code",
        "year",
        "seed",
        "checkpoint_step",
        "source_train_year",
        "run_status",
        "baseline_comparison_status",
        "final_grain_kg_ha",
        "wp_et_kg_m3",
        "pfp_n_kg_kg",
        "irrigation_event_total_mm",
        "nitrogen_event_total_kg_ha",
        "gap_yield",
        "gap_wp_et",
        "gap_pfp_n",
        "advisor_any_metric_strict_winner",
        "winning_metrics",
    ]
    if not compared.empty and "advisor_any_metric_strict_winner" in compared.columns:
        counts = compared.groupby("station_code").agg(
            rows=("year", "count"),
            compared_ok=("baseline_comparison_status", lambda s: int((s == "ok").sum())),
            any_metric_wins=("advisor_any_metric_strict_winner", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            yield_wins=("yield_strict_win", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()) if "yield_strict_win" in compared.columns else 0),
            wp_wins=("wp_et_strict_win", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()) if "wp_et_strict_win" in compared.columns else 0),
            pfp_wins=("pfp_n_strict_win", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()) if "pfp_n_strict_win" in compared.columns else 0),
        ).reset_index()
    else:
        counts = pd.DataFrame()
    lines = [
        "# 031_34 Four-site all-year frozen MaskablePPO transfer record",
        "",
        f"Mode: `{mode}`",
        "",
        "## Scope",
        "",
        "- Frozen candidate checkpoints from 031_34a 95% yield-guardrailed reselection.",
        "- No training.",
        "- Deterministic transfer evaluation only.",
        "- Computes ETCP/WP_ET via saved DSSAT snapshot.",
        "",
        "## Non-scientific execution notes",
        "",
        "- First smoke attempt failed before DSSAT evaluation because the 031_34a reselection table stores station/seed/checkpoint only and does not include `year` or `model_path`.",
        "- The script was fixed to merge 031_34a candidates back to `031_33_checkpoint_eval_summary.csv` for immutable checkpoint metadata. This changes metadata lookup only, not candidate choice or evaluation logic.",
        "",
        "## Compact evaluation table",
        "",
        compared[[c for c in cols if c in compared.columns]].to_string(index=False) if not compared.empty else "No rows.",
        "",
        "## Station-level comparison counts",
        "",
        counts.to_string(index=False) if not counts.empty else "No counts.",
        "",
        "## Boundary",
        "",
        "This is within-station all-year transfer, not cross-station generalization.",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = parser.parse_args()

    ensure_dirs()
    meta = direct_ppo.load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    base_config_path = ROOT / meta["base_config"]
    run_config = direct_ppo.load_yaml(base_config_path)
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    shutil.copyfile(base_config_path, OUT / "configs" / base_config_path.name)

    candidates = load_candidates(meta, args.mode)
    stations = sorted(candidates["station_code"].unique().tolist())
    selection = available_years(meta, stations, args.mode)
    selection.to_csv(OUT / "configs" / f"031_34_{args.mode}_scenario_selection.csv", index=False, encoding="utf-8-sig")
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = build_env_config(run_config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"031_34_{args.mode}_resolved_env_config.yaml")

    rows: list[dict[str, Any]] = []
    years_by_station = {s: sorted(g["year"].astype(int).unique().tolist()) for s, g in selection.groupby("station_code")}
    site_map = {str(k): str(v) for k, v in meta["station_to_site"].items()}
    for cand in candidates.sort_values(["station_code", "seed"]).itertuples(index=False):
        cand_series = pd.Series(cand._asdict())
        for year in years_by_station[str(cand_series["station_code"])]:
            rows.append(evaluate_model_year(run_config, env_config, cand_series, int(year), site_map))
            # Save partial after every evaluation; useful for long full mode.
            pd.DataFrame(rows).to_csv(OUT / "evaluation" / f"031_34_{args.mode}_candidate_summary_partial.csv", index=False, encoding="utf-8-sig")

    eval_df = pd.DataFrame(rows)
    eval_path = OUT / "evaluation" / f"031_34_{args.mode}_candidate_summary.csv"
    eval_df.to_csv(eval_path, index=False, encoding="utf-8-sig")
    baselines = load_baselines(meta)
    baseline_path = OUT / "evaluation" / "031_34_baseline_rows_reused.csv"
    baselines.to_csv(baseline_path, index=False, encoding="utf-8-sig")
    compared = add_baseline_comparison(eval_df, baselines)
    compared_path = OUT / "evaluation" / f"031_34_{args.mode}_candidate_vs_baseline.csv"
    compared.to_csv(compared_path, index=False, encoding="utf-8-sig")
    write_record(args.mode, eval_df, compared)
    result = {
        "task": "031_34_four_site_all_year_frozen_maskableppo_transfer",
        "mode": args.mode,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "candidate_summary": str(eval_path.relative_to(ROOT)).replace("\\", "/"),
        "candidate_vs_baseline": str(compared_path.relative_to(ROOT)).replace("\\", "/"),
        "baseline_rows": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
        "row_count": int(len(eval_df)),
    }
    (OUT / f"031_34_{args.mode}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not compared.empty:
        print(compared[[c for c in ["station_code", "year", "seed", "checkpoint_step", "run_status", "final_grain_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "gap_yield", "gap_wp_et", "gap_pfp_n", "advisor_any_metric_strict_winner", "winning_metrics"] if c in compared.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
