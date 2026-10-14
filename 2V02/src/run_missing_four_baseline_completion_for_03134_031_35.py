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
import run_extension_expert_baseline_018_03 as extension
import run_free_timing_discrete_maskableppo_ncost2x_scaled_reward_031_17 as base
import run_four_site_all_year_frozen_maskableppo_transfer_031_34 as transfer34
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_action_safety import normalize_action


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_031_35_missing_four_baseline_completion_for_03134.yaml"
OUT = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134"
DOC = ROOT / "docs" / "031_35_missing_four_baseline_completion_for_03134_record.md"


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "daily_outputs", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, values)) + " |" for values in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def clean_scenario(value: Any) -> str:
    s = str(value).strip()
    if s in {"", "nan", "NaN", "None"}:
        return "null"
    mapping = {
        "recorded": "recorded_farmer",
        "recorded_shifted": "recorded_farmer_template",
        "extension_expert_fixed_dap": "official_extension_expert",
        "Official extension expert fixed DAP": "official_extension_expert",
        "Null": "null",
        "DSSAT auto": "dssat_auto",
        "Recorded/farmer practice": "recorded_farmer",
    }
    return mapping.get(s, s)


def first_present(row: pd.Series, names: list[str], default=np.nan) -> Any:
    for name in names:
        if name in row.index and str(row[name]).strip() != "":
            value = row[name]
            if not pd.isna(value):
                return value
    return default


def standardize_existing(path: Path, meta: dict[str, Any]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        site = str(first_present(row, ["site"], "")).strip()
        if not site:
            continue
        station = meta["site_to_station"].get(site)
        if station not in meta["target_stations"]:
            continue
        year = pd.to_numeric(first_present(row, ["year", "requested_year"], np.nan), errors="coerce")
        if pd.isna(year) or int(year) < int(meta.get("min_year", 2000)):
            continue
        scenario = clean_scenario(first_present(row, ["scenario", "scenario_family", "scenario_label"], ""))
        if scenario == "recorded_farmer_template":
            scenario = "recorded_farmer_template_existing"
        if scenario not in {"null", "recorded_farmer", "recorded_farmer_template_existing", "dssat_auto", "official_extension_expert"}:
            continue
        y = pd.to_numeric(first_present(row, ["grain_yield_kg_ha", "final_grain_kg_ha", "final_gwad", "final_gwad_daily", "HWAM"], np.nan), errors="coerce")
        b = pd.to_numeric(first_present(row, ["biomass_kg_ha", "final_biomass_kg_ha", "final_cwad", "final_cwad_daily", "CWAM"], np.nan), errors="coerce")
        i = pd.to_numeric(first_present(row, ["irrigation_mm", "actual_irrigation_mm", "event_irrigation_total", "IRCM"], np.nan), errors="coerce")
        n = pd.to_numeric(first_present(row, ["nitrogen_kg_ha", "actual_nitrogen_kg_ha", "event_fertilizer_total", "NICM"], np.nan), errors="coerce")
        etcp = pd.to_numeric(first_present(row, ["ETCP", "etcp_mm"], np.nan), errors="coerce")
        wp = pd.to_numeric(first_present(row, ["WP_ET_kg_m3", "wp_et_kg_m3"], np.nan), errors="coerce")
        pfp = pd.to_numeric(first_present(row, ["PFP_N_kg_kg", "pfp_n_kg_kg"], np.nan), errors="coerce")
        if pd.isna(wp) and not pd.isna(etcp) and float(etcp) > 0 and not pd.isna(y):
            wp = float(y) / float(etcp) / 10.0
        if pd.isna(pfp) and not pd.isna(n) and float(n) > 0 and not pd.isna(y):
            pfp = float(y) / float(n)
        rows.append(
            {
                "station_code": station,
                "site": site,
                "year": int(year),
                "scenario": scenario,
                "grain_yield_kg_ha": float(y) if not pd.isna(y) else np.nan,
                "biomass_kg_ha": float(b) if not pd.isna(b) else np.nan,
                "actual_irrigation_mm": float(i) if not pd.isna(i) else np.nan,
                "actual_nitrogen_kg_ha": float(n) if not pd.isna(n) else np.nan,
                "etcp_mm": float(etcp) if not pd.isna(etcp) else np.nan,
                "WP_ET_kg_m3": float(wp) if not pd.isna(wp) else np.nan,
                "PFP_N_kg_kg": float(pfp) if not pd.isna(pfp) else np.nan,
                "max_water_stress": pd.to_numeric(first_present(row, ["max_water_stress", "max_wspd"], np.nan), errors="coerce"),
                "max_nitrogen_stress": pd.to_numeric(first_present(row, ["max_nitrogen_stress", "max_nstd"], np.nan), errors="coerce"),
                "source_status": "reused_existing",
                "source_file": str(path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return pd.DataFrame(rows)


def missing_station_years(meta: dict[str, Any], mode: str) -> pd.DataFrame:
    comp = pd.read_csv(ROOT / meta["ppo_full_comparison_csv"], keep_default_na=False)
    miss = comp.loc[~comp["baseline_comparison_status"].eq("ok"), ["station_code", "year"]].drop_duplicates()
    miss["year"] = pd.to_numeric(miss["year"], errors="coerce").astype(int)
    if mode == "smoke":
        sm = meta["smoke"]
        miss = miss[miss["station_code"].eq(str(sm["station_code"])) & miss["year"].eq(int(sm["year"]))].copy()
    return miss.sort_values(["station_code", "year"]).reset_index(drop=True)


def build_env_config(run_config: dict[str, Any], meta: dict[str, Any], years: pd.DataFrame) -> dict[str, Any]:
    pool = pd.read_csv(ROOT / meta["scenario_pool_csv"])
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selection = pool.merge(years, on=["station_code", "year"], how="inner")
    selection["selected_for_train"] = False
    selection["selected_for_eval"] = True
    selection["selection_reason"] = "031_35_missing_four_baseline_completion"
    return direct_ppo.build_env_config(run_config, selection)


def recorded_templates(meta: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    path = ROOT / meta["recorded_template_daily_csv"]
    df = pd.read_csv(path, keep_default_na=False)
    out: dict[str, dict[int, dict[str, float]]] = {}
    for site, group in df[df["scenario"].eq("recorded_farmer")].groupby("site"):
        schedule: dict[int, dict[str, float]] = {}
        for _, row in group.iterrows():
            i = pd.to_numeric(row.get("irrigation_executed_mm", 0), errors="coerce")
            n = pd.to_numeric(row.get("nitrogen_executed_kg_ha", 0), errors="coerce")
            if (pd.isna(i) or float(i) == 0.0) and (pd.isna(n) or float(n) == 0.0):
                continue
            dap = max(1, int(round(pd.to_numeric(row.get("dap", 1), errors="coerce"))))
            schedule.setdefault(dap, {"amir": 0.0, "anfer": 0.0})
            schedule[dap]["amir"] += 0.0 if pd.isna(i) else float(i)
            schedule[dap]["anfer"] += 0.0 if pd.isna(n) else float(n)
        out[str(site)] = schedule
    return out


def expert_schedule(site: str, meta: dict[str, Any]) -> dict[int, dict[str, float]]:
    region = meta["site_regions"][site]
    sched = extension.build_region_schedule()
    sched = sched[sched["region"].eq(region)].copy()
    sched.insert(0, "site", site)
    sched.insert(1, "station", site)
    sched.insert(2, "year", 0)
    return extension.split_irrigation_events(sched)


def metrics_from_snapshot_lenient(snapshot: Path, final_yield: float) -> dict[str, Any]:
    rows = siteppo.parse_summary_out(snapshot / "Summary.OUT")
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = siteppo.num(row, "HWAM")
        if hwam is None:
            continue
        candidates.append((abs(float(hwam) - float(final_yield)), -idx, row))
    if not candidates:
        raise ValueError(f"No usable Summary.OUT row in {snapshot}")
    _, neg_idx, row = min(candidates, key=lambda item: (item[0], item[1]))
    ircm = float(siteppo.num(row, "IRCM") or 0.0)
    nicm = float(siteppo.num(row, "NICM") or 0.0)
    etcp = siteppo.num(row, "ETCP")
    ypem = siteppo.num(row, "YPEM")
    ypnam = siteppo.num(row, "YPNAM")
    if etcp is None or float(etcp) <= 0:
        raise ValueError(f"Invalid ETCP in {snapshot}: {etcp}")
    wp = float(ypem) * 0.1 if ypem is not None and float(ypem) >= 0 else float(final_yield) / float(etcp) / 10.0
    pfp = float(ypnam) if nicm > 0 and ypnam is not None and float(ypnam) >= 0 else math.nan
    return {
        "summary_irrigation_total": ircm,
        "summary_nitrogen_total": nicm,
        "etcp_mm": float(etcp),
        "WP_ET_kg_m3": wp,
        "PFP_N_kg_kg": pfp,
        "summary_match_score": abs(float(siteppo.num(row, "HWAM") or final_yield) - float(final_yield)),
        "summary_row_index": int(-neg_idx),
    }


def schedule_for(scenario: str, station: str, site: str, meta: dict[str, Any], templates: dict[str, dict[int, dict[str, float]]]) -> dict[int, dict[str, float]]:
    if scenario == "null":
        return {}
    if scenario == "recorded_farmer_template_02705":
        return templates.get(site, {})
    if scenario == "official_extension_expert":
        return expert_schedule(site, meta)
    raise ValueError(scenario)


def eval_fixed(run_config: dict[str, Any], env_config: dict[str, Any], meta: dict[str, Any], station: str, year: int, scenario: str, schedule: dict[int, dict[str, float]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    site = meta["station_to_site"][station]
    env = base.direct_ppo.make_base_env(env_config, station, int(year), int(meta["seed"]), f"{station}_{year}_031_35_{scenario}", evaluation=True)
    weather = direct_ppo.weather_for_daily(run_config)
    rows: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, station, int(year))["planting_date"])
        while not done and step_count < int(meta["max_steps"]):
            latest_pre = base.latest_observation_dict(env, obs, info)
            dap_raw = base.scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            action_real = schedule.get(dap, {"amir": 0.0, "anfer": 0.0})
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action_real)
            obs, reward, terminated, truncated, info = env.step(norm)
            done = bool(terminated or truncated)
            latest = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(station)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            rows.append(
                {
                    "station_code": station,
                    "site": site,
                    "year": int(year),
                    "scenario": scenario,
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": base.scalar(wrow.get("rain"), np.nan),
                    "srad": base.scalar(wrow.get("srad"), np.nan),
                    "tmax": base.scalar(wrow.get("tmax"), np.nan),
                    "tmin": base.scalar(wrow.get("tmin"), np.nan),
                    "grnwt": base.scalar(latest.get("grnwt")),
                    "topwt": base.scalar(latest.get("topwt")),
                    "swfac": base.scalar(latest.get("swfac")),
                    "nstres": base.scalar(latest.get("nstres")),
                    "irrigation_executed_mm": float(action_real.get("amir", 0.0)),
                    "nitrogen_executed_kg_ha": float(action_real.get("anfer", 0.0)),
                    "reward": float(reward),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(rows)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"Fixed baseline did not finish: {station}{year} {scenario}")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        total_i = float(pd.to_numeric(daily["irrigation_executed_mm"], errors="coerce").fillna(0).sum())
        total_n = float(pd.to_numeric(daily["nitrogen_executed_kg_ha"], errors="coerce").fillna(0).sum())
        snapshot_tmp = siteppo.snapshot_from_env(env)
        snapshot = OUT / "snapshots" / station / str(year) / scenario
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
        metrics = metrics_from_snapshot_lenient(snapshot, final_y)
        actual_i = float(metrics["summary_irrigation_total"])
        actual_n = float(metrics["summary_nitrogen_total"])
        summary = {
            "station_code": station,
            "site": site,
            "year": int(year),
            "scenario": scenario,
            "grain_yield_kg_ha": final_y,
            "biomass_kg_ha": final_b,
            "actual_irrigation_mm": actual_i,
            "actual_nitrogen_kg_ha": actual_n,
            "max_water_stress": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nitrogen_stress": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "source_status": "generated_031_35_fixed_policy",
            "source_file": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
            "requested_irrigation_mm": total_i,
            "requested_nitrogen_kg_ha": total_n,
            **metrics,
        }
        return daily, summary
    finally:
        env.close()


def dedupe_baselines(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    priority = {
        "generated_031_35_fixed_policy": 0,
        "reused_existing": 1,
    }
    out = df.copy()
    out["_scenario_rank"] = out["scenario"].map({"null": 0, "recorded_farmer": 1, "recorded_farmer_template_02705": 2, "recorded_farmer_template_existing": 3, "official_extension_expert": 4, "dssat_auto": 5}).fillna(9)
    out["_source_rank"] = out["source_status"].map(priority).fillna(5)
    out = out.sort_values(["station_code", "year", "scenario", "_source_rank", "source_file"])
    out = out.drop_duplicates(["station_code", "year", "scenario"], keep="first")
    return out.drop(columns=["_scenario_rank", "_source_rank"])


def make_template_aware_comparison_baselines(unified: pd.DataFrame) -> pd.DataFrame:
    """Add a comparison-only recorded_farmer surrogate from recorded templates.

    The original unified table is kept unchanged.  This helper only prepares a
    second comparison view for the user's explicit "reuse existing recorded
    farmer template in other years" requirement.  If a true recorded_farmer row
    already exists for a site-year, it is preferred and no surrogate is added.
    """
    if unified.empty:
        return unified.copy()
    out = unified.copy()
    add_rows: list[pd.Series] = []
    template_priority = {
        "recorded_farmer_template_02705": 0,
        "recorded_farmer_template_existing": 1,
    }
    for (site, year), group in out.groupby(["site", "year"], dropna=False):
        scenarios = set(group["scenario"].astype(str))
        if "recorded_farmer" in scenarios:
            continue
        templates = group[group["scenario"].astype(str).isin(template_priority)].copy()
        if templates.empty:
            continue
        templates["_template_rank"] = templates["scenario"].map(template_priority).fillna(9)
        chosen = templates.sort_values(["_template_rank", "source_file"]).iloc[0].drop(labels=["_template_rank"])
        chosen = chosen.copy()
        chosen["scenario"] = "recorded_farmer"
        chosen["source_status"] = f"{chosen.get('source_status', '')}; comparison_only_recorded_template_surrogate"
        add_rows.append(chosen)
    if add_rows:
        out = pd.concat([out, pd.DataFrame(add_rows)], ignore_index=True, sort=False)
    return out


def baseline_comparison_inputs(unified: pd.DataFrame) -> pd.DataFrame:
    return unified.rename(
        columns={
            "grain_yield_kg_ha": "final_grain_kg_ha",
            "WP_ET_kg_m3": "wp_et_kg_m3",
            "PFP_N_kg_kg": "pfp_n_kg_kg",
        }
    )


def write_comparison_outputs(mode: str, meta: dict[str, Any], unified: pd.DataFrame, generated: pd.DataFrame, missing: pd.DataFrame, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    ppo = pd.read_csv(ROOT / meta["ppo_full_summary_csv"], keep_default_na=False)
    for col in ["final_grain_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg"]:
        if col in ppo.columns:
            ppo[col] = pd.to_numeric(ppo[col], errors="coerce")
    strict_baselines = baseline_comparison_inputs(unified)
    compared = transfer34.add_baseline_comparison(ppo, strict_baselines)
    compared.to_csv(OUT / "evaluation" / f"031_35_{mode}_ppo_vs_expanded_baselines.csv", index=False, encoding="utf-8-sig")

    template_aware_unified = make_template_aware_comparison_baselines(unified)
    template_aware_unified.to_csv(OUT / "evaluation" / f"031_35_{mode}_template_aware_unified_baseline_summary.csv", index=False, encoding="utf-8-sig")
    template_compared = transfer34.add_baseline_comparison(ppo, baseline_comparison_inputs(template_aware_unified))
    template_compared.to_csv(OUT / "evaluation" / f"031_35_{mode}_ppo_vs_expanded_template_aware_baselines.csv", index=False, encoding="utf-8-sig")

    counts = compared.groupby(["station_code", "baseline_comparison_status"]).size().reset_index(name="n")
    template_counts = template_compared.groupby(["station_code", "baseline_comparison_status"]).size().reset_index(name="n")
    wins = compared[compared["baseline_comparison_status"].eq("ok")].groupby("station_code").agg(
        compared_ok=("year", "count"),
        any_metric_wins=("advisor_any_metric_strict_winner", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
    ).reset_index()
    template_wins = template_compared[template_compared["baseline_comparison_status"].eq("ok")].groupby("station_code").agg(
        compared_ok=("year", "count"),
        any_metric_wins=("advisor_any_metric_strict_winner", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
    ).reset_index()
    lines = [
        "# 031_35 Missing four-baseline completion for 031_34 record",
        "",
        f"Mode: `{mode}`",
        "",
        "## Scope",
        "",
        "- No PPO/DQN training.",
        "- Fixed-policy baselines were generated only for missing station-years from 031_34.",
        "- Existing true DSSAT auto rows are reused where available; missing true auto rows are not faked.",
        "- `recorded_farmer_template_02705` is a transferred template, not true yearly observed farmer management.",
        "- Generated fixed-policy actions are executed through the gym action interface; very large transferred recorded-template single events may be clipped by the environment action bounds.",
        "- A second `template_aware` comparison table treats transferred recorded templates as comparison-only recorded farmer surrogates when true recorded-farmer rows are absent.",
        "",
        "## Missing station-years processed",
        "",
        md_table(missing, 120),
        "",
        "## Generated/reused coverage status",
        "",
        md_table(pd.DataFrame(manifest).groupby(["scenario", "status"]).size().reset_index(name="n") if manifest else pd.DataFrame(), 80),
        "",
        "## Strict expanded PPO comparison status",
        "",
        md_table(counts, 80),
        "",
        "## Strict wins among comparable rows",
        "",
        md_table(wins, 80),
        "",
        "## Template-aware PPO comparison status",
        "",
        md_table(template_counts, 80),
        "",
        "## Template-aware wins among comparable rows",
        "",
        md_table(template_wins, 80),
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "strict_compared": compared,
        "template_compared": template_compared,
        "strict_counts": counts,
        "template_counts": template_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--postprocess-existing", action="store_true", help="Reuse existing 031_35 output CSVs and only rebuild comparison/record files.")
    args = parser.parse_args()
    ensure_dirs()
    meta = direct_ppo.load_yaml(CONFIG)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    if args.postprocess_existing:
        missing = pd.read_csv(OUT / "evaluation" / f"031_35_{args.mode}_missing_station_years.csv", keep_default_na=False)
        generated = pd.read_csv(OUT / "evaluation" / f"031_35_{args.mode}_generated_baseline_summary.csv", keep_default_na=False)
        unified = pd.read_csv(OUT / "evaluation" / f"031_35_{args.mode}_unified_baseline_summary.csv", keep_default_na=False)
        manifest_df = pd.read_csv(OUT / "evaluation" / f"031_35_{args.mode}_coverage_manifest.csv", keep_default_na=False)
        outputs = write_comparison_outputs(args.mode, meta, unified, generated, missing, manifest_df.to_dict("records"))
        result = {
            "task": "031_35",
            "mode": args.mode,
            "postprocess_existing": True,
            "missing_year_rows": int(len(missing)),
            "generated_rows": int(len(generated)),
            "unified_rows": int(len(unified)),
            "comparison_rows": int(len(outputs["strict_compared"])),
            "template_aware_comparison_rows": int(len(outputs["template_compared"])),
            "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        }
        (OUT / f"031_35_{args.mode}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(outputs["template_counts"].to_string(index=False))
        return
    base_config_path = ROOT / meta["base_config"]
    run_config = direct_ppo.load_yaml(base_config_path)
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    missing = missing_station_years(meta, args.mode)
    missing.to_csv(OUT / "evaluation" / f"031_35_{args.mode}_missing_station_years.csv", index=False, encoding="utf-8-sig")
    existing = pd.concat([standardize_existing(ROOT / p, meta) for p in meta["existing_baseline_sources"]], ignore_index=True)
    existing.to_csv(OUT / "evaluation" / "031_35_reused_existing_baselines_raw_standardized.csv", index=False, encoding="utf-8-sig")
    env_config = build_env_config(run_config, meta, missing)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"031_35_{args.mode}_resolved_env_config.yaml")
    templates = recorded_templates(meta)
    daily_frames: list[pd.DataFrame] = []
    generated_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for row in missing.itertuples(index=False):
        station, year = str(row.station_code), int(row.year)
        site = meta["station_to_site"][station]
        for scenario in meta["scenarios_to_generate"]:
            try:
                daily, summary = eval_fixed(run_config, env_config, meta, station, year, scenario, schedule_for(scenario, station, site, meta, templates))
                daily_frames.append(daily)
                generated_rows.append(summary)
                manifest.append({"station_code": station, "site": site, "year": year, "scenario": scenario, "status": "generated_031_35_fixed_policy", "details": summary["source_file"]})
                pd.DataFrame(generated_rows).to_csv(OUT / "evaluation" / f"031_35_{args.mode}_generated_baseline_summary_partial.csv", index=False, encoding="utf-8-sig")
            except Exception as exc:
                manifest.append({"station_code": station, "site": site, "year": year, "scenario": scenario, "status": "failed", "details": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    generated = pd.DataFrame(generated_rows)
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    generated.to_csv(OUT / "evaluation" / f"031_35_{args.mode}_generated_baseline_summary.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(OUT / "evaluation" / f"031_35_{args.mode}_generated_baseline_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(OUT / "evaluation" / f"031_35_{args.mode}_coverage_manifest.csv", index=False, encoding="utf-8-sig")
    unified = dedupe_baselines(pd.concat([existing, generated], ignore_index=True, sort=False))
    unified.to_csv(OUT / "evaluation" / f"031_35_{args.mode}_unified_baseline_summary.csv", index=False, encoding="utf-8-sig")
    outputs = write_comparison_outputs(args.mode, meta, unified, generated, missing, manifest)
    result = {"task": "031_35", "mode": args.mode, "missing_year_rows": int(len(missing)), "generated_rows": int(len(generated)), "unified_rows": int(len(unified)), "comparison_rows": int(len(outputs["strict_compared"])), "template_aware_comparison_rows": int(len(outputs["template_compared"])), "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/")}
    (OUT / f"031_35_{args.mode}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(outputs["strict_counts"].to_string(index=False))


if __name__ == "__main__":
    main()
