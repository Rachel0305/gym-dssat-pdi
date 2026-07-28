from __future__ import annotations

import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
PROMPT = ROOT / "prompts" / "032_07_lc2010_seed0_50k_cross_year_transfer.md"
MODEL = ROOT / "benchmark_results" / "032_04_lc2010_stress_aware_ppo_multiseed_200k" / "models" / "LCA" / "LCA_2010_stress_aware_maskableppo_seed0_ckpt50000.zip"
BASE_DAILY = ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134" / "evaluation" / "031_35_full_generated_baseline_daily.csv"
AUTO_DAILY = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_generated_dssat_auto_daily.csv"
OUT = ROOT / "benchmark_results" / "032_07_lc2010_seed0_50k_cross_year_transfer"
DOC = ROOT / "docs" / "032_07_lc2010_seed0_50k_cross_year_transfer_record.md"

STATION = "LCA"
SOURCE_YEAR = 2010
SOURCE_SEED = 0
SOURCE_CHECKPOINT = 50000
TARGET_YEARS = [2012, 2013, 2014, 2015]

SCENARIOS = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer": "Recorded farmer",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "rl_candidate": "MaskablePPO transfer",
}
COLORS = {
    "null": "#3B3B3B",
    "recorded_farmer": "#B33A3A",
    "dssat_auto": "#C28B00",
    "official_extension_expert": "#6650A4",
    "rl_candidate": "#18864B",
}
STYLES = {
    "null": "-",
    "recorded_farmer": "--",
    "dssat_auto": "-.",
    "official_extension_expert": ":",
    "rl_candidate": "-",
}


def ensure_dirs() -> None:
    for rel in ["configs", "daily_outputs/LCA", "tables", "figures", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_config(year: int) -> dict[str, Any]:
    cfg = direct_ppo.load_yaml(CONFIG)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = SOURCE_SEED
    cfg["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cfg["runtime"]["smoke_station"] = STATION
    cfg["runtime"]["smoke_year"] = int(year)
    return cfg


def stress_summary(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ["swfac", "nstres"]:
        s = pd.to_numeric(daily.get(col, pd.Series(dtype=float)), errors="coerce")
        out[f"max_{col}"] = float(s.max()) if len(s) else np.nan
        for threshold in [0.001, 0.01, 0.05]:
            out[f"{col}_days_gt_{str(threshold).replace('.', 'p')}"] = int((s > threshold).sum())
    return out


def evaluate_transfer_year(year: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    daily_path = OUT / "daily_outputs" / STATION / f"{STATION}_{year}_source{SOURCE_YEAR}_seed{SOURCE_SEED}_ckpt{SOURCE_CHECKPOINT}_transfer_daily.csv"
    if daily_path.exists():
        daily = pd.read_csv(daily_path)
        out = base.summarize_daily("MaskablePPO", daily, daily_path, MODEL)
        out.update({"station_code": STATION, "target_year": year, "source_year": SOURCE_YEAR, "seed": SOURCE_SEED, "checkpoint_step": SOURCE_CHECKPOINT, "run_status": "ok_existing"})
        out.update(stress_summary(daily))
        return out

    if not MODEL.exists():
        return {"station_code": STATION, "target_year": year, "run_status": "missing_model", "model_path": str(MODEL.relative_to(ROOT))}

    config = load_config(year)
    selection = base.make_selection(config)
    selection_path = OUT / "configs" / f"032_07_selection_{STATION}_{year}.csv"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"032_07_resolved_env_config_{STATION}_{year}.yaml")
    weather = direct_ppo.weather_for_daily(config)
    model = MaskablePPO.load(str(MODEL), device="cpu")

    records: list[dict[str, Any]] = []
    env = None
    try:
        env = base.make_env(config, env_config, STATION, year, SOURCE_SEED, f"{STATION}_{year}_032_07_transfer_eval", evaluation=True)
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, year)["planting_date"])
        while not done and step_count < int(config["runtime"]["max_steps"]):
            latest = base.latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = base.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            action_info = dict(getattr(env, "last_action_info", {}) or {})
            records.append(
                {
                    "station_code": STATION,
                    "year": year,
                    "source_year": SOURCE_YEAR,
                    "source_seed": SOURCE_SEED,
                    "source_checkpoint_step": SOURCE_CHECKPOINT,
                    "algorithm": "MaskablePPO",
                    "date": date.strftime("%Y-%m-%d"),
                    "doy": int(date.dayofyear),
                    "dap": dap,
                    "rain": scalar(wrow.get("rain"), np.nan),
                    "srad": scalar(wrow.get("srad"), np.nan),
                    "tmax": scalar(wrow.get("tmax"), np.nan),
                    "tmin": scalar(wrow.get("tmin"), np.nan),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "topwt": scalar(latest.get("topwt")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "xlai": scalar(latest.get("xlai")),
                    "reward": float(reward),
                    **action_info,
                    "done": done,
                    "info": json.dumps(info, ensure_ascii=False, default=str),
                }
            )
            step_count += 1
    except Exception:
        err_path = OUT / "evaluation" / f"032_07_{STATION}_{year}_transfer_error.txt"
        err_path.write_text(traceback.format_exc(), encoding="utf-8")
        return {"station_code": STATION, "target_year": year, "run_status": "failed", "notes": str(err_path.relative_to(ROOT))}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    daily = pd.DataFrame(records)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    out = base.summarize_daily("MaskablePPO", daily, daily_path, MODEL)
    out.update({"station_code": STATION, "target_year": year, "source_year": SOURCE_YEAR, "seed": SOURCE_SEED, "checkpoint_step": SOURCE_CHECKPOINT, "run_status": "ok"})
    out.update(stress_summary(daily))
    return out


def normalize_baseline_daily() -> pd.DataFrame:
    if not BASE_DAILY.exists():
        raise FileNotFoundError(BASE_DAILY)
    if not AUTO_DAILY.exists():
        raise FileNotFoundError(AUTO_DAILY)
    base_daily = pd.read_csv(BASE_DAILY, keep_default_na=False)
    auto_daily = pd.read_csv(AUTO_DAILY, keep_default_na=False)
    base = base_daily[
        base_daily["station_code"].eq(STATION)
        & base_daily["year"].isin(TARGET_YEARS)
        & base_daily["scenario"].isin(["null", "official_extension_expert", "recorded_farmer_template_02705", "recorded_farmer"])
    ].copy()
    base["scenario"] = base["scenario"].replace({"recorded_farmer_template_02705": "recorded_farmer"})
    auto = auto_daily[
        auto_daily["station_code"].eq(STATION)
        & auto_daily["year"].isin(TARGET_YEARS)
        & auto_daily["scenario"].eq("dssat_auto")
    ].copy()
    auto = auto.rename(
        columns={
            "external_irrigation_action_mm": "irrigation_executed_mm",
            "external_nitrogen_action_kg_ha": "nitrogen_executed_kg_ha",
        }
    )
    combined = pd.concat([base, auto], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(["year", "scenario", "dap"], keep="first")
    return normalize_daily_columns(combined, "baseline")


def normalize_daily_columns(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    out = df.copy()
    renames = {
        "rain": "rainfall_mm",
        "tmax": "tmax_c",
        "tmin": "tmin_c",
        "grnwt": "grain_yield_kg_ha",
        "topwt": "biomass_kg_ha",
        "swfac": "water_stress_index_wspd",
        "nstres": "nitrogen_stress_index_nstd",
        "safe_action_amir": "irrigation_executed_mm",
        "safe_action_anfer": "nitrogen_executed_kg_ha",
    }
    out = out.rename(columns={k: v for k, v in renames.items() if k in out.columns})
    out["algorithm"] = algorithm
    out["site"] = "LC"
    out["station"] = "Luancheng"
    out["requested_year"] = out["year"]
    out["soil_water_mm"] = np.nan
    for col in [
        "irrigation_executed_mm",
        "nitrogen_executed_kg_ha",
        "rainfall_mm",
        "tmax_c",
        "tmin_c",
        "grain_yield_kg_ha",
        "biomass_kg_ha",
        "water_stress_index_wspd",
        "nitrogen_stress_index_nstd",
        "soil_water_mm",
    ]:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["irrigation_executed_mm"] = out["irrigation_executed_mm"].fillna(0)
    out["nitrogen_executed_kg_ha"] = out["nitrogen_executed_kg_ha"].fillna(0)
    out["cumulative_irrigation_mm"] = out.groupby(["year", "scenario"])["irrigation_executed_mm"].cumsum()
    out["cumulative_nitrogen_kg_ha"] = out.groupby(["year", "scenario"])["nitrogen_executed_kg_ha"].cumsum()
    keep = [
        "site",
        "station",
        "station_code",
        "requested_year",
        "year",
        "algorithm",
        "scenario",
        "date",
        "doy",
        "dap",
        "rainfall_mm",
        "tmax_c",
        "tmin_c",
        "soil_water_mm",
        "water_stress_index_wspd",
        "nitrogen_stress_index_nstd",
        "irrigation_executed_mm",
        "nitrogen_executed_kg_ha",
        "grain_yield_kg_ha",
        "biomass_kg_ha",
        "cumulative_irrigation_mm",
        "cumulative_nitrogen_kg_ha",
    ]
    for col in keep:
        if col not in out:
            out[col] = np.nan
    return out[keep]


def normalize_ppo_daily(year: int) -> pd.DataFrame:
    path = OUT / "daily_outputs" / STATION / f"{STATION}_{year}_source{SOURCE_YEAR}_seed{SOURCE_SEED}_ckpt{SOURCE_CHECKPOINT}_transfer_daily.csv"
    raw = pd.read_csv(path)
    out = raw.copy()
    out["scenario"] = "rl_candidate"
    return normalize_daily_columns(out, "MaskablePPO_transfer")


def endpoint_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (year, scenario), g in daily.groupby(["year", "scenario"]):
        ordered = g.sort_values("dap")
        y = float(pd.to_numeric(ordered["grain_yield_kg_ha"], errors="coerce").max())
        b = float(pd.to_numeric(ordered["biomass_kg_ha"], errors="coerce").max())
        i = float(pd.to_numeric(ordered["irrigation_executed_mm"], errors="coerce").fillna(0).sum())
        n = float(pd.to_numeric(ordered["nitrogen_executed_kg_ha"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "station_code": STATION,
                "site": "LC",
                "year": int(year),
                "scenario": scenario,
                "final_grain_kg_ha": y,
                "final_biomass_kg_ha": b,
                "irrigation_event_total_mm": i,
                "nitrogen_event_total_kg_ha": n,
                "PFP_N": y / n if n > 0 and np.isfinite(y) else math.nan,
                "max_water_stress_wspd": float(pd.to_numeric(ordered["water_stress_index_wspd"], errors="coerce").max()),
                "max_nitrogen_stress_nstd": float(pd.to_numeric(ordered["nitrogen_stress_index_nstd"], errors="coerce").max()),
                "irrigation_event_count": int((pd.to_numeric(ordered["irrigation_executed_mm"], errors="coerce").fillna(0) > 0).sum()),
                "nitrogen_event_count": int((pd.to_numeric(ordered["nitrogen_executed_kg_ha"], errors="coerce").fillna(0) > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def compare_transfer(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, g in summary.groupby("year"):
        ppo = g[g["scenario"].eq("rl_candidate")]
        expert = g[g["scenario"].eq("official_extension_expert")]
        if ppo.empty or expert.empty:
            continue
        p = ppo.iloc[0]
        e = expert.iloc[0]
        baselines = g[g["scenario"].ne("rl_candidate")]
        best_y = float(baselines["final_grain_kg_ha"].max())
        best_pfp = float(baselines["PFP_N"].max(skipna=True))
        rows.append(
            {
                "year": int(year),
                "ppo_yield": p["final_grain_kg_ha"],
                "expert_yield": e["final_grain_kg_ha"],
                "delta_yield_vs_expert": p["final_grain_kg_ha"] - e["final_grain_kg_ha"],
                "delta_yield_vs_best_baseline": p["final_grain_kg_ha"] - best_y,
                "ppo_irrigation": p["irrigation_event_total_mm"],
                "expert_irrigation": e["irrigation_event_total_mm"],
                "irrigation_saving_vs_expert": e["irrigation_event_total_mm"] - p["irrigation_event_total_mm"],
                "ppo_n": p["nitrogen_event_total_kg_ha"],
                "expert_n": e["nitrogen_event_total_kg_ha"],
                "n_saving_vs_expert": e["nitrogen_event_total_kg_ha"] - p["nitrogen_event_total_kg_ha"],
                "ppo_PFP_N": p["PFP_N"],
                "expert_PFP_N": e["PFP_N"],
                "delta_PFP_N_vs_expert": p["PFP_N"] - e["PFP_N"],
                "delta_PFP_N_vs_best_baseline": p["PFP_N"] - best_pfp if np.isfinite(best_pfp) else np.nan,
                "ppo_max_wspd": p["max_water_stress_wspd"],
                "ppo_max_nstd": p["max_nitrogen_stress_nstd"],
            }
        )
    return pd.DataFrame(rows)


def plot_year(daily: pd.DataFrame, year: int) -> list[Path]:
    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    weather = daily[daily["scenario"].eq("null")].sort_values("dap")
    ax = axes[0, 0]
    ax.bar(weather["dap"], pd.to_numeric(weather["rainfall_mm"], errors="coerce"), color="#3977A8", alpha=0.58, label="Rain")
    ax.set_ylabel("Rain (mm)")
    ax2 = ax.twinx()
    ax2.plot(weather["dap"], pd.to_numeric(weather["tmax_c"], errors="coerce"), color="#C23B32", lw=1.3, label="Tmax")
    ax2.plot(weather["dap"], pd.to_numeric(weather["tmin_c"], errors="coerce"), color="#686868", lw=1.3, ls="--", label="Tmin")
    ax2.set_ylabel("Temperature (°C)")
    ax.set_title("Weather", loc="left", fontweight="bold")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, ncol=3, fontsize=8, loc="upper right")

    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty:
            continue
        label = LABELS[scenario]
        color = COLORS[scenario]
        style = STYLES[scenario]
        axes[0, 1].plot(sub["dap"], pd.to_numeric(sub["soil_water_mm"], errors="coerce"), color=color, ls=style, lw=1.35, label=label)
        axes[1, 0].plot(sub["dap"], pd.to_numeric(sub["water_stress_index_wspd"], errors="coerce"), color=color, ls=style, lw=1.35, label=label)
        axes[1, 1].plot(sub["dap"], pd.to_numeric(sub["nitrogen_stress_index_nstd"], errors="coerce"), color=color, ls=style, lw=1.35, label=label)
        for ax_ev, col in ((axes[2, 0], "irrigation_executed_mm"), (axes[2, 1], "nitrogen_executed_kg_ha")):
            vals = pd.to_numeric(sub[col], errors="coerce").fillna(0)
            ev = sub[vals.gt(0)].copy()
            vals_ev = pd.to_numeric(ev[col], errors="coerce")
            ax_ev.vlines(ev["dap"], 0, vals_ev, color=color, lw=2, alpha=0.88)
            ax_ev.scatter(ev["dap"], vals_ev, color=color, marker="D" if scenario == "rl_candidate" else "o", s=24, label=label)
        axes[3, 0].plot(sub["dap"], pd.to_numeric(sub["grain_yield_kg_ha"], errors="coerce"), color=color, ls=style, lw=1.4, label=f"{label} grain")
        axes[3, 0].plot(sub["dap"], pd.to_numeric(sub["biomass_kg_ha"], errors="coerce"), color=color, ls=style, lw=0.9, alpha=0.42)
        axes[3, 1].plot(sub["dap"], pd.to_numeric(sub["cumulative_irrigation_mm"], errors="coerce"), color=color, ls=style, lw=1.35, label=f"{label} I")
        axes[3, 1].plot(sub["dap"], pd.to_numeric(sub["cumulative_nitrogen_kg_ha"], errors="coerce"), color=color, ls=style, lw=0.9, alpha=0.55)

    axes[0, 1].text(0.02, 0.90, "SWTD not available in current completed baseline daily files", transform=axes[0, 1].transAxes, fontsize=8)
    titles = [
        (axes[0, 1], "Soil water", "SWTD (mm)"),
        (axes[1, 0], "Water stress index", "WSPD (0=no stress)"),
        (axes[1, 1], "Nitrogen stress index", "NSTD (0=no stress)"),
        (axes[2, 0], "Irrigation events", "mm/event"),
        (axes[2, 1], "Nitrogen application events", "kg/ha/event"),
        (axes[3, 0], "Grain and biomass trajectories", "kg/ha"),
        (axes[3, 1], "Cumulative irrigation and nitrogen", "mm or kg/ha"),
    ]
    for axx, title, ylabel in titles:
        axx.set_title(title, loc="left", fontweight="bold")
        axx.set_ylabel(ylabel)
        axx.grid(color="#E8E8E8", linewidth=0.65)
    for ax_ev in axes[2, :]:
        handles, labels = ax_ev.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_ev.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[3, 1].legend(fontsize=6, ncol=2)
    axes[3, 0].text(0.01, 0.97, "Thin companion lines are biomass; thick lines are grain.", transform=axes[3, 0].transAxes, va="top", fontsize=7)
    axes[3, 1].text(0.01, 0.97, "Thick lines: cumulative irrigation; thin lines: cumulative nitrogen.", transform=axes[3, 1].transAxes, va="top", fontsize=7)
    axes[3, 0].set_xlabel("DAP")
    axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"LC{year} MaskablePPO five-scenario daily process (LC2010 seed0 ckpt50k transfer)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    base_path = OUT / "figures" / f"032_07_lc{year}_source2010_seed0_ckpt50000_transfer_five_scenario_daily"
    paths = [base_path.with_suffix(".png"), base_path.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    ensure_dirs()
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    shutil.copyfile(CONFIG, OUT / "configs" / CONFIG.name)
    direct_ppo.OUTPUT_ROOT = OUT

    eval_rows = []
    for year in TARGET_YEARS:
        eval_rows.append(evaluate_transfer_year(year))
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(OUT / "evaluation" / "032_07_transfer_eval_summary.csv", index=False, encoding="utf-8-sig")

    baseline = normalize_baseline_daily()
    all_daily_parts = []
    manifest_rows: list[dict[str, Any]] = []
    checks_rows: list[dict[str, Any]] = []
    for year in TARGET_YEARS:
        year_baseline = baseline[baseline["year"].eq(year)].copy()
        ppo_daily_path = OUT / "daily_outputs" / STATION / f"{STATION}_{year}_source{SOURCE_YEAR}_seed{SOURCE_SEED}_ckpt{SOURCE_CHECKPOINT}_transfer_daily.csv"
        if not ppo_daily_path.exists():
            checks_rows.append({"year": year, "check": "ppo_daily_exists", "passed": False, "value": str(ppo_daily_path.relative_to(ROOT))})
            continue
        ppo = normalize_ppo_daily(year)
        daily = pd.concat([year_baseline, ppo], ignore_index=True, sort=False)
        scenarios_present = sorted(daily["scenario"].dropna().unique().tolist())
        full_present = set(scenarios_present) == set(SCENARIOS)
        checks_rows.append({"year": year, "check": "five_scenarios_present", "passed": full_present, "value": ",".join(scenarios_present)})
        daily_path = OUT / "tables" / f"032_07_lc{year}_five_scenario_daily.csv"
        summary_path = OUT / "tables" / f"032_07_lc{year}_five_scenario_summary.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        summary = endpoint_summary(daily)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        figs = plot_year(daily, year) if full_present else []
        all_daily_parts.append(daily)
        manifest_rows.append(
            {
                "year": year,
                "daily_csv": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
                "summary_csv": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
                "figures": ";".join(str(p.relative_to(ROOT)).replace("\\", "/") for p in figs),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    checks = pd.DataFrame(checks_rows)
    all_daily = pd.concat(all_daily_parts, ignore_index=True, sort=False) if all_daily_parts else pd.DataFrame()
    all_summary = endpoint_summary(all_daily) if not all_daily.empty else pd.DataFrame()
    comparison = compare_transfer(all_summary) if not all_summary.empty else pd.DataFrame()
    manifest.to_csv(OUT / "032_07_manifest.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(OUT / "tables" / "032_07_checks.csv", index=False, encoding="utf-8-sig")
    all_summary.to_csv(OUT / "tables" / "032_07_all_year_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "tables" / "032_07_transfer_vs_expert_comparison.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 032_07 LC2010 seed0/50k cross-year transfer record",
        "",
        "## Scope",
        "",
        "- Frozen source model: LC2010 MaskablePPO seed0 checkpoint 50k from 032_04.",
        "- Target years: LC2012-LC2015.",
        "- No retraining, no reward change, no checkpoint reselection.",
        "- LC2011 is excluded because current completed-baseline daily files do not contain DSSAT auto for that year.",
        "- Mixed cumulative reward is not plotted; figures use cumulative irrigation/nitrogen per 032_02.",
        "",
        "## Transfer evaluation summary",
        "",
        md_table(eval_df),
        "",
        "## Five-scenario endpoint summary",
        "",
        md_table(all_summary[["year", "scenario", "final_grain_kg_ha", "irrigation_event_total_mm", "nitrogen_event_total_kg_ha", "PFP_N", "max_water_stress_wspd", "max_nitrogen_stress_nstd", "irrigation_event_count", "nitrogen_event_count"]], 80) if not all_summary.empty else "No summary rows.",
        "",
        "## PPO transfer vs official expert / best baseline",
        "",
        md_table(comparison),
        "",
        "## Checks",
        "",
        md_table(checks),
        "",
        "## Outputs",
        "",
        md_table(manifest),
        "",
        "## Interpretation boundary",
        "",
        "- This is a frozen-policy transfer diagnostic, not a new training result.",
        "- Any favorable or unfavorable year should be interpreted as LC2010-policy transfer behavior.",
        "- Soil water SWTD is not available in the completed generated baseline daily files used here; the soil-water panel is therefore marked unavailable rather than fabricated.",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")
    result = {
        "task": "032_07_lc2010_seed0_50k_cross_year_transfer",
        "training_run": False,
        "dssat_run": True,
        "source_model": str(MODEL.relative_to(ROOT)).replace("\\", "/"),
        "target_years": TARGET_YEARS,
        "eval_summary": str((OUT / "evaluation" / "032_07_transfer_eval_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "comparison": str((OUT / "tables" / "032_07_transfer_vs_expert_comparison.csv").relative_to(ROOT)).replace("\\", "/"),
        "manifest": str((OUT / "032_07_manifest.csv").relative_to(ROOT)).replace("\\", "/"),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "032_07_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not comparison.empty:
        print("\nComparison:")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
