from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROMPT = ROOT / "prompts" / "032_06_lc2010_guardrail_selected_five_scenario_daily.md"
BASE_DAILY = ROOT / "benchmark_results" / "031_39_representative_free_timing_ppo_five_scenario_daily" / "031_39_lca2010_current_free_timing_ppo_five_scenario_daily.csv"
BASE_SUMMARY = ROOT / "benchmark_results" / "031_39_representative_free_timing_ppo_five_scenario_daily" / "031_39_lca2010_current_free_timing_ppo_five_scenario_summary.csv"
SELECTION = ROOT / "benchmark_results" / "032_05_lc2010_checkpoint_guardrail_audit" / "tables" / "032_05_guardrail_selected_checkpoints.csv"
OUT = ROOT / "benchmark_results" / "032_06_lc2010_guardrail_selected_five_scenario_daily"
DOC = ROOT / "docs" / "032_06_lc2010_guardrail_selected_five_scenario_daily_record.md"

SCENARIOS = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert", "rl_candidate"]
LABELS = {
    "null": "Null",
    "recorded_farmer": "Recorded farmer",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "Official expert",
    "rl_candidate": "MaskablePPO candidate",
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


@dataclass(frozen=True)
class Candidate:
    seed: int
    checkpoint_step: int
    final_grnwt: float
    total_irrigation: float
    total_n: float
    max_nstres: float
    action_sequence: str
    daily_csv_path: Path


def ensure_dirs() -> None:
    for rel in ["configs", "tables", "figures"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def load_candidates() -> list[Candidate]:
    sel = pd.read_csv(SELECTION)
    sel = sel[
        sel["yield_guardrail"].eq("95pct_expert")
        & sel["nstress_guardrail"].eq("moderate_nstress")
        & sel["selection_status"].eq("selected")
    ].copy()
    if len(sel) != 3:
        raise RuntimeError(f"Expected 3 selected rows for 95pct_expert/moderate_nstress, got {len(sel)}")
    candidates: list[Candidate] = []
    for _, row in sel.sort_values("seed").iterrows():
        seed = int(row["seed"])
        ckpt = int(row["checkpoint_step"])
        daily = ROOT / "benchmark_results" / "032_04_lc2010_stress_aware_ppo_multiseed_200k" / "daily_outputs" / "LCA" / f"LCA_2010_seed{seed}_ckpt{ckpt}_daily.csv"
        if not daily.exists():
            raise FileNotFoundError(daily)
        candidates.append(
            Candidate(
                seed=seed,
                checkpoint_step=ckpt,
                final_grnwt=float(row["final_grnwt"]),
                total_irrigation=float(row["total_irrigation"]),
                total_n=float(row["total_n"]),
                max_nstres=float(row["max_nstres"]),
                action_sequence=str(row.get("action_sequence", "")),
                daily_csv_path=daily,
            )
        )
    return candidates


def normalize_ppo_daily(candidate: Candidate, base_daily: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(candidate.daily_csv_path)
    weather = base_daily[base_daily["scenario"].eq("null")][["dap", "rainfall_mm", "tmax_c", "tmin_c", "date"]].drop_duplicates("dap")
    out = raw.rename(
        columns={
            "safe_action_amir": "irrigation_executed_mm",
            "safe_action_anfer": "nitrogen_executed_kg_ha",
            "swfac": "water_stress_index_wspd",
            "nstres": "nitrogen_stress_index_nstd",
            "grnwt": "grain_yield_kg_ha",
            "topwt": "biomass_kg_ha",
        }
    ).copy()
    out["scenario"] = "rl_candidate"
    out["site"] = "LC"
    out["station"] = "Luancheng"
    out["requested_year"] = 2010
    out["algorithm"] = "MaskablePPO"
    out["seed"] = candidate.seed
    out["checkpoint"] = candidate.checkpoint_step
    out = out.merge(weather, on="dap", how="left", suffixes=("", "_baseline"))
    for col, fallback in [("rainfall_mm", "rain"), ("tmax_c", "tmax"), ("tmin_c", "tmin")]:
        if col not in out:
            out[col] = np.nan
        if fallback in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(pd.to_numeric(out[fallback], errors="coerce"))
    if "date_baseline" in out:
        out["date"] = out.get("date", pd.Series([None] * len(out))).fillna(out["date_baseline"])
    elif "date" not in out:
        out["date"] = ""
    out["soil_water_mm"] = np.nan
    # The 032_04 daily CSV does not include SWTD. Keep baseline SWTD panels valid
    # for four scenarios and leave the PPO line absent rather than fabricating it.
    out["temperature_source_qc"] = "pass"
    out["cumulative_irrigation_mm"] = pd.to_numeric(out["irrigation_executed_mm"], errors="coerce").fillna(0).cumsum()
    out["cumulative_nitrogen_kg_ha"] = pd.to_numeric(out["nitrogen_executed_kg_ha"], errors="coerce").fillna(0).cumsum()
    keep = [
        "site",
        "station",
        "requested_year",
        "algorithm",
        "seed",
        "checkpoint",
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
        "temperature_source_qc",
    ]
    for col in keep:
        if col not in out:
            out[col] = np.nan
    return out[keep]


def prepare_case(candidate: Candidate) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_daily = pd.read_csv(BASE_DAILY, keep_default_na=False)
    base_summary = pd.read_csv(BASE_SUMMARY, keep_default_na=False)
    base_part = base_daily[~base_daily["scenario"].eq("rl_candidate")].copy()
    for col in ["irrigation_executed_mm", "nitrogen_executed_kg_ha"]:
        base_part[col] = pd.to_numeric(base_part[col], errors="coerce").fillna(0)
    base_part["cumulative_irrigation_mm"] = base_part.groupby("scenario")[["irrigation_executed_mm"]].cumsum()
    base_part["cumulative_nitrogen_kg_ha"] = base_part.groupby("scenario")[["nitrogen_executed_kg_ha"]].cumsum()
    ppo = normalize_ppo_daily(candidate, base_part)
    daily = pd.concat([base_part, ppo], ignore_index=True, sort=False)

    rows = []
    for scenario, g in daily.groupby("scenario"):
        ordered = g.sort_values("dap")
        n_total = float(pd.to_numeric(ordered["nitrogen_executed_kg_ha"], errors="coerce").sum())
        y = float(pd.to_numeric(ordered["grain_yield_kg_ha"], errors="coerce").max())
        i_total = float(pd.to_numeric(ordered["irrigation_executed_mm"], errors="coerce").sum())
        rows.append(
            {
                "scenario": scenario,
                "final_grain_kg_ha": y,
                "final_biomass_kg_ha": float(pd.to_numeric(ordered["biomass_kg_ha"], errors="coerce").max()),
                "irrigation_event_total_mm": i_total,
                "nitrogen_event_total_kg_ha": n_total,
                "PFP_N": y / n_total if n_total > 0 else math.nan,
                "max_water_stress_wspd": float(pd.to_numeric(ordered["water_stress_index_wspd"], errors="coerce").max()),
                "max_nitrogen_stress_nstd": float(pd.to_numeric(ordered["nitrogen_stress_index_nstd"], errors="coerce").max()),
            }
        )
    summary = pd.DataFrame(rows)
    checks = pd.DataFrame(
        [
            {"seed": candidate.seed, "checkpoint": candidate.checkpoint_step, "check": "base_daily_exists", "passed": BASE_DAILY.exists(), "value": str(BASE_DAILY.relative_to(ROOT))},
            {"seed": candidate.seed, "checkpoint": candidate.checkpoint_step, "check": "ppo_daily_exists", "passed": candidate.daily_csv_path.exists(), "value": str(candidate.daily_csv_path.relative_to(ROOT))},
            {"seed": candidate.seed, "checkpoint": candidate.checkpoint_step, "check": "five_scenarios_present", "passed": set(daily["scenario"]) == set(SCENARIOS), "value": ",".join(sorted(set(daily["scenario"])))},
            {"seed": candidate.seed, "checkpoint": candidate.checkpoint_step, "check": "ppo_irrigation_matches_selection", "passed": abs(summary.loc[summary.scenario.eq("rl_candidate"), "irrigation_event_total_mm"].iloc[0] - candidate.total_irrigation) < 1e-6, "value": candidate.total_irrigation},
            {"seed": candidate.seed, "checkpoint": candidate.checkpoint_step, "check": "ppo_n_matches_selection", "passed": abs(summary.loc[summary.scenario.eq("rl_candidate"), "nitrogen_event_total_kg_ha"].iloc[0] - candidate.total_n) < 1e-6, "value": candidate.total_n},
        ]
    )
    return daily, summary, checks


def plot_daily(daily: pd.DataFrame, candidate: Candidate) -> list[Path]:
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
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    names = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, names, ncol=3, fontsize=8, loc="upper right")

    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
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
        handles, names = ax_ev.get_legend_handles_labels()
        unique = dict(zip(names, handles))
        ax_ev.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[3, 1].legend(fontsize=6, ncol=2)
    axes[3, 0].text(0.01, 0.97, "Thin companion lines are biomass; thick lines are grain.", transform=axes[3, 0].transAxes, va="top", fontsize=7)
    axes[3, 1].text(0.01, 0.97, "Thick lines: cumulative irrigation; thin lines: cumulative nitrogen.", transform=axes[3, 1].transAxes, va="top", fontsize=7)
    axes[3, 0].set_xlabel("DAP")
    axes[3, 1].set_xlabel("DAP")
    fig.suptitle(f"LC2010 MaskablePPO five-scenario daily process (guardrail seed{candidate.seed}, ckpt{candidate.checkpoint_step})", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    base = OUT / "figures" / f"032_06_lc2010_seed{candidate.seed}_ckpt{candidate.checkpoint_step}_five_scenario_daily"
    paths = [base.with_suffix(".png"), base.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, values)) + " |" for values in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    ensure_dirs()
    shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)
    candidates = load_candidates()
    manifest_rows: list[dict[str, Any]] = []
    all_summary = []
    all_checks = []
    for candidate in candidates:
        daily, summary, checks = prepare_case(candidate)
        daily_path = OUT / "tables" / f"032_06_lc2010_seed{candidate.seed}_five_scenario_daily.csv"
        summary_path = OUT / "tables" / f"032_06_lc2010_seed{candidate.seed}_five_scenario_summary.csv"
        checks_path = OUT / "tables" / f"032_06_lc2010_seed{candidate.seed}_checks.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        summary.insert(0, "seed", candidate.seed)
        summary.insert(1, "checkpoint_step", candidate.checkpoint_step)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
        figs = plot_daily(daily, candidate)
        all_summary.append(summary)
        all_checks.append(checks)
        manifest_rows.append(
            {
                "seed": candidate.seed,
                "checkpoint_step": candidate.checkpoint_step,
                "daily_csv": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
                "summary_csv": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
                "checks_csv": str(checks_path.relative_to(ROOT)).replace("\\", "/"),
                "figures": ";".join(str(p.relative_to(ROOT)).replace("\\", "/") for p in figs),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_checks = pd.concat(all_checks, ignore_index=True)
    manifest.to_csv(OUT / "032_06_manifest.csv", index=False, encoding="utf-8-sig")
    combined_summary.to_csv(OUT / "tables" / "032_06_lc2010_all_seed_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    combined_checks.to_csv(OUT / "tables" / "032_06_lc2010_all_seed_checks.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 032_06 LC2010 guardrail-selected five-scenario daily record",
        "",
        "## Scope",
        "",
        "- No training.",
        "- No new DSSAT runs.",
        "- Four baseline scenarios reuse 031_39 daily evidence.",
        "- PPO candidates use 032_05 guardrail-selected checkpoints from 032_04 daily outputs.",
        "- The old cumulative reward panel is not reused; final panel shows cumulative irrigation and nitrogen.",
        "",
        "## Selected candidates",
        "",
        md_table(pd.DataFrame([c.__dict__ | {"daily_csv_path": str(c.daily_csv_path.relative_to(ROOT)).replace("\\", "/")} for c in candidates])),
        "",
        "## Five-scenario endpoint summaries",
        "",
        md_table(combined_summary[["seed", "checkpoint_step", "scenario", "final_grain_kg_ha", "irrigation_event_total_mm", "nitrogen_event_total_kg_ha", "PFP_N", "max_water_stress_wspd", "max_nitrogen_stress_nstd"]], 60),
        "",
        "## Evidence checks",
        "",
        md_table(combined_checks, 40),
        "",
        "## Outputs",
        "",
        md_table(manifest, 20),
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")
    result = {
        "task": "032_06_lc2010_guardrail_selected_five_scenario_daily",
        "training_run": False,
        "dssat_run": False,
        "manifest": str((OUT / "032_06_manifest.csv").relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
    }
    (OUT / "032_06_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
