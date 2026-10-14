"""Exploratory, post hoc five-site PPO management screen (read only inputs)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "five_site_strategy_screen_233"
RESULTS = ROOT / "results"
BENCH = ROOT / "benchmark_results"


def read(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def actions(table: pd.DataFrame, irrigation: str, nitrogen: str,
            scenario: str | None = None) -> dict:
    d = table.loc[table.scenario.eq(scenario)].copy() if scenario else table.copy()
    if len(d.year.unique()) != 10:
        raise ValueError(f"Expected 10 validation years; got {sorted(d.year.unique())}")
    irrig = d.loc[d[irrigation] > 0]
    signatures = []
    for _, year in d.groupby("year"):
        year = year.sort_values(["dap", irrigation, nitrogen])
        signatures.append(tuple(zip(year.dap.round(3), year[irrigation].round(3),
                                    year[nitrogen].round(3))))
    return {
        "irrigation_events_per_year": len(irrig) / 10,
        "small_15mm_fraction": float((irrig[irrigation] <= 15).mean()) if len(irrig) else 0.0,
        "distinct_action_schedules": len(set(signatures)),
    }


def conventional(site: str) -> tuple[float, float, float, float, float]:
    if site == "SY":
        base = read("benchmark_results/046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000/tables/046_06_sya_five_scenario_season_summary.csv")
        y, i, n = "final_grain_kg_ha", "irrigation_mm", "nitrogen_kg_ha"
        event_path = "benchmark_results/046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000/tables/046_06_sya_five_scenario_management_events.csv"
    elif site == "LC":
        base = read("benchmark_results/053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/053_03_lca_five_scenario_season_summary.csv")
        y, i, n = "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha"
        event_path = "benchmark_results/053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/053_03_lca_five_scenario_management_events.csv"
    elif site == "YC":
        base = read("benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_season_summary.csv")
        y, i, n = "grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha"
        event_path = "benchmark_results/055_03_yca_lowIC_055_00_yca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000/tables/055_03_yca_five_scenario_management_events.csv"
    else:
        root = ("results/hl_fq_8seed_cross_site/055_03_five_scenario/HL"
                if site == "HL" else
                "results/fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ")
        stem = "hl" if site == "HL" else "fq"
        base = read(f"{root}/best_seed_seed0/tables/{stem}_seed0_yearly_five_scenario_metrics.csv")
        y, i, n = "grain_yield_kg_ha", "total_irrigation_mm", "total_n_kg_ha"
        event_path = f"{root}/best_seed_seed0/tables/{stem}_seed0_five_scenario_management_events.csv"
    means = base.groupby("scenario")[[y, i, n]].mean()
    control = means.loc[["official_extension_expert", "recorded_farmer_template", "dssat_auto_external_n"]]
    expert = means.loc["official_extension_expert"]
    e = read(event_path)
    expert_events = len(e.loc[e.scenario.eq("official_extension_expert") & e.irrigation_executed_mm.gt(0)]) / 10
    return float(control[y].max()), float(expert[y]), float(expert[i]), float(expert[n]), expert_events


def sy_lc(site: str, baselines: tuple) -> list[dict]:
    package = RESULTS / "sy_lc_8seed_extension_231" / "five_scenario_figures_v2" / site
    means = pd.read_csv(package / "eight_seed_ten_year_means.csv")
    frozen = (BENCH / ("046_06_sya_originIC_046_10_sya_originIC_expanded_action_maskableppo_five_scenario_daily_ckpt100000"
                       if site == "SY" else
                       "053_03_lca_lowIC_053_00_lca_lowIC_expanded_action_maskableppo_auto_nstd050_minimal_five_scenario_figures_ckpt100000"))
    out = []
    for seed in range(8):
        if seed == 0:
            event = next((frozen / "tables").glob("*management_events.csv"))
        else:
            event = next((package / f"seed_{seed:02d}" / "tables").glob("*management_events.csv"))
        row = means.loc[means.seed.eq(seed)].iloc[0]
        m = actions(pd.read_csv(event), "irrigation_executed_mm", "nitrogen_executed_kg_ha", "rl_candidate")
        out.append(dict(site=site, regime="historical", seed=seed,
                        yield_kg_ha=float(row.grain_yield_kg_ha), irrigation_mm=float(row.actual_irrigation_mm),
                        nitrogen_kg_ha=float(row.actual_nitrogen_kg_ha), WP_ET=float(row.WP_ET_kg_m3),
                        PFP_N=float(row.PFP_N_kg_kg), source=str(event.relative_to(ROOT)), **m))
    return out


def hl_fq(site: str, regime: str) -> list[dict]:
    if site == "HL" and regime == "historical":
        root = RESULTS / "hl_fq_8seed_cross_site" / "055_03_five_scenario" / "HL"
    elif site == "HL":
        root = RESULTS / "hla_wgen_8seed_053" / "055_03_five_scenario" / "HL"
    else:
        root = RESULTS / "fqa_multiyear_wgen_ppo_052_8seed" / "055_03_five_scenario" / "FQ"
    stem = site.lower()
    out = []
    for seed in range(8):
        table = root / f"best_seed_seed{seed}" / "tables"
        metric = pd.read_csv(table / f"{stem}_seed{seed}_yearly_five_scenario_metrics.csv")
        p = metric.loc[metric.scenario.eq(f"ppo_seed_{seed}")]
        if len(p) != 10:
            raise ValueError(f"{site} {regime} seed{seed}: {len(p)} PPO seasons")
        event = table / f"{stem}_seed{seed}_five_scenario_management_events.csv"
        m = actions(pd.read_csv(event), "irrigation_executed_mm", "nitrogen_executed_kg_ha", f"ppo_seed_{seed}")
        out.append(dict(site=site, regime=regime, seed=seed,
                        yield_kg_ha=float(p.grain_yield_kg_ha.mean()),
                        irrigation_mm=float(p.total_irrigation_mm.mean()), nitrogen_kg_ha=float(p.total_n_kg_ha.mean()),
                        WP_ET=float(p.WP_ET_kg_m3.mean()), PFP_N=float(p.PFP_N_kg_grain_per_kg_N.mean()),
                        source=str(event.relative_to(ROOT)), **m))
    return out


def yc() -> list[dict]:
    p = RESULTS / "yc_random_weather_ppo" / "004_05" / "all_evaluation_episode_level_0_7.csv"
    d = pd.read_csv(p)
    d = d.loc[d.training_regime.eq("RANDOM_WEATHER_WGEN") & d.evaluation_weather_type.eq("observed_weather")]
    out = []
    for seed in range(8):
        q = d.loc[d.ppo_seed.eq(seed)]
        if len(q) != 10:
            raise ValueError(f"YC seed{seed}: {len(q)} PPO seasons")
        event = RESULTS / "yc_random_weather_ppo" / "004_15_75mm_all_seed_five_scenario_figures" / f"seed_{seed}" / f"seed{seed}_management_events.csv"
        m = actions(pd.read_csv(event), "irrigation", "nitrogen")
        out.append(dict(site="YC", regime="wgen_004_05", seed=seed,
                        yield_kg_ha=float(q["yield"].mean()), irrigation_mm=float(q.total_irrigation.mean()),
                        nitrogen_kg_ha=float(q.total_fertilizer.mean()), WP_ET=float("nan"),
                        PFP_N=float((q["yield"] / q.total_fertilizer.replace(0, float("nan"))).mean()),
                        source=str(event.relative_to(ROOT)), **m))
    return out


def main() -> None:
    baselines = {site: conventional(site) for site in ["SY", "LC", "YC", "HL", "FQ"]}
    rows = sy_lc("SY", baselines["SY"]) + sy_lc("LC", baselines["LC"]) + yc() + hl_fq("HL", "historical") + hl_fq("HL", "wgen") + hl_fq("FQ", "wgen")
    anchors = {"SY": ("historical", 0), "LC": ("historical", 0), "YC": ("wgen_004_05", 5),
               "HL": ("historical", 5), "FQ": ("wgen", 4)}
    lookup = {(r["site"], r["regime"], r["seed"]): r for r in rows}
    for r in rows:
        site = r["site"]
        best_y, expert_y, expert_i, expert_n, expert_events = baselines[site]
        anchor = lookup[(site, *anchors[site])]
        y, i, n = r["yield_kg_ha"], r["irrigation_mm"], r["nitrogen_kg_ha"]
        r.update(best_conventional_yield=best_y, expert_yield=expert_y,
                 expert_irrigation=expert_i, expert_nitrogen=expert_n,
                 expert_irrigation_events_per_year=expert_events,
                 anchor_regime=anchors[site][0], anchor_seed=anchors[site][1])
        # Two routes: top conventional yield, or <=5% yield loss with >=10% saving
        # in at least one input. No input may exceed expert use by >10%.
        resource_cap = i <= 1.10 * expert_i and n <= 1.10 * expert_n
        resource_gain = i <= 0.90 * expert_i or n <= 0.90 * expert_n
        top_yield = y >= best_y and resource_cap and (i < expert_i or n < expert_n)
        conservation = y >= 0.95 * best_y and resource_cap and resource_gain
        r["yield_resource_route"] = "top_yield" if top_yield else "conservation" if conservation else "none"
        r["operation_gate"] = (r["irrigation_events_per_year"] <= 1.5 * expert_events
                               and r["small_15mm_fraction"] <= 2 / 3)
        r["temporal_gate"] = site == "YC" or r["distinct_action_schedules"] >= 2
        # When a candidate yields less than the already frozen/preferred local
        # reference, a tiny resource change is not treated as a meaningful gain.
        r["anchor_materiality_gate"] = (y >= anchor["yield_kg_ha"] or
                                        i <= 0.90 * anchor["irrigation_mm"] or
                                        n <= 0.90 * anchor["nitrogen_kg_ha"])
        r["provisional_pass"] = bool(r["yield_resource_route"] != "none" and
                                      r["operation_gate"] and r["temporal_gate"] and
                                      r["anchor_materiality_gate"])
        r["failure_reasons"] = ";".join(
            name for name, failed in [("yield_resource", r["yield_resource_route"] == "none"),
                                      ("operations", not r["operation_gate"]),
                                      ("fixed_10_year_schedule", not r["temporal_gate"]),
                                      ("no_material_gain_vs_anchor", not r["anchor_materiality_gate"])] if failed)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing screen: {OUT}")
    OUT.mkdir(parents=True)
    frame = pd.DataFrame(rows).sort_values(["site", "regime", "seed"])
    frame.to_csv(OUT / "all_48_strategy_screen.csv", index=False)
    frame.loc[frame.provisional_pass].to_csv(OUT / "provisional_candidates.csv", index=False)
    (OUT / "rule_manifest.json").write_text(json.dumps({
        "status": "post_hoc_exploratory_not_confirmatory",
        "validation_years": "2014-2023",
        "site_local_reference": "four fixed conventional scenarios; expert is resource reference",
        "yield_floor": "at least 95% of highest conventional mean yield",
        "resource_cap": "I and N each <= 110% of expert mean",
        "conservation": "at least 10% lower I or N than expert mean",
        "operation_limit": "mean irrigation events <= 1.5x expert and <= 2/3 of irrigation events <=15mm",
        "temporal": "at least 2 unique DAP+I+N schedules across 10 years, except YC",
        "anchor": anchors,
        "anchor_materiality": "if lower yield than anchor, >=10% lower I or N required",
        "yc_source_note": "004_05 official results; 004_15 event table is a plotting derivative of 004_05",
        "limitations": ["FQ 2018 frozen weather gives zero yield in all scenarios",
                        "YC WP_ET unavailable for five of eight seeds; not a screen gate",
                        "thresholds were formulated after viewing outcomes and require sensitivity review"]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.groupby(["site", "regime"]).provisional_pass.agg(["sum", "count"]).to_string())
    print("Candidates:", frame.loc[frame.provisional_pass, ["site", "regime", "seed", "yield_resource_route"]].to_dict("records"))


if __name__ == "__main__":
    main()
