"""Summarize the eight archived seeds of the isolated HL/FQ revisions."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "HL": ROOT / "results/hla_wgen_8seed_053_no_auto_baseline_231",
    "FQ": ROOT / "results/fqa_wgen_052_corrected2018_231",
}
METRICS = ["grain_yield_kg_ha", "total_irrigation_mm", "total_n_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_grain_per_kg_N"]


def main() -> None:
    for site, root in SOURCES.items():
        rows = []
        for seed in range(8):
            folder = root / f"seed_{seed:02d}"
            table = folder / "yearly_metrics.csv" if site == "HL" else folder / "four_scenario_ten_year/yearly_metrics.csv"
            frame = pd.read_csv(table, keep_default_na=False)
            assert len(frame) == 40
            ppo = frame[frame.scenario.eq(f"ppo_seed_{seed}")]
            expert = frame[frame.scenario.eq("official_extension_expert")]
            assert len(ppo) == len(expert) == 10
            row = {"site": site, "seed": seed, "validation_years": 10}
            for metric in METRICS:
                p = pd.to_numeric(ppo[metric], errors="coerce")
                e = pd.to_numeric(expert[metric], errors="coerce")
                row[f"ppo_mean_{metric}"] = p.mean()
                row[f"expert_mean_{metric}"] = e.mean()
                row[f"delta_ppo_minus_expert_{metric}"] = p.mean() - e.mean()
            rows.append(row)
        output = root / "eight_seed_comparison.csv"
        if not output.exists():
            pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
        print(output)

    fq_root = SOURCES["FQ"]
    repair_rows = []
    for seed in range(8):
        original = ROOT / f"results/fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ/best_seed_seed{seed}/tables/fq_seed{seed}_yearly_five_scenario_metrics.csv"
        old = pd.read_csv(original, keep_default_na=False)
        old = old[old.year.eq(2018) & old.scenario.eq(f"ppo_seed_{seed}")].iloc[0]
        new = pd.read_csv(fq_root / f"seed_{seed:02d}/corrected2018_metrics.csv").iloc[0]
        repair_rows.append({
            "seed": seed, "frozen_2018_yield_kg_ha": old.grain_yield_kg_ha,
            "corrected_2018_yield_kg_ha": new.grain_yield_kg_ha,
            "corrected_2018_irrigation_mm": new.total_irrigation_mm,
            "corrected_2018_n_kg_ha": new.total_n_kg_ha,
            "corrected_2018_WP_ET_kg_m3": new.WP_ET_kg_m3,
            "weather_sha256": new.weather_sha256,
        })
    repair_output = fq_root / "fq2018_original_vs_corrected_eight_seeds.csv"
    if repair_output.exists():
        raise FileExistsError(repair_output)
    pd.DataFrame(repair_rows).to_csv(repair_output, index=False, encoding="utf-8-sig")
    print(repair_output)


if __name__ == "__main__":
    main()
