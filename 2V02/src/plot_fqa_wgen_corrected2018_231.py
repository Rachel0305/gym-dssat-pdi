"""Build ten-year FQA tables/figures with corrected 2018, preserving 052."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/fqa_multiyear_wgen_ppo_052_8seed/055_03_five_scenario/FQ"
OUT = ROOT / "results/fqa_wgen_052_corrected2018_231"
CORRECTED_BASE = ROOT / "benchmark_results/088_00_fq2018_input_quality_correction_run_corrected2018/evaluation/051_02_baseline_summary.csv"
BASES = ["null", "recorded_farmer_template", "official_extension_expert"]
NAMES = {"null": "Null", "recorded_farmer_template": "Recorded", "official_extension_expert": "Expert"}
METRICS = [
    ("grain_yield_kg_ha", "Yield (kg/ha)"),
    ("total_irrigation_mm", "Irrigation (mm)"),
    ("total_n_kg_ha", "N (kg/ha)"),
    ("WP_ET_kg_m3", "WP_ET (kg/m3)"),
    ("PFP_N_kg_grain_per_kg_N", "PFP_N (kg/kg)"),
]


def main() -> None:
    base = pd.read_csv(CORRECTED_BASE, keep_default_na=False)
    base = base[base.scenario.isin(BASES)].copy()
    assert len(base) == 3 and set(base.year) == {2018}
    if (OUT / "ten_year_revision_audit.json").exists():
        raise FileExistsError("Completed corrected FQ report exists")
    audit = []
    for seed in range(8):
        original = SOURCE / f"best_seed_seed{seed}/tables/fq_seed{seed}_yearly_five_scenario_metrics.csv"
        ppo = OUT / f"seed_{seed:02d}/corrected2018_metrics.csv"
        if not ppo.exists():
            raise FileNotFoundError(ppo)
        frame = pd.read_csv(original, keep_default_na=False)
        order = BASES + [f"ppo_seed_{seed}"]
        frame = frame[frame.scenario.isin(order) & frame.year.ne(2018)].copy()
        ppo_row = pd.read_csv(ppo, keep_default_na=False)
        corrected_rows = []
        for row in base.to_dict(orient="records"):
            corrected_rows.append({
                "site": "FQ", "station_code": "FQA", "seed": seed,
                "year": 2018, "checkpoint_step": 100000,
                "scenario": row["scenario"], "scenario_label": NAMES[row["scenario"]],
                "grain_yield_kg_ha": row["grain_yield_kg_ha"],
                "total_irrigation_mm": row["actual_irrigation_mm"],
                "total_n_kg_ha": row["actual_nitrogen_kg_ha"],
                "etcp_mm": row["etcp_mm"], "WP_ET_kg_m3": row["WP_ET_kg_m3"],
                "PFP_N_kg_grain_per_kg_N": row["PFP_N_kg_kg"],
                "snapshot_path": row["snapshot_path"],
                "source_status": "088 isolated corrected FQ2018 native baseline",
            })
        revision = pd.concat([frame, pd.DataFrame(corrected_rows), ppo_row], ignore_index=True, sort=False)
        revision["scenario_label"] = revision.scenario.map({**NAMES, f"ppo_seed_{seed}": f"PPO seed {seed}"})
        revision = revision.sort_values(["year", "scenario"]).reset_index(drop=True)
        assert len(revision) == 40 and revision.groupby(["year", "scenario"]).size().eq(1).all()
        report = OUT / f"seed_{seed:02d}" / "four_scenario_ten_year"
        report.mkdir(parents=True, exist_ok=True)
        if (report / "yearly_metrics.csv").exists():
            raise FileExistsError(report / "yearly_metrics.csv")
        revision.to_csv(report / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
        agg = revision.groupby(["scenario", "scenario_label"], as_index=False)[[m for m, _ in METRICS]].mean(numeric_only=True)
        agg.to_csv(report / "ten_year_means.csv", index=False, encoding="utf-8-sig")
        years = sorted(map(int, revision.year.unique()))
        colors = ["#666666", "#c44e52", "#7e63b6", "#2a9d55"]
        for metric, label in METRICS:
            fig, ax = plt.subplots(figsize=(12, 4.7))
            x = np.arange(len(years))
            for i, scenario in enumerate(order):
                rows = revision[revision.scenario.eq(scenario)].set_index("year").reindex(years)
                value = pd.to_numeric(rows[metric], errors="coerce").to_numpy()
                ax.bar(x + (i - 1.5) * 0.2, value, 0.19, color=colors[i], label=NAMES.get(scenario, f"PPO seed {seed}"))
            ax.set_xticks(x, years)
            ax.set_ylabel(label)
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(report / f"yearly_{metric}.png", dpi=180)
            plt.close(fig)
        audit.append({"seed": seed, "rows": len(revision), "corrected2018_ppo_yield": float(ppo_row.grain_yield_kg_ha.iloc[0])})
    (OUT / "ten_year_revision_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# FQA WGEN 052 eight seeds: corrected 2018 validation\n\n"
        "The frozen 052 figures and models remain unchanged. The 2018 FQ weather copy "
        "uses the 088 documented 15-day summer TMIN sign correction. Every 100K model "
        "was replayed for 2018 and its DSSAT snapshot, daily CSV, and exact Summary.OUT/ETCP "
        "metrics are archived under its seed directory. The ten-year tables and figures "
        "use 2014-2017 and 2019-2023 frozen results plus the corrected 2018 replay. "
        "They compare Null, recorded template, expert and PPO. The auto baseline is "
        "excluded because the existing 088 corrected auto baseline uses a different N contract.\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
