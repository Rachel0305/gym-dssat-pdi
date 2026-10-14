"""Replot frozen HLA WGEN validation without the automatic baseline.

Reads the archived five-scenario CSVs. Never changes training or snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/hla_wgen_8seed_053/055_03_five_scenario/HL"
OUT = ROOT / "results/hla_wgen_8seed_053_no_auto_baseline_231"
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
    if (OUT / "audit.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed report: {OUT}")
    audit = []
    for seed in range(8):
        source = SOURCE / f"best_seed_seed{seed}/tables/hl_seed{seed}_yearly_five_scenario_metrics.csv"
        frame = pd.read_csv(source, keep_default_na=False)
        keep = BASES + [f"ppo_seed_{seed}"]
        frame = frame[frame.scenario.isin(keep)].copy()
        assert len(frame) == 40 and set(frame.scenario) == set(keep)
        frame["scenario_label"] = frame.scenario.map({**NAMES, f"ppo_seed_{seed}": f"PPO seed {seed}"})
        dest = OUT / f"seed_{seed:02d}"
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / "yearly_metrics.csv").exists():
            frame.to_csv(dest / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
        agg = frame.groupby(["scenario", "scenario_label"], as_index=False)[[m for m, _ in METRICS]].mean(numeric_only=True)
        if not (dest / "ten_year_means.csv").exists():
            agg.to_csv(dest / "ten_year_means.csv", index=False, encoding="utf-8-sig")
        years = sorted(frame.year.unique())
        colors = ["#666666", "#c44e52", "#7e63b6", "#2a9d55"]
        for metric, label in METRICS:
            plot_path = dest / f"yearly_{metric}.png"
            if plot_path.exists():
                continue
            fig, ax = plt.subplots(figsize=(12, 4.7))
            x = np.arange(len(years))
            for i, scenario in enumerate(keep):
                rows = frame[frame.scenario.eq(scenario)].set_index("year").reindex(years)
                val = pd.to_numeric(rows[metric], errors="coerce").to_numpy()
                ax.bar(x + (i - 1.5) * 0.2, val, 0.19, color=colors[i], label=NAMES.get(scenario, f"PPO seed {seed}"))
            ax.set_xticks(x, years)
            ax.set_ylabel(label)
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
        audit.append({"seed": seed, "rows": len(frame), "years": [int(y) for y in years], "source": str(source.relative_to(ROOT))})
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# HLA WGEN 100K eight seeds: four scenario replot\n\n"
        "Retains Null, recorded template, official expert and PPO. The archived "
        "`dssat_auto_external_n` comparison is excluded from these tables and figures. "
        "PPO checkpoint, replay, weather and original five-scenario archive are unchanged. "
        "All ten validation years and all eight seeds are retained.\n\n"
        "Source: `results/hla_wgen_8seed_053/055_03_five_scenario/HL`.\n"
        "Run: `python src/plot_hla_wgen_no_auto_231.py`.\n",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
