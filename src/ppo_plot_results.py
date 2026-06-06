from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_episode(daily: pd.DataFrame, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = pd.to_numeric(daily["dap"], errors="coerce")
    specs = [
        ("dap_swfac_irrigation_reward.png", ["swfac", "real_action_amir", "reward"], "Water stress, irrigation, reward"),
        ("dap_nstres_fertilization_reward.png", ["nstres", "real_action_anfer", "reward"], "Nitrogen stress, fertilization, reward"),
        ("crop_growth_timeseries.png", ["topwt", "grnwt", "xlai"], "Crop growth"),
        ("cumulative_water_nitrogen.png", ["totir", "tofer"], "Cumulative water and nitrogen"),
        ("daily_actions.png", ["real_action_amir", "real_action_anfer"], "Daily actions"),
    ]
    paths: list[Path] = []
    for filename, cols, title in specs:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for col in cols:
            if col in daily.columns:
                ax.plot(x, pd.to_numeric(daily[col], errors="coerce"), label=col)
        ax.set_title(title)
        ax.set_xlabel("DAP")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figure_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
