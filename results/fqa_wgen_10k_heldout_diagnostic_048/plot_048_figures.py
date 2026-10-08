"""Draw an HL-055_03-style figure set from the bounded FQA 048 evidence."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
COLORS = {"2k": "#7E63B6", "10k": "#2A9D55"}
LABELS = {"2k": "PPO seed 0 · 2K smoke", "10k": "PPO seed 0 · 10K multi-year"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = {label: read_csv(BASE / label / "endpoint_metrics.csv") for label in COLORS}
    pairs = {label: {int(row["weather_seed"]): row for row in values} for label, values in rows.items()}
    seeds = [1081, 1100]
    x = np.arange(len(seeds))
    width = 0.34

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.5), sharex=True)
    panels = [
        ("yield_kg_ha", "Grain yield (kg/ha)", 0),
        ("pfp_n_kg_kg", "PFP_N (kg grain/kg N)", 1),
        ("episode_return", "Evaluation episode return (reward units)", 2),
    ]
    for ax, (field, ylabel, _) in zip(axes, panels):
        for offset, label in zip((-width / 2, width / 2), COLORS):
            vals = [float(pairs[label][seed][field]) for seed in seeds]
            bars = ax.bar(x + offset, vals, width, color=COLORS[label], label=LABELS[label])
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}" if field == "episode_return" else f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.24)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[-1].set_xticks(x, [f"2007 · WGEN seed {seed}" for seed in seeds])
    axes[-1].set_xlabel("Paired heldout weather realization")
    fig.suptitle("FQA | 2K smoke vs 10K multi-year PPO | heldout endpoint diagnostics", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.005, "Two heldout realizations only. PFP_N uses wrapper cumulative N; WP_ET unavailable without ETCP replay.", fontsize=8)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])
    save(fig, "fqa_seed0_2k_vs_10k_heldout_metrics.png")

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    for ax, field, ylabel in (
        (axes[0], "irrigation_mm", "Wrapper cumulative irrigation (mm)"),
        (axes[1], "nitrogen_kg_ha", "Wrapper cumulative nitrogen (kg N/ha)"),
    ):
        for offset, label in zip((-width / 2, width / 2), COLORS):
            vals = [float(pairs[label][seed][field]) for seed in seeds]
            bars = ax.bar(x + offset, vals, width, color=COLORS[label], label=LABELS[label])
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:g}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.24)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    axes[-1].set_xticks(x, [f"WGEN seed {seed}" for seed in seeds])
    axes[-1].set_xlabel("2007 heldout weather realization")
    fig.suptitle("FQA | water and nitrogen totals | PPO seed 0", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.005, "Totals are taken from the safety wrapper; DSSAT Summary.OUT closure was not checked in 048.", fontsize=8)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    save(fig, "fqa_seed0_2k_vs_10k_heldout_management.png")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for label, marker in (("2k", "o"), ("10k", "s")):
        vals = [float(pairs[label][seed]["episode_return"]) for seed in seeds]
        ax.plot(x, vals, color=COLORS[label], marker=marker, lw=2, ms=7, label=LABELS[label])
        for xi, val in zip(x, vals):
            ax.annotate(f"{val:.3f}", (xi, val), xytext=(0, 8 if label == "10k" else -15), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, [f"WGEN seed {seed}" for seed in seeds])
    ax.set_xlabel("2007 heldout weather realization")
    ax.set_ylabel("Evaluation episode return (reward units)")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    fig.suptitle("FQA | evaluation reward comparison | PPO seed 0", x=0.02, ha="left", fontweight="bold")
    fig.text(0.02, 0.005, "Deterministic policy evaluation; two weather realizations do not establish generalization.", fontsize=8)
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    save(fig, "fqa_seed0_2k_vs_10k_heldout_reward.png")

    for seed in seeds:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, field, ylabel in (
            (axes[0], "irrigation_mm", "Irrigation (mm)"),
            (axes[1], "nitrogen_kg_ha", "Nitrogen (kg N/ha)"),
        ):
            vals = [float(pairs[label][seed][field]) for label in COLORS]
            bars = ax.bar(range(2), vals, color=[COLORS[label] for label in COLORS], width=0.64)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:g}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(range(2), ["2K smoke", "10K multi-year"])
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.22)
        fig.suptitle(f"FQA 2007 · WGEN seed {seed} | management totals | PPO seed 0", x=0.02, ha="left", fontweight="bold")
        fig.text(0.02, 0.005, "Same realized daily weather in both runs; wrapper totals, not Summary.OUT-verified applications.", fontsize=8)
        fig.tight_layout(rect=[0, 0.04, 1, 0.92])
        save(fig, f"fqa_seed0_{seed}_heldout_management_bars.png")

    for seed in seeds:
        weather = read_csv(BASE / "10k/weather_daily" / f"episode_{seeds.index(seed)+1:04d}.csv")
        doy = np.array([int(row["DOY"]) for row in weather])
        rain = np.array([float(row["RAIN"]) for row in weather])
        srad = np.array([float(row["SRAD"]) for row in weather])
        tmax = np.array([float(row["TMAX"]) for row in weather])
        tmin = np.array([float(row["TMIN"]) for row in weather])
        fig, axes = plt.subplots(3, 1, figsize=(11.5, 8), sharex=True)
        axes[0].bar(doy, rain, width=0.85, color="#5B9BD5")
        axes[0].set_ylabel("Rain (mm/day)")
        axes[1].plot(doy, tmax, color="#C44E52", lw=1.5, label="TMAX")
        axes[1].plot(doy, tmin, color="#4C72B0", lw=1.5, label="TMIN")
        axes[1].set_ylabel("Temperature (°C)")
        axes[1].legend(frameon=False, ncol=2)
        axes[2].plot(doy, srad, color="#D8A305", lw=1.4)
        axes[2].set_ylabel("SRAD (MJ/m²/day)")
        axes[2].set_xlabel("Day of year")
        for ax in axes:
            ax.grid(axis="y", alpha=0.22)
        fig.suptitle(f"FQA 2007 · WGEN seed {seed} | archived daily weather", x=0.02, ha="left", fontweight="bold")
        fig.text(0.02, 0.005, "Weather trace used by both paired checkpoints; daily crop states and action events were not archived in 048.", fontsize=8)
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])
        save(fig, f"fqa_seed0_{seed}_heldout_daily_weather.png")

    print(f"figures={len(list(OUT.glob('*.png')))} output={OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
