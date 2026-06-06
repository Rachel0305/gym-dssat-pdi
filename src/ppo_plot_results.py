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
    if {"raw_real_action_amir", "safe_real_action_amir", "raw_real_action_anfer", "safe_real_action_anfer"}.issubset(daily.columns):
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
        axes[0].plot(x, pd.to_numeric(daily["raw_real_action_amir"], errors="coerce"), label="raw_real_action_amir", alpha=0.8)
        axes[0].plot(x, pd.to_numeric(daily["safe_real_action_amir"], errors="coerce"), label="safe_real_action_amir", alpha=0.8)
        axes[0].set_ylabel("Irrigation")
        axes[0].grid(alpha=0.25)
        axes[0].legend(frameon=False)
        axes[1].plot(x, pd.to_numeric(daily["raw_real_action_anfer"], errors="coerce"), label="raw_real_action_anfer", alpha=0.8)
        axes[1].plot(x, pd.to_numeric(daily["safe_real_action_anfer"], errors="coerce"), label="safe_real_action_anfer", alpha=0.8)
        axes[1].set_ylabel("N fertilizer")
        axes[1].set_xlabel("DAP")
        axes[1].grid(alpha=0.25)
        axes[1].legend(frameon=False)
        fig.suptitle("Raw vs safe actions")
        fig.tight_layout()
        path = figure_dir / "daily_actions_raw_vs_safe.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    safety_cols = [col for col in ["action_clipped_amir", "action_clipped_anfer"] if col in daily.columns]
    if safety_cols:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for col in safety_cols:
            ax.plot(x, pd.to_numeric(daily[col], errors="coerce"), label=col)
        if "safety_rule_triggered" in daily.columns:
            triggered = daily["safety_rule_triggered"].fillna("").astype(str).str.len().gt(0).astype(int)
            ax.plot(x, triggered, label="any_safety_rule_triggered", alpha=0.7)
        ax.set_title("Action safety triggers")
        ax.set_xlabel("DAP")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figure_dir / "action_safety_triggers.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
