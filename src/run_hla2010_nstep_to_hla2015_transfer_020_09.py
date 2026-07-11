from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stable_baselines3 import DQN

import run_hla2010_to_2015_dqn_transfer_eval_015_17 as old


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_to_hla2015_transfer_020_09"
FIG = OUT / "figures"
DOC = ROOT / "docs" / "2026-07-10_020_09_hla2010_nstep_to_hla2015_transfer_record.md"

TRANSFER = [(0, 30000), (1, 10000), (2, 20000)]
COLORS = {"null": "#222222", "expert_2007_shifted": "#C9252D", "dssat_auto": "#B8860B", "local_dqn_seed0": "#255C99", "local_dqn_seed1": "#7B3F98", "transfer_2010_seed0": "#1F7A3A", "transfer_2010_seed1": "#3A7CA5", "transfer_2010_seed2": "#8B5A2B"}
LABELS = {"null": "Null", "expert_2007_shifted": "Official expert", "dssat_auto": "DSSAT auto", "local_dqn_seed0": "Local DQN seed0", "local_dqn_seed1": "Local DQN seed1", "transfer_2010_seed0": "n-step HLA2010 seed0", "transfer_2010_seed1": "n-step HLA2010 seed1", "transfer_2010_seed2": "n-step HLA2010 seed2"}


def model_path(seed: int, step: int) -> Path:
    return ROOT / "DSSAT_auto_validation" / "HLA_2004" / f"hla2010_nstep_dqn_seed{seed}_020_0{6 if seed == 1 else 7 if seed == 2 else 8}" / f"nstep5_seed{seed}_50000steps" / "models" / f"nstep5_checkpoint_{step}.zip"


def plot_process(daily: pd.DataFrame, path: Path) -> None:
    rain = daily[daily["scenario"].eq("null")][["dap", "rain"]].drop_duplicates().sort_values("dap")
    order = list(LABELS)
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [0.7, 1, 1, 0.9, 1.25]})
    fig.suptitle("HLA2010 n-step DQN transfer to HLA2015", x=0.06, ha="left", fontsize=12, fontweight="bold")
    axes[0].bar(rain["dap"], rain["rain"], width=0.9, color="#BFC5CF", edgecolor="#68717D", linewidth=.25)
    axes[0].set_ylabel("Rain\n(mm)")
    for scenario in order:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("dap")
        if sub.empty: continue
        color = COLORS[scenario]; ls = "--" if "expert" in scenario or scenario.endswith("seed1") else "-"
        axes[1].plot(sub["dap"], sub["water_stress"], color=color, ls=ls, lw=1.5, label=LABELS[scenario])
        axes[2].plot(sub["dap"], sub["nitrogen_stress"], color=color, ls=ls, lw=1.5)
        axes[4].plot(sub["dap"], sub["grain_kg_ha"], color=color, ls=ls, lw=1.6)
        axes[4].plot(sub["dap"], sub["biomass_kg_ha"], color=color, ls=":", lw=1.2)
        irrig = sub[sub["irrigation_mm"].fillna(0) > 0]; fert = sub[sub["fertilizer_kg_ha"].fillna(0) > 0]
        if not irrig.empty: axes[3].vlines(irrig["dap"], 0, irrig["irrigation_mm"], color=color, ls=ls, lw=1.8)
        if not fert.empty: axes[3].scatter(fert["dap"], fert["fertilizer_kg_ha"], color=color, s=24, edgecolor="white", linewidth=.3)
    axes[1].set_ylabel("Water\nstress"); axes[2].set_ylabel("Nitrogen\nstress"); axes[3].set_ylabel("Mgmt\namount"); axes[4].set_ylabel("kg/ha"); axes[4].set_xlabel("DAP")
    axes[1].legend(loc="upper left", ncol=2, fontsize=7)
    for ax in axes:
        ax.grid(True, axis="x", color="#E2E7EF", lw=.55); ax.grid(True, axis="y", color="#EDF1F5", lw=.45, ls="--"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, .98]); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight"); fig.savefig(path.with_suffix(".svg"), bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def main() -> None:
    old.configure()
    old.LABELS.update(LABELS); old.COLORS.update(COLORS)
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    env_args = old.prepare_test_case(OUT / "test_case_hla2015")
    transfer_daily, transfer_summary = [], []
    for seed, step in TRANSFER:
        path = model_path(seed, step)
        if not path.exists(): raise FileNotFoundError(path)
        model = DQN.load(str(path), device="cpu")
        daily, summary = old.evaluate_transfer(model, env_args, seed, step, OUT)
        summary["days"] = float(len(daily))
        summary["final_dap"] = float(daily["dap"].dropna().max()) if not daily.empty else np.nan
        transfer_daily.append(daily); transfer_summary.append(summary)
    baseline_daily, baseline_summary = old.load_baseline_and_local()
    null_daily = baseline_daily[baseline_daily["scenario"].eq("null")].sort_values("dap")
    if not null_daily.empty:
        null_summary = pd.DataFrame([{
            "requested_year": 2015,
            "scenario": "null",
            "label": "Null",
            "final_gwad": float(null_daily["grain_kg_ha"].dropna().iloc[-1]),
            "final_cwad": float(null_daily["biomass_kg_ha"].dropna().iloc[-1]),
            "irrigation_total": float(null_daily["irrigation_mm"].fillna(0).sum()),
            "fertilizer_total": float(null_daily["fertilizer_kg_ha"].fillna(0).sum()),
            "max_water_stress": float(null_daily["water_stress"].max()),
            "max_nitrogen_stress": float(null_daily["nitrogen_stress"].max()),
            "total_reward": np.nan,
        }])
        baseline_summary = pd.concat([null_summary, baseline_summary], ignore_index=True, sort=False)
    all_daily = pd.concat([baseline_daily, *transfer_daily], ignore_index=True, sort=False)
    all_summary = pd.concat([baseline_summary, pd.DataFrame(transfer_summary)], ignore_index=True, sort=False)
    all_daily.to_csv(OUT / "020_09_hla2010_nstep_to_hla2015_daily.csv", index=False, encoding="utf-8-sig")
    all_summary.to_csv(OUT / "020_09_hla2010_nstep_to_hla2015_summary.csv", index=False, encoding="utf-8-sig")
    plot_process(all_daily, FIG / "020_09_hla2010_nstep_to_hla2015_process")
    view = all_summary[all_summary["scenario"].isin(["null", "expert_2007_shifted", "dssat_auto", "transfer_2010_seed0", "transfer_2010_seed1", "transfer_2010_seed2"])]
    lines = ["# 020_09 HLA2010 n-step DQN → HLA2015 跨年份迁移记录", "", "本轮不重新训练，只加载 HLA2010 n-step 三个成功 checkpoint，在HLA2015环境中直接评估。", "", "## 结果", "", old.markdown_table(view), "", "## 输出", "", f"- Daily: `{(OUT / '020_09_hla2010_nstep_to_hla2015_daily.csv').relative_to(ROOT)}`", f"- Summary: `{(OUT / '020_09_hla2010_nstep_to_hla2015_summary.csv').relative_to(ROOT)}`", f"- Figure: `{(FIG / '020_09_hla2010_nstep_to_hla2015_process.png').relative_to(ROOT)}`", "", "迁移结果只用于判断同站点跨年份泛化，不等同于跨站点泛化。"]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(all_summary.to_string(index=False))


if __name__ == "__main__": main()
