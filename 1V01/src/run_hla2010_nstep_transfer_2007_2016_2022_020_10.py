from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

import run_hla2010_to_2016_2022_dqn_transfer_eval_015_19 as old

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_transfer_2007_2016_2022_020_10"
DOC = ROOT / "docs" / "2026-07-10_020_10_hla2010_nstep_transfer_record.md"
YEARS = [2007, 2016, 2022]
MODELS = [(0, 30000), (1, 10000), (2, 20000)]
MODEL_DIRS = {0: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed0_020_08" / "nstep5_seed0_50000steps", 1: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed1_020_06" / "nstep5_seed1_50000steps", 2: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed2_020_07" / "nstep5_seed2_50000steps"}
COLORS = {"null": "#222222", "recorded": "#C9252D", "dssat_auto": "#B8860B", "transfer_2010_seed0": "#1F7A3A", "transfer_2010_seed1": "#3A7CA5", "transfer_2010_seed2": "#8B5A2B"}
LABELS = {"null": "Null", "recorded": "Recorded expert", "dssat_auto": "DSSAT auto", "transfer_2010_seed0": "n-step seed0", "transfer_2010_seed1": "n-step seed1", "transfer_2010_seed2": "n-step seed2"}


def model_path(seed: int, step: int) -> Path:
    return MODEL_DIRS[seed] / "models" / f"nstep5_checkpoint_{step}.zip"


def baseline_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    old07 = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_to_2007_dqn_transfer_eval_015_18"
    old1622 = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_to_2016_2022_dqn_transfer_eval_015_19"
    d07 = pd.read_csv(old07 / "hla2010_to_2007_transfer_eval_daily.csv", keep_default_na=False)
    s07 = pd.read_csv(old07 / "hla2010_to_2007_transfer_eval_summary.csv", keep_default_na=False)
    d16 = pd.read_csv(old1622 / "hla2010_to_2016_2022_transfer_eval_daily.csv", keep_default_na=False)
    s16 = pd.read_csv(old1622 / "hla2010_to_2016_2022_transfer_eval_summary.csv", keep_default_na=False)
    for frame in (d07, s07, d16, s16):
        frame["scenario"] = frame["scenario"].replace({"": "null"}).fillna("null")
    d07 = d07[d07["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    d16 = d16[d16["scenario"].isin(["null", "dssat_auto"])].copy()
    s07 = s07[s07["scenario"].isin(["null", "recorded", "dssat_auto"])].copy()
    s16 = s16[s16["scenario"].isin(["null", "dssat_auto"])].copy()
    return pd.concat([d07, d16], ignore_index=True, sort=False), pd.concat([s07, s16], ignore_index=True, sort=False)


def plot_year(daily: pd.DataFrame, year: int, path: Path) -> None:
    suball = daily[daily["requested_year"].astype(int).eq(year)].copy()
    rain = suball[suball["scenario"].eq("null")][["dap", "rain"]].drop_duplicates().sort_values("dap")
    order = ["null", "recorded", "dssat_auto", "transfer_2010_seed0", "transfer_2010_seed1", "transfer_2010_seed2"]
    fig, axes = plt.subplots(5, 1, figsize=(9.5, 9.5), sharex=True, gridspec_kw={"height_ratios": [0.7, 1, 1, .9, 1.25]})
    fig.suptitle(f"HLA2010 n-step DQN transfer to HLA{year}", x=.06, ha="left", fontsize=12, fontweight="bold")
    axes[0].bar(rain["dap"], rain["rain"], width=.9, color="#BFC5CF", edgecolor="#68717D", linewidth=.25)
    axes[0].set_ylabel("Rain\n(mm)")
    for sc in order:
        s = suball[suball["scenario"].eq(sc)].sort_values("dap")
        if s.empty: continue
        color = COLORS[sc]; ls = "--" if sc in {"recorded", "transfer_2010_seed1"} else "-"
        axes[1].plot(s["dap"], s["water_stress"], color=color, ls=ls, lw=1.5, label=LABELS[sc]); axes[2].plot(s["dap"], s["nitrogen_stress"], color=color, ls=ls, lw=1.5)
        axes[4].plot(s["dap"], s["grain_kg_ha"], color=color, ls=ls, lw=1.6); axes[4].plot(s["dap"], s["biomass_kg_ha"], color=color, ls=":", lw=1.2)
        i = s[s["irrigation_mm"].fillna(0) > 0]; n = s[s["fertilizer_kg_ha"].fillna(0) > 0]
        if not i.empty: axes[3].vlines(i["dap"], 0, i["irrigation_mm"], color=color, ls=ls, lw=1.8)
        if not n.empty: axes[3].scatter(n["dap"], n["fertilizer_kg_ha"], color=color, s=24, edgecolor="white", linewidth=.3)
    axes[1].set_ylabel("Water\nstress"); axes[2].set_ylabel("Nitrogen\nstress"); axes[3].set_ylabel("Mgmt\namount"); axes[4].set_ylabel("kg/ha"); axes[4].set_xlabel("DAP"); axes[1].legend(loc="upper left", ncol=2, fontsize=7)
    for ax in axes:
        ax.grid(True, axis="x", color="#E2E7EF", lw=.55); ax.grid(True, axis="y", color="#EDF1F5", lw=.45, ls="--"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, .98]); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight"); fig.savefig(path.with_suffix(".svg"), bbox_inches="tight"); fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)


def main() -> None:
    old.configure(); old.LABELS.update(LABELS); old.COLORS.update(COLORS)
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    base_d, base_s = baseline_daily()
    all_d, all_s = [base_d], [base_s]
    for year in YEARS:
        env_args = old.prepare_test_case(year, OUT / "test_cases" / str(year))
        null_yield = old.get_null_baseline_yield(base_s, year)
        rain = old.rain_lookup(base_d, year)
        for seed, step in MODELS:
            model = DQN.load(str(model_path(seed, step)), device="cpu")
            daily, summary = old.evaluate_transfer(model, env_args, year, null_yield, seed, step, OUT, rain)
            all_d.append(daily); all_s.append(pd.DataFrame([summary]))
    daily = pd.concat(all_d, ignore_index=True, sort=False); summary = pd.concat(all_s, ignore_index=True, sort=False)
    daily.to_csv(OUT / "020_10_hla2010_nstep_transfer_daily.csv", index=False, encoding="utf-8-sig"); summary.to_csv(OUT / "020_10_hla2010_nstep_transfer_summary.csv", index=False, encoding="utf-8-sig")
    for year in YEARS: plot_year(daily, year, OUT / "figures" / f"020_10_hla2010_nstep_to_{year}_process")
    view = summary[summary["scenario"].isin(["null", "recorded", "dssat_auto", "transfer_2010_seed0", "transfer_2010_seed1", "transfer_2010_seed2"])]
    old.markdown_table = getattr(old, "markdown_table")
    lines = ["# 020_10 HLA2010 n-step DQN → HLA2007/2016/2022 跨年份迁移记录", "", "本轮复用015_18/015_19已有null/auto（及2007 recorded）基线，只加载HLA2010 n-step三个成功模型，不重新训练。", "", "## 汇总", "", old.markdown_table(view), "", "## 输出", "", f"- Daily: `{(OUT / '020_10_hla2010_nstep_transfer_daily.csv').relative_to(ROOT)}`", f"- Summary: `{(OUT / '020_10_hla2010_nstep_transfer_summary.csv').relative_to(ROOT)}`", f"- Figures: `{(OUT / 'figures').relative_to(ROOT)}`", "", "结果仅用于HLA站点内跨年份迁移，不等同于跨站点泛化。"]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__": main()
