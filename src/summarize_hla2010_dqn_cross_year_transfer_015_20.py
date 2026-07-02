from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004"
OUT_DIR = BASE / "hla2010_dqn_cross_year_transfer_summary_015_20"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-02_015_20_hla2010_dqn_cross_year_transfer_summary.md"

SOURCES = [
    BASE / "hla2010_to_2007_dqn_transfer_eval_015_18" / "hla2010_to_2007_transfer_eval_summary.csv",
    BASE / "hla2010_to_2015_dqn_transfer_eval_015_17" / "hla2010_to_2015_transfer_eval_summary.csv",
    BASE / "hla2010_to_2016_2022_dqn_transfer_eval_015_19" / "hla2010_to_2016_2022_transfer_eval_summary.csv",
]
SUPPLEMENTAL_2015_SUMMARY = (
    BASE
    / "hla_2010_2015_final_dqn_four_scenario_015_16"
    / "hla_2010_2015_four_scenario_final_dqn_all_summary.csv"
)

SCENARIO_LABELS = {
    "null": "Null",
    "recorded": "Recorded",
    "expert_2007_shifted": "Recorded",
    "dssat_auto": "DSSAT auto",
    "transfer_2010_seed0": "DQN transfer s0",
    "transfer_2010_seed1": "DQN transfer s1",
}

SCENARIO_ORDER = ["null", "recorded", "dssat_auto", "transfer_2010_seed0", "transfer_2010_seed1"]

COLORS = {
    "null": "#222222",
    "recorded": "#C9252D",
    "dssat_auto": "#B8860B",
    "transfer_2010_seed0": "#255C99",
    "transfer_2010_seed1": "#7B3F98",
}

HATCHES = {
    "null": "",
    "recorded": "///",
    "dssat_auto": "",
    "transfer_2010_seed0": "",
    "transfer_2010_seed1": "\\\\\\",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def load_all() -> pd.DataFrame:
    frames = []
    for source in SOURCES:
        if not source.exists():
            raise FileNotFoundError(source)
        df = pd.read_csv(source)
        df["source_file"] = str(source.relative_to(PROJECT_ROOT))
        frames.append(df)
    data = pd.concat(frames, ignore_index=True, sort=False)
    if SUPPLEMENTAL_2015_SUMMARY.exists():
        supp = pd.read_csv(SUPPLEMENTAL_2015_SUMMARY)
        supp["source_file"] = str(SUPPLEMENTAL_2015_SUMMARY.relative_to(PROJECT_ROOT))
        supp["scenario"] = supp["scenario"].fillna("null").replace({"": "null"})
        supp = supp[supp["requested_year"].eq(2015) & supp["scenario"].eq("null")].copy()
        if not supp.empty:
            data = pd.concat([data, supp.head(1)], ignore_index=True, sort=False)
    data["scenario"] = data["scenario"].fillna("null").replace({"": "null"})
    data["scenario"] = data["scenario"].replace({"expert_2007_shifted": "recorded"})
    data = data[data["scenario"].isin(SCENARIO_ORDER)].copy()
    data["scenario_label"] = data["scenario"].map(SCENARIO_LABELS)
    for col in [
        "requested_year",
        "final_gwad",
        "final_cwad",
        "rain_total",
        "irrigation_total",
        "fertilizer_total",
        "max_water_stress",
        "max_nitrogen_stress",
        "total_reward",
        "train_seed",
        "train_checkpoint_step",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = (
        data.sort_values(["requested_year", "scenario", "train_seed"])
        .drop_duplicates(subset=["requested_year", "scenario", "train_seed"], keep="last")
        .reset_index(drop=True)
    )
    return data


def add_derived(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    auto = (
        out[out["scenario"].eq("dssat_auto")]
        .set_index("requested_year")[["final_gwad", "irrigation_total", "fertilizer_total"]]
        .rename(
            columns={
                "final_gwad": "auto_gwad",
                "irrigation_total": "auto_irrigation",
                "fertilizer_total": "auto_fertilizer",
            }
        )
    )
    null = (
        out[out["scenario"].eq("null")]
        .set_index("requested_year")[["final_gwad"]]
        .rename(columns={"final_gwad": "null_gwad"})
    )
    out = out.join(auto, on="requested_year").join(null, on="requested_year")
    out["yield_diff_vs_auto"] = out["final_gwad"] - out["auto_gwad"]
    out["yield_gain_vs_null"] = out["final_gwad"] - out["null_gwad"]
    out["irrigation_saving_vs_auto"] = out["auto_irrigation"] - out["irrigation_total"]
    out["fertilizer_saving_vs_auto"] = out["auto_fertilizer"] - out["fertilizer_total"]
    out["yield_ratio_vs_auto_pct"] = out["final_gwad"] / out["auto_gwad"] * 100.0
    return out


def best_transfer(data: pd.DataFrame) -> pd.DataFrame:
    dqn = data[data["scenario"].str.startswith("transfer_2010_seed")].copy()
    dqn = dqn.sort_values(
        ["requested_year", "yield_diff_vs_auto", "irrigation_saving_vs_auto", "total_reward"],
        ascending=[True, False, False, False],
    )
    return dqn.groupby("requested_year", as_index=False).head(1).reset_index(drop=True)


def plot_summary(data: pd.DataFrame, best: pd.DataFrame) -> Path:
    years = sorted(int(y) for y in data["requested_year"].dropna().unique())
    x = np.arange(len(years))
    width = 0.15
    offsets = {
        "null": -2 * width,
        "recorded": -width,
        "dssat_auto": 0.0,
        "transfer_2010_seed0": width,
        "transfer_2010_seed1": 2 * width,
    }

    fig = plt.figure(figsize=(8.6, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.24)
    ax_yield = fig.add_subplot(gs[0, :])
    ax_irrig = fig.add_subplot(gs[1, 0])
    ax_delta = fig.add_subplot(gs[1, 1])

    for scenario in SCENARIO_ORDER:
        sub = data[data["scenario"].eq(scenario)].copy()
        if sub.empty:
            continue
        vals = []
        irrig = []
        xpos = []
        for i, year in enumerate(years):
            row = sub[sub["requested_year"].astype(int).eq(year)]
            if row.empty:
                vals.append(np.nan)
                irrig.append(np.nan)
            else:
                vals.append(float(row.iloc[0]["final_gwad"]))
                irrig.append(float(row.iloc[0]["irrigation_total"]))
            xpos.append(x[i] + offsets[scenario])
        ax_yield.bar(
            xpos,
            vals,
            width=width * 0.92,
            color=COLORS[scenario],
            edgecolor="#333333",
            linewidth=0.45,
            hatch=HATCHES[scenario],
            label=SCENARIO_LABELS[scenario],
        )
        ax_irrig.bar(
            xpos,
            irrig,
            width=width * 0.92,
            color=COLORS[scenario],
            edgecolor="#333333",
            linewidth=0.45,
            hatch=HATCHES[scenario],
        )

    ax_yield.set_title("a  HLA2010-trained DQN reaches the auto yield plateau across four candidate years", loc="left", fontweight="bold")
    ax_yield.set_ylabel("Grain yield GWAD (kg ha$^{-1}$)")
    ax_yield.set_xticks(x)
    ax_yield.set_xticklabels([str(y) for y in years])
    ax_yield.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=6.5)

    ax_irrig.set_title("b  Transfer policies use less irrigation than DSSAT auto", loc="left", fontweight="bold")
    ax_irrig.set_ylabel("Irrigation (mm)")
    ax_irrig.set_xticks(x)
    ax_irrig.set_xticklabels([str(y) for y in years])

    dqn = data[data["scenario"].str.startswith("transfer_2010_seed")].copy()
    for scenario in ["transfer_2010_seed0", "transfer_2010_seed1"]:
        sub = dqn[dqn["scenario"].eq(scenario)].sort_values("requested_year")
        ax_delta.scatter(
            sub["irrigation_saving_vs_auto"],
            sub["yield_diff_vs_auto"],
            s=42,
            color=COLORS[scenario],
            edgecolor="#222222",
            linewidth=0.45,
            label=SCENARIO_LABELS[scenario],
            marker="o" if scenario.endswith("seed0") else "s",
        )
        for _, r in sub.iterrows():
            ax_delta.text(
                float(r["irrigation_saving_vs_auto"]) + 1.5,
                float(r["yield_diff_vs_auto"]) + 1.5,
                str(int(r["requested_year"])),
                fontsize=6.2,
                color=COLORS[scenario],
            )
    ax_delta.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
    ax_delta.axvline(0, color="#888888", linewidth=0.7, linestyle="--")
    ax_delta.set_title("c  Yield parity is achieved with irrigation savings", loc="left", fontweight="bold")
    ax_delta.set_xlabel("Irrigation saving vs DSSAT auto (mm)")
    ax_delta.set_ylabel("Yield difference vs DSSAT auto (kg ha$^{-1}$)")
    ax_delta.legend(loc="lower right", fontsize=6.5)

    for ax in [ax_yield, ax_irrig, ax_delta]:
        ax.grid(True, axis="y", color="#E8ECF2", linewidth=0.55, linestyle="--")
        ax.grid(True, axis="x", color="#F0F3F7", linewidth=0.4)

    note = (
        "Training year: HLA2010. Validation years: all HLA candidate years passing the management-response screen. "
        "No additional training was performed for 2007/2015/2016/2022."
    )
    fig.text(0.01, 0.005, note, fontsize=6.2, color="#444444")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "hla2010_dqn_cross_year_transfer_summary"
    fig.savefig(f"{out}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def to_md_table(df: pd.DataFrame, cols: list[str]) -> str:
    use = df[cols].copy()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use.iterrows():
        vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                vals.append("" if pd.isna(val) else f"{float(val):.2f}")
            else:
                vals.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_record(data: pd.DataFrame, best: pd.DataFrame, fig_path: Path) -> None:
    cols = [
        "requested_year",
        "scenario_label",
        "final_gwad",
        "irrigation_total",
        "fertilizer_total",
        "yield_diff_vs_auto",
        "irrigation_saving_vs_auto",
        "yield_ratio_vs_auto_pct",
    ]
    best_cols = [
        "requested_year",
        "scenario_label",
        "train_seed",
        "train_checkpoint_step",
        "final_gwad",
        "irrigation_total",
        "fertilizer_total",
        "yield_diff_vs_auto",
        "irrigation_saving_vs_auto",
        "total_reward",
    ]
    lines = [
        "# 015_20 HLA2010 DQN 跨年迁移验证总结",
        "",
        "## 图的核心结论",
        "",
        "HLA2010-trained DQN 在 HLA 所有通过管理响应筛选的 4 个独立候选年份上均达到或贴近 DSSAT auto 产量平台，同时减少灌溉且不施氮，说明当前策略具有同站点跨年份泛化迹象。",
        "",
        "## 证据链",
        "",
        "- 训练年：HLA2010。",
        "- 迁移验证年：HLA2007、HLA2015、HLA2016、HLA2022。",
        "- 这些年份是 HLA 筛选表中通过管理响应标准的全部独立候选年份。",
        "- 迁移验证不重新训练，只加载 HLA2010 checkpoint。",
        "",
        "## 输出文件",
        "",
        f"- 汇总总表：`{(OUT_DIR / 'hla2010_dqn_cross_year_transfer_all_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- best-transfer 表：`{(OUT_DIR / 'hla2010_dqn_cross_year_transfer_best_by_year.csv').relative_to(PROJECT_ROOT)}`",
        f"- 总图：`{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 全部情景汇总",
        "",
        to_md_table(data.sort_values(["requested_year", "scenario"]), cols),
        "",
        "## 每年最佳迁移策略",
        "",
        to_md_table(best.sort_values("requested_year"), best_cols),
        "",
        "## 谨慎表述",
        "",
        "- 该结果支持 HLA 同站点跨年份迁移，不等于已经证明跨站点泛化。",
        "- 2016/2022 没有 recorded expert，因此与 recorded 的比较只限于 2007/2015。",
        "- DQN 的主要优势不是大幅提高产量，而是在达到 auto 产量平台附近时减少灌溉和施氮。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = add_derived(load_all())
    best = best_transfer(data)
    data.to_csv(OUT_DIR / "hla2010_dqn_cross_year_transfer_all_summary.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "hla2010_dqn_cross_year_transfer_best_by_year.csv", index=False, encoding="utf-8-sig")
    fig_path = plot_summary(data, best)
    write_record(data, best, fig_path)
    print(data.sort_values(["requested_year", "scenario"]).to_string(index=False))
    print("\nBEST TRANSFER")
    print(best.sort_values("requested_year").to_string(index=False))
    print(f"\nFigure: {fig_path}")
    print(f"Record: {DOC_PATH}")


if __name__ == "__main__":
    main()
