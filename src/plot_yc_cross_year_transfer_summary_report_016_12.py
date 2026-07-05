from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_CSV = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_cross_year_transfer_success_plots_016_11_fixed" / "summary_selected_years.csv"
FULL_SUMMARY_CSV = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_station_level3_true_model_transfer_016_04" / "yc2014_true_model_transfer_summary.csv"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "yc2014_cross_year_transfer_summary_report_016_12_fixed"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-05_016_12_yc_cross_year_transfer_summary_report_fixed.md"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def build_summary_table() -> pd.DataFrame:
    selected = pd.read_csv(SELECTED_CSV)
    full = pd.read_csv(FULL_SUMMARY_CSV)
    full["scenario"] = full["scenario"].fillna("null")

    rows = []
    for _, sel in selected.iterrows():
        year = int(sel["year"])
        transfer_scenario = str(sel["scenario"])
        transfer_seed = float(sel["train_seed"])
        transfer_ckpt = float(sel["checkpoint_step"])
        transfer_selection = str(sel["selection"])

        null_row = full[(full["year"] == year) & (full["scenario"] == "null")].iloc[0]
        rec_row = full[(full["year"] == year) & (full["scenario"] == "recorded_shifted")].iloc[0]
        auto_row = full[(full["year"] == year) & (full["scenario"] == "dssat_auto")].iloc[0]
        dqn_row = full[
            (full["year"] == year)
            & (full["scenario"] == transfer_scenario)
            & (full["train_seed"].fillna(-1) == transfer_seed)
            & (full["checkpoint_step"].fillna(-1) == transfer_ckpt)
            & (full["selection"].fillna("") == transfer_selection)
        ].iloc[0]

        def parse_event_totals(row: pd.Series) -> tuple[float, float]:
            run_dir = PROJECT_ROOT / str(row["run_dir"])
            mgmt_path = run_dir / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT"
            if not mgmt_path.exists():
                return float(row.get("irrigation_total", 0.0) or 0.0), float(row.get("fertilizer_total", 0.0) or 0.0)
            irrig = 0.0
            fert = 0.0
            seen: set[tuple[str, int, float]] = set()
            pattern = re.compile(r"^\s*\d+\s+\w{3}\s+\d+,\s+\d{4}\s+\d+\s+\d+\s+(\d+)\s+\w+\s+(.*)$")
            for line in mgmt_path.read_text(encoding="latin-1", errors="ignore").splitlines():
                if "Irrigation" not in line and "Fertilizer" not in line:
                    continue
                match = pattern.match(line)
                if not match:
                    continue
                dap = int(match.group(1))
                tail = match.group(2)
                if "Irrigation" in tail:
                    amt_match = re.search(r"Irrigation\s+([0-9.]+)", tail)
                    if not amt_match:
                        continue
                    amount = float(amt_match.group(1))
                    key = ("Irrigation", dap, amount)
                    if key in seen:
                        continue
                    seen.add(key)
                    irrig += amount
                else:
                    amt_match = re.search(r"Fertilizer\s+([0-9.]+)", tail)
                    if not amt_match:
                        continue
                    amount = float(amt_match.group(1))
                    key = ("Fertilizer", dap, amount)
                    if key in seen:
                        continue
                    seen.add(key)
                    fert += amount
            if irrig == 0 and fert == 0:
                return float(row.get("irrigation_total", 0.0) or 0.0), float(row.get("fertilizer_total", 0.0) or 0.0)
            return irrig, fert

        null_irrig, null_n = parse_event_totals(null_row)
        rec_irrig, rec_n = parse_event_totals(rec_row)
        auto_irrig, auto_n = parse_event_totals(auto_row)
        dqn_irrig, dqn_n = parse_event_totals(dqn_row)

        rows.append(
            {
                "year": year,
                "dqn_seed": int(transfer_seed),
                "dqn_checkpoint": int(transfer_ckpt),
                "null_yield": float(null_row["final_grain_kg_ha"]),
                "expert_yield": float(rec_row["final_grain_kg_ha"]),
                "auto_yield": float(auto_row["final_grain_kg_ha"]),
                "dqn_yield": float(dqn_row["final_grain_kg_ha"]),
                "null_irrigation": null_irrig,
                "expert_irrigation": rec_irrig,
                "auto_irrigation": auto_irrig,
                "dqn_irrigation": dqn_irrig,
                "null_n": null_n,
                "expert_n": rec_n,
                "auto_n": auto_n,
                "dqn_n": dqn_n,
                "yield_diff_vs_auto": float(dqn_row["final_grain_kg_ha"] - auto_row["final_grain_kg_ha"]),
                "yield_diff_vs_expert": float(dqn_row["final_grain_kg_ha"] - rec_row["final_grain_kg_ha"]),
                "yield_gain_vs_null": float(dqn_row["final_grain_kg_ha"] - null_row["final_grain_kg_ha"]),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def plot_summary(df: pd.DataFrame, out_png: Path) -> None:
    years = df["year"].astype(int).tolist()
    x = np.arange(len(years))
    width = 0.18

    fig = plt.figure(figsize=(14.5, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0], hspace=0.32, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    colors = {
        "null": "#4A4A4A",
        "expert": "#C73E3A",
        "auto": "#B8860B",
        "dqn": "#2E8B57",
    }

    # a) yield grouped bar
    ax1.bar(x - 1.5 * width, df["null_yield"], width=width, color=colors["null"], label="Null")
    ax1.bar(x - 0.5 * width, df["expert_yield"], width=width, color=colors["expert"], label="Recorded expert")
    ax1.bar(x + 0.5 * width, df["auto_yield"], width=width, color=colors["auto"], label="DSSAT auto")
    ax1.bar(x + 1.5 * width, df["dqn_yield"], width=width, color=colors["dqn"], label="DQN transfer")
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_ylabel("Grain yield (kg/ha)")
    ax1.set_title("a  Yield comparison across transferred years", loc="left", fontweight="bold")
    ax1.legend(ncol=2, loc="upper left")

    # b) DQN minus auto/expert
    ax2.axhline(0, color="#888888", lw=1.0, ls="--")
    ax2.bar(x - width/2, df["yield_diff_vs_auto"], width=width, color="#5DA5DA", label="DQN - auto")
    ax2.bar(x + width/2, df["yield_diff_vs_expert"], width=width, color="#F17CB0", label="DQN - expert")
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.set_ylabel("Yield difference (kg/ha)")
    ax2.set_title("b  Yield advantage of transferred DQN", loc="left", fontweight="bold")
    ax2.legend(loc="upper left")

    # c) irrigation comparison
    ax3.plot(x, df["null_irrigation"], color=colors["null"], lw=2, marker="o", label="Null")
    ax3.plot(x, df["expert_irrigation"], color=colors["expert"], lw=2, marker="o", label="Recorded expert")
    ax3.plot(x, df["auto_irrigation"], color=colors["auto"], lw=2, marker="o", label="DSSAT auto")
    ax3.plot(x, df["dqn_irrigation"], color=colors["dqn"], lw=2, marker="o", label="DQN transfer")
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    ax3.set_ylabel("Irrigation (mm)")
    ax3.set_title("c  Irrigation input", loc="left", fontweight="bold")

    # d) nitrogen comparison
    ax4.plot(x, df["null_n"], color=colors["null"], lw=2, marker="o", label="Null")
    ax4.plot(x, df["expert_n"], color=colors["expert"], lw=2, marker="o", label="Recorded expert")
    ax4.plot(x, df["auto_n"], color=colors["auto"], lw=2, marker="o", label="DSSAT auto")
    ax4.plot(x, df["dqn_n"], color=colors["dqn"], lw=2, marker="o", label="DQN transfer")
    ax4.set_xticks(x)
    ax4.set_xticklabels(years)
    ax4.set_ylabel("Nitrogen (kg/ha)")
    ax4.set_title("d  Nitrogen input", loc="left", fontweight="bold")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True, color="#E6E8F0", linewidth=0.8, alpha=0.9)

    fig.suptitle("YC2014-trained DQN cross-year transfer summary", x=0.06, ha="left", fontsize=16, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_doc(df: pd.DataFrame, fig_path: Path) -> None:
    lines = [
        "# 016_12 YC 跨年份迁移总对比汇总图表",
        "",
        "## 目的",
        "",
        "把 016_11 已筛出的 4 个代表年份（2006/2009/2015/2018）进一步整理成可直接汇报的总表和总图。",
        "",
        "## 汇总表",
        "",
        df.to_markdown(index=False),
        "",
        "## 图文件",
        "",
        f"- `{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 解释口径",
        "",
        "- 这不是重新训练，而是基于 016_04 已有迁移结果做的汇总展示；",
        "- DQN transfer 表示用 YC2014 训练出的模型迁移到对应年份后，从这些年份里挑出的代表性最好结果；",
        "- 汇总图重点看三件事：产量位置、相对 auto/expert 的差值、以及水氮投入结构。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = build_summary_table()
    table_path = OUT_DIR / "yc_cross_year_transfer_summary_table.csv"
    fig_path = fig_dir / "yc_cross_year_transfer_summary_report.png"
    df.to_csv(table_path, index=False, encoding="utf-8-sig")
    plot_summary(df, fig_path)
    write_doc(df, fig_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
