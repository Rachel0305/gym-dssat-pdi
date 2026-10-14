from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "sy_local_dqn_train_cross_year_transfer_017_08"
INPUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "sy2014_dqn_resource_space_017_09"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-06_017_09_sy2014_dqn_resource_space_record.md"

YEAR = 2014
WATER_COST = 1.0
N_COST = 5.0

SCENARIO_ORDER = ["null", "recorded", "dssat_auto", "dqn"]
SCENARIO_LABELS = {
    "null": "Null",
    "recorded": "Recorded expert",
    "dssat_auto": "DSSAT auto",
    "dqn": "DQN ckpt15000",
}
SCENARIO_COLORS = {
    "null": "#222222",
    "recorded": "#D62728",
    "dssat_auto": "#B8860B",
    "dqn": "#238B45",
}
SCENARIO_STYLES = {
    "null": "-",
    "recorded": "--",
    "dssat_auto": "-",
    "dqn": "-",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Keep literal scenario name "null"; pandas otherwise treats it as missing.
    base_daily = pd.read_csv(SOURCE_DIR / "017_08_sy_baseline_screen_daily.csv", keep_default_na=False)
    dqn_daily = pd.read_csv(SOURCE_DIR / "017_08_sy_dqn_transfer_daily.csv", keep_default_na=False)
    base_events = pd.read_csv(SOURCE_DIR / "017_08_sy_baseline_screen_events.csv", keep_default_na=False)
    dqn_events = pd.read_csv(SOURCE_DIR / "017_08_sy_dqn_transfer_events.csv", keep_default_na=False)
    summary = pd.read_csv(SOURCE_DIR / "017_08_sy_combined_summary.csv", keep_default_na=False)

    base_daily = base_daily[(base_daily["year"].eq(YEAR)) & (base_daily["scenario"].isin(["null", "recorded", "dssat_auto"]))].copy()
    dqn_daily = dqn_daily[(dqn_daily["year"].eq(YEAR)) & (dqn_daily["scenario"].eq("transfer_SY2014_ckpt15000"))].copy()
    dqn_daily["scenario"] = "dqn"
    daily = pd.concat([base_daily, dqn_daily], ignore_index=True)

    base_events = base_events[(base_events["year"].eq(YEAR)) & (base_events["scenario"].isin(["null", "recorded", "dssat_auto"]))].copy()
    dqn_events = dqn_events[(dqn_events["year"].eq(YEAR)) & (dqn_events["scenario"].eq("transfer_SY2014_ckpt15000"))].copy()
    dqn_events["scenario"] = "dqn"
    events = pd.concat([base_events, dqn_events], ignore_index=True)

    summary = summary[(summary["year"].eq(YEAR)) & (summary["scenario"].isin(["null", "recorded", "dssat_auto", "transfer_SY2014_ckpt15000"]))].copy()
    summary.loc[summary["scenario"].eq("transfer_SY2014_ckpt15000"), "scenario"] = "dqn"
    return daily, events, summary


def parse_weather_rain(year: int) -> pd.DataFrame:
    wth = INPUT_ROOT / f"CNSY{year % 100:02d}01.WTH"
    rows: list[dict[str, float]] = []
    if not wth.exists():
        return pd.DataFrame(columns=["doy", "rain"])
    header: list[str] = []
    in_data = False
    for line in wth.read_text(encoding="latin-1", errors="ignore").splitlines():
        if line.startswith("@"):
            header = line.replace("@", "", 1).split()
            in_data = "DATE" in header
            continue
        if in_data and line.strip() and not line.startswith("*") and not line.startswith("!"):
            parts = line.split()
            if len(parts) < len(header):
                continue
            rec = dict(zip(header, parts[: len(header)]))
            try:
                rows.append({"doy": int(str(rec["DATE"])[-3:]), "rain": float(rec.get("RAIN", 0.0))})
            except Exception:
                continue
    return pd.DataFrame(rows)


def infer_planting_doy(events: pd.DataFrame) -> int:
    """Infer DOY at DAP 0 from MgmtEvent text; fallback to SY2014 recorded PDATE."""
    zero = events[events["dap"].eq(0)]
    for raw in zero.get("operation", pd.Series(dtype=str)).astype(str):
        tokens = raw.split()
        # Example has "... 2014  108  18  0 ..."; the token after year is DOY.
        for i, tok in enumerate(tokens):
            if tok == str(YEAR) and i + 1 < len(tokens):
                try:
                    return int(tokens[i + 1])
                except ValueError:
                    pass
    return 108


def make_event_matrix(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["scenario", "dap", "irrigation_event_mm", "fertilizer_event_kg_ha"])
    out = events.copy()
    out["irrigation_event_mm"] = np.where(out["unit"].eq("mm"), out["amount"].astype(float), 0.0)
    out["fertilizer_event_kg_ha"] = np.where(out["unit"].astype(str).str.contains("kg", na=False), out["amount"].astype(float), 0.0)
    return (
        out.groupby(["scenario", "dap"], as_index=False)[["irrigation_event_mm", "fertilizer_event_kg_ha"]]
        .sum()
        .sort_values(["scenario", "dap"])
    )


def add_reward_proxy(daily: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    planting_doy = infer_planting_doy(events)
    rain = parse_weather_rain(YEAR)
    if not rain.empty:
        rain["dap"] = rain["doy"] - planting_doy
        rain = rain[(rain["dap"] >= 0)].copy()
        daily = daily.drop(columns=["rain"], errors="ignore")
        daily = daily.merge(rain[["dap", "rain"]], on="dap", how="left")
        daily["rain"] = daily["rain"].fillna(0.0)
    event_matrix = make_event_matrix(events)
    daily = daily.drop(columns=["irrigation_event_mm", "fertilizer_event_kg_ha"], errors="ignore")
    daily = daily.merge(event_matrix, on=["scenario", "dap"], how="left")
    daily["irrigation_event_mm"] = daily["irrigation_event_mm"].fillna(0.0)
    daily["fertilizer_event_kg_ha"] = daily["fertilizer_event_kg_ha"].fillna(0.0)
    daily["daily_proxy_reward"] = -WATER_COST * daily["irrigation_event_mm"] - N_COST * daily["fertilizer_event_kg_ha"]

    null_gwad = float(summary.loc[summary["scenario"].eq("null"), "final_gwad"].iloc[0])
    terminal_map = {}
    for _, row in summary.iterrows():
        terminal_map[row["scenario"]] = max(0.0, float(row["final_gwad"]) - null_gwad)

    daily["terminal_bonus"] = 0.0
    for scenario, bonus in terminal_map.items():
        mask = daily["scenario"].eq(scenario)
        if mask.any():
            last_idx = daily.loc[mask, "dap"].idxmax()
            daily.loc[last_idx, "terminal_bonus"] = bonus
    daily["proxy_reward"] = daily["daily_proxy_reward"] + daily["terminal_bonus"]
    daily["cumulative_proxy_reward"] = daily.groupby("scenario")["proxy_reward"].cumsum()
    return daily


def make_summary(daily: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    event_matrix = make_event_matrix(events)
    totals = event_matrix.groupby("scenario", as_index=False)[["irrigation_event_mm", "fertilizer_event_kg_ha"]].sum()
    totals = totals.rename(columns={"irrigation_event_mm": "irrigation_total_mm", "fertilizer_event_kg_ha": "fertilizer_total_kg_ha"})
    stress = daily.groupby("scenario", as_index=False).agg(
        max_water_stress=("swfac", "max"),
        mean_water_stress=("swfac", "mean"),
        max_nitrogen_stress=("nstres", "max"),
        mean_nitrogen_stress=("nstres", "mean"),
        final_cumulative_proxy_reward=("cumulative_proxy_reward", "last"),
    )
    out = summary[["scenario", "final_gwad", "final_cwad", "rain_total"]].merge(totals, on="scenario", how="left").merge(stress, on="scenario", how="left")
    out[["irrigation_total_mm", "fertilizer_total_kg_ha"]] = out[["irrigation_total_mm", "fertilizer_total_kg_ha"]].fillna(0.0)
    null = out.loc[out["scenario"].eq("null")].iloc[0]
    rec = out.loc[out["scenario"].eq("recorded")].iloc[0]
    auto = out.loc[out["scenario"].eq("dssat_auto")].iloc[0]
    out["yield_diff_vs_null"] = out["final_gwad"] - float(null["final_gwad"])
    out["yield_diff_vs_recorded"] = out["final_gwad"] - float(rec["final_gwad"])
    out["yield_diff_vs_dssat_auto"] = out["final_gwad"] - float(auto["final_gwad"])
    out["irrigation_saving_vs_recorded"] = float(rec["irrigation_total_mm"]) - out["irrigation_total_mm"]
    out["fertilizer_saving_vs_recorded"] = float(rec["fertilizer_total_kg_ha"]) - out["fertilizer_total_kg_ha"]
    out["resource_success_vs_recorded"] = (
        (out["yield_diff_vs_recorded"] > 0)
        & (out["irrigation_saving_vs_recorded"] >= 0)
        & (out["fertilizer_saving_vs_recorded"] >= 0)
    )
    out["scenario_label"] = out["scenario"].map(SCENARIO_LABELS)
    order = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    return out.sort_values("scenario", key=lambda x: x.map(order)).reset_index(drop=True)


def plot_process(daily: pd.DataFrame, events: pd.DataFrame, summary: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(10.5, 12.5),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.05, 1.05, 1.15, 1.15, 1.0]},
    )
    fig.patch.set_facecolor("white")
    max_dap = int(max(1, daily["dap"].max()))

    # Rainfall: rebuild from WTH because the original daily export may not carry DOY.
    planting_doy = infer_planting_doy(events)
    rain = parse_weather_rain(YEAR)
    if not rain.empty:
        rain["dap"] = rain["doy"] - planting_doy
        rain = rain[(rain["dap"] >= 0) & (rain["dap"] <= max_dap)].copy()
    else:
        rain = pd.DataFrame({"dap": [], "rain": []})
    axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#BFC7D1", edgecolor="#8E99A8", linewidth=0.25)
    axes[0].set_ylabel("Rain\n(mm)")
    axes[0].set_title("SY2014 four-scenario process plot: DQN uses full budget to gain yield", loc="left", fontsize=12, fontweight="bold", pad=14)
    axes[0].text(0, 1.03, "High-contrast diagnostic figure; reward panel uses a unified DQN proxy, not DSSAT native reward.", transform=axes[0].transAxes, fontsize=8, color="#555555")

    # Stress curves
    for sc in SCENARIO_ORDER:
        sub = daily[daily["scenario"].eq(sc)].sort_values("dap")
        if sub.empty:
            continue
        axes[1].plot(sub["dap"], sub["swfac"], label=SCENARIO_LABELS[sc], color=SCENARIO_COLORS[sc], linestyle=SCENARIO_STYLES[sc], linewidth=1.5)
        axes[2].plot(sub["dap"], sub["nstres"], label=SCENARIO_LABELS[sc], color=SCENARIO_COLORS[sc], linestyle=SCENARIO_STYLES[sc], linewidth=1.5)
        axes[4].plot(sub["dap"], sub["grnwt"], color=SCENARIO_COLORS[sc], linestyle=SCENARIO_STYLES[sc], linewidth=1.6)
        axes[4].plot(sub["dap"], sub["topwt"], color=SCENARIO_COLORS[sc], linestyle=":", linewidth=1.3, alpha=0.95)
        axes[5].plot(sub["dap"], sub["cumulative_proxy_reward"], color=SCENARIO_COLORS[sc], linestyle=SCENARIO_STYLES[sc], linewidth=1.6)

    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[1].legend(ncol=2, loc="upper left")
    axes[1].set_ylim(bottom=-0.03)
    axes[2].set_ylim(bottom=-0.03)

    # Management events
    axes[3].set_ylabel("Mgmt\namount")
    axes[3].set_title("Management events: irrigation = vertical lines; fertilization = triangle markers", loc="left", fontsize=9)
    offsets = {"null": -1.2, "recorded": -0.4, "dssat_auto": 0.4, "dqn": 1.2}
    for sc in SCENARIO_ORDER:
        sub = events[events["scenario"].eq(sc)]
        color = SCENARIO_COLORS[sc]
        for _, row in sub.iterrows():
            dap = float(row["dap"]) + offsets[sc]
            amount = float(row["amount"])
            unit = str(row["unit"])
            if unit == "mm":
                axes[3].vlines(dap, 0, amount, colors=color, linestyles=SCENARIO_STYLES[sc], linewidth=2.0, alpha=0.9)
            else:
                axes[3].scatter(dap, amount, marker="^", s=32, color=color, edgecolor="white", linewidth=0.35, zorder=4)
    axes[3].set_ylim(bottom=-5)

    axes[4].set_ylabel("Crop\nkg/ha")
    axes[4].set_title("Crop outcome: solid = grain weight (GWAD); dotted = aboveground biomass (CWAD)", loc="left", fontsize=9)
    axes[5].set_ylabel("Cumulative\nreward proxy")
    axes[5].set_xlabel("DAP")
    axes[5].set_title("Unified reward proxy: terminal max(0, GWAD - null GWAD) minus water and nitrogen costs", loc="left", fontsize=9)

    for ax in axes:
        ax.grid(True, which="major", axis="both", linestyle="--", color="#E4E8EF", linewidth=0.6, alpha=0.8)
        ax.set_xlim(-2, max_dap + 4)
    axes[-1].set_xticks(np.arange(0, max_dap + 1, 20))

    # Summary text box
    dqn = summary[summary["scenario"].eq("dqn")].iloc[0]
    rec = summary[summary["scenario"].eq("recorded")].iloc[0]
    text = (
        f"DQN vs recorded: +{dqn['yield_diff_vs_recorded']:.0f} kg/ha GWAD, "
        f"{-dqn['irrigation_saving_vs_recorded']:.0f} mm more irrigation, "
        f"{-dqn['fertilizer_saving_vs_recorded']:.0f} kg/ha more N"
    )
    axes[5].text(
        0.01,
        0.90,
        text,
        transform=axes[5].transAxes,
        fontsize=8.5,
        color="#238B45",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.5},
    )

    fig.tight_layout(h_pad=0.8)
    out_base = FIG_DIR / "017_09_sy2014_four_scenario_process"
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_base.with_suffix(".png")


def write_record(summary: pd.DataFrame, fig_path: Path) -> None:
    dqn = summary[summary["scenario"].eq("dqn")].iloc[0]
    table_cols = [
        "scenario_label",
        "final_gwad",
        "final_cwad",
        "irrigation_total_mm",
        "fertilizer_total_kg_ha",
        "max_water_stress",
        "max_nitrogen_stress",
        "final_cumulative_proxy_reward",
        "yield_diff_vs_recorded",
        "irrigation_saving_vs_recorded",
        "fertilizer_saving_vs_recorded",
        "resource_success_vs_recorded",
    ]
    table = summary[table_cols].copy()
    for col in table.columns:
        if pd.api.types.is_bool_dtype(table[col]):
            table[col] = table[col].map(lambda x: "True" if bool(x) else "False")
            continue
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{float(x):.2f}")
    md_table = []
    md_table.append("| " + " | ".join(table.columns) + " |")
    md_table.append("| " + " | ".join(["---"] * len(table.columns)) + " |")
    for _, row in table.iterrows():
        md_table.append("| " + " | ".join(str(row[col]) for col in table.columns) + " |")
    lines = [
        "# 017_09 SY2014 DQN 资源使用与四情景过程图记录",
        "",
        "## 目的",
        "",
        "复用 017_08 已有结果，不重新训练，检查 SY2014 最佳 DQN checkpoint 的过程、资源使用和是否具有节水节氮空间。",
        "",
        "## 输入",
        "",
        f"- 来源目录：`{SOURCE_DIR.relative_to(PROJECT_ROOT)}`",
        "- DQN 情景：`transfer_SY2014_ckpt15000`，在本记录中重命名为 `dqn`。",
        "- 四情景：null、recorded、DSSAT auto、DQN。",
        "",
        "## 统一 reward proxy",
        "",
        "为了让四个情景可以按同一目标函数比较，本阶段重新计算 proxy reward：",
        "",
        "```text",
        "daily_proxy_reward = -1.0 * irrigation_mm - 5.0 * fertilizer_kg_ha",
        "terminal_bonus = max(0, final_GWAD - null_GWAD)",
        "```",
        "",
        "注意：这是 DQN 目标函数的统一代理值，不是 DSSAT 原生 reward。",
        "",
        "## 汇总结果",
        "",
        "\n".join(md_table),
        "",
        "## 判断",
        "",
        f"- DQN 相对 recorded 的产量变化：{dqn['yield_diff_vs_recorded']:.0f} kg/ha。",
        f"- DQN 相对 recorded 的灌溉节省：{dqn['irrigation_saving_vs_recorded']:.1f} mm（负值表示用水更多）。",
        f"- DQN 相对 recorded 的施氮节省：{dqn['fertilizer_saving_vs_recorded']:.1f} kg/ha（负值表示施氮更多）。",
        "- 因此，当前 SY2014 DQN 可以表述为“高产型成功”：产量超过 recorded 和 DSSAT auto。",
        "- 但不能表述为“节水节氮型成功”：它使用满 I120/N300 预算，且相对 recorded 多用水、略多施氮。",
        "- 下一步若要追求节水节氮，需要做水氮成本/预算敏感性或候选 checkpoint 筛选，而不是直接继续增加训练步数。",
        "",
        "## 输出",
        "",
        f"- 四情景日值表：`{(OUT_DIR / '017_09_sy2014_four_scenario_daily.csv').relative_to(PROJECT_ROOT)}`",
        f"- 四情景事件表：`{(OUT_DIR / '017_09_sy2014_four_scenario_events.csv').relative_to(PROJECT_ROOT)}`",
        f"- 四情景汇总表：`{(OUT_DIR / '017_09_sy2014_four_scenario_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 过程图：`{fig_path.relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    daily, events, raw_summary = load_inputs()
    daily = add_reward_proxy(daily, events, raw_summary)
    summary = make_summary(daily, events, raw_summary)

    daily.to_csv(OUT_DIR / "017_09_sy2014_four_scenario_daily.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "017_09_sy2014_four_scenario_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "017_09_sy2014_four_scenario_summary.csv", index=False, encoding="utf-8-sig")
    fig_path = plot_process(daily, events, summary)
    write_record(summary, fig_path)
    print(summary.to_string(index=False))
    print(f"[figure] {fig_path}")
    print(f"[record] {DOC_PATH}")


if __name__ == "__main__":
    main()
