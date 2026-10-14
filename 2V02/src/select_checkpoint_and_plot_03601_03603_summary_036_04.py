from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "036_04_select_checkpoint_and_plot_03601_03603_summary"
OUT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
DOC = ROOT / "docs" / f"{TASK}_record.md"

PPO_WITH_WP = ROOT / "benchmark_results" / "036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018" / "tables" / "036_03_corrected_checkpoint_validation_with_wp_et.csv"
BY_STATION = ROOT / "benchmark_results" / "036_03_replay_03601_checkpoints_for_wp_et_and_fqa2018" / "tables" / "036_03_by_station_checkpoint_with_wp_et.csv"
BASELINE_NON_SY = ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134" / "evaluation" / "031_36_full_completed_template_aware_unified_baseline_summary.csv"
BASELINE_SY = ROOT / "benchmark_results" / "031_29_sy_auto_and_recorded_template_completion" / "evaluation" / "031_29_sy_baseline_summary.csv"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scenario_as_text(value: object) -> str:
    if pd.isna(value):
        return "null"
    text = str(value)
    return "null" if text.lower() == "nan" else text


def canonical_group(scenario: object) -> str:
    text = scenario_as_text(scenario)
    if text == "null":
        return "null"
    if text == "dssat_auto":
        return "dssat_auto"
    if text == "official_extension_expert":
        return "official_extension_expert"
    if text.startswith("recorded_farmer"):
        return "recorded_farmer_or_template"
    return text


def load_baseline() -> pd.DataFrame:
    frames = []
    for path in [BASELINE_NON_SY, BASELINE_SY]:
        frame = pd.read_csv(path)
        frame["scenario"] = frame["scenario"].map(scenario_as_text)
        frame["canonical_group"] = frame["scenario"].map(canonical_group)
        frames.append(frame)
    baseline = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "station_code",
        "year",
        "scenario",
        "canonical_group",
        "grain_yield_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
    }
    missing = sorted(required - set(baseline.columns))
    if missing:
        raise RuntimeError(f"Baseline missing required columns: {missing}")
    return baseline


def select_checkpoints(station_summary: pd.DataFrame) -> pd.DataFrame:
    frame = station_summary.copy()
    for col in [
        "any_metric_win_count",
        "yield_win_count",
        "wp_et_win_count",
        "pfp_n_win_count",
        "mean_gap_yield",
        "mean_gap_wp_et",
        "mean_gap_pfp_n",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["tie_break_score"] = (
        frame["mean_gap_yield"].fillna(-1e9)
        + 1000.0 * frame["mean_gap_wp_et"].fillna(-1e9)
        + 10.0 * frame["mean_gap_pfp_n"].fillna(-1e9)
    )
    frame = frame.sort_values(
        [
            "station_code",
            "any_metric_win_count",
            "yield_win_count",
            "wp_et_win_count",
            "pfp_n_win_count",
            "tie_break_score",
            "checkpoint_step",
        ],
        ascending=[True, False, False, False, False, False, True],
    )
    selected = frame.groupby("station_code", as_index=False).head(1).copy()
    selected["selection_rule"] = (
        "max any_metric_win_count; then yield/wp_et/pfp counts; "
        "then mean_gap_yield + 1000*mean_gap_wp_et + 10*mean_gap_pfp_n; then earlier checkpoint"
    )
    return selected.reset_index(drop=True)


def build_selected_years(ppo: pd.DataFrame, selected: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    keys = selected[["station_code", "checkpoint_step"]].drop_duplicates()
    merged = ppo.merge(keys, on=["station_code", "checkpoint_step"], how="inner").copy()

    # Official expert for resource-saving reference.
    expert = baseline.loc[baseline["canonical_group"].eq("official_extension_expert")].copy()
    expert = (
        expert.sort_values(["station_code", "year", "scenario"])
        .groupby(["station_code", "year"], as_index=False)
        .agg(
            expert_yield=("grain_yield_kg_ha", "max"),
            expert_irrigation=("actual_irrigation_mm", "max"),
            expert_n=("actual_nitrogen_kg_ha", "max"),
            expert_wp_et=("WP_ET_kg_m3", "max"),
            expert_pfp_n=("PFP_N_kg_kg", "max"),
        )
    )
    merged = merged.merge(expert, on=["station_code", "year"], how="left")
    merged["water_saving_vs_expert_mm"] = merged["expert_irrigation"] - merged["total_irrigation"]
    merged["n_saving_vs_expert_kg_ha"] = merged["expert_n"] - merged["total_n"]
    merged["yield_gap_vs_expert"] = merged["final_grnwt"] - merged["expert_yield"]
    merged["wp_et_gap_vs_expert"] = merged["WP_ET_kg_m3"] - merged["expert_wp_et"]
    merged["pfp_n_gap_vs_expert"] = merged["PFP_N_kg_kg"] - merged["expert_pfp_n"]

    # Prefer the corrected with-WP gaps from 036_03.
    rename = {
        "gap_yield_vs_available_baseline_max_with_wp": "yield_gap_vs_four_max",
        "gap_wp_et_vs_available_baseline_max_with_wp": "wp_et_gap_vs_four_max",
        "gap_pfp_n_vs_available_baseline_max_with_wp": "pfp_n_gap_vs_four_max",
        "any_metric_win_available_baseline_with_wp": "any_metric_win_four_max",
    }
    for old, new in rename.items():
        if old in merged.columns:
            merged[new] = merged[old]
    return merged


def summarize_station_years(selected_years: pd.DataFrame) -> pd.DataFrame:
    def count_positive(s: pd.Series) -> int:
        return int((pd.to_numeric(s, errors="coerce") > 0).sum())

    rows = []
    for station, g in selected_years.groupby("station_code"):
        rows.append(
            {
                "station_code": station,
                "checkpoint_step": int(g["checkpoint_step"].iloc[0]),
                "validation_years": int(g["year"].nunique()),
                "any_metric_win_count": int(g["any_metric_win_four_max"].fillna(False).astype(bool).sum()),
                "yield_win_count": count_positive(g["yield_gap_vs_four_max"]),
                "wp_et_win_count": count_positive(g["wp_et_gap_vs_four_max"]),
                "pfp_n_win_count": count_positive(g["pfp_n_gap_vs_four_max"]),
                "mean_final_grnwt": float(g["final_grnwt"].mean()),
                "mean_total_irrigation": float(g["total_irrigation"].mean()),
                "mean_total_n": float(g["total_n"].mean()),
                "mean_WP_ET_kg_m3": float(g["WP_ET_kg_m3"].mean()),
                "mean_PFP_N_kg_kg": float(g["PFP_N_kg_kg"].mean()),
                "mean_water_saving_vs_expert_mm": float(g["water_saving_vs_expert_mm"].mean()),
                "std_water_saving_vs_expert_mm": float(g["water_saving_vs_expert_mm"].std(ddof=1)),
                "mean_n_saving_vs_expert_kg_ha": float(g["n_saving_vs_expert_kg_ha"].mean()),
                "std_n_saving_vs_expert_kg_ha": float(g["n_saving_vs_expert_kg_ha"].std(ddof=1)),
                "mean_yield_gap_vs_four_max": float(g["yield_gap_vs_four_max"].mean()),
                "mean_wp_et_gap_vs_four_max": float(g["wp_et_gap_vs_four_max"].mean()),
                "mean_pfp_n_gap_vs_four_max": float(g["pfp_n_gap_vs_four_max"].mean()),
                "zero_yield_years": ",".join(map(str, sorted(g.loc[pd.to_numeric(g["final_grnwt"], errors="coerce").le(0), "year"].unique()))),
            }
        )
    return pd.DataFrame(rows)


def savefig(fig: plt.Figure, stem: str) -> list[str]:
    paths = []
    for suffix in ["png", "svg"]:
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def plot_station(station: str, data: pd.DataFrame) -> list[str]:
    data = data.sort_values("year")
    years = data["year"].astype(str).tolist()
    x = np.arange(len(data))
    colors = {"yield": "#4C78A8", "wp": "#F58518", "pfp": "#54A24B", "water": "#72B7B2", "n": "#B279A2"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    fig.suptitle(f"{station}: selected checkpoint {int(data['checkpoint_step'].iloc[0])} validation summary", x=0.02, ha="left", fontsize=13, fontweight="bold")

    metric_specs = [
        ("yield_gap_vs_four_max", "Yield gap vs best of four scenarios", "kg/ha", colors["yield"], axes[0, 0]),
        ("wp_et_gap_vs_four_max", "WP_ET gap vs best of four scenarios", "kg/m³", colors["wp"], axes[0, 1]),
        ("pfp_n_gap_vs_four_max", "PFP_N gap vs best of four scenarios", "kg/kg", colors["pfp"], axes[1, 0]),
    ]
    for col, title, unit, color, ax in metric_specs:
        vals = pd.to_numeric(data[col], errors="coerce")
        ax.axhline(0, color="#333333", lw=0.8)
        bar_colors = [color if v >= 0 else "#C9CDD3" for v in vals.fillna(-1e9)]
        ax.bar(x, vals, color=bar_colors, edgecolor="#333333", linewidth=0.3)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.22)

    ax = axes[1, 1]
    width = 0.38
    water = pd.to_numeric(data["water_saving_vs_expert_mm"], errors="coerce")
    nitro = pd.to_numeric(data["n_saving_vs_expert_kg_ha"], errors="coerce")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.bar(x - width / 2, water, width=width, label="Water saved vs expert (mm)", color=colors["water"], edgecolor="#333333", linewidth=0.3)
    ax.bar(x + width / 2, nitro, width=width, label="N saved vs expert (kg/ha)", color=colors["n"], edgecolor="#333333", linewidth=0.3)
    ax.set_title("Resource saving vs official expert")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(fontsize=8)

    return savefig(fig, f"036_04_{station.lower()}_selected_checkpoint_metric_gaps")


def plot_overall(station_summary: pd.DataFrame) -> list[str]:
    data = station_summary.sort_values("station_code")
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.bar(
        x - width / 2,
        data["mean_water_saving_vs_expert_mm"],
        yerr=data["std_water_saving_vs_expert_mm"].fillna(0),
        width=width,
        color="#72B7B2",
        edgecolor="#333333",
        linewidth=0.35,
        capsize=3,
        label="Water saved vs expert (mm)",
    )
    ax.bar(
        x + width / 2,
        data["mean_n_saving_vs_expert_kg_ha"],
        yerr=data["std_n_saving_vs_expert_kg_ha"].fillna(0),
        width=width,
        color="#B279A2",
        edgecolor="#333333",
        linewidth=0.35,
        capsize=3,
        label="N saved vs expert (kg/ha)",
    )
    for i, row in data.reset_index(drop=True).iterrows():
        ax.text(i, ax.get_ylim()[1] * 0.92, f"{int(row['any_metric_win_count'])}/{int(row['validation_years'])}", ha="center", va="top", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(data["station_code"])
    ax.set_title("036_04 selected checkpoints: mean resource saving vs official expert")
    ax.set_ylabel("Mean saving across validation years; error bars are year-to-year SD")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    return savefig(fig, "036_04_overall_water_n_saving_vs_expert")


def write_record(selected: pd.DataFrame, station_summary: pd.DataFrame, figure_paths: list[str], zero_rows: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_None._"
        return df.to_markdown(index=False)

    lines = [
        "# 036_04 checkpoint选择与五站点汇总绘图记录",
        "",
        "## 任务边界",
        "",
        "- 本任务不训练模型。",
        "- 本任务不修改reward、动作空间、约束条件或DSSAT输入。",
        "- 本任务只整理036_01正式重跑与036_03 replay后的结果。",
        "- FQA2018零产量异常保留，不静默删除。",
        "",
        "## 固定checkpoint选择规则",
        "",
        "每站点代表checkpoint按以下顺序选择：`any_metric_win_count`、`yield_win_count`、`wp_et_win_count`、`pfp_n_win_count`、`mean_gap_yield + 1000*mean_gap_wp_et + 10*mean_gap_pfp_n`，最后选更早checkpoint。",
        "",
        "## 选中的checkpoint",
        "",
        md_table(selected[[
            "station_code",
            "checkpoint_step",
            "validation_years",
            "any_metric_win_count",
            "yield_win_count",
            "wp_et_win_count",
            "pfp_n_win_count",
            "mean_gap_yield",
            "mean_gap_wp_et",
            "mean_gap_pfp_n",
        ]].round(4)),
        "",
        "## 站点汇总",
        "",
        md_table(station_summary.round(4)),
        "",
        "## 零产量/异常保留",
        "",
        md_table(zero_rows[["station_code", "year", "checkpoint_step", "final_grnwt", "total_irrigation", "total_n", "WP_ET_kg_m3", "PFP_N_kg_kg"]].round(4)),
        "",
        "## 输出图件",
        "",
    ]
    lines.extend([f"- `{p}`" for p in figure_paths])
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 图中“超过四情景最高值”使用036_03的available-baseline最大值口径；若同一年存在多个recorded/template版本，取可用基线最大值是保守口径，但不等同于单一canonical recorded farmer。",
            "- 节水/节氮图默认相对official expert计算，因为导师当前关注PPO是否能在专家措施基础上节水节氮；若后续要求相对四情景最省资源值，可另行生成。",
        ]
    )
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ppo = pd.read_csv(PPO_WITH_WP)
    by_station = pd.read_csv(BY_STATION)
    baseline = load_baseline()

    selected = select_checkpoints(by_station)
    selected_years = build_selected_years(ppo, selected, baseline)
    station_summary = summarize_station_years(selected_years)
    zero_rows = selected_years.loc[pd.to_numeric(selected_years["final_grnwt"], errors="coerce").le(0)].copy()

    selected.to_csv(TABLE_DIR / "036_04_selected_checkpoints.csv", index=False, encoding="utf-8-sig")
    selected_years.to_csv(TABLE_DIR / "036_04_selected_year_level_comparison.csv", index=False, encoding="utf-8-sig")
    station_summary.to_csv(TABLE_DIR / "036_04_selected_station_summary.csv", index=False, encoding="utf-8-sig")

    figure_paths: list[str] = []
    for station, g in selected_years.groupby("station_code"):
        figure_paths.extend(plot_station(station, g))
    figure_paths.extend(plot_overall(station_summary))

    write_record(selected, station_summary, figure_paths, zero_rows)
    print(
        json.dumps(
            {
                "task": TASK,
                "selected_checkpoints": selected[["station_code", "checkpoint_step"]].to_dict(orient="records"),
                "selected_rows": int(len(selected_years)),
                "zero_yield_rows": int(len(zero_rows)),
                "figures": len(figure_paths),
                "record_md": str(DOC.relative_to(ROOT)),
                "out_dir": str(OUT.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
