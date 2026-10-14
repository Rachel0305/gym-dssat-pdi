"""Summarize 039_02 lowIC baseline audit into RL-input usability classes.

This script does not run DSSAT and does not modify any input files.  It reads
the full originalIC-vs-lowIC baseline outputs from 039_02 and produces a
traceable classification table answering:

    Is the manually lowered initial soil water/mineral nitrogen setting suitable
    as an RL training input for each station-year?

The classification intentionally uses null and official expert as the main
guardrails.  DSSAT auto is reported as a comparator but is not used as a hard
exclusion criterion because 039_02 showed DSSAT auto often applies no nitrogen
under lowIC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "039_02_original_vs_lowIC_three_baseline_audit"
OUT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
DOC = OUT / "039_02_lowIC_usability_classification_record.md"

SUMMARY_CSV = TABLE_DIR / "039_02_full_summary.csv"
EFFECT_CSV = TABLE_DIR / "039_02_full_lowIC_minus_original_effect.csv"

STRESS_DELTA_MIN = 0.05
YIELD_DROP_SIGNAL_KG_HA = -500.0
EXPERT_YIELD_DROP_TOO_SEVERE_KG_HA = -1000.0
EXPERT_MAX_WATER_TOO_SEVERE = 0.50
EXPERT_MAX_N_TOO_SEVERE = 0.20

SCENARIO_LABELS = {
    "null": "Null",
    "official_extension_expert": "Official expert",
    "dssat_auto": "DSSAT auto",
}


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(SUMMARY_CSV)
    if not EFFECT_CSV.exists():
        raise FileNotFoundError(EFFECT_CSV)
    summary = pd.read_csv(SUMMARY_CSV, keep_default_na=False)
    effect = pd.read_csv(EFFECT_CSV, keep_default_na=False)
    for frame in [summary, effect]:
        for col in frame.columns:
            if col.startswith(("original_", "lowIC_", "delta_")) or col in {
                "grain_yield_kg_ha",
                "biomass_kg_ha",
                "actual_irrigation_mm",
                "actual_nitrogen_kg_ha",
                "etcp_mm",
                "WP_ET_kg_m3",
                "PFP_N_kg_kg",
                "max_water_stress",
                "max_nitrogen_stress",
                "year",
            }:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ["lowIC_increased_stress", "lowIC_too_severe_candidate"]:
        if col in effect.columns:
            effect[col] = effect[col].astype(str).str.lower().eq("true")
    return summary, effect


def station_scenario_aggregate(effect: pd.DataFrame) -> pd.DataFrame:
    return (
        effect.groupby(["station_code", "site", "scenario"], dropna=False)
        .agg(
            n=("year", "count"),
            mean_delta_yield=("delta_grain_yield_kg_ha", "mean"),
            min_delta_yield=("delta_grain_yield_kg_ha", "min"),
            max_delta_yield=("delta_grain_yield_kg_ha", "max"),
            mean_delta_WSPD=("delta_max_water_stress", "mean"),
            mean_delta_NSTD=("delta_max_nitrogen_stress", "mean"),
            stress_increased_n=("lowIC_increased_stress", "sum"),
            too_severe_candidate_n=("lowIC_too_severe_candidate", "sum"),
            mean_lowIC_yield=("lowIC_grain_yield_kg_ha", "mean"),
            mean_original_yield=("original_grain_yield_kg_ha", "mean"),
        )
        .reset_index()
        .sort_values(["station_code", "scenario"])
    )


def row_or_none(effect: pd.DataFrame, station: str, year: int, scenario: str) -> dict[str, Any] | None:
    sub = effect[
        effect["station_code"].astype(str).eq(station)
        & effect["year"].eq(int(year))
        & effect["scenario"].astype(str).eq(scenario)
    ]
    if len(sub) != 1:
        return None
    return sub.iloc[0].to_dict()


def classify_one(null_row: dict[str, Any] | None, expert_row: dict[str, Any] | None, auto_row: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if null_row is None or expert_row is None:
        out.update(
            {
                "lowIC_usability_class": "D_incomplete",
                "lowIC_usability_reason": "missing null or official expert original-lowIC pair",
                "null_has_learning_signal": False,
                "expert_can_buffer_lowIC": False,
            }
        )
        return out

    null_delta_w = float(null_row.get("delta_max_water_stress", 0.0))
    null_delta_n = float(null_row.get("delta_max_nitrogen_stress", 0.0))
    null_delta_y = float(null_row.get("delta_grain_yield_kg_ha", 0.0))
    null_signal = (
        null_delta_w >= STRESS_DELTA_MIN
        or null_delta_n >= STRESS_DELTA_MIN
        or null_delta_y <= YIELD_DROP_SIGNAL_KG_HA
    )

    expert_delta_y = float(expert_row.get("delta_grain_yield_kg_ha", 0.0))
    expert_low_w = float(expert_row.get("lowIC_max_water_stress", 0.0))
    expert_low_n = float(expert_row.get("lowIC_max_nitrogen_stress", 0.0))
    expert_too_severe = (
        expert_delta_y <= EXPERT_YIELD_DROP_TOO_SEVERE_KG_HA
        or expert_low_w >= EXPERT_MAX_WATER_TOO_SEVERE
        or expert_low_n >= EXPERT_MAX_N_TOO_SEVERE
    )
    expert_buffers = not expert_too_severe

    if not null_signal:
        cls = "C_weak_lowIC_response"
        reason = "lowIC does not create enough null stress/yield contrast"
    elif expert_too_severe:
        cls = "B_lowIC_too_severe_for_direct_training"
        reason = "official expert is also strongly harmed or stressed under lowIC"
    else:
        cls = "A_RL_training_candidate"
        reason = "lowIC creates null stress/yield contrast and official expert remains buffered"

    out.update(
        {
            "lowIC_usability_class": cls,
            "lowIC_usability_reason": reason,
            "null_has_learning_signal": bool(null_signal),
            "expert_can_buffer_lowIC": bool(expert_buffers),
            "null_delta_yield": null_delta_y,
            "null_delta_WSPD": null_delta_w,
            "null_delta_NSTD": null_delta_n,
            "expert_delta_yield": expert_delta_y,
            "expert_lowIC_WSPD": expert_low_w,
            "expert_lowIC_NSTD": expert_low_n,
        }
    )
    if auto_row is not None:
        out.update(
            {
                "auto_delta_yield": float(auto_row.get("delta_grain_yield_kg_ha", float("nan"))),
                "auto_lowIC_WSPD": float(auto_row.get("lowIC_max_water_stress", float("nan"))),
                "auto_lowIC_NSTD": float(auto_row.get("lowIC_max_nitrogen_stress", float("nan"))),
                "auto_too_severe_candidate": bool(auto_row.get("lowIC_too_severe_candidate", False)),
            }
        )
    return out


def build_usability_table(effect: pd.DataFrame) -> pd.DataFrame:
    keys = effect[["station_code", "site", "year"]].drop_duplicates().sort_values(["station_code", "year"])
    rows: list[dict[str, Any]] = []
    for rec in keys.itertuples(index=False):
        station = str(rec.station_code)
        year = int(rec.year)
        null_row = row_or_none(effect, station, year, "null")
        expert_row = row_or_none(effect, station, year, "official_extension_expert")
        auto_row = row_or_none(effect, station, year, "dssat_auto")
        row = {"station_code": station, "site": rec.site, "year": year}
        row.update(classify_one(null_row, expert_row, auto_row))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_usability(usability: pd.DataFrame) -> Path:
    counts = usability.groupby(["station_code", "lowIC_usability_class"]).size().reset_index(name="n")
    class_order = [
        "A_RL_training_candidate",
        "B_lowIC_too_severe_for_direct_training",
        "C_weak_lowIC_response",
        "D_incomplete",
    ]
    colors = {
        "A_RL_training_candidate": "#2CA25F",
        "B_lowIC_too_severe_for_direct_training": "#DE2D26",
        "C_weak_lowIC_response": "#FDAE6B",
        "D_incomplete": "#9E9E9E",
    }
    stations = sorted(usability["station_code"].unique())
    bottom = {station: 0 for station in stations}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cls in class_order:
        vals = []
        for station in stations:
            sub = counts[counts["station_code"].eq(station) & counts["lowIC_usability_class"].eq(cls)]
            vals.append(int(sub["n"].iloc[0]) if len(sub) else 0)
        ax.bar(stations, vals, bottom=[bottom[s] for s in stations], color=colors[cls], label=cls)
        for station, val in zip(stations, vals):
            bottom[station] += val
    ax.set_title("039_02 lowIC usability classes by station-year", fontweight="bold")
    ax.set_ylabel("Number of years")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "039_02_lowIC_usability_classification_by_station.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_record(aggregate: pd.DataFrame, usability: pd.DataFrame, fig_path: Path) -> None:
    class_counts = usability.groupby(["station_code", "lowIC_usability_class"]).size().reset_index(name="n")
    overall = usability["lowIC_usability_class"].value_counts().rename_axis("lowIC_usability_class").reset_index(name="n")
    lines = [
        "# 039_02 lowIC 是否适合作为 RL 训练输入：判定表记录",
        "",
        "## 数据来源",
        "",
        f"- summary: `{SUMMARY_CSV.relative_to(ROOT).as_posix()}`",
        f"- effect: `{EFFECT_CSV.relative_to(ROOT).as_posix()}`",
        "",
        "## 判定规则",
        "",
        f"- null 学习信号：lowIC-null 相比 original-null 的 WSPD 或 NSTD 增加至少 {STRESS_DELTA_MIN}，或产量下降至少 {abs(YIELD_DROP_SIGNAL_KG_HA):.0f} kg/ha。",
        f"- expert 过强判据：lowIC-expert 相比 original-expert 产量下降超过 {abs(EXPERT_YIELD_DROP_TOO_SEVERE_KG_HA):.0f} kg/ha，或 lowIC-expert 最大 WSPD ≥ {EXPERT_MAX_WATER_TOO_SEVERE}，或最大 NSTD ≥ {EXPERT_MAX_N_TOO_SEVERE}。",
        "- `A_RL_training_candidate`：有 null 学习信号，且 official expert 没被 lowIC 压垮。",
        "- `B_lowIC_too_severe_for_direct_training`：official expert 在 lowIC 下也明显受害或强胁迫。",
        "- `C_weak_lowIC_response`：lowIC 对 null 的响应太弱。",
        "- `D_incomplete`：缺少必要对照。",
        "- DSSAT auto 只作为辅助比较，不作为一票否决，因为 039_02 显示 auto 在 lowIC 下经常不施氮。",
        "",
        "## 总体分类",
        "",
        md_table(overall, 20),
        "",
        "## 分站点分类",
        "",
        md_table(class_counts, 80),
        "",
        "## 分站点/情景 lowIC-original 聚合",
        "",
        md_table(aggregate, 120),
        "",
        "## 图件",
        "",
        f"- `{fig_path.relative_to(ROOT).as_posix()}`",
        "",
        "## 输出文件",
        "",
        f"- `benchmark_results/{TASK}/tables/039_02_station_scenario_aggregate.csv`",
        f"- `benchmark_results/{TASK}/tables/039_02_lowIC_usability_classification.csv`",
        f"- `benchmark_results/{TASK}/tables/039_02_lowIC_usability_class_counts.csv`",
        f"- `benchmark_results/{TASK}/039_02_lowIC_usability_summary.json`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    summary, effect = load_tables()
    aggregate = station_scenario_aggregate(effect)
    usability = build_usability_table(effect)
    class_counts = usability.groupby(["station_code", "lowIC_usability_class"]).size().reset_index(name="n")
    fig_path = plot_usability(usability)

    aggregate.to_csv(TABLE_DIR / "039_02_station_scenario_aggregate.csv", index=False, encoding="utf-8-sig")
    usability.to_csv(TABLE_DIR / "039_02_lowIC_usability_classification.csv", index=False, encoding="utf-8-sig")
    class_counts.to_csv(TABLE_DIR / "039_02_lowIC_usability_class_counts.csv", index=False, encoding="utf-8-sig")
    write_record(aggregate, usability, fig_path)

    result = {
        "task": "039_02_lowIC_usability_classification",
        "input_summary_rows": int(len(summary)),
        "input_effect_rows": int(len(effect)),
        "classified_station_years": int(len(usability)),
        "class_counts": usability["lowIC_usability_class"].value_counts().to_dict(),
        "outputs": {
            "aggregate_csv": str((TABLE_DIR / "039_02_station_scenario_aggregate.csv").relative_to(ROOT)).replace("\\", "/"),
            "usability_csv": str((TABLE_DIR / "039_02_lowIC_usability_classification.csv").relative_to(ROOT)).replace("\\", "/"),
            "class_counts_csv": str((TABLE_DIR / "039_02_lowIC_usability_class_counts.csv").relative_to(ROOT)).replace("\\", "/"),
            "figure": str(fig_path.relative_to(ROOT)).replace("\\", "/"),
            "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    (OUT / "039_02_lowIC_usability_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
