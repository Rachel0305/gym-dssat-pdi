from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "041_05"
TASK_NAME = "sya_lowIC_teacher_warmstart_drift_audit"
SOURCE = ROOT / "benchmark_results" / "041_04_sya_lowIC_teacher_warmstart_balanced_bc_maskableppo"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
YEARS = list(range(2014, 2024))


def ensure_dirs() -> None:
    for rel in ["tables", "figures"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def day_series(df: pd.DataFrame) -> pd.Series:
    dap = pd.to_numeric(df.get("dap"), errors="coerce")
    if dap.notna().any() and float(dap.max()) > 10:
        return dap.fillna(pd.to_numeric(df["step"], errors="coerce") + 1)
    return pd.to_numeric(df["step"], errors="coerce") + 1


def stage_bin(day: float) -> str:
    if day <= 30:
        return "D1_30"
    if day <= 60:
        return "D31_60"
    if day <= 90:
        return "D61_90"
    return "D91_plus"


def read_daily(year: int, ckpt: int) -> pd.DataFrame:
    p = SOURCE / "daily_outputs" / "SYA" / f"041_03_SYA_{year}_ckpt{ckpt}_daily.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["day_for_audit"] = day_series(df)
    df["stage_bin"] = df["day_for_audit"].apply(stage_bin)
    return df


def summarize_daily(df: pd.DataFrame, year: int, label: str) -> dict[str, object]:
    irr = pd.to_numeric(df["irrigation_mm_action"], errors="coerce").fillna(0)
    nit = pd.to_numeric(df["nitrogen_kg_ha_action"], errors="coerce").fillna(0)
    day = pd.to_numeric(df["day_for_audit"], errors="coerce")
    irr_events = df[irr > 0].copy()
    nit_events = df[nit > 0].copy()
    out: dict[str, object] = {
        "year": year,
        "policy_label": label,
        "irrigation_total": float(irr.sum()),
        "nitrogen_total": float(nit.sum()),
        "irrigation_event_count": int((irr > 0).sum()),
        "nitrogen_event_count": int((nit > 0).sum()),
        "first_irrigation_day": float(day[irr > 0].min()) if (irr > 0).any() else np.nan,
        "last_irrigation_day": float(day[irr > 0].max()) if (irr > 0).any() else np.nan,
        "first_nitrogen_day": float(day[nit > 0].min()) if (nit > 0).any() else np.nan,
        "last_nitrogen_day": float(day[nit > 0].max()) if (nit > 0).any() else np.nan,
        "max_swfac": float(pd.to_numeric(df["swfac"], errors="coerce").max()),
        "max_nstres": float(pd.to_numeric(df["nstres"], errors="coerce").max()),
    }
    for stage in ["D1_30", "D31_60", "D61_90", "D91_plus"]:
        m = df["stage_bin"] == stage
        out[f"irrigation_{stage}"] = float(irr[m].sum())
        out[f"nitrogen_{stage}"] = float(nit[m].sum())
    return out


def build_metric_delta() -> pd.DataFrame:
    p = SOURCE / "evaluation" / "041_03_checkpoint_validation_summary.csv"
    df = pd.read_csv(p)
    keep = df[df["checkpoint_step"].isin([0, 100000])].copy()
    cols = [
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "summary_irrigation_total",
        "summary_nitrogen_total",
        "gap_yield_vs_four_max",
        "gap_wp_et_vs_four_max",
        "gap_pfp_n_vs_four_max",
        "yield_win_vs_four_max",
        "wp_et_win_vs_four_max",
        "pfp_n_win_vs_four_max",
        "all3_win_vs_four_max",
    ]
    delta_cols = {
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "summary_irrigation_total",
        "summary_nitrogen_total",
        "gap_yield_vs_four_max",
        "gap_wp_et_vs_four_max",
        "gap_pfp_n_vs_four_max",
    }
    wide = keep.pivot(index="year", columns="checkpoint_step", values=cols)
    rows = []
    for year in YEARS:
        row: dict[str, object] = {"year": year}
        for col in cols:
            bc_value = wide.loc[year, (col, 0)]
            ppo_value = wide.loc[year, (col, 100000)]
            row[f"bc_{col}"] = bc_value
            row[f"ppo100k_{col}"] = ppo_value
            if col in delta_cols:
                bc_num = pd.to_numeric(pd.Series([bc_value]), errors="coerce").iloc[0]
                ppo_num = pd.to_numeric(pd.Series([ppo_value]), errors="coerce").iloc[0]
                if pd.notna(bc_num) and pd.notna(ppo_num):
                    row[f"delta_{col}"] = float(ppo_num - bc_num)
        rows.append(row)
    return pd.DataFrame(rows)


def build_event_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    event_rows = []
    for year in YEARS:
        for ckpt, label in [(0, "bc_init"), (100000, "ppo100k")]:
            df = read_daily(year, ckpt)
            rows.append(summarize_daily(df, year, label))
            events = df[
                (pd.to_numeric(df["irrigation_mm_action"], errors="coerce").fillna(0) > 0)
                | (pd.to_numeric(df["nitrogen_kg_ha_action"], errors="coerce").fillna(0) > 0)
            ].copy()
            if not events.empty:
                events["policy_label"] = label
                event_rows.append(
                    events[
                        [
                            "year",
                            "policy_label",
                            "step",
                            "day_for_audit",
                            "stage_bin",
                            "action_index",
                            "irrigation_mm_action",
                            "nitrogen_kg_ha_action",
                            "swfac",
                            "nstres",
                            "grnwt",
                            "topwt",
                        ]
                    ]
                )
    return pd.DataFrame(rows), pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()


def write_figures(metric_delta: pd.DataFrame, event_summary: pd.DataFrame) -> list[str]:
    figures: list[str] = []

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(metric_delta))
    years = metric_delta["year"].astype(str).tolist()
    axes[0].bar(x, metric_delta["delta_grain_yield_kg_ha"], color="#4c78a8")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Δ yield\nkg/ha")
    axes[1].bar(x, metric_delta["delta_WP_ET_kg_m3"], color="#59a14f")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Δ WP_ET")
    axes[2].bar(x, metric_delta["delta_PFP_N_kg_kg"], color="#f28e2b")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Δ PFP_N")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(years, rotation=45)
    fig.suptitle("041_05 PPO 100K minus BC init: metric drift")
    fig.tight_layout()
    p = OUT / "figures" / "041_05_metric_drift_ppo100k_minus_bcinit.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    figures.append(p.relative_to(ROOT).as_posix())

    piv = event_summary.pivot(index="year", columns="policy_label", values=["irrigation_total", "nitrogen_total"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    width = 0.38
    for ax, metric, title in [
        (axes[0], "irrigation_total", "Irrigation total"),
        (axes[1], "nitrogen_total", "Nitrogen total"),
    ]:
        bc = piv[(metric, "bc_init")].reindex(YEARS)
        pp = piv[(metric, "ppo100k")].reindex(YEARS)
        ax.bar(x - width / 2, bc, width, label="BC init")
        ax.bar(x + width / 2, pp, width, label="PPO 100K")
        ax.set_ylabel(title)
        ax.legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(years, rotation=45)
    fig.suptitle("041_05 Resource totals: BC init vs PPO 100K")
    fig.tight_layout()
    p = OUT / "figures" / "041_05_resource_totals_bcinit_vs_ppo100k.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    figures.append(p.relative_to(ROOT).as_posix())

    stages = ["D1_30", "D31_60", "D61_90", "D91_plus"]
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    for resource, prefix in [("irrigation", "irrigation"), ("nitrogen", "nitrogen")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
        for ax, label in zip(axes, ["bc_init", "ppo100k"]):
            sub = event_summary[event_summary["policy_label"] == label].set_index("year").reindex(YEARS)
            bottom = np.zeros(len(YEARS))
            for stage, color in zip(stages, colors):
                vals = sub[f"{prefix}_{stage}"].fillna(0).to_numpy()
                ax.bar(x, vals, bottom=bottom, label=stage, color=color)
                bottom += vals
            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels(years, rotation=45)
            ax.set_ylabel(resource)
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.suptitle(f"041_05 {resource} stage allocation")
        fig.tight_layout()
        p = OUT / "figures" / f"041_05_{resource}_stage_allocation.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        figures.append(p.relative_to(ROOT).as_posix())

    return figures


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number", "bool"]).columns:
        if work[col].dtype == bool:
            continue
        work[col] = pd.to_numeric(work[col], errors="coerce").round(3)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    if len(df) > max_rows:
        lines.append(f"\n仅显示前 {max_rows} 行，共 {len(df)} 行。")
    return "\n".join(lines)


def write_record(result: dict[str, object], metric_delta: pd.DataFrame, event_summary: pd.DataFrame) -> None:
    ppo = metric_delta
    lines = [
        "# 041_05 SYA lowIC teacher warm-start 漂移审计记录",
        "",
        "## 结论",
        "",
        "- 本任务不训练、不重新运行 DSSAT，只读取 041_04 已有评估与日值表。",
        "- 目标是解释：为什么 041_04 的 PPO fine-tune 相比 BC init 产量下降、PFP_N 上升。",
        f"- 输出分支：`{result['branch']}`",
        "",
        "## 主要数字",
        "",
        f"- 100K 相比 BC init，平均产量变化：{ppo['delta_grain_yield_kg_ha'].mean():.1f} kg/ha。",
        f"- 100K 相比 BC init，平均 WP_ET 变化：{ppo['delta_WP_ET_kg_m3'].mean():.3f} kg/m3。",
        f"- 100K 相比 BC init，平均 PFP_N 变化：{ppo['delta_PFP_N_kg_kg'].mean():.3f} kg/kg。",
        f"- 100K 相比 BC init，平均灌溉变化：{ppo['delta_summary_irrigation_total'].mean():.1f} mm。",
        f"- 100K 相比 BC init，平均施氮变化：{ppo['delta_summary_nitrogen_total'].mean():.1f} kg/ha。",
        "",
        "## 每年指标变化：PPO 100K - BC init",
        "",
        md_table(
            metric_delta[
                [
                    "year",
                    "delta_grain_yield_kg_ha",
                    "delta_WP_ET_kg_m3",
                    "delta_PFP_N_kg_kg",
                    "delta_summary_irrigation_total",
                    "delta_summary_nitrogen_total",
                    "ppo100k_all3_win_vs_four_max",
                ]
            ],
            20,
        ),
        "",
        "## 每年管理总量与阶段分配",
        "",
        md_table(event_summary, 25),
        "",
        "## 输出文件",
        "",
    ]
    for key, value in result["outputs"].items():
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {key}: `{item}`")
        else:
            lines.append(f"- {key}: `{value}`")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    metric_delta = build_metric_delta()
    event_summary, events = build_event_tables()
    metric_delta_path = OUT / "tables" / "041_05_metric_delta_ppo100k_minus_bcinit.csv"
    event_summary_path = OUT / "tables" / "041_05_event_stage_summary.csv"
    events_path = OUT / "tables" / "041_05_management_events_bcinit_vs_ppo100k.csv"
    metric_delta.to_csv(metric_delta_path, index=False)
    event_summary.to_csv(event_summary_path, index=False)
    events.to_csv(events_path, index=False)
    figures = write_figures(metric_delta, event_summary)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": "A_drift_audit_completed",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "years": YEARS,
        "outputs": {
            "metric_delta": metric_delta_path.relative_to(ROOT).as_posix(),
            "event_summary": event_summary_path.relative_to(ROOT).as_posix(),
            "events": events_path.relative_to(ROOT).as_posix(),
            "figures": figures,
            "record_md": DOC.relative_to(ROOT).as_posix(),
        },
    }
    (OUT / "041_05_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_record(result, metric_delta, event_summary)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
