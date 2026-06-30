from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HLA_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004"
OUT_DIR = HLA_ROOT / "hla_all_year_screen_and_dqn_transfer_014_02"
FIG_DIR = OUT_DIR / "figures"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_014_02_hla_all_year_screen_and_dqn_transfer_record.md"

SOURCE_CANDIDATE = (
    HLA_ROOT
    / "hla_ic1_yearly_diagnostics_2004_2023"
    / "hla_original_ic1_null_vs_auto_candidate_years.csv"
)
SOURCE_YEARLY_SUMMARY = (
    HLA_ROOT / "hla_ic1_yearly_diagnostics_2004_2023" / "hla_ic1_yearly_summary.csv"
)
SOURCE_FOUR_SCENARIO = (
    HLA_ROOT
    / "hla_2010_2015_four_scenario_with_ppo"
    / "hla_2010_2015_four_scenario_summary.csv"
)
SOURCE_HLA2010_DQN = (
    HLA_ROOT
    / "hla2010_dqn_economic_reward_probe_012_03"
    / "hla2010_dqn_economic_compare_summary.csv"
)
SOURCE_HLA2015_DQN = (
    HLA_ROOT
    / "hla2010_dqn_economic_reward_probe_012_03"
    / "figures_012_05_hla2015"
    / "hla2015_economic_dqn_four_scenario_summary.csv"
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt_float(x: object, digits: int = 1) -> str:
    try:
        val = float(x)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_无数据。_"
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda v: fmt_float(v, 1))
        else:
            show[col] = show[col].astype(str)
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in show.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_candidate_table(candidate: pd.DataFrame) -> pd.DataFrame:
    df = candidate.copy()
    numeric_cols = [
        "year",
        "null_gwad",
        "auto_gwad",
        "yield_gain",
        "auto_irrig",
        "null_max_wspd",
        "null_max_nstd",
        "rain",
        "score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["valid_for_training_screen"] = (
        df["null_gwad"].gt(0)
        & df["auto_gwad"].gt(0)
        & df["yield_gain"].gt(200)
        & df["null_max_wspd"].gt(0.2)
    )
    df["exclude_reason"] = ""
    df.loc[df["null_gwad"].le(0), "exclude_reason"] += "null产量为0或失败；"
    df.loc[df["auto_gwad"].le(0), "exclude_reason"] += "auto产量为0或失败；"
    df.loc[df["yield_gain"].le(200), "exclude_reason"] += "auto相对null增产不足200；"
    df.loc[df["null_max_wspd"].le(0.2), "exclude_reason"] += "null水分胁迫不明显；"
    df.loc[df["valid_for_training_screen"], "exclude_reason"] = "候选"
    df = df.sort_values(["valid_for_training_screen", "yield_gain"], ascending=[False, False])
    cols = [
        "year",
        "null_gwad",
        "auto_gwad",
        "yield_gain",
        "auto_irrig",
        "null_max_wspd",
        "null_max_nstd",
        "rain",
        "valid_for_training_screen",
        "exclude_reason",
    ]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def build_existing_dqn_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    four = read_csv(SOURCE_FOUR_SCENARIO)
    if not four.empty:
        for _, row in four.iterrows():
            rows.append(
                {
                    "year": int(row["requested_year"]),
                    "source": "existing_four_scenario_ppo_or_baseline",
                    "case": row["scenario"],
                    "yield_kg_ha": row.get("final_gwad"),
                    "biomass_kg_ha": row.get("final_cwad"),
                    "irrigation_mm": row.get("irrigation_total"),
                    "nitrogen_kg_ha": row.get("fertilizer_total"),
                    "note": "HLA 2010/2015 旧四情景结果，PPO为旧策略/旧线索，不作为当前DQN结论",
                }
            )

    dqn2010 = read_csv(SOURCE_HLA2010_DQN)
    if not dqn2010.empty:
        for _, row in dqn2010.iterrows():
            rows.append(
                {
                    "year": 2010,
                    "source": "existing_hla2010_economic_dqn_012_03",
                    "case": row.get("case"),
                    "yield_kg_ha": row.get("yield"),
                    "biomass_kg_ha": row.get("daily_topwt"),
                    "irrigation_mm": row.get("I"),
                    "nitrogen_kg_ha": row.get("N"),
                    "note": "已有HLA2010经济奖励DQN；seed0 5K曾达到DSSAT auto同量级，但仍需统一新流程复核",
                }
            )

    dqn2015 = read_csv(SOURCE_HLA2015_DQN)
    if not dqn2015.empty:
        for _, row in dqn2015.iterrows():
            rows.append(
                {
                    "year": 2015,
                    "source": "existing_hla2015_economic_dqn_012_05",
                    "case": row.get("scenario"),
                    "yield_kg_ha": row.get("yield_kg_ha"),
                    "biomass_kg_ha": row.get("biomass_kg_ha"),
                    "irrigation_mm": row.get("irrigation_mm"),
                    "nitrogen_kg_ha": row.get("nitrogen_kg_ha"),
                    "note": "已有HLA2015经济DQN结果：seed0未优于null，说明2015当前设置下不稳",
                }
            )
    return pd.DataFrame(rows)


def plot_candidate_yield(candidate: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]})
    ordered = candidate.sort_values("year")
    x = np.arange(len(ordered))
    width = 0.38
    axes[0].bar(x - width / 2, ordered["null_gwad"], width, color="#BFC5CF", edgecolor="#333333", label="Null")
    axes[0].bar(x + width / 2, ordered["auto_gwad"], width, color="#6F95D8", edgecolor="#333333", label="DSSAT auto irrigation")
    axes[0].set_ylabel("GWAD / kg ha$^{-1}$")
    axes[0].set_title("HLA IC=1 yearly screening: null vs DSSAT auto", loc="left")
    axes[0].grid(True, axis="y", color="#E6E8F0")
    axes[0].legend(frameon=False, ncol=2)

    colors = np.where(ordered["valid_for_training_screen"], "#2E7D32", "#A0A4AA")
    axes[1].bar(x, ordered["yield_gain"], color=colors, edgecolor="#333333")
    axes[1].axhline(200, color="#8B0000", linestyle="--", linewidth=1.1, label="Candidate threshold: +200 kg/ha")
    axes[1].set_ylabel("Auto - null")
    axes[1].set_xlabel("Year")
    axes[1].grid(True, axis="y", color="#E6E8F0")
    axes[1].legend(frameon=False, loc="upper right")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(int(y)) for y in ordered["year"]], rotation=45)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "014_02_hla_null_auto_yield_gain.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_doc(candidate: pd.DataFrame, existing_dqn: pd.DataFrame, fig_path: Path) -> None:
    selected = candidate[candidate["valid_for_training_screen"]].copy()
    top = selected.sort_values("yield_gain", ascending=False).head(8)
    dqn_focus = existing_dqn[existing_dqn["year"].isin([2010, 2015])].copy()
    lines = [
        "# 014_02 HLA 全年份筛选与 DQN 迁移准备记录",
        "",
        "## 本轮目标",
        "",
        "在不重新训练、不重新运行 DSSAT 的前提下，整理 HLA 2004–2023 已有 IC=1 null/auto 诊断结果，判断海伦站是否还有值得继续做 DQN 的年份。",
        "",
        "## 输入",
        "",
        f"- 年份筛选来源：`{SOURCE_CANDIDATE.relative_to(PROJECT_ROOT)}`",
        f"- 年度日值来源：`{SOURCE_YEARLY_SUMMARY.relative_to(PROJECT_ROOT)}`",
        f"- 旧四情景来源：`{SOURCE_FOUR_SCENARIO.relative_to(PROJECT_ROOT)}`",
        f"- HLA2010 DQN 来源：`{SOURCE_HLA2010_DQN.relative_to(PROJECT_ROOT)}`",
        f"- HLA2015 DQN 来源：`{SOURCE_HLA2015_DQN.relative_to(PROJECT_ROOT)}`",
        "",
        "## 筛选规则",
        "",
        "- null 与 auto 均需产生非零产量。",
        "- DSSAT auto 相对 null 增产超过 200 kg/ha，说明该年份存在低成本可见的管理响应空间。",
        "- null 最大水分胁迫大于 0.2，避免选择几乎没有水分管理需求的年份。",
        "- 2004、2012 这类 null 或 auto 明显失败/极端异常年份不作为第一批训练候选。",
        "",
        "## 候选年份排序",
        "",
        dataframe_to_markdown(top),
        "",
        "## 全部年份筛选表",
        "",
        dataframe_to_markdown(candidate),
        "",
        "## 已有 HLA DQN/四情景结果整理",
        "",
        dataframe_to_markdown(dqn_focus),
        "",
        "## 初步结论",
        "",
        "- HLA 仍然有可探索年份，但不建议回到 2004；2004 null 产量为 0，太极端，不适合作为第一批 DQN 训练验证。",
        "- 按已有 IC=1 null/auto 诊断，优先候选是 2007、2015、2010，其次是 2016、2022。",
        "- 已有 HLA2010 经济奖励 DQN seed0 5K 曾达到 DSSAT auto 同量级，并且用水更少、施氮 50 kg/ha；这说明 HLA 不是完全没有希望。",
        "- 已有 HLA2015 经济奖励 DQN seed0 结果接近 null，说明 2015 对当前 DQN 设置不稳定，不能直接当成功案例。",
        "- 下一步如果继续 HLA，建议先选择 2010 做统一新流程复核；如果 2010 复核成立，再跑 2007 或 2016，而不是直接多年份大训练。",
        "",
        "## 图件",
        "",
        f"- `{fig_path.relative_to(PROJECT_ROOT)}`",
        "",
        "## 输出文件",
        "",
        f"- `{(OUT_DIR / '014_02_hla_candidate_year_screen_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- `{(OUT_DIR / '014_02_hla_existing_dqn_summary.csv').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_raw = read_csv(SOURCE_CANDIDATE)
    if candidate_raw.empty:
        raise FileNotFoundError(f"缺少年份筛选输入：{SOURCE_CANDIDATE}")
    candidate = build_candidate_table(candidate_raw)
    existing_dqn = build_existing_dqn_summary()
    candidate.to_csv(OUT_DIR / "014_02_hla_candidate_year_screen_summary.csv", index=False, encoding="utf-8-sig")
    existing_dqn.to_csv(OUT_DIR / "014_02_hla_existing_dqn_summary.csv", index=False, encoding="utf-8-sig")
    fig_path = plot_candidate_yield(candidate)
    write_doc(candidate, existing_dqn, fig_path)
    print("候选年份：")
    print(candidate[candidate["valid_for_training_screen"]].head(8).to_string(index=False))
    print(f"\n图件：{fig_path}")
    print(f"记录：{DOC_PATH}")


if __name__ == "__main__":
    main()
