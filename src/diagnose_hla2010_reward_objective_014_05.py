from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004"
RUN_ROOT = ROOT / "hla2010_dqn_unified_recheck_014_03" / "2010"
OUT_DIR = ROOT / "hla2010_reward_objective_diagnosis_014_05"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_014_05_hla2010_reward_objective_diagnosis_record.md"

WATER_COST = 1.0
NITROGEN_COST = 5.0

TRAJECTORIES = {
    "dqn_free_daily_200step_smoke": RUN_ROOT / "free_daily_seed0_200steps" / "dqn_eval_daily.csv",
    "dqn_free_daily_5000step": RUN_ROOT / "free_daily_seed0_5000steps" / "dqn_eval_daily.csv",
    "dqn_agronomic_window_200step_smoke": RUN_ROOT / "agronomic_window_seed0_200steps" / "dqn_eval_daily.csv",
}

BASELINE_SUMMARY = ROOT / "hla_2010_2015_four_scenario_with_ppo" / "hla_2010_2015_four_scenario_summary.csv"


def read_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["grnwt", "topwt", "safe_amir", "safe_anfer", "reward", "swfac", "nstres"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarize_dqn(label: str, path: Path, null_yield: float) -> dict[str, float | str]:
    df = read_daily(path)
    final_yield = float(df["grnwt"].dropna().iloc[-1])
    final_biomass = float(df["topwt"].dropna().iloc[-1])
    irrigation = float(df["safe_amir"].sum())
    nitrogen = float(df["safe_anfer"].sum())
    yield_gain_vs_null = final_yield - null_yield
    water_cost_term = WATER_COST * irrigation
    nitrogen_cost_term = NITROGEN_COST * nitrogen
    current_net = final_yield - water_cost_term - nitrogen_cost_term
    net_gain_vs_null = current_net - null_yield
    break_even_n_cost = np.nan
    if nitrogen > 0:
        break_even_n_cost = (yield_gain_vs_null - WATER_COST * irrigation) / nitrogen
    break_even_w_cost = np.nan
    if irrigation > 0:
        break_even_w_cost = (yield_gain_vs_null - NITROGEN_COST * nitrogen) / irrigation
    return {
        "case": label,
        "type": "DQN trajectory",
        "final_yield": final_yield,
        "final_biomass": final_biomass,
        "irrigation": irrigation,
        "nitrogen": nitrogen,
        "yield_gain_vs_null": yield_gain_vs_null,
        "water_cost_term": water_cost_term,
        "nitrogen_cost_term": nitrogen_cost_term,
        "current_net_objective": current_net,
        "net_gain_vs_null_under_current_reward": net_gain_vs_null,
        "break_even_n_cost_given_water_cost_1": break_even_n_cost,
        "break_even_w_cost_given_n_cost_5": break_even_w_cost,
        "max_water_stress": float(df["swfac"].max()),
        "max_nitrogen_stress": float(df["nstres"].max()),
    }


def baseline_rows(null_yield: float) -> list[dict[str, float | str]]:
    if not BASELINE_SUMMARY.exists():
        return []
    df = pd.read_csv(BASELINE_SUMMARY)
    df = df[(df["requested_year"].astype(int) == 2010)].copy()
    rows: list[dict[str, float | str]] = []
    for _, r in df.iterrows():
        scenario = str(r["scenario"]) if pd.notna(r["scenario"]) else "null"
        final_yield = float(r["final_gwad"])
        irrigation = float(r["irrigation_total"])
        nitrogen = float(r["fertilizer_total"])
        yield_gain_vs_null = final_yield - null_yield
        water_cost_term = WATER_COST * irrigation
        nitrogen_cost_term = NITROGEN_COST * nitrogen
        current_net = final_yield - water_cost_term - nitrogen_cost_term
        rows.append(
            {
                "case": f"baseline_{scenario}",
                "type": "baseline summary",
                "final_yield": final_yield,
                "final_biomass": float(r["final_cwad"]),
                "irrigation": irrigation,
                "nitrogen": nitrogen,
                "yield_gain_vs_null": yield_gain_vs_null,
                "water_cost_term": water_cost_term,
                "nitrogen_cost_term": nitrogen_cost_term,
                "current_net_objective": current_net,
                "net_gain_vs_null_under_current_reward": current_net - null_yield,
                "break_even_n_cost_given_water_cost_1": np.nan if nitrogen <= 0 else (yield_gain_vs_null - irrigation) / nitrogen,
                "break_even_w_cost_given_n_cost_5": np.nan if irrigation <= 0 else (yield_gain_vs_null - 5.0 * nitrogen) / irrigation,
                "max_water_stress": float(r["max_wspd"]),
                "max_nitrogen_stress": float(r["max_nstd"]),
            }
        )
    return rows


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无数据。_"
    show = df.copy()
    keep = [
        "case",
        "final_yield",
        "irrigation",
        "nitrogen",
        "yield_gain_vs_null",
        "water_cost_term",
        "nitrogen_cost_term",
        "current_net_objective",
        "net_gain_vs_null_under_current_reward",
        "break_even_n_cost_given_water_cost_1",
    ]
    show = show[[c for c in keep if c in show.columns]]
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        else:
            show[col] = show[col].astype(str)
    lines = [
        "| " + " | ".join(show.columns) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_doc(df: pd.DataFrame, null_yield: float) -> None:
    sorted_df = df.sort_values("current_net_objective", ascending=False)
    best = sorted_df.iloc[0]
    lines = [
        "# 014_05 HLA2010 奖励目标诊断记录",
        "",
        "## 目的",
        "",
        "诊断 HLA2010 DQN 5K 退化为 no-op，是否是当前经济奖励函数下的理性结果。",
        "",
        "当前奖励口径：",
        "",
        "```text",
        "reward = ΔGRNWT - 1.0 × irrigation - 5.0 × nitrogen",
        "```",
        "",
        f"null 参考产量：{null_yield:.2f} kg/ha。",
        "",
        "## 重评分结果",
        "",
        df_to_md(sorted_df),
        "",
        "## 关键结论",
        "",
        f"- 在当前 water_cost=1、nitrogen_cost=5 的口径下，最高 current_net_objective 是 `{best['case']}`，数值为 {best['current_net_objective']:.2f}。",
        "- HLA2010 的 5K DQN 选择 no-op，并不是动作链路故障；从当前经济奖励看，no-op 确实比 DQN smoke 的施氮轨迹更划算。",
        "- 200 step smoke 的 `free_daily` 轨迹虽然产量高约 897 kg/ha，但用了 120 mm 水和 200 kg/ha 氮；在 N cost=5 下，氮成本为 1000，已经吃掉全部增产收益。",
        "- 要让 `free_daily 200step` 这种 I120/N200 轨迹相对 no-op 不亏，在 water_cost=1 下，氮成本需要低于约 3.89 kg grain/kg N。",
        "- 因此 HLA 当前问题首先是奖励目标定义问题，不是继续加 seed 或继续训练能解决的问题。",
        "",
        "## 对下一步的含义",
        "",
        "- 如果论文目标是经济净收益最大化，那么 HLA2010 no-op 是合理结果，不能强行说 DQN 失败。",
        "- 如果论文目标是约束内产量最大化或节水节氮下保产，那么当前 `ΔGRNWT - 水氮成本` 奖励不适合 HLA，需要统一修改所有站点的奖励口径，而不能只给 HLA 特调。",
        "- 下一步应先和导师确定主目标：经济净收益、约束内产量最大化，还是多目标折中。",
        "",
        "## 输出",
        "",
        f"- `{(OUT_DIR / '014_05_hla2010_reward_rescore_summary.csv').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(BASELINE_SUMMARY)
    null_row = baseline[(baseline["requested_year"].astype(int) == 2010) & (baseline["scenario"].isna())]
    if null_row.empty:
        null_row = baseline[(baseline["requested_year"].astype(int) == 2010) & (baseline["scenario"].astype(str).str.lower() == "null")]
    null_yield = float(null_row["final_gwad"].iloc[0])
    rows = baseline_rows(null_yield)
    for label, path in TRAJECTORIES.items():
        rows.append(summarize_dqn(label, path, null_yield))
    df = pd.DataFrame(rows)
    df = df.sort_values("current_net_objective", ascending=False).reset_index(drop=True)
    out = OUT_DIR / "014_05_hla2010_reward_rescore_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    write_doc(df, null_yield)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
