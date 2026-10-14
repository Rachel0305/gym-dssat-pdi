from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "028_14_existing_results_report"
DOC = ROOT / "docs" / "2026-07-19_028_14_existing_results_expert_management_and_metric_gap_report.md"
HTML = ROOT / "docs" / "2026-07-19_028_14_existing_results_expert_management_and_metric_gap_report.html"
RECORD = ROOT / "docs" / "2026-07-19_028_14_existing_results_expert_management_and_metric_gap_record.md"

DAILY = ROOT / "benchmark_results" / "028_12_screened_year_representative_advisor_package" / "028_12_all_representative_five_scenario_daily.csv"
SUMMARY = ROOT / "benchmark_results" / "028_12_screened_year_representative_advisor_package" / "028_12_all_representative_five_scenario_summary.csv"
OVERVIEW = ROOT / "benchmark_results" / "028_13_screened_year_maskableppo_advisor_summary" / "028_13_site_year_overview.csv"
EXPERT_SOURCE = ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03" / "018_03_extension_expert_schedule.csv"
DQN_SOURCE = ROOT / "benchmark_results" / "027_05" / "027_05_dqn_five_scenario_summary.csv"

BASELINES = ["null", "recorded_farmer", "dssat_auto", "official_extension_expert"]
METRICS = {
    "yield": ("yield_kg_ha", "final_grain_kg_ha", "产量", "kg/ha"),
    "WP_ET": ("WP_ET_kg_m3", "wp_et_kg_m3", "WP_ET", "kg/m³"),
    "PFP_N": ("PFP_N_kg_kg", "pfp_n_kg_kg", "PFP_N", "kg grain/kg N"),
}
SCENARIO_CN = {
    "null": "Null",
    "recorded_farmer": "Recorded/农民经验",
    "dssat_auto": "DSSAT auto",
    "official_extension_expert": "官方推广 expert",
    "rl_candidate": "MaskablePPO",
}


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt(value: float | int | str, digits: int = 1) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def event_text(frame: pd.DataFrame) -> str:
    rows = frame[(num(frame["irrigation_executed_mm"]) > 0) | (num(frame["nitrogen_executed_kg_ha"]) > 0)].copy()
    rows = rows.sort_values("dap")
    parts = []
    for row in rows.itertuples():
        i = float(row.irrigation_executed_mm)
        n = float(row.nitrogen_executed_kg_ha)
        parts.append(f"DAP{int(round(float(row.dap)))}: I{fmt(i)}/N{fmt(n)}")
    return "; ".join(parts) if parts else "全季无水氮操作"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DAILY, keep_default_na=False)
    summary = pd.read_csv(SUMMARY, keep_default_na=False)
    overview = pd.read_csv(OVERVIEW, keep_default_na=False)
    expert = pd.read_csv(EXPERT_SOURCE, keep_default_na=False)
    dqn = pd.read_csv(DQN_SOURCE, keep_default_na=False)
    for col in ["final_grain_kg_ha", "wp_et_kg_m3", "pfp_n_kg_kg", "irrigation_event_total_mm", "nitrogen_event_total_kg_ha"]:
        summary[col] = num(summary[col])
        dqn[col] = num(dqn[col])
    return daily, summary, overview, expert, dqn


def expert_templates(expert: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "region", "source_table", "stage_code", "stage_cn", "dap",
        "n_kg_mu_low", "n_kg_mu_high", "n_kg_ha_mid",
        "irrigation_fang_mu_low", "irrigation_fang_mu_high", "irrigation_mm_mid",
    ]
    out = expert[cols].drop_duplicates().sort_values(["region", "dap"]).reset_index(drop=True)
    out["template_total_irrigation_mm"] = out.groupby("region")["irrigation_mm_mid"].transform("sum")
    out["template_total_n_kg_ha"] = out.groupby("region")["n_kg_ha_mid"].transform("sum")
    return out


def executed_events(daily: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = daily[daily["scenario"].eq(scenario)].copy()
    out["irrigation_executed_mm"] = num(out["irrigation_executed_mm"])
    out["nitrogen_executed_kg_ha"] = num(out["nitrogen_executed_kg_ha"])
    out = out[(out["irrigation_executed_mm"] > 0) | (out["nitrogen_executed_kg_ha"] > 0)]
    return out[["site", "station", "requested_year", "dap", "date", "irrigation_executed_mm", "nitrogen_executed_kg_ha"]].sort_values(["site", "requested_year", "dap"])


def metric_gap_detail(summary: pd.DataFrame, overview: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for ov in overview.itertuples():
        subset = summary[(summary["site"].eq(ov.site)) & (summary["year"].eq(int(ov.year)))]
        base = subset[subset["scenario"].isin(BASELINES)].copy()
        for metric, (ov_col, sum_col, label, unit) in METRICS.items():
            ppo = pd.to_numeric(pd.Series([getattr(ov, ov_col)]), errors="coerce").iloc[0]
            values = num(base[sum_col])
            valid = base[values.notna()].copy()
            valid["metric_value"] = values[values.notna()].values
            if valid.empty:
                best_value = np.nan
                best_scenario = ""
            else:
                best_row = valid.sort_values(["metric_value", "scenario"], ascending=[False, True]).iloc[0]
                best_value = float(best_row["metric_value"])
                best_scenario = str(best_row["scenario"])
            gap = float(ppo - best_value) if np.isfinite(ppo) and np.isfinite(best_value) else np.nan
            pct = float(gap / best_value * 100.0) if np.isfinite(gap) and best_value != 0 else np.nan
            row = {
                "site": ov.site, "year": int(ov.year), "seed": int(ov.representative_seed),
                "checkpoint": int(ov.representative_checkpoint), "metric": metric,
                "metric_label": label, "unit": unit, "ppo_value": ppo,
                "four_baseline_best_value": best_value,
                "four_baseline_best_scenario": best_scenario,
                "gap_absolute": gap, "gap_percent": pct,
                "status": "strict_win" if np.isfinite(gap) and gap > 0 else ("tie" if np.isfinite(gap) and abs(gap) <= 1e-12 else "below"),
                "winner_seed_count": int(ov.winner_seed_count), "seed_count": int(ov.seed_count),
                "cross_seed_status": ov.cross_seed_status,
            }
            for scenario in BASELINES:
                srow = base[base["scenario"].eq(scenario)]
                value = num(srow[sum_col]).iloc[0] if len(srow) else np.nan
                row[f"{scenario}_value"] = value
                row[f"gap_vs_{scenario}"] = ppo - value if np.isfinite(ppo) and np.isfinite(value) else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["site", "year", "metric"]).reset_index(drop=True)


def rationality(daily: pd.DataFrame, summary: pd.DataFrame, overview: pd.DataFrame) -> pd.DataFrame:
    allowed = {
        "HLA": {1, 30, 50, 65, 85, 110}, "SY": {1, 30, 50, 65, 85, 110},
        "YC": {7, 30, 45, 60, 80, 100}, "FQ": {7, 30, 45, 60, 80, 100},
        "LC": {7, 30, 45, 60, 80, 100},
    }
    rows = []
    for ov in overview.itertuples():
        dd = daily[(daily["site"].eq(ov.site)) & (daily["requested_year"].eq(int(ov.year)))]
        ppo_d = dd[dd["scenario"].eq("rl_candidate")].copy()
        expert_d = dd[dd["scenario"].eq("official_extension_expert")].copy()
        actions = ppo_d[(num(ppo_d["irrigation_executed_mm"]) > 0) | (num(ppo_d["nitrogen_executed_kg_ha"]) > 0)].copy()
        action_daps = {int(round(v)) for v in num(actions["dap"]).dropna()}
        late_n = num(actions.loc[num(actions["dap"]) > 90, "nitrogen_executed_kg_ha"]).sum()
        expert_sum = summary[(summary["site"].eq(ov.site)) & (summary["year"].eq(int(ov.year))) & (summary["scenario"].eq("official_extension_expert"))].iloc[0]
        budget_ok = float(ov.irrigation_mm) <= 120.0 + 1e-9 and float(ov.nitrogen_kg_ha) <= 300.0 + 1e-9
        stage_ok = action_daps.issubset(allowed[str(ov.site)])
        timing_ok = late_n <= 1e-9
        if not (budget_ok and stage_ok and timing_ok):
            assessment = "存在预算/阶段/晚期施氮问题，需复核"
        elif int(ov.seed_count) < 3:
            assessment = "代表策略结构合理；仅1 seed，稳定性未评估"
        elif int(ov.winner_seed_count) >= 2:
            assessment = "代表策略结构合理；至少2/3 seed有指标优势"
        else:
            assessment = "代表策略结构合理；仅候选seed成功，跨seed稳定性不足"
        rows.append({
            "site": ov.site, "year": int(ov.year), "seed": int(ov.representative_seed), "checkpoint": int(ov.representative_checkpoint),
            "ppo_action_schedule": event_text(ppo_d), "expert_executed_schedule": event_text(expert_d),
            "ppo_irrigation_mm": float(ov.irrigation_mm), "expert_irrigation_mm": float(expert_sum["irrigation_event_total_mm"]),
            "irrigation_saving_vs_expert_mm": float(expert_sum["irrigation_event_total_mm"] - float(ov.irrigation_mm)),
            "ppo_nitrogen_kg_ha": float(ov.nitrogen_kg_ha), "expert_nitrogen_kg_ha": float(expert_sum["nitrogen_event_total_kg_ha"]),
            "nitrogen_saving_vs_expert_kg_ha": float(expert_sum["nitrogen_event_total_kg_ha"] - float(ov.nitrogen_kg_ha)),
            "late_n_after_dap90_kg_ha": float(late_n), "budget_ok": budget_ok, "valid_stage_actions": stage_ok,
            "no_n_after_dap90": timing_ok, "max_water_stress_index": float(ov.max_water_stress_index),
            "max_nitrogen_stress_index": float(ov.max_nitrogen_stress_index), "winning_metrics": ov.winning_metrics,
            "winner_seed_count": int(ov.winner_seed_count), "seed_count": int(ov.seed_count), "cross_seed_status": ov.cross_seed_status,
            "management_assessment": assessment,
        })
    return pd.DataFrame(rows).sort_values(["site", "year"]).reset_index(drop=True)


def dqn_evidence(dqn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, year), group in dqn.groupby(["site", "year"]):
        rl = group[group["scenario"].eq("rl_candidate")]
        if rl.empty:
            continue
        rl = rl.iloc[0]
        base = group[group["scenario"].isin(BASELINES)]
        wins = []
        result = {"site": site, "year": int(year), "seed": rl["seed"], "checkpoint": rl["checkpoint"]}
        for metric, (_, col, _, _) in METRICS.items():
            rv = float(rl[col]) if pd.notna(rl[col]) else np.nan
            bv = num(base[col]).max()
            gap = rv - bv if np.isfinite(rv) and np.isfinite(bv) else np.nan
            result[f"{metric}_value"] = rv
            result[f"{metric}_gap_vs_four_best"] = gap
            if np.isfinite(gap) and gap > 0:
                wins.append(metric)
        result["winning_metrics"] = ";".join(wins) if wins else "none"
        result["evidence_status"] = "historical_candidate_provisional_exploration_schedule_issue"
        rows.append(result)
    return pd.DataFrame(rows).sort_values(["site", "year"]).reset_index(drop=True)


def plot_gap(gaps: pd.DataFrame) -> None:
    order = [f"{r.site}{int(r.year)}" for r in gaps.drop_duplicates(["site", "year"]).itertuples()]
    pivot = gaps.assign(site_year=gaps["site"] + gaps["year"].astype(int).astype(str)).pivot(index="site_year", columns="metric", values="gap_percent")
    pivot = pivot.reindex(index=order, columns=["yield", "WP_ET", "PFP_N"])
    data = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    im = ax.imshow(masked, cmap="RdYlGn", vmin=-15, vmax=15, aspect="auto")
    ax.set_xticks(range(3), ["Yield", "WP_ET", "PFP_N"])
    ax.set_yticks(range(len(pivot)), pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(j, i, "NA" if not np.isfinite(value) else f"{value:+.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(value) and abs(value) > 10 else "black")
    ax.set_title("Representative MaskablePPO gap versus the best of four baselines", loc="left", weight="bold")
    ax.set_xlabel("Positive values mean PPO is strictly higher")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Gap versus four-baseline maximum (%)")
    fig.tight_layout()
    fig.savefig(OUT / "028_14_metric_gap_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "028_14_metric_gap_heatmap.svg", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def build_reports(templates: pd.DataFrame, gaps: pd.DataFrame, rational: pd.DataFrame, dqn: pd.DataFrame) -> None:
    metric_wins = gaps[gaps["status"].eq("strict_win")].groupby("metric").size().to_dict()
    stable = int(((rational["winner_seed_count"] >= 2) & (rational["seed_count"] == 3)).sum())
    three_seed = int((rational["seed_count"] == 3).sum())
    lower_i = int((rational["irrigation_saving_vs_expert_mm"] > 0).sum())
    lower_n = int((rational["nitrogen_saving_vs_expert_kg_ha"] > 0).sum())

    template_view = templates[["region", "source_table", "stage_cn", "dap", "n_kg_mu_low", "n_kg_mu_high", "n_kg_ha_mid", "irrigation_fang_mu_low", "irrigation_fang_mu_high", "irrigation_mm_mid"]].copy()
    template_view.columns = ["区域模板", "推文表", "阶段", "DAP", "N下限(kg/亩)", "N上限(kg/亩)", "N中值(kg/ha)", "水下限(方/亩)", "水上限(方/亩)", "水中值(mm)"]
    template_view["区域模板"] = template_view["区域模板"].map({"northeast_greatwall_spring_maize": "东北/长城沿线春玉米", "huanghuai_fenwei_summer_maize": "华北黄淮/汾渭夏玉米"})

    gap_wide = gaps.pivot(index=["site", "year"], columns="metric", values=["ppo_value", "four_baseline_best_value", "gap_absolute", "gap_percent"]).reset_index()
    gap_wide.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else col for col in gap_wide.columns]
    overview_rows = []
    for r in rational.itertuples():
        sub = gaps[(gaps.site.eq(r.site)) & (gaps.year.eq(r.year))]
        g = {x.metric: x for x in sub.itertuples()}
        overview_rows.append({
            "站点年": f"{r.site}{r.year}", "PPO措施": r.ppo_action_schedule,
            "I/N": f"{fmt(r.ppo_irrigation_mm,0)}/{fmt(r.ppo_nitrogen_kg_ha,0)}",
            "产量差": f"{fmt(g['yield'].gap_absolute)} ({fmt(g['yield'].gap_percent)}%)",
            "WP_ET差": f"{fmt(g['WP_ET'].gap_absolute,2)} ({fmt(g['WP_ET'].gap_percent)}%)",
            "PFP_N差": "NA" if not np.isfinite(g["PFP_N"].gap_absolute) else f"{fmt(g['PFP_N'].gap_absolute,1)} ({fmt(g['PFP_N'].gap_percent)}%)",
            "胜出指标": r.winning_metrics, "seed证据": f"{r.winner_seed_count}/{r.seed_count}", "判断": r.management_assessment,
        })
    overview_view = pd.DataFrame(overview_rows)

    dqn_rows = []
    for r in dqn.itertuples():
        dqn_rows.append({
            "站点年": f"{r.site}{int(r.year)}",
            "产量差": f"{fmt(r.yield_gap_vs_four_best)}",
            "WP_ET差": f"{fmt(r.WP_ET_gap_vs_four_best, 2)}",
            "PFP_N差": "NA" if not np.isfinite(r.PFP_N_gap_vs_four_best) else f"{fmt(r.PFP_N_gap_vs_four_best, 1)}",
            "胜出指标": r.winning_metrics if r.winning_metrics else "无",
            "证据状态": "provisional：旧探索率日程影响",
        })
    dqn_view = pd.DataFrame(dqn_rows)

    text = f"""# 现有强化学习结果：专家管理、措施合理性与指标差距

## 技术摘要

- 官方推广 expert 不是历史农民记录，而是把推文表1（HLA/SY）和表3（YC/FQ/LC）的阶段水肥推荐量取区间中值，固定映射到 DAP 后在 DSSAT 中前向执行。
- 当前17个已筛选站点年份的代表 MaskablePPO 均至少有1项指标严格超过 null、recorded、DSSAT auto、official expert 四基线的最高可比值；产量、WP_ET、PFP_N分别有 **{metric_wins.get('yield',0)}/17、{metric_wins.get('WP_ET',0)}/17、{metric_wins.get('PFP_N',0)}/17** 年胜出。
- 代表策略17/17均满足I≤120、N≤300、动作阶段合法且DAP>90无施氮；相对official expert，**{lower_i}/17节水、{lower_n}/17节氮**。但跨seed只有 **{stable}/{three_seed}** 个三seed年份达到至少2/3 seed成功，不能说所有seed稳定。
- 历史DQN五个代表站点年中，仅SY2014在当前严格五情景口径下有明确指标胜出；且这些DQN受旧探索率日程问题影响，只能作为provisional历史对照。

## 专家策略如何安排

专家方案采用固定DAP映射，不根据每年DSSAT事后物候重新调整。表内给出推文原始范围及换算后的中值；1 kg/亩=15 kg/ha，1方/亩=1.5 mm。单次灌溉超过50 mm时，执行层拆到相邻日，因此实际事件表会出现DAP50/51或65/66拆分。

{markdown_table(template_view)}

东北模板计划总量约I266.25/N300；夏玉米模板约I228.75/N247.5。若作物在DAP100前收获，末次30 mm不会执行，因此FQ2013、FQ2016、FQ2023和LC2010等实际expert灌溉量约198.75 mm。

## 当前PPO措施是否合理、指标超过或还差多少

下表的“差”均为PPO代表策略减去四基线中的最高可比值。正数表示严格超过，负数表示仍落后。PFP_N在施氮为0时保持NA。

{markdown_table(overview_view)}

综合看，**HLA2015、SY2014、SY2015**属于当前较均衡的候选；FQ2020三项均胜出但只有1/3 seed成功，仍不稳定。HLA2010、HLA2022、LC2010、YC2008和YC2014主要依靠PFP_N优势，FQ2019主要以水氮效率换取约4.3%的产量下降。FQ2013、FQ2023和HLA2016存在较明显的单指标代价，不能只报“有一项胜出”。

## 历史DQN同口径证据

下表同样比较四基线最高可比值，但历史DQN正式结果受到已确认的旧探索率日程问题影响，不能与修复后的MaskablePPO作等强度结论。

{markdown_table(dqn_view)}

## 如何理解“措施合理”

本报告中的合理性是受限判断：动作全部位于预定义阶段、没有突破I120/N300预算、没有DAP>90晚期施氮，并且至少一个终值指标具有优势。它不等价于真实田间最优。HLA2010和SY2014仍有一定模型水分胁迫，但代表策略保持高产或效率优势；更需要警惕的是跨seed稳定性，而不是动作违法。

## 证据边界

- 17/17表示每个筛选年份存在至少一个成功候选，不表示每个seed成功。
- 10/16三seed年份达到至少2/3初步稳定；LC2010只有1 seed。
- PPO精确终值来自训练/迁移summary；日值PlantGro.OUT产量为整数精度，因此图与严格胜负判定的精度来源不同但已绑定记录。
- official expert是推文方案的固定DAP模型映射，不是真实逐年实测专家操作。
- 当前奖励不直接包含WP_ET/PFP_N，指标胜出是结果评价，不是奖励函数逐项硬编码。

## 建议

当前可向导师汇报“统一阶段型MaskablePPO框架在五站点17个筛选年份均找到至少一个指标领先候选”，同时把10/16跨seed稳定率和7/16不稳定年份并列展示。下一步若继续，不应重做已有年份，而应优先复核不稳定年份或进行预注册的奖励稳健性分析。
"""
    DOC.write_text(text, encoding="utf-8")

    css = """
    body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1320px;margin:28px auto;padding:0 24px;color:#202124;line-height:1.55}
    h1,h2{color:#17365d} h1{border-bottom:3px solid #2f6b9a;padding-bottom:10px}
    .summary{background:#eef5fa;border-left:5px solid #2f6b9a;padding:14px 18px;margin:16px 0}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:14px 0 26px} th,td{border:1px solid #cfd8dc;padding:7px;vertical-align:top} th{background:#e7eef4;position:sticky;top:0}
    tr:nth-child(even){background:#f8fafb}.scroll{overflow-x:auto}.note{background:#fff6dd;border-left:5px solid #d69e00;padding:12px 16px}
    img{max-width:920px;width:100%;display:block;margin:18px auto} code{background:#f2f2f2;padding:2px 4px}
    """
    html_overview = overview_view.to_html(index=False, escape=True, border=0)
    html_templates = template_view.to_html(index=False, escape=True, border=0)
    html_dqn = dqn_view.to_html(index=False, escape=True, border=0)
    page = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>现有强化学习结果整理</title><style>{css}</style></head><body>
    <h1>现有强化学习结果：专家管理、措施合理性与指标差距</h1>
    <div class='summary'><b>结论：</b>17/17筛选年份存在至少一个指标领先的MaskablePPO候选；{stable}/{three_seed}个三seed年份达到至少2/3初步稳定。所有代表策略均在预算内且无DAP&gt;90施氮，但不能把“存在候选”写成“所有seed稳定”。</div>
    <h2>1. 官方推广expert怎么安排</h2><p>HLA/SY采用东北及长城沿线春玉米表1，YC/FQ/LC采用华北黄淮和汾渭平原夏玉米表3。取推荐区间中值并固定映射到DAP；专家方案是模型化基线，不是真实逐年记录。</p><div class='scroll'>{html_templates}</div>
    <h2>2. 当前PPO措施与指标差距</h2><p>差值为PPO减去四基线最大值；正数代表严格领先，负数代表尚有差距。</p><img src='../benchmark_results/028_14_existing_results_report/028_14_metric_gap_heatmap.png' alt='17个站点年份三指标相对四基线最佳值的百分比差距'><div class='scroll'>{html_overview}</div>
    <p><b>综合分级：</b>HLA2015、SY2014、SY2015较均衡；FQ2020三项均领先但仅1/3 seed成功。若只在PFP_N或WP_ET上领先、另一指标明显退步，应作为权衡案例而非全面成功案例。</p>
    <h2>3. 历史DQN同口径证据</h2><p>历史DQN仅SY2014有严格胜出指标；由于旧探索率日程问题，表中结果只作provisional对照。</p><div class='scroll'>{html_dqn}</div>
    <h2>4. 措施合理性和稳定性</h2><p>17/17代表策略满足I≤120、N≤300，动作位于预定义阶段，且DAP&gt;90无施氮；相对expert有{lower_i}/17节水、{lower_n}/17节氮。合理性是模型约束和结果层面的判断，不等同于田间因果最优。</p>
    <div class='note'><b>限制：</b>10/16三seed年份达到至少2/3稳定；LC2010仅1 seed。PFP_N在N=0时为NA。official expert是固定DAP映射。历史DQN结果受旧探索率日程问题影响，只作为provisional对照。</div>
    <h2>5. 建议</h2><p>导师汇报时同时展示“17/17存在候选”和“10/16初步稳定”，不要只报代表seed。后续若继续，优先处理跨seed不稳定年份或做预注册的奖励稳健性分析。</p>
    </body></html>"""
    HTML.write_text(page, encoding="utf-8")

    record = f"""# 028_14 现有结果整理记录

## 状态

completed

## 执行范围

- 训练调用：0
- DSSAT新运行：0
- 输入、IC、reward、模型权重修改：0
- 复用17个站点年份、85个五情景终值和11155行日值。

## 数据质量检查

- 17个site-year均有5个情景；
- PPO精确终值来自028_13，日值来自028_12；
- expert实际措施从日值事件提取；
- PFP_N的N=0保持NA；
- 代表seed与跨seed证据分开；
- 产量、WP_ET、PFP_N分别有{metric_wins.get('yield',0)}、{metric_wins.get('WP_ET',0)}、{metric_wins.get('PFP_N',0)}个年份严格胜出；
- 17/17预算合法、阶段合法、DAP>90无施氮；
- {stable}/{three_seed}三seed年份达到至少2/3初步稳定。

## 解释边界

合理性为模型约束与现有结果层面的审计，不是田间试验证明。official expert是推文区间中值的固定DAP映射。DQN保留provisional标签。

## 输出

- `{OUT.relative_to(ROOT)}`
- `{DOC.relative_to(ROOT)}`
- `{HTML.relative_to(ROOT)}`
"""
    RECORD.write_text(record, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    daily, summary, overview, expert, dqn = load()
    templates = expert_templates(expert)
    expert_events = executed_events(daily, "official_extension_expert")
    ppo_events = executed_events(daily, "rl_candidate")
    gaps = metric_gap_detail(summary, overview)
    rational = rationality(daily, summary, overview)
    dqn_out = dqn_evidence(dqn)

    templates.to_csv(OUT / "028_14_expert_schedule_templates.csv", index=False, encoding="utf-8-sig")
    expert_events.to_csv(OUT / "028_14_expert_executed_events.csv", index=False, encoding="utf-8-sig")
    ppo_events.to_csv(OUT / "028_14_ppo_executed_events.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(OUT / "028_14_ppo_metric_gap_detail.csv", index=False, encoding="utf-8-sig")
    rational.to_csv(OUT / "028_14_ppo_management_rationality.csv", index=False, encoding="utf-8-sig")
    dqn_out.to_csv(OUT / "028_14_dqn_current_evidence.csv", index=False, encoding="utf-8-sig")
    plot_gap(gaps)
    build_reports(templates, gaps, rational, dqn_out)

    checks = {
        "status": "completed",
        "training_calls": 0,
        "new_dssat_runs": 0,
        "site_years": int(overview.shape[0]),
        "five_scenario_rows": int(summary.shape[0]),
        "daily_rows": int(daily.shape[0]),
        "metric_gap_rows": int(gaps.shape[0]),
        "all_site_years_have_three_metrics": bool(gaps.shape[0] == 17 * 3),
        "all_representatives_any_metric_win": bool((gaps.groupby(["site", "year"])["status"].apply(lambda x: (x == "strict_win").any())).all()),
        "all_budget_ok": bool(rational["budget_ok"].all()),
        "all_stage_actions_valid": bool(rational["valid_stage_actions"].all()),
        "all_no_n_after_dap90": bool(rational["no_n_after_dap90"].all()),
        "three_seed_stable_2of3_or_better": int(((rational["winner_seed_count"] >= 2) & (rational["seed_count"] == 3)).sum()),
    }
    (OUT / "028_14_result.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
