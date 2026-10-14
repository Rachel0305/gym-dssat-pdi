from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DSSAT_auto_validation" / "five_site_strategy_design_019_01"
DOC = ROOT / "docs" / "2026-07-10_019_09_five_site_evidence_status_refresh.md"

BASELINE_PATH = (
    ROOT
    / "DSSAT_auto_validation"
    / "extension_expert_baseline_018_03"
    / "018_03_clean_multisite_comparison_with_extension_expert.csv"
)
SEED_HYF_PATH = (
    ROOT
    / "DSSAT_auto_validation"
    / "extension_expert_baseline_018_03"
    / "018_10_hla_yc_fq_seed_recheck"
    / "018_10_site_recheck_status.csv"
)
LC_SEED_PATH = (
    ROOT
    / "DSSAT_auto_validation"
    / "extension_expert_baseline_018_03"
    / "018_06_lc2010_seed_stability_audit"
    / "018_06_lc2010_seed_best_summary.csv"
)
SY_SEED_PATH = (
    ROOT
    / "DSSAT_auto_validation"
    / "sy2014_seed1_minimal_reproduction_018_08"
    / "018_08_seed0_vs_seed1_comparison.csv"
)
NCOST_PATH = (
    ROOT
    / "DSSAT_auto_validation"
    / "reward_sensitivity_019_02"
    / "019_02_site_level_interpretation.csv"
)
LEACHING_DECISION_PATH = ROOT / "docs" / "2026-07-10_019_08_keep_leaching_as_sensitivity_not_main_reward.md"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def md_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def scenario_row(baselines: pd.DataFrame, site: str, label: str) -> pd.Series:
    rows = baselines[(baselines["site"] == site) & (baselines["scenario_label"] == label)]
    if len(rows) != 1:
        raise ValueError(f"{site}/{label} expected exactly one row, found {len(rows)}")
    return rows.iloc[0]


def build_seed_status() -> dict[str, str]:
    hyf = pd.read_csv(SEED_HYF_PATH)
    seed_status = dict(zip(hyf["site"], hyf["recheck_status"]))

    lc = pd.read_csv(LC_SEED_PATH)
    seed0_n = float(lc.loc[lc["seed"] == 0, "nitrogen_kg_ha"].iloc[0])
    seed1_n = float(lc.loc[lc["seed"] == 1, "nitrogen_kg_ha"].iloc[0])
    seed_status["LC"] = (
        "yield_stable_resource_unstable" if seed0_n != seed1_n else "stable_across_seed"
    )

    sy = pd.read_csv(SY_SEED_PATH)
    if set(sy["seed"].astype(int)) >= {0, 1}:
        seed_status["SY"] = "stable_high_yield_across_seed_but_high_resource"
    else:
        seed_status["SY"] = "missing_seed_evidence"
    return seed_status


def main() -> None:
    required = [
        BASELINE_PATH,
        SEED_HYF_PATH,
        LC_SEED_PATH,
        SY_SEED_PATH,
        NCOST_PATH,
        LEACHING_DECISION_PATH,
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required evidence:\n" + "\n".join(missing))

    baselines = pd.read_csv(BASELINE_PATH)
    ncost = pd.read_csv(NCOST_PATH)
    seed_status = build_seed_status()

    site_years = {"HLA": 2010, "YC": 2014, "FQ": 2016, "SY": 2014, "LC": 2010}
    station_names = {
        "HLA": "Hailun",
        "YC": "Yucheng",
        "FQ": "Fengqiu",
        "SY": "Shenyang",
        "LC": "Luancheng",
    }
    optimization_space = {
        "HLA": "明确",
        "YC": "明确",
        "FQ": "局部且年份敏感",
        "SY": "明确但资源投入高",
        "LC": "有限但真实",
    }
    ic_status = {
        "HLA": "IC=1 主线；不因结果继续修改",
        "YC": "当前输入可用；不优先修改",
        "FQ": "年份敏感；禁止为提高胜率随意修改",
        "SY": "2014 使用经诊断的 IC=2；必须保留敏感性说明",
        "LC": "土壤ID/日期链路已修；保留修复记录",
    }
    evidence_classification = {
        "HLA": "资源效率候选；跨seed产量不稳定",
        "YC": "产量跨seed稳定；施氮量不稳定",
        "FQ": "单seed近平台且节氮；尚未跨seed复现",
        "SY": "跨seed高产复现；未对auto实现投入量占优",
        "LC": "产量跨seed稳定；资源投入不稳定",
    }

    rows = []
    for site, year in site_years.items():
        dqn = scenario_row(baselines, site, "DQN best checkpoint") if site != "SY" else scenario_row(baselines, site, "DQN ckpt15000")
        auto = scenario_row(baselines, site, "DSSAT auto")
        ext = scenario_row(baselines, site, "Official extension expert fixed DAP")
        ncost_row = ncost[ncost["site"] == site]
        ncost_status = ncost_row["status"].iloc[0] if len(ncost_row) == 1 else "missing"

        dy_auto = float(dqn["grain_yield_kg_ha"] - auto["grain_yield_kg_ha"])
        dy_ext = float(dqn["grain_yield_kg_ha"] - ext["grain_yield_kg_ha"])
        input_dom_auto = (
            float(dqn["irrigation_mm"]) <= float(auto["irrigation_mm"])
            and float(dqn["nitrogen_kg_ha"]) <= float(auto["nitrogen_kg_ha"])
        )
        input_dom_ext = (
            float(dqn["irrigation_mm"]) <= float(ext["irrigation_mm"])
            and float(dqn["nitrogen_kg_ha"]) <= float(ext["nitrogen_kg_ha"])
        )
        pareto_auto_1kg_tolerance = dy_auto >= -1.0 and input_dom_auto
        pareto_ext_1kg_tolerance = dy_ext >= -1.0 and input_dom_ext

        rows.append(
            {
                "site": site,
                "station": station_names[site],
                "representative_year": year,
                "optimization_space": optimization_space[site],
                "dqn_yield": float(dqn["grain_yield_kg_ha"]),
                "dqn_irrigation": float(dqn["irrigation_mm"]),
                "dqn_nitrogen": float(dqn["nitrogen_kg_ha"]),
                "yield_diff_vs_auto": dy_auto,
                "yield_diff_vs_extension": dy_ext,
                "input_dominates_auto": input_dom_auto,
                "input_dominates_extension": input_dom_ext,
                "pareto_dominates_auto_1kg_tolerance": pareto_auto_1kg_tolerance,
                "pareto_dominates_extension_1kg_tolerance": pareto_ext_1kg_tolerance,
                "seed_status": seed_status.get(site, "missing"),
                "n_cost_offline_result": ncost_status,
                "ic_status": ic_status[site],
                "evidence_classification": evidence_classification[site],
                "wue_nue_claim": "未统一计算；当前只能声称产量/投入量差异",
                "leaching_reward": "技术链路已通；暂不纳入正式reward",
                "baseline_source": str(dqn["source_file"]),
            }
        )

    board = pd.DataFrame(rows)

    gaps = pd.DataFrame(
        [
            {
                "issue": "WUE/NUE定义缺失",
                "severity": "high",
                "evidence": "现有总表比较产量、I和N总量，未统一计算WUE/NUE；auto常出现N=0",
                "impact": "不能把投入量较低直接写成水氮利用效率已超过",
                "minimum_next_action": "先与导师确定WUE/NUE公式、分母和N=0处理规则，再离线计算",
            },
            {
                "issue": "跨seed稳定性不完整",
                "severity": "high",
                "evidence": "HLA产量不稳定；YC氮用量不稳定；FQ仅seed1成功；LC资源用量不稳定",
                "impact": "除SY高产复现外，多数站点不能称为稳定成功",
                "minimum_next_action": "仅对最终候选框架补最小seed，不重复旧训练",
            },
            {
                "issue": "SY自动管理弱基线",
                "severity": "medium",
                "evidence": "SY2014 auto产量显著低于recorded/extension，且N=0",
                "impact": "超过auto不能单独证明DQN优越；应重点比较官方推广expert",
                "minimum_next_action": "汇报时分别列auto和extension，不合并成单一专家结论",
            },
            {
                "issue": "N-cost离线重评分无选择变化",
                "severity": "medium",
                "evidence": "019_02中五站从N cost 5提高到20均未改变现有checkpoint选择",
                "impact": "不能宣称单纯调高N成本已解决高N策略",
                "minimum_next_action": "保持正式reward；若导师要求再做统一小规模重训练，而非站点单独调参",
            },
            {
                "issue": "淋洗惩罚训练不稳定",
                "severity": "medium",
                "evidence": "019_07中FQ2016 leaching_cost=20在2000步较好，但4000/5000步退化",
                "impact": "淋洗项目前不适合作正式reward组成",
                "minimum_next_action": "保留为环境效益敏感性；正式主线暂不加入",
            },
        ]
    )

    inventory_rows = []
    for path in required:
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            row_count = len(frame)
            column_count = len(frame.columns)
        else:
            row_count = "n/a"
            column_count = "n/a"
        inventory_rows.append(
            {
                "source": rel(path),
                "exists": path.exists(),
                "rows": row_count,
                "columns": column_count,
                "size_bytes": path.stat().st_size,
            }
        )
    inventory = pd.DataFrame(inventory_rows)

    duplicate_count = int(baselines.duplicated(["site", "scenario_label"]).sum())
    negative_count = int(
        (
            (baselines["grain_yield_kg_ha"] < 0)
            | (baselines["irrigation_mm"] < 0)
            | (baselines["nitrogen_kg_ha"] < 0)
        ).sum()
    )
    counts = baselines.groupby("site").size()
    complete_five_rows = bool((counts == 5).all() and len(counts) == 5)
    dqn_label_present = bool(
        baselines["scenario_label"].isin({"DQN best checkpoint", "DQN ckpt15000"}).groupby(baselines["site"]).any().all()
    )
    basename_only_sources = int(
        baselines["source_file"].fillna("").map(lambda x: "/" not in x and "\\" not in x).sum()
    )
    quality = pd.DataFrame(
        [
            {"check": "五站每站五情景", "result": "pass" if complete_five_rows else "fail", "value": str(counts.to_dict()), "severity_if_failed": "critical"},
            {"check": "站点-情景键唯一", "result": "pass" if duplicate_count == 0 else "fail", "value": duplicate_count, "severity_if_failed": "critical"},
            {"check": "产量/灌溉/施氮非负", "result": "pass" if negative_count == 0 else "fail", "value": negative_count, "severity_if_failed": "high"},
            {"check": "每站存在DQN情景", "result": "pass" if dqn_label_present else "fail", "value": dqn_label_present, "severity_if_failed": "critical"},
            {"check": "仅文件名的source_file", "result": "warn" if basename_only_sources else "pass", "value": basename_only_sources, "severity_if_failed": "low"},
        ]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUT / "019_09_current_five_site_decision_board.csv", index=False, encoding="utf-8-sig")
    gaps.to_csv(OUT / "019_09_evidence_quality_and_gaps.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(OUT / "019_09_source_inventory.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUT / "019_09_data_quality_checks.csv", index=False, encoding="utf-8-sig")

    doc_lines = [
        "# 019_09 五站点已有证据状态刷新",
        "",
        "## 结论先行",
        "",
        "本轮没有重新审计或重新训练，而是刷新 019_01 已有证据矩阵。五站都已有代表年份和 DQN 证据，但目前不能说五站都已稳定满足导师目标。",
        "",
        "当前最重要的新口径是：现有证据能直接支持产量、灌溉总量和施氮总量比较，但尚未形成统一 WUE/NUE 指标，因此不能把投入量占优自动表述为利用效率已经占优。",
        "在正式 WUE/NUE 口径确定前，本表用 Pareto 判据做初筛：DQN 产量不低于基线（仅允许 1 kg/ha 数值容差），且灌溉量和施氮量均不高于基线。该判据评价的是产量-投入组合，不等同于正式 WUE/NUE。",
        "",
        "## 当前五站决策表",
        "",
        md_table(board),
        "",
        "## 证据质量与剩余缺口",
        "",
        md_table(gaps),
        "",
        "## 数据源完整性",
        "",
        md_table(inventory),
        "",
        "## 数据质量检查",
        "",
        md_table(quality),
        "",
        "## 已关闭的重复工作",
        "",
        "- 不再重做五站优化空间审计；019_01 已完成。",
        "- 不再重复 gym/PDI 与 DSSAT 传输核查；旧链路证据继续有效。",
        "- 不再通过随意降低 IC 制造优化空间。",
        "- 不继续把淋洗惩罚并入正式 reward；019_08 已决定保留为敏感性扩展。",
        "- 不把离线 N-cost 重评分解释成新策略已经学会节氮。",
        "",
        "## 下一步最小工作",
        "",
        "1. 先确定论文中 WUE/NUE 的正式定义和 N=0 情况的处理方式，并利用现有数据离线计算，不训练。",
        "2. 将五站按证据等级分组：HLA/YC/FQ/LC 保留各自的稳定性限制；SY 是跨 seed 高产复现，但不是对 auto 的投入量双占优。",
        "3. 只有在指标口径确定后，才决定哪一个站点需要补最小 seed 或统一 reward 小试。",
        "",
    ]
    DOC.write_text("\n".join(doc_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
