from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from calculate_five_site_wue_nue_from_summary_019_10 import (
    ROOT,
    native_or_none,
    num,
    parse_summary_out,
    select_matching_row,
    valid_positive,
)


OUT = ROOT / "DSSAT_auto_validation" / "five_site_checkpoint_pareto_020_01"
BASELINE_METRICS = (
    ROOT
    / "DSSAT_auto_validation"
    / "five_site_wue_nue_019_10"
    / "019_10_native_wue_nue_metrics.csv"
)
DOC = ROOT / "docs" / "2026-07-10_020_01_five_site_checkpoint_pareto_screen_record.md"


@dataclass(frozen=True)
class RunConfig:
    site: str
    station: str
    year: int
    seed: int
    summary_csv: Path
    raw_mode: str
    raw_root: Path | None = None


def p(relative: str) -> Path:
    return ROOT / relative


CONFIGS = [
    RunConfig("HLA", "Hailun", 2010, 0, p("DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed0_50000steps/checkpoint_summary.csv"), "checkpoint_dir", p("DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed0_50000steps")),
    RunConfig("HLA", "Hailun", 2010, 1, p("DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps/checkpoint_summary.csv"), "checkpoint_dir", p("DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed1_50000steps")),
    RunConfig("YC", "Yucheng", 2014, 0, p("DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/seed0/015_04_yc2014_unified_dqn_checkpoint_summary.csv"), "checkpoint_dir", p("DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/seed0")),
    RunConfig("YC", "Yucheng", 2014, 1, p("DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_seed1_015_05/seed1/015_05_yc2014_unified_dqn_checkpoint_seed1_summary.csv"), "checkpoint_dir", p("DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_seed1_015_05/seed1")),
    RunConfig("FQ", "Fengqiu", 2016, 0, p("DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed0_50000steps/checkpoint_summary.csv"), "eval_suffix", p("DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed0_50000steps")),
    RunConfig("FQ", "Fengqiu", 2016, 1, p("DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/checkpoint_summary.csv"), "eval_suffix", p("DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps")),
    RunConfig("SY", "Shenyang", 2014, 0, p("DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08/017_08_sy_dqn_checkpoint_summary.csv"), "run_dir"),
    RunConfig("SY", "Shenyang", 2014, 1, p("DSSAT_auto_validation/sy2014_seed1_minimal_reproduction_018_08/018_08_formal_checkpoint_summary.csv"), "run_dir"),
    RunConfig("LC", "Luancheng", 2010, 0, p("DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed0_5000steps/checkpoint_summary.csv"), "eval_suffix", p("DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed0_5000steps")),
    RunConfig("LC", "Luancheng", 2010, 1, p("DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed1_5000steps/checkpoint_summary.csv"), "eval_suffix", p("DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed1_5000steps")),
]


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def normalize_checkpoint_summary(config: RunConfig) -> pd.DataFrame:
    df = pd.read_csv(config.summary_csv)
    if "site" in df.columns:
        df = df[(df["site"] == config.site) & (df["year"].astype(int) == config.year)]
    if "seed" in df.columns:
        df = df[df["seed"].astype(int) == config.seed]

    step_col = find_column(df, ["checkpoint_step", "checkpoint"])
    yield_col = find_column(df, ["final_grain_kg_ha", "final_gwad"])
    irrigation_col = find_column(df, ["action_irrigation_total", "irrigation_total", "event_irrigation_total"])
    nitrogen_col = find_column(df, ["action_fertilizer_total", "fertilizer_total", "event_fertilizer_total"])
    reward_col = find_column(df, ["total_reward"])

    out = pd.DataFrame(
        {
            "checkpoint_step": df[step_col].astype(int),
            "expected_yield": df[yield_col].astype(float),
            "expected_irrigation": df[irrigation_col].astype(float),
            "expected_nitrogen": df[nitrogen_col].astype(float),
            "training_reward": df[reward_col].astype(float),
        }
    )
    return out.drop_duplicates("checkpoint_step").sort_values("checkpoint_step")


def raw_summary_path(config: RunConfig, checkpoint: int, original_row: pd.Series | None = None) -> Path:
    if config.raw_mode == "checkpoint_dir":
        return config.raw_root / f"checkpoint_{checkpoint}" / "pdi_tmp_snapshot_eval" / "Summary.OUT"
    if config.raw_mode == "eval_suffix":
        return config.raw_root / f"pdi_tmp_snapshot_eval_{checkpoint}" / "Summary.OUT"
    if config.raw_mode == "run_dir":
        df = pd.read_csv(config.summary_csv)
        step_col = find_column(df, ["checkpoint_step", "checkpoint"])
        subset = df[df[step_col].astype(int) == checkpoint]
        if "seed" in subset.columns:
            subset = subset[subset["seed"].astype(int) == config.seed]
        if "site" in subset.columns:
            subset = subset[subset["site"] == config.site]
        if len(subset) != 1:
            raise ValueError(f"{config.site} seed{config.seed} checkpoint {checkpoint}: run_dir row count={len(subset)}")
        return ROOT / str(subset.iloc[0]["run_dir"]) / "pdi_tmp_snapshot_eval" / "Summary.OUT"
    raise ValueError(config.raw_mode)


def extract_metrics(config: RunConfig, checkpoint_row: pd.Series) -> tuple[dict, dict]:
    checkpoint = int(checkpoint_row["checkpoint_step"])
    source = raw_summary_path(config, checkpoint)
    if not source.exists():
        raise FileNotFoundError(source)
    parsed = parse_summary_out(source)
    row, score, selected_index = select_matching_row(
        parsed,
        float(checkpoint_row["expected_yield"]),
        float(checkpoint_row["expected_irrigation"]),
        float(checkpoint_row["expected_nitrogen"]),
    )

    hwam = num(row, "HWAM")
    ircm = num(row, "IRCM")
    nicm = num(row, "NICM")
    nucm = num(row, "NUCM")
    nlcm = num(row, "NLCM")
    ypem = native_or_none(num(row, "YPEM"))
    ypim = native_or_none(num(row, "YPIM"))
    ypnam = native_or_none(num(row, "YPNAM"))
    ypnum = native_or_none(num(row, "YPNUM"))

    metric = {
        "site": config.site,
        "station": config.station,
        "year": config.year,
        "seed": config.seed,
        "checkpoint_step": checkpoint,
        "HWAM_kg_ha": hwam,
        "irrigation_mm": float(checkpoint_row["expected_irrigation"]),
        "nitrogen_kg_ha": float(checkpoint_row["expected_nitrogen"]),
        "NUCM_kg_ha": nucm,
        "NLCM_kg_ha": nlcm,
        "ETCP_mm": num(row, "ETCP"),
        "WP_ET_kg_m3": ypem * 0.1 if ypem is not None else None,
        "IWP_gross_kg_m3": ypim * 0.1 if valid_positive(ircm) and ypim is not None else None,
        "PFP_N_kg_kg": ypnam if valid_positive(nicm) and ypnam is not None else None,
        "NUtE_kg_kg": ypnum if valid_positive(nucm) and ypnum is not None else None,
        "training_reward": float(checkpoint_row["training_reward"]),
        "source_summary_out": source.relative_to(ROOT).as_posix(),
    }
    audit = {
        "site": config.site,
        "seed": config.seed,
        "checkpoint_step": checkpoint,
        "parsed_rows": len(parsed),
        "selected_row_index": selected_index,
        "match_score_H_W_N": score,
        "expected_yield": float(checkpoint_row["expected_yield"]),
        "selected_HWAM": hwam,
        "expected_irrigation": float(checkpoint_row["expected_irrigation"]),
        "selected_IRCM": ircm,
        "expected_nitrogen": float(checkpoint_row["expected_nitrogen"]),
        "selected_NICM": nicm,
        "source_summary_out": source.relative_to(ROOT).as_posix(),
    }
    return metric, audit


def pareto_flags(df: pd.DataFrame, group_columns: list[str]) -> pd.Series:
    flags = pd.Series(False, index=df.index)
    maximize = ["HWAM_kg_ha", "WP_ET_kg_m3"]
    minimize = ["irrigation_mm", "nitrogen_kg_ha", "NLCM_kg_ha"]
    for _, group in df.groupby(group_columns):
        for idx, candidate in group.iterrows():
            dominated = False
            for other_idx, other in group.iterrows():
                if idx == other_idx:
                    continue
                no_worse = all(other[c] >= candidate[c] for c in maximize) and all(
                    other[c] <= candidate[c] for c in minimize
                )
                strictly_better = any(other[c] > candidate[c] for c in maximize) or any(
                    other[c] < candidate[c] for c in minimize
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            flags.loc[idx] = not dominated
    return flags


def add_baseline_flags(metrics: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    for baseline_name, label in [
        ("auto", "DSSAT auto"),
        ("extension", "Official extension expert fixed DAP"),
    ]:
        baseline_lookup = baselines[baselines["scenario"] == label].set_index("site")
        yield_ok = []
        wp_ok = []
        input_ok = []
        leach_ok = []
        for _, row in out.iterrows():
            base = baseline_lookup.loc[row["site"]]
            yield_ok.append(row["HWAM_kg_ha"] >= base["HWAM_kg_ha"] - 1.0)
            wp_ok.append(row["WP_ET_kg_m3"] >= base["WP_ET_kg_m3_DSSAT_YPEM"] - 0.01)
            input_ok.append(
                row["irrigation_mm"] <= base["irrigation_mm_event_table"]
                and row["nitrogen_kg_ha"] <= base["NICM_kg_ha"]
            )
            leach_ok.append(row["NLCM_kg_ha"] <= base["NLCM_kg_ha"] + 0.01)
        out[f"yield_ok_vs_{baseline_name}"] = yield_ok
        out[f"WP_ET_ok_vs_{baseline_name}"] = wp_ok
        out[f"input_ok_vs_{baseline_name}"] = input_ok
        out[f"leaching_ok_vs_{baseline_name}"] = leach_ok
        out[f"strict_success_vs_{baseline_name}"] = (
            out[f"yield_ok_vs_{baseline_name}"]
            & out[f"WP_ET_ok_vs_{baseline_name}"]
            & out[f"input_ok_vs_{baseline_name}"]
            & out[f"leaching_ok_vs_{baseline_name}"]
        )
    out["strict_success_vs_both"] = out["strict_success_vs_auto"] & out["strict_success_vs_extension"]
    out["yield_WP_ok_vs_both"] = (
        out["yield_ok_vs_auto"]
        & out["WP_ET_ok_vs_auto"]
        & out["yield_ok_vs_extension"]
        & out["WP_ET_ok_vs_extension"]
    )
    return out


def md_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
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


def deduplicate_pareto_strategies(pareto: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical outcome/resource signatures for presentation only."""
    signature = [
        "site", "HWAM_kg_ha", "WP_ET_kg_m3", "irrigation_mm",
        "nitrogen_kg_ha", "NLCM_kg_ha",
    ]
    rows = []
    for _, group in pareto.groupby(signature, dropna=False, sort=True):
        first = group.sort_values(["seed", "checkpoint_step"]).iloc[0]
        rows.append(
            {
                **{column: first[column] for column in signature},
                "NUtE_kg_kg": first["NUtE_kg_kg"],
                "occurrence_count": len(group),
                "seed_count": group["seed"].nunique(),
                "seed_checkpoint_members": "; ".join(
                    f"seed{int(row.seed)}@{int(row.checkpoint_step)}"
                    for row in group.sort_values(["seed", "checkpoint_step"]).itertuples()
                ),
                "strict_success_vs_both": bool(group["strict_success_vs_both"].any()),
                "yield_WP_ok_vs_both": bool(group["yield_WP_ok_vs_both"].any()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["site", "strict_success_vs_both", "HWAM_kg_ha", "WP_ET_kg_m3"],
        ascending=[True, False, False, False],
    )


def main() -> None:
    required = [BASELINE_METRICS] + [config.summary_csv for config in CONFIGS]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required prior evidence:\n" + "\n".join(missing))

    metric_rows = []
    audit_rows = []
    for config in CONFIGS:
        normalized = normalize_checkpoint_summary(config)
        for _, checkpoint_row in normalized.iterrows():
            metric, audit = extract_metrics(config, checkpoint_row)
            metric_rows.append(metric)
            audit_rows.append(audit)

    metrics = pd.DataFrame(metric_rows).sort_values(["site", "seed", "checkpoint_step"]).reset_index(drop=True)
    if metrics.duplicated(["site", "seed", "checkpoint_step"]).any():
        raise ValueError("Duplicate site-seed-checkpoint keys")
    if metrics[["HWAM_kg_ha", "WP_ET_kg_m3", "irrigation_mm", "nitrogen_kg_ha", "NLCM_kg_ha"]].isna().any().any():
        raise ValueError("Missing core Pareto metric")

    metrics["is_pareto_within_seed"] = pareto_flags(metrics, ["site", "seed"])
    metrics["is_pareto_within_site_all_seeds"] = pareto_flags(metrics, ["site"])

    baselines = pd.read_csv(BASELINE_METRICS)
    flagged = add_baseline_flags(metrics, baselines)
    pareto = flagged[flagged["is_pareto_within_site_all_seeds"]].copy()
    pareto_unique = deduplicate_pareto_strategies(pareto)

    summary_rows = []
    for site, group in flagged.groupby("site"):
        front = group[group["is_pareto_within_site_all_seeds"]]
        strict = group[group["strict_success_vs_both"]]
        yield_wp = group[group["yield_WP_ok_vs_both"]]
        strict_seed_count = strict["seed"].nunique()
        yield_wp_seed_count = yield_wp["seed"].nunique()
        if len(strict) and strict_seed_count == group["seed"].nunique():
            evidence_status = "strict_success_across_seed"
        elif len(strict):
            evidence_status = "strict_success_single_seed_only"
        elif len(yield_wp) and yield_wp_seed_count == group["seed"].nunique():
            evidence_status = "yield_WP_reproduced_but_resource_not_dominant"
        elif len(yield_wp):
            evidence_status = "yield_WP_single_seed_only"
        else:
            evidence_status = "no_checkpoint_meets_yield_WP_vs_both"
        candidate_pool = strict if len(strict) else (yield_wp if len(yield_wp) else front)
        candidate = candidate_pool.sort_values(
            ["HWAM_kg_ha", "WP_ET_kg_m3", "nitrogen_kg_ha", "irrigation_mm", "checkpoint_step"],
            ascending=[False, False, True, True, True],
        ).iloc[0]
        summary_rows.append(
            {
                "site": site,
                "checkpoint_count": len(group),
                "seed_count": group["seed"].nunique(),
                "pareto_checkpoint_count": len(front),
                "strict_success_vs_both_count": len(strict),
                "strict_success_seed_count": strict_seed_count,
                "yield_WP_ok_vs_both_count": len(yield_wp),
                "yield_WP_ok_seed_count": yield_wp_seed_count,
                "candidate_seed": int(candidate["seed"]),
                "candidate_checkpoint": int(candidate["checkpoint_step"]),
                "candidate_yield": candidate["HWAM_kg_ha"],
                "candidate_WP_ET": candidate["WP_ET_kg_m3"],
                "candidate_irrigation": candidate["irrigation_mm"],
                "candidate_nitrogen": candidate["nitrogen_kg_ha"],
                "candidate_NUtE": candidate["NUtE_kg_kg"],
                "candidate_N_leaching": candidate["NLCM_kg_ha"],
                "candidate_basis": "strict_success" if len(strict) else ("yield_WP" if len(yield_wp) else "pareto_front_only"),
                "evidence_status": evidence_status,
            }
        )
    site_summary = pd.DataFrame(summary_rows).sort_values("site")
    audit = pd.DataFrame(audit_rows).sort_values(["site", "seed", "checkpoint_step"])

    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "020_01_all_checkpoint_native_metrics.csv", index=False, encoding="utf-8-sig")
    flagged.to_csv(OUT / "020_01_checkpoint_baseline_flags.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(OUT / "020_01_pareto_front.csv", index=False, encoding="utf-8-sig")
    pareto_unique.to_csv(OUT / "020_01_pareto_unique_strategies.csv", index=False, encoding="utf-8-sig")
    site_summary.to_csv(OUT / "020_01_site_summary.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "020_01_source_match_audit.csv", index=False, encoding="utf-8-sig")

    """Legacy corrupted report block retained inert for audit.
    report_cols = [
        "site", "checkpoint_count", "seed_count", "pareto_checkpoint_count",
        "strict_success_vs_both_count", "yield_WP_ok_vs_both_count",
        "strict_success_seed_count", "yield_WP_ok_seed_count",
        "candidate_seed", "candidate_checkpoint", "candidate_yield", "candidate_WP_ET",
        "candidate_irrigation", "candidate_nitrogen", "candidate_NUtE",
        "candidate_N_leaching", "candidate_basis", "evidence_status",
    ]
    lines = [
        "# 020_01 五站现有 DQN checkpoint Pareto 筛选记录",
        "",
        "## 任务边界",
        "",
        "本轮先核对了旧实验。早期项目存在固定方案/专家库Pareto搜索，但不存在按019_10新WUE/NUE口径对五站正式DQN checkpoint的统一筛选。因此本轮不重复训练，只复用已保存的正式checkpoint及Summary.OUT。",
        "",
        "淋洗惩罚是否进入reward已由019_08关闭：本轮只把NLCM作为评价指标，不修改正式reward。",
        "",
        "## 五站汇总",
        "",
        md_table(site_summary[report_cols]),
        "",
        "## 口径说明",
        "",
        "- Pareto目标：最大化产量和WP_ET，最小化灌溉、施氮和NLCM。",
        "- IWP/PFP_N/NUtE用于解释，不进入支配判定，以避免零投入NA与氮胁迫高NUtE造成误判。",
        "- strict_success要求对auto和官方expert都同时满足产量、WP_ET、水氮投入和淋洗不劣。",
        "- candidate只是现有checkpoint候选，不自动等于跨seed稳定成功。",
        "",
        "## 数据质量",
        "",
        f"- checkpoint总数：{len(metrics)}。",
        f"- site-seed-checkpoint重复键：{int(metrics.duplicated(['site', 'seed', 'checkpoint_step']).sum())}。",
        f"- 核心Pareto指标缺失单元格：{int(metrics[['HWAM_kg_ha', 'WP_ET_kg_m3', 'irrigation_mm', 'nitrogen_kg_ha', 'NLCM_kg_ha']].isna().sum().sum())}。",
        f"- Summary.OUT匹配最大H/W/N误差和：{audit['match_score_H_W_N'].max():.3f}。",
        "",
        "## 下一步纪律",
        "",
        "1. 先根据本表判断哪些站点已有可用候选，不重做旧优化空间审计。",
        "2. 只有确有候选但跨seed不稳定时才补最小seed或稳定性训练。",
        "3. 只有确定性空间存在更优解、但所有checkpoint均未学到时，才讨论训练结构/reward参数。",
        "4. 不通过修改IC制造优化空间；不重新开启leaching-cost主reward实验。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    """

    report_cols = [
        "site", "checkpoint_count", "seed_count", "pareto_checkpoint_count",
        "strict_success_vs_both_count", "yield_WP_ok_vs_both_count",
        "strict_success_seed_count", "yield_WP_ok_seed_count",
        "candidate_seed", "candidate_checkpoint", "candidate_yield", "candidate_WP_ET",
        "candidate_irrigation", "candidate_nitrogen", "candidate_NUtE",
        "candidate_N_leaching", "candidate_basis", "evidence_status",
    ]
    decision_rows = pd.DataFrame(
        [
            {"站点": "HLA", "现有证据": "严格全面占优仅见于 seed0", "下一步": "只做最小稳定性复核，不改 IC/reward"},
            {"站点": "YC", "现有证据": "产量与 WP_ET 跨 seed 达标，但资源投入未全面占优", "下一步": "先复用既有资源响应/上界证据，再决定是否训练"},
            {"站点": "FQ", "现有证据": "产量与 WP_ET 跨 seed 达标，但资源投入未全面占优", "下一步": "先复用既有资源响应/上界证据，再决定是否训练"},
            {"站点": "SY", "现有证据": "产量与 WP_ET 跨 seed 达标，但资源投入未全面占优", "下一步": "先复用既有资源响应/上界证据，再决定是否训练"},
            {"站点": "LC", "现有证据": "产量与 WP_ET 仅 seed0 达标，且仅有 5K smoke", "下一步": "先核查既有 LC 证据，不直接上长训练"},
        ]
    )
    lines = [
        "# 020_01 五站现有 DQN checkpoint Pareto 筛选记录",
        "",
        "## 任务边界",
        "",
        "本轮先核对了旧实验。早期项目存在固定方案或专家库 Pareto 搜索，但不存在按 019_10 新 WUE/NUE 口径对五站正式 DQN checkpoint 的统一筛选。因此本轮不重复训练，只复用已保存的正式 checkpoint 和 Summary.OUT。",
        "",
        "淋洗惩罚是否进入 reward 已由 019_08 关闭：本轮只把 NLCM 作为评价指标，不修改正式 reward。",
        "",
        "## 五站汇总",
        "",
        md_table(site_summary[report_cols]),
        "",
        "## 证据解释",
        "",
        "- HLA 是唯一出现严格全面占优 checkpoint 的站点，但 3 个 checkpoint 全部来自 seed0，因此只能称为单 seed 成功，不能称为跨 seed 稳定成功。",
        "- YC、FQ、SY 均有两个 seed 达到产量与 WP_ET 条件，但没有 checkpoint 同时在水、氮投入和 NLCM 上对 auto 与官方 expert 全面不劣。",
        "- LC 只有 seed0 的一个 checkpoint 达到产量与 WP_ET 条件，而且现有证据仅为 5K smoke，不足以直接进入正式长训练。",
        "- `candidate` 是按固定排序挑出的代表性现有 checkpoint，不等同于全局最优，也不等同于稳定成功。",
        "",
        "## 下一步决策表",
        "",
        md_table(decision_rows),
        "",
        "## 指标口径",
        "",
        "- Pareto 目标：最大化产量和 WP_ET，最小化灌溉、施氮和 NLCM。",
        "- IWP、PFP_N、NUtE 用于解释，不进入支配判定，避免零投入 NA 与氮胁迫导致高 NUtE 的误判。",
        "- `strict_success` 要求相对 DSSAT auto 和官方 expert 同时满足：产量、WP_ET、水氮投入和淋洗均不劣。",
        "- 原始 Pareto 表保留所有 checkpoint；去重表只合并五个核心指标完全相同的策略结果，用于展示，不删除证据。",
        "",
        "## 数据质量",
        "",
        f"- checkpoint 总数：{len(metrics)}。",
        f"- site-seed-checkpoint 重复键：{int(metrics.duplicated(['site', 'seed', 'checkpoint_step']).sum())}。",
        f"- 核心 Pareto 指标缺失单元格：{int(metrics[['HWAM_kg_ha', 'WP_ET_kg_m3', 'irrigation_mm', 'nitrogen_kg_ha', 'NLCM_kg_ha']].isna().sum().sum())}。",
        f"- Summary.OUT 匹配最大 H/W/N 误差和：{audit['match_score_H_W_N'].max():.3f}。",
        f"- 原始 Pareto checkpoint 数：{len(pareto)}；去重后策略结果数：{len(pareto_unique)}。",
        "",
        "## 后续纪律",
        "",
        "1. 先依据本表判断已有证据，不重做旧优化空间审计。",
        "2. 只有确有候选但跨 seed 不稳定时，才补最小 seed 或稳定性训练。",
        "3. 只有确定性空间存在更优解、但所有 checkpoint 均未学到时，才讨论训练结构或 reward 参数。",
        "4. 不通过修改 IC 制造优化空间；不重新开启 leaching-cost reward 实验。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
