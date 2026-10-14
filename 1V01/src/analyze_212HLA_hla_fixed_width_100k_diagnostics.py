"""Consolidate the corrected HLA augmentation, 100K process, and exact metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "benchmark_results" / "054_00_hla_lowIC_expanded_action_maskableppo"
V1 = ROOT / "benchmark_results" / "204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo"
V1_DIAG = ROOT / "benchmark_results" / "205HLA_hla_weather_aug_100k_diagnostics"
V2 = ROOT / "benchmark_results" / "210HLA_hla_lowIC_10y_six_weather_fixed_width_balanced_maskableppo"
OUT = ROOT / "benchmark_results" / "212HLA_hla_fixed_width_100k_diagnostics"
TABLES = OUT / "tables"
RECORD = ROOT / "docs" / "212HLA_hla_fixed_width_100k_diagnostics_record.md"
CHECKPOINTS = [10_000, 25_000, 50_000, 75_000, 100_000]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def finite(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def phase_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    work = metrics.copy()
    numeric = [
        "num_timesteps", "approx_kl", "explained_variance", "value_loss", "requested_noop_rate",
        "requested_action_entropy_normalized", "safe_nonzero_rate", "scenario_coverage", "rss_mb",
    ]
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["phase"] = pd.cut(
        work["num_timesteps"],
        bins=[0, 10_000, 25_000, 50_000, 75_000, 100_080],
        labels=["0-10K", "10-25K", "25-50K", "50-75K", "75-100K"],
        include_lowest=True,
        right=True,
    )
    rows: list[dict[str, Any]] = []
    for phase, group in work.groupby("phase", observed=True):
        ev = group["explained_variance"]
        rows.append(
            {
                "phase": str(phase), "updates": len(group),
                "step_start": int(group["num_timesteps"].min()), "step_end": int(group["num_timesteps"].max()),
                "mean_approx_kl": finite(group["approx_kl"].mean()), "p95_approx_kl": finite(group["approx_kl"].quantile(0.95)),
                "median_explained_variance": finite(ev.median()), "negative_explained_variance_rate": finite((ev < 0).mean()),
                "explained_variance_below_minus1_rate": finite((ev < -1).mean()),
                "median_value_loss": finite(group["value_loss"].median()), "p95_value_loss": finite(group["value_loss"].quantile(0.95)),
                "mean_requested_action_entropy_normalized": finite(group["requested_action_entropy_normalized"].mean()),
                "mean_requested_noop_rate": finite(group["requested_noop_rate"].mean()),
                "mean_safe_nonzero_rate": finite(group["safe_nonzero_rate"].mean()),
                "final_scenario_coverage": int(group["scenario_coverage"].iloc[-1]), "peak_rss_mb": finite(group["rss_mb"].max()),
            }
        )
    return pd.DataFrame(rows)


def exact_v2(checkpoint: int) -> dict[str, Any]:
    candidates = sorted(
        path for path in (ROOT / "benchmark_results").glob(f"054_03_hla_lowIC_210HLA_*ckpt{checkpoint}_exact211")
        if (path / "tables" / "054_03_hla_five_scenario_season_summary.csv").exists()
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one exact211 replay for {checkpoint}, got {len(candidates)}")
    directory = candidates[0]
    table = pd.read_csv(directory / "tables" / "054_03_hla_five_scenario_season_summary.csv")
    ppo = table[table["scenario"].astype(str).eq("rl_candidate")].copy()
    return {
        "WP_ET_kg_m3": finite(pd.to_numeric(ppo["WP_ET_kg_m3"], errors="coerce").mean()),
        "ETCP_mm": finite(pd.to_numeric(ppo["etcp_mm"], errors="coerce").mean()),
        "grain_yield_kg_ha": finite(pd.to_numeric(ppo["grain_yield_kg_ha"], errors="coerce").mean()),
        "root": rel(directory),
    }


def checkpoint_table() -> pd.DataFrame:
    process = pd.read_csv(V2 / "audits" / "210HLA_checkpoint_process_summary.csv")
    process["checkpoint_step"] = pd.to_numeric(process["checkpoint_step"], errors="raise").astype(int)
    old_detail = pd.read_csv(OLD / "evaluation" / "054_00_checkpoint_validation_summary.csv")
    for column in ["checkpoint_step", "final_grnwt", "PFP_N", "total_irrigation", "total_n"]:
        old_detail[column] = pd.to_numeric(old_detail[column], errors="coerce")
    v1 = pd.read_csv(V1_DIAG / "tables" / "205HLA_checkpoint_old_vs_augmented.csv")
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        new = process[process["checkpoint_step"].eq(checkpoint)].iloc[0]
        old = old_detail[old_detail["checkpoint_step"].eq(checkpoint)]
        v1row = v1[pd.to_numeric(v1["checkpoint_step"], errors="coerce").eq(checkpoint)].iloc[0]
        exact = exact_v2(checkpoint)
        rows.append(
            {
                "checkpoint_step": checkpoint,
                "v2_yield_kg_ha": finite(new["mean_final_grnwt"]), "v2_WP_ET_kg_m3": exact["WP_ET_kg_m3"],
                "v2_PFP_N_kg_kg": finite(new["mean_PFP_N"]), "v2_irrigation_mm": finite(new["mean_total_irrigation"]),
                "v2_nitrogen_kg_ha": finite(new["mean_total_n"]), "v2_unique_action_signatures": int(new["unique_full_action_signatures"]),
                "v2_validation_noop_rate": finite(new["mean_requested_noop_rate"]), "v2_max_swfac": finite(new["max_swfac"]),
                "v2_max_nstres": finite(new["max_nstres"]),
                "v1_invalid_yield_kg_ha": finite(v1row["new_mean_yield_kg_ha"]), "v1_invalid_WP_ET_kg_m3": finite(v1row["new_mean_WP_ET_kg_m3"]),
                "v1_invalid_PFP_N_kg_kg": finite(v1row["new_mean_PFP_N_kg_kg"]), "v1_invalid_unique_action_signatures": int(v1row["new_unique_action_signatures"]),
                "old_available": not old.empty,
                "old_yield_kg_ha": finite(old["final_grnwt"].mean()) if not old.empty else np.nan,
                "old_WP_ET_kg_m3": finite(v1row["old_mean_WP_ET_kg_m3"]) if not old.empty else np.nan,
                "old_PFP_N_kg_kg": finite(old["PFP_N"].mean()) if not old.empty else np.nan,
                "old_irrigation_mm": finite(old["total_irrigation"].mean()) if not old.empty else np.nan,
                "old_nitrogen_kg_ha": finite(old["total_n"].mean()) if not old.empty else np.nan,
                "old_unique_action_signatures": int(old["action_sequence"].astype(str).nunique()) if not old.empty else 0,
                "v2_exact_replay_root": exact["root"],
            }
        )
    frame = pd.DataFrame(rows)
    for metric in ["yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
        frame[f"v2_delta_{metric}_vs_old"] = frame[f"v2_{metric}"] - frame[f"old_{metric}"]
    return frame


def variant_rewards() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, root, task in [
        ("v1_invalid_width7", V1, "204HLA"),
        ("v2_fixed_width6", V2, "210HLA"),
    ]:
        episodes = pd.read_csv(root / "logs" / f"{task}_training_episode_metrics.csv")
        for column in ["episode_reward", "total_irrigation_mm", "total_n_kg_ha"]:
            episodes[column] = pd.to_numeric(episodes[column], errors="coerce")
        grouped = episodes.groupby("weather_variant", as_index=False).agg(
            episodes=("episode_index", "count"), mean_episode_reward=("episode_reward", "mean"),
            mean_irrigation_mm=("total_irrigation_mm", "mean"), mean_nitrogen_kg_ha=("total_n_kg_ha", "mean"),
        )
        grouped.insert(0, "dataset", label)
        rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows)


def main() -> None:
    required = [
        V2 / "210HLA_formal_result.json", V2 / "logs" / "210HLA_training_update_metrics.csv",
        V2 / "audits" / "210HLA_checkpoint_process_summary.csv",
        V1_DIAG / "tables" / "205HLA_checkpoint_old_vs_augmented.csv",
        ROOT / "benchmark_results" / "209HLA_hla_lowIC_v2_weather_physical_gate" / "209HLA_result.json",
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing 212HLA inputs: " + "; ".join(missing))
    if OUT.exists() or RECORD.exists():
        raise FileExistsError("212HLA refuses to overwrite existing diagnostics")
    TABLES.mkdir(parents=True, exist_ok=False)

    phases = phase_summary(pd.read_csv(V2 / "logs" / "210HLA_training_update_metrics.csv"))
    checkpoints = checkpoint_table()
    variants = variant_rewards()
    v1_phases = pd.read_csv(V1_DIAG / "tables" / "205HLA_training_phase_process_metrics.csv")
    v1_phases.insert(0, "dataset", "v1_invalid_width7")
    v2_phases = phases.copy()
    v2_phases.insert(0, "dataset", "v2_fixed_width6")
    phase_compare = pd.concat([v1_phases, v2_phases], ignore_index=True)

    checkpoints.to_csv(TABLES / "212HLA_checkpoint_old_v1_v2_comparison.csv", index=False, encoding="utf-8-sig")
    phases.to_csv(TABLES / "212HLA_v2_training_phase_process_metrics.csv", index=False, encoding="utf-8-sig")
    phase_compare.to_csv(TABLES / "212HLA_v1_v2_training_phase_comparison.csv", index=False, encoding="utf-8-sig")
    variants.to_csv(TABLES / "212HLA_v1_v2_variant_episode_rewards.csv", index=False, encoding="utf-8-sig")

    best_yield = checkpoints.loc[checkpoints["v2_yield_kg_ha"].idxmax()]
    best_wp = checkpoints.loc[checkpoints["v2_WP_ET_kg_m3"].idxmax()]
    best_pfp = checkpoints.loc[checkpoints["v2_PFP_N_kg_kg"].idxmax()]
    collapsed = checkpoints[checkpoints["v2_unique_action_signatures"].le(1)]
    candidate10 = checkpoints[checkpoints["checkpoint_step"].eq(10_000)].iloc[0]
    candidate100 = checkpoints[checkpoints["checkpoint_step"].eq(100_000)].iloc[0]
    old25 = checkpoints[checkpoints["checkpoint_step"].eq(25_000)].iloc[0]
    common = checkpoints[checkpoints["old_available"]].copy()
    all_three = common[
        common["v2_delta_yield_kg_ha_vs_old"].gt(0)
        & common["v2_delta_WP_ET_kg_m3_vs_old"].gt(0)
        & common["v2_delta_PFP_N_kg_kg_vs_old"].gt(0)
    ]
    result = {
        "task": "212HLA_hla_fixed_width_100k_diagnostics",
        "status": "completed",
        "root_cause": "202/204 v1 wrote DSSAT WTH weather fields at width 7 instead of the source-compatible fixed width 6; DSSAT misread the weather columns.",
        "root_cause_evidence": {
            "source_and_v2_data_line_width": 31,
            "v1_data_line_width": 35,
            "v1_augmented_variant_mean_rewards_all_negative": True,
            "v2_all_six_variant_mean_rewards_positive": bool((variants[variants["dataset"].eq("v2_fixed_width6")]["mean_episode_reward"] > 0).all()),
            "physical_gate": "209HLA allow_2k_smoke=true",
        },
        "best_v2_checkpoint_by_yield": int(best_yield["checkpoint_step"]),
        "best_v2_checkpoint_by_WP_ET": int(best_wp["checkpoint_step"]),
        "best_v2_checkpoint_by_PFP_N": int(best_pfp["checkpoint_step"]),
        "collapsed_checkpoint_steps": collapsed["checkpoint_step"].astype(int).tolist(),
        "recovery_at_100k": bool(75_000 in collapsed["checkpoint_step"].tolist() and int(candidate100["v2_unique_action_signatures"]) > 1),
        "behavior_first_candidate": 10_000,
        "behavior_first_reason": "10K has 10/10 validation action signatures, the highest exact WP_ET, yield within 0.27% of 100K, and slightly less irrigation than 100K.",
        "100k_interpretation": "100K recovers from the 75K collapse and improves yield/WP_ET/diversity versus old 100K, but returns to I240/N240 and lowers PFP_N.",
        "best_frontier_improvement_vs_old_not_proven": bool(
            candidate10["v2_yield_kg_ha"] < old25["old_yield_kg_ha"]
            and candidate10["v2_WP_ET_kg_m3"] < old25["old_WP_ET_kg_m3"]
        ),
        "common_checkpoints_improving_yield_WP_ET_PFP_N_together": all_three["checkpoint_step"].astype(int).tolist(),
        "WP_ET_rule": "exact Summary.OUT/ETCP replay only",
        "outputs": {
            "checkpoint_comparison": rel(TABLES / "212HLA_checkpoint_old_v1_v2_comparison.csv"),
            "phase_metrics": rel(TABLES / "212HLA_v2_training_phase_process_metrics.csv"),
            "variant_rewards": rel(TABLES / "212HLA_v1_v2_variant_episode_rewards.csv"),
            "record": rel(RECORD),
        },
    }
    (OUT / "212HLA_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    display = checkpoints[[
        "checkpoint_step", "v2_yield_kg_ha", "v2_WP_ET_kg_m3", "v2_PFP_N_kg_kg", "v2_irrigation_mm",
        "v2_nitrogen_kg_ha", "v2_unique_action_signatures", "v2_validation_noop_rate",
    ]].copy()
    variant_display = variants[["dataset", "weather_variant", "episodes", "mean_episode_reward"]].copy()
    record = [
        "# 212HLA HLA 固定宽度天气增强 100K 综合诊断",
        "", "## 结论先说", "",
        "- 修复后的增强数据可以训练出非固定 HLA PPO：10K 为 10/10 动作序列，100K 为 6/10；75K 的坍缩在 100K 恢复。",
        "- 100K 延长训练有帮助，但恢复的是高投入策略：yield=6840.5 kg/ha、WP_ET=1.49 kg/m³、PFP_N=28.50、I=240 mm、N=240 kg/ha。",
        "- 行为优先候选为 10K：10 种动作序列、最高 WP_ET=1.50、yield=6822.3，且与 100K 的产量差仅约 0.27%。",
        "- 不能声称增强改善了旧 HLA 最优前沿：旧 25K 的 yield=6844.4、WP_ET=1.514、PFP_N=28.52、I=225、N=240，仍略优于新 10K。",
        "", "## v1 根因与 v2 修复", "",
        "- v1 天气数据行 35 字符：DATE 后四个天气字段各宽 7；DSSAT 原始固定格式应为各宽 6、整行 31 字符。",
        "- v1 五个伪年份变体训练 reward 均为负；v2 六变体 mean reward 均为正，209HLA 也验证全部正常成熟。",
        "- 因此 204HLA 100K 只保留为负面工程诊断，不能作为有效增强效果。",
        "", "## 修复后 checkpoint", "", display.to_markdown(index=False),
        "", "## v1/v2 训练情景 reward", "", variant_display.to_markdown(index=False),
        "", "## 过程解释", "",
        f"- 坍缩 checkpoint（验证动作序列≤1）：`{result['collapsed_checkpoint_steps']}`；100K 恢复：`{result['recovery_at_100k']}`。",
        "- critic 在各阶段都有短暂 explained variance<0/value-loss 尖峰，但 KL 与训练动作分布未同步发散；它是风险信号，不是坍缩的单一因果证明。",
        "- 下一轮不应先改 reward；应冻结 v2 数据格式和场景门槛，再做 HLA 专属资源约束/奖励的小规模 10K 对照。",
        "", "## 复现文件", "",
        "- v2 数据生成：`src/build_208HLA_hla_lowIC_weather_scenario_bank_v2_fixed_width.py`。",
        "- 物理门槛：`src/audit_209HLA_hla_lowIC_v2_weather_physical_gate.py`。",
        "- 训练：`src/run_210HLA_hla_lowIC_10y_six_weather_fixed_width_balanced_maskableppo.py`。",
        "- 精确 WP_ET：`src/run_211HLA_hla_v2_exact_wp_et_replays.py`。",
        f"- 本汇总：`{rel(Path(__file__))}`；机器结果：`{rel(OUT / '212HLA_result.json')}`。",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
