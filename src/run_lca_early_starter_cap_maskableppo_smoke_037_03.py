from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base032


TASK = "037_03_lca_early_starter_cap_maskableppo_smoke_clean_retrain"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = "LCA"
SITE = "LC"
SEED = 0
TOTAL_TIMESTEPS = 100_000
CHECKPOINT_STEPS = [25_000, 50_000, 75_000, 100_000]
EARLY_DAP_MAX = 10
EARLY_IRRIGATION_CAP_MM = 30.0
EARLY_N_CAP_KG_HA = 40.0

SELECTED_036 = (
    ROOT
    / "benchmark_results"
    / "036_04_select_checkpoint_and_plot_03601_03603_summary"
    / "tables"
    / "036_04_selected_year_level_comparison.csv"
)


OriginalStressAwareDiscreteWrapper = base032.StressAwareDiscreteWrapper


class EarlyStarterCapWrapper(OriginalStressAwareDiscreteWrapper):
    """036 wrapper plus a DAP1-10 cumulative starter cap."""

    def _action_is_legal_without_clipping(self, raw: dict[str, float], dap: int) -> bool:
        if not super()._action_is_legal_without_clipping(raw, dap):
            return False
        if int(dap) <= EARLY_DAP_MAX:
            i = float(raw.get("amir", 0.0))
            n = float(raw.get("anfer", 0.0))
            if i > 0 and float(self.safety_state.cumulative_irrigation) + i - EARLY_IRRIGATION_CAP_MM > 1e-9:
                return False
            if n > 0 and float(self.safety_state.cumulative_n) + n - EARLY_N_CAP_KG_HA > 1e-9:
                return False
        return True


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "logs", f"models/{STATION}", f"daily_outputs/{STATION}", f"tensorboard/{STATION}", "tables"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)


def patch_base_wrapper() -> None:
    base032.StressAwareDiscreteWrapper = EarlyStarterCapWrapper


def load_config() -> dict[str, Any]:
    config = batch.load_config()
    config = json.loads(json.dumps(config))
    config["seed"] = SEED
    config["total_timesteps"] = TOTAL_TIMESTEPS
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = 2005
    config["action_safety"]["early_starter_cap_enabled"] = True
    config["action_safety"]["early_starter_cap_dap_max"] = EARLY_DAP_MAX
    config["action_safety"]["early_starter_irrigation_cap_mm"] = EARLY_IRRIGATION_CAP_MM
    config["action_safety"]["early_starter_n_cap_kg_ha"] = EARLY_N_CAP_KG_HA
    return config


def selected_split() -> tuple[pd.DataFrame, list[int], list[int]]:
    split = batch.load_split()
    split = split[split["station_code"].eq(STATION)].copy()
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    train_years = split.loc[split["split"].eq("train"), "year"].astype(int).tolist()
    val_years = split.loc[split["split"].eq("validation"), "year"].astype(int).tolist()
    return split, train_years, val_years


def add_early_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        out = row.to_dict()
        path = ROOT / str(row.get("daily_csv_path", ""))
        if path.exists():
            daily = pd.read_csv(path)
            dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce")
            irr = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            n = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            early = dap <= EARLY_DAP_MAX
            total_i = float(irr.sum())
            total_n = float(n.sum())
            out["early_dap1_10_irrigation"] = float(irr[early].sum())
            out["early_dap1_10_n"] = float(n[early].sum())
            out["early_irrigation_share"] = out["early_dap1_10_irrigation"] / total_i if total_i > 0 else np.nan
            out["early_n_share"] = out["early_dap1_10_n"] / total_n if total_n > 0 else np.nan
            out["late_n_after_dap90_total"] = float(n[dap > 90].sum())
        rows.append(out)
    return pd.DataFrame(rows)


def compare_to_036(eval_df: pd.DataFrame) -> pd.DataFrame:
    if not SELECTED_036.exists() or eval_df.empty:
        return eval_df
    old = pd.read_csv(SELECTED_036)
    old = old[old["station_code"].eq(STATION)].copy()
    keep = [
        "year",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N_kg_kg",
        "yield_gap_vs_four_max",
        "wp_et_gap_vs_four_max",
        "pfp_n_gap_vs_four_max",
        "any_metric_win_four_max",
        "action_sequence",
    ]
    old = old[[c for c in keep if c in old.columns]].rename(
        columns={
            "final_grnwt": "old036_final_grnwt",
            "total_irrigation": "old036_total_irrigation",
            "total_n": "old036_total_n",
            "PFP_N_kg_kg": "old036_PFP_N_kg_kg",
            "yield_gap_vs_four_max": "old036_yield_gap_vs_four_max",
            "wp_et_gap_vs_four_max": "old036_wp_et_gap_vs_four_max",
            "pfp_n_gap_vs_four_max": "old036_pfp_n_gap_vs_four_max",
            "any_metric_win_four_max": "old036_any_metric_win_four_max",
            "action_sequence": "old036_action_sequence",
        }
    )
    merged = eval_df.merge(old, on="year", how="left", validate="many_to_one")
    merged["delta_final_grnwt_vs_036"] = merged["final_grnwt"] - merged["old036_final_grnwt"]
    merged["delta_irrigation_vs_036"] = merged["total_irrigation"] - merged["old036_total_irrigation"]
    merged["delta_n_vs_036"] = merged["total_n"] - merged["old036_total_n"]
    return merged


def summarize_by_checkpoint(eval_df: pd.DataFrame) -> pd.DataFrame:
    ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")].copy()
    if ok.empty:
        return pd.DataFrame()
    return (
        ok.groupby("checkpoint_step", as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_PFP_N=("PFP_N", "mean"),
            mean_early_irrigation=("early_dap1_10_irrigation", "mean"),
            mean_early_n=("early_dap1_10_n", "mean"),
            mean_early_irrigation_share=("early_irrigation_share", "mean"),
            mean_early_n_share=("early_n_share", "mean"),
            mean_delta_yield_vs_036=("delta_final_grnwt_vs_036", "mean"),
            mean_delta_i_vs_036=("delta_irrigation_vs_036", "mean"),
            mean_delta_n_vs_036=("delta_n_vs_036", "mean"),
            max_swfac=("max_swfac", "max"),
            max_nstres=("max_nstres", "max"),
        )
        .sort_values("checkpoint_step")
    )


def markdown_table(df: pd.DataFrame, round_digits: int | None = None) -> str:
    """Render a small DataFrame as a Markdown table without optional tabulate."""
    if df.empty:
        return "无记录"
    show = df.copy()
    if round_digits is not None:
        show = show.round(round_digits)
    show = show.astype(str)
    headers = list(show.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in show.values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_record(
    config: dict[str, Any],
    split: pd.DataFrame,
    train_df: pd.DataFrame,
    reset_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    by_ckpt: pd.DataFrame,
    train_years: list[int],
    val_years: list[int],
) -> None:
    lines: list[str] = []
    lines.append("# 037_03：LCA early starter cap 自由时序 MaskablePPO clean retrain 记录\n")
    lines.append("\n## 任务边界\n")
    lines.append("- 本任务只做 LCA 单站点 smoke，不扩展五站点。\n")
    lines.append("- 相对 036 主线唯一新增约束：DAP1-10 累计灌溉 ≤30mm、累计施氮 ≤40kg/ha。\n")
    lines.append("- 不修改 reward、不修改 PPO 超参数、不修改训练/验证年份划分。\n")
    lines.append("\n## 配置\n")
    lines.append(f"- 站点：{STATION}/{SITE}。\n")
    lines.append(f"- seed：{SEED}。\n")
    lines.append(f"- 训练步数：{TOTAL_TIMESTEPS}。\n")
    lines.append(f"- checkpoint：{', '.join(map(str, CHECKPOINT_STEPS))}。\n")
    lines.append(f"- 训练年份：{', '.join(map(str, train_years))}。\n")
    lines.append(f"- 验证年份：{', '.join(map(str, val_years))}。\n")
    lines.append("- 继承 036：IC=1、IRRIG=L/FERTI=L、自由时序、7天最小间隔、单季水氮上限、DAP90后禁氮、原 stress-aware reward。\n")
    lines.append("\n## 训练 checkpoint\n")
    lines.append(markdown_table(train_df))
    lines.append("\n\n## 训练年份采样次数\n")
    lines.append(markdown_table(reset_df) if not reset_df.empty else "无记录")
    lines.append("\n\n## 验证年均值按 checkpoint 汇总\n")
    lines.append(markdown_table(by_ckpt, round_digits=4) if not by_ckpt.empty else "无成功评估")
    lines.append("\n\n## 每年评估明细\n")
    cols = [
        "year",
        "checkpoint_step",
        "run_status",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
        "early_dap1_10_irrigation",
        "early_dap1_10_n",
        "early_irrigation_share",
        "early_n_share",
        "delta_final_grnwt_vs_036",
        "delta_irrigation_vs_036",
        "delta_n_vs_036",
        "max_swfac",
        "max_nstres",
        "action_sequence",
    ]
    lines.append(markdown_table(eval_df[[c for c in cols if c in eval_df.columns]], round_digits=4) if not eval_df.empty else "无记录")
    lines.append("\n\n## 初步判读规则\n")
    lines.append("- 如果 early_dap1_10 水氮显著下降，但产量相对 036 大幅下降，则说明约束过强或 reward 不支持后续补偿。\n")
    lines.append("- 如果 early dump 被压住且产量/资源效率维持，则可考虑五站点扩展。\n")
    lines.append("- 如果 agent 改成晚期集中补偿或产生高胁迫，则不能直接扩展，需要继续审计。\n")
    DOC.write_text("".join(lines), encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    patch_base_wrapper()
    batch.OUT = OUT
    batch.SEED = SEED
    batch.TOTAL_TIMESTEPS = TOTAL_TIMESTEPS
    batch.CHECKPOINT_STEPS = CHECKPOINT_STEPS
    shutil.copyfile(batch.CONFIG, OUT / "configs" / batch.CONFIG.name)
    config = load_config()
    split, train_years, val_years = selected_split()
    direct_ppo.OUTPUT_ROOT = OUT
    selection_all = batch.build_selection(split)
    env_config = direct_ppo.build_env_config(config, selection_all)
    direct_ppo.write_yaml(config, OUT / "configs" / "037_03_config_snapshot.yaml")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "037_03_resolved_env_config.yaml")
    split.to_csv(OUT / "configs" / "037_03_lca_half_split_years.csv", index=False, encoding="utf-8-sig")

    rows: list[dict[str, Any]] = []
    reset_df = pd.DataFrame()
    try:
        train_df, reset_df = batch.train_station(config, env_config, STATION, train_years)
        train_df.to_csv(OUT / "evaluation" / "037_03_training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
        reset_df.to_csv(OUT / "logs" / "037_03_training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
        for _, row in train_df.iterrows():
            if str(row.get("run_status", "")).startswith("ok"):
                for year in val_years:
                    rows.append(batch.evaluate_checkpoint(config, env_config, row, int(year)))
    except Exception:
        err = traceback.format_exc()
        (OUT / "037_03_error.txt").write_text(err, encoding="utf-8")
        raise

    eval_df = pd.DataFrame(rows)
    if not eval_df.empty:
        eval_df = add_early_metrics(eval_df)
        eval_df = compare_to_036(eval_df)
    eval_df.to_csv(OUT / "evaluation" / "037_03_lca_validation_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    by_ckpt = summarize_by_checkpoint(eval_df)
    by_ckpt.to_csv(OUT / "tables" / "037_03_lca_by_checkpoint_summary.csv", index=False, encoding="utf-8-sig")
    write_record(config, split, train_df, reset_df, eval_df, by_ckpt, train_years, val_years)
    result = {
        "task": TASK,
        "station": STATION,
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoints": CHECKPOINT_STEPS,
        "early_dap_max": EARLY_DAP_MAX,
        "early_irrigation_cap_mm": EARLY_IRRIGATION_CAP_MM,
        "early_n_cap_kg_ha": EARLY_N_CAP_KG_HA,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "eval_summary": str((OUT / "evaluation" / "037_03_lca_validation_checkpoint_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "by_checkpoint": str((OUT / "tables" / "037_03_lca_by_checkpoint_summary.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "037_03_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not by_ckpt.empty:
        print(by_ckpt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
