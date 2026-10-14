from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as base
import run_linked_free_timing_ppo_dqn_train_smoke_034_04 as linked03404
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline03400


TASK_ID = "035_02"
STATION = "FQA"
YEAR = 2014
SEED = 0
CHECKPOINTS = [10000, 20000, 30000, 40000, 50000]
OUT = ROOT / "benchmark_results" / "035_02_fqa2014_linked_maskableppo_checkpoint_guardrail"
DOC = ROOT / "docs" / "035_02_fqa2014_linked_maskableppo_checkpoint_guardrail_record.md"
PROMPT = ROOT / "prompts" / "035_02_fqa2014_linked_maskableppo_checkpoint_guardrail.md"
BASELINE_SUMMARY = (
    ROOT
    / "benchmark_results"
    / "034_00_multisite_input_ic1_four_baseline_rebuild"
    / "evaluation"
    / "034_00_full_baseline_summary.csv"
)


def ensure_dirs() -> None:
    for rel in ["configs", "models/FQA", "daily_outputs/FQA", "snapshots/FQA/2014", "tensorboard/FQA", "evaluation"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copyfile(PROMPT, OUT / "configs" / PROMPT.name)


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


def load_config_and_env() -> tuple[dict[str, Any], dict[str, Any]]:
    config, env_config = linked03404.load_config_and_env()
    config["seed"] = SEED
    config["total_timesteps"] = max(CHECKPOINTS)
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    config["runtime"]["smoke_station"] = STATION
    config["runtime"]["smoke_year"] = YEAR
    direct_ppo.write_yaml(config, OUT / "configs" / "035_02_train_config.yaml")
    direct_ppo.write_yaml(env_config, OUT / "configs" / "035_02_resolved_env_config.yaml")
    return config, env_config


def configure_output_globals() -> None:
    base.OUT = OUT
    linked03404.OUT = OUT
    linked03404.TASK_ID = TASK_ID
    linked03404.DOC = DOC
    linked03404.PROMPT = PROMPT
    direct_ppo.OUTPUT_ROOT = OUT


def checkpoint_model_path(step: int) -> Path:
    return OUT / "models" / STATION / f"maskableppo_linked_seed{SEED}_step{step}.zip"


def train_checkpoints(config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    from sb3_contrib import MaskablePPO

    configure_output_globals()
    env = None
    rows: list[dict[str, Any]] = []
    try:
        env = base.make_env(config, env_config, STATION, YEAR, SEED, f"{STATION}_{YEAR}_{TASK_ID}_train", evaluation=False)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED,
            tensorboard_log=str(OUT / "tensorboard" / STATION),
            **base.ppo_kwargs(config),
        )
        prev = 0
        for step in CHECKPOINTS:
            chunk = int(step - prev)
            model.learn(total_timesteps=chunk, reset_num_timesteps=(prev == 0), progress_bar=False)
            path = checkpoint_model_path(step)
            model.save(str(path.with_suffix("")))
            rows.append(
                {
                    "algorithm": "MaskablePPO",
                    "station_code": STATION,
                    "year": YEAR,
                    "seed": SEED,
                    "checkpoint_step": step,
                    "chunk_timesteps": chunk,
                    "run_status": "ok",
                    "model_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "task": TASK_ID,
                    "linked_management_expected": True,
                }
            )
            prev = step
    except Exception:
        rows.append(
            {
                "algorithm": "MaskablePPO",
                "station_code": STATION,
                "year": YEAR,
                "seed": SEED,
                "checkpoint_step": prev,
                "run_status": "failed",
                "model_path": "",
                "task": TASK_ID,
                "notes": traceback.format_exc()[-5000:],
            }
        )
    finally:
        if env is not None:
            env.close()
    train = pd.DataFrame(rows)
    train.to_csv(OUT / "evaluation" / "035_02_training_checkpoints.csv", index=False, encoding="utf-8-sig")
    return train


def overview_modes(snapshot: Path) -> str:
    overview = snapshot / "OVERVIEW.OUT"
    if not overview.exists():
        return ""
    lines = [
        line.strip()
        for line in overview.read_text(encoding="utf-8", errors="ignore").splitlines()
        if "MANAGEMENT OPT" in line
    ]
    return " | ".join(lines[:3])


def add_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    out = eval_df.copy()
    for col in ["etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg", "simple_profit"]:
        if col not in out:
            out[col] = pd.NA
    for idx, row in out.iterrows():
        snapshot = ROOT / str(row.get("snapshot_path", "") or "")
        final_y = float(row.get("final_grnwt", row.get("grain_yield_kg_ha", float("nan"))))
        if snapshot.exists():
            metrics = baseline03400.metrics_from_snapshot(snapshot, final_y)
            for key, value in metrics.items():
                out.loc[idx, key] = value
    out["grain_yield_kg_ha"] = pd.to_numeric(out["final_grnwt"], errors="coerce")
    out["actual_irrigation_mm"] = pd.to_numeric(out["summary_irrigation_mm"], errors="coerce")
    out["actual_nitrogen_kg_ha"] = pd.to_numeric(out["summary_n_kg_ha"], errors="coerce")
    out["etcp_mm"] = pd.to_numeric(out["etcp_mm"], errors="coerce")
    out["WP_ET_kg_m3"] = pd.to_numeric(out["WP_ET_kg_m3"], errors="coerce")
    out["PFP_N_kg_kg"] = pd.to_numeric(out["PFP_N_kg_kg"], errors="coerce")
    out["simple_profit"] = out["grain_yield_kg_ha"] - 1.1 * out["actual_irrigation_mm"] - 1.58 * out["actual_nitrogen_kg_ha"]
    return out


def evaluate_checkpoints(config: dict[str, Any], env_config: dict[str, Any], train: pd.DataFrame) -> pd.DataFrame:
    configure_output_globals()
    rows: list[pd.DataFrame] = []
    for _, row in train.iterrows():
        if str(row.get("run_status")) != "ok":
            continue
        eval_row = row.copy()
        eval_row["train_years"] = str(YEAR)
        try:
            result = linked03404.evaluate_with_snapshot(config, env_config, eval_row)
            result["checkpoint_step"] = int(row["checkpoint_step"])
            # Move snapshot to checkpoint-specific folder, then update path.
            old_snapshot = ROOT / str(result.loc[0, "snapshot_path"])
            new_snapshot = OUT / "snapshots" / STATION / str(YEAR) / f"maskableppo_step{int(row['checkpoint_step'])}"
            if old_snapshot.exists() and old_snapshot.resolve() != new_snapshot.resolve():
                if new_snapshot.exists():
                    shutil.rmtree(new_snapshot)
                shutil.copytree(old_snapshot, new_snapshot)
                result.loc[0, "snapshot_path"] = str(new_snapshot.relative_to(ROOT)).replace("\\", "/")
            result.loc[0, "overview_management_opt"] = overview_modes(new_snapshot)
            result.loc[0, "overview_is_linked"] = "IRRIG   :L" in overview_modes(new_snapshot) and "FERT :L" in overview_modes(new_snapshot)
            rows.append(result)
        except Exception:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "algorithm": "MaskablePPO",
                            "checkpoint_step": int(row["checkpoint_step"]),
                            "run_status": "eval_failed",
                            "notes": traceback.format_exc()[-5000:],
                        }
                    ]
                )
            )
    eval_df = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    eval_df = add_metrics(eval_df) if len(eval_df) else eval_df
    eval_df.to_csv(OUT / "evaluation" / "035_02_checkpoint_eval_summary.csv", index=False, encoding="utf-8-sig")
    return eval_df


def expert_row() -> pd.Series:
    baseline = pd.read_csv(BASELINE_SUMMARY, keep_default_na=False)
    baseline["year"] = pd.to_numeric(baseline["year"], errors="coerce").astype(int)
    row = baseline[
        baseline["station_code"].astype(str).eq(STATION)
        & baseline["year"].eq(YEAR)
        & baseline["scenario"].astype(str).eq("official_extension_expert")
    ]
    if len(row) != 1:
        raise ValueError(f"未找到 {STATION}{YEAR} official_extension_expert")
    out = row.iloc[0].copy()
    for col in ["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["simple_profit"] = float(out["grain_yield_kg_ha"]) - 1.1 * float(out["actual_irrigation_mm"]) - 1.58 * float(out["actual_nitrogen_kg_ha"])
    return out


def guardrail(eval_df: pd.DataFrame) -> pd.DataFrame:
    expert = expert_row()
    out = eval_df.copy()
    out["delta_yield_vs_expert"] = out["grain_yield_kg_ha"] - float(expert["grain_yield_kg_ha"])
    out["delta_i_vs_expert"] = out["actual_irrigation_mm"] - float(expert["actual_irrigation_mm"])
    out["delta_n_vs_expert"] = out["actual_nitrogen_kg_ha"] - float(expert["actual_nitrogen_kg_ha"])
    out["delta_wp_vs_expert"] = out["WP_ET_kg_m3"] - float(expert["WP_ET_kg_m3"])
    out["delta_pfp_vs_expert"] = out["PFP_N_kg_kg"] - float(expert["PFP_N_kg_kg"])
    out["delta_profit_vs_expert"] = out["simple_profit"] - float(expert["simple_profit"])
    out["beats_expert_yield"] = out["delta_yield_vs_expert"] >= 0
    out["beats_expert_wp"] = out["delta_wp_vs_expert"] > 0
    out["beats_expert_pfp"] = out["delta_pfp_vs_expert"] > 0
    out["guardrail_pass"] = (
        out["interface_pass"].astype(bool)
        & out["beats_expert_yield"].astype(bool)
        & (out[["beats_expert_yield", "beats_expert_wp", "beats_expert_pfp"]].any(axis=1))
    )
    out = out.sort_values(
        ["guardrail_pass", "simple_profit", "actual_nitrogen_kg_ha", "actual_irrigation_mm"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    out["guardrail_rank"] = range(1, len(out) + 1)
    out.to_csv(OUT / "evaluation" / "035_02_guardrail_ranking.csv", index=False, encoding="utf-8-sig")
    return out


def write_record(train: pd.DataFrame, eval_df: pd.DataFrame, guard: pd.DataFrame, elapsed: float) -> None:
    passed = guard[guard["guardrail_pass"].astype(bool)] if len(guard) else pd.DataFrame()
    best = guard.iloc[0].to_dict() if len(guard) else {}
    lines = [
        "# 035_02 FQA2014 linked MaskablePPO checkpoint guardrail 重训记录",
        "",
        "## 结论先说",
        "",
        f"- 已训练 checkpoint 数：{len(train[train['run_status'].eq('ok')])}。",
        f"- 已评估 checkpoint 数：{len(eval_df)}。",
        f"- guardrail 通过数：{len(passed)}。",
        f"- 当前 guardrail 排名第一：step {best.get('checkpoint_step', 'NA')}。",
        "- 本任务只跑 seed0/FQA2014，不代表跨 seed、跨年、跨站点结论。",
        "",
        "## 训练 checkpoint",
        "",
        md_table(train, 20),
        "",
        "## checkpoint 评估",
        "",
        md_table(
            eval_df[
                [
                    "checkpoint_step",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                    "reward_stress_aware_sum",
                    "interface_pass",
                    "action_sequence",
                ]
            ]
            if len(eval_df)
            else eval_df,
            20,
        ),
        "",
        "## guardrail 排名",
        "",
        md_table(
            guard[
                [
                    "guardrail_rank",
                    "checkpoint_step",
                    "guardrail_pass",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                    "delta_yield_vs_expert",
                    "delta_wp_vs_expert",
                    "delta_pfp_vs_expert",
                    "delta_profit_vs_expert",
                ]
            ]
            if len(guard)
            else guard,
            20,
        ),
        "",
        "## 解释边界",
        "",
        "- 若没有 checkpoint 通过，本轮不能现场追加步数或调 reward。",
        "- 若有 checkpoint 通过，只能作为 FQA2014 seed0 候选，后续仍需 seed/年份验证。",
        "- 所有 checkpoint 均要求 linked interface_pass，否则不得作为候选。",
        "",
        f"耗时：{elapsed:.1f} 秒。",
    ]
    text = "\n".join(lines) + "\n"
    DOC.write_text(text, encoding="utf-8")
    (OUT / DOC.name).write_text(text, encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config = load_config_and_env()
    train = train_checkpoints(config, env_config)
    eval_df = evaluate_checkpoints(config, env_config, train)
    guard = guardrail(eval_df) if len(eval_df) else pd.DataFrame()
    write_record(train, eval_df, guard, time.time() - start)
    result = {
        "task": TASK_ID,
        "trained_checkpoints": int(len(train[train["run_status"].eq("ok")])) if len(train) else 0,
        "evaluated_checkpoints": int(len(eval_df)),
        "guardrail_pass": int(guard["guardrail_pass"].sum()) if len(guard) and "guardrail_pass" in guard else 0,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "guardrail_csv": str((OUT / "evaluation" / "035_02_guardrail_ranking.csv").relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "035_02_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if len(guard):
        print(
            guard[
                [
                    "guardrail_rank",
                    "checkpoint_step",
                    "guardrail_pass",
                    "grain_yield_kg_ha",
                    "actual_irrigation_mm",
                    "actual_nitrogen_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "simple_profit",
                    "action_sequence",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

