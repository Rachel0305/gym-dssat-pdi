"""Task 190: one from-scratch 50K PPO learning curve, exactly matching Task 188.

This intentionally explicit wrapper reuses the proven Task-188 runtime chain:
six 2005-derived weather scenarios, direct-rain observations, dense daily-GRNWT
reward, seed 0, and the one-day irrigation gap.  The only experimental change
is the training length and the checkpoint schedule.  It does not resume a 2K
policy, mutate Task188 files, or launch workers in parallel.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_188_sya_balanced_irrigation_gap1_smoke2k as task188


TASK_ID = "190"
TASK_NAME = "sya_balanced_irrigation_gap1_50k_learning_curve"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_balanced_irrigation_gap1_50k_learning_curve.md"
TASK188_CONFIG = ROOT / "configs" / "188_sya_balanced_irrigation_gap1_smoke2k.json"
TASK188_RESULT = ROOT / "benchmark_results" / "188_sya_balanced_irrigation_gap1_smoke2k" / "188_result.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
STATION = "SYA"
CHECKPOINT_STEPS = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
TARGET_DAPS = [97, 104]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无记录。"
    work = frame.copy()
    for column in work.select_dtypes(include=["number"]).columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    return work.to_markdown(index=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(cfg: dict[str, Any]) -> None:
    """Guard the one-factor design before creating an output directory."""
    expected = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "station_code": STATION,
        "source_year": 2005,
        "total_timesteps": 50000,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "seed": 0,
        "scenario_sampling": "deterministic_round_robin",
        "scenario_order": ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"],
        "direct_rain_observation": True,
        "run_training": True,
        "from_scratch": True,
        "overwrite_existing_output": False,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Task190 config mismatch for {key}: expected {value!r}, got {cfg.get(key)!r}")
    safety = cfg.get("action_safety_override", {})
    if (int(safety.get("min_days_between_irrigation", -1)), int(safety.get("min_days_between_fertilization", -1)), int(safety.get("season_irrigation_cap_mm", -1))) != (1, 7, 240):
        raise ValueError("Task190 requires Task188 safety: irrigation gap=1, N gap=7, seasonal I cap=240.")
    fixed = cfg.get("fixed_ppo_parameters", {})
    expected_ppo = {"learning_rate": 0.0003, "gamma": 1.0, "gae_lambda": 1.0, "n_steps": 144, "batch_size": 144, "n_epochs": 5, "clip_range": 0.2}
    if fixed != expected_ppo or cfg.get("ppo_overrides") != {"ent_coef": 0.02}:
        raise ValueError("Task190 PPO parameters must exactly match Task188.")
    base = load_json(TASK188_CONFIG)
    for key in ["station_code", "source_year", "seed", "scenario_manifest", "input_root", "scenario_sampling", "scenario_order", "direct_rain_observation", "ppo_overrides", "fixed_ppo_parameters"]:
        if cfg.get(key) != base.get(key):
            raise ValueError(f"Task190 drifted from Task188 at {key}: {cfg.get(key)!r} != {base.get(key)!r}")
    if not TASK188_RESULT.exists():
        raise FileNotFoundError(f"Task188 comparison result missing: {rel(TASK188_RESULT)}")


def preflight(cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    validate_config(cfg)
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing Task190 output: {rel(OUT)}")
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    task177 = task188.task186.task177
    gate = task177.task174.task173.task172.task168.verify_gate(cfg)
    manifest = task177.task174.task173.task172.task168.read_manifest(cfg)
    split = task177.task174.task173.task172.task168.load_split(manifest)
    if split["year"].astype(int).tolist() != [2005, 2006, 2007, 2008, 2009, 2010]:
        raise ValueError("Scenario manifest/order is not the required six-scenario fixed order.")
    if not bool(gate.get("physical_response_passed")):
        raise RuntimeError("Physical weather-response gate is not passed.")
    return gate, manifest, split


def set_task177_globals(task177: Any) -> dict[str, Any]:
    old = {key: getattr(task177, key) for key in ["TASK_ID", "TASK_NAME", "OUT", "DOC", "PROMPT", "CHECKPOINT_STEPS"]}
    task177.TASK_ID = TASK_ID
    task177.TASK_NAME = TASK_NAME
    task177.OUT = OUT
    task177.DOC = DOC
    task177.PROMPT = PROMPT
    task177.CHECKPOINT_STEPS = list(CHECKPOINT_STEPS)
    return old


def restore_task177_globals(task177: Any, old: dict[str, Any]) -> None:
    for key, value in old.items():
        setattr(task177, key, value)


def rename_output(source: Path, target: Path) -> None:
    if source.exists():
        if target.exists():
            raise FileExistsError(f"Refuse to overwrite generated output: {rel(target)}")
        source.rename(target)


def rename_inherited_outputs() -> None:
    pairs = [
        ("evaluation/032_22_training_checkpoint_inventory.csv", "evaluation/190_training_checkpoint_inventory.csv"),
        ("evaluation/173_scenario_episode_metrics.csv", "evaluation/190_scenario_episode_metrics.csv"),
        ("evaluation/173_action_pair_detail.csv", "evaluation/190_action_pair_detail.csv"),
        ("evaluation/173_action_change_summary.csv", "evaluation/190_action_change_summary.csv"),
        ("evaluation/177_grnwt_increment_summary.csv", "evaluation/190_grnwt_increment_summary.csv"),
        ("logs/032_22_training_year_reset_counts.csv", "logs/190_training_scenario_reset_counts.csv"),
        ("logs/032_22_SYA_training_year_switch_log.csv", "logs/190_SYA_training_scenario_switch_log.csv"),
        ("logs/186_episode_diagnostics.csv", "logs/190_episode_diagnostics.csv"),
        ("logs/186_decision_point_policy_values.csv", "logs/190_decision_point_policy_values.csv"),
    ]
    for old_name, new_name in pairs:
        rename_output(OUT / old_name, OUT / new_name)


def daily_cap_and_mask_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics.itertuples(index=False):
        daily_path = ROOT / str(metric.daily_csv_path)
        if not daily_path.exists():
            continue
        daily = pd.read_csv(daily_path, keep_default_na=False).sort_values("dap").reset_index(drop=True)
        cumulative = pd.to_numeric(daily.get("season_cumulative_irrigation"), errors="coerce")
        cap_rows = daily[cumulative.ge(240.0 - 1e-6)]
        post_mask = pd.to_numeric(daily.get("mask_valid_action_count"), errors="coerce")
        pre_mask = post_mask.shift(1).where(daily["dap"].diff().eq(1))
        late = daily.assign(pre_action_mask_count=pre_mask).loc[lambda x: x["dap"].ge(90)]
        rows.append(
            {
                "checkpoint_step": int(metric.checkpoint_step),
                "weather_variant": str(metric.weather_variant),
                "reached_i240": not cap_rows.empty,
                "first_i240_dap": int(cap_rows["dap"].iloc[0]) if not cap_rows.empty else np.nan,
                "late_pre_action_noop_only_rate": float(pd.to_numeric(late["pre_action_mask_count"], errors="coerce").eq(1).mean()) if not late.empty else np.nan,
                "target_dap97_104_noop_only": bool(pd.to_numeric(late.loc[late["dap"].isin(TARGET_DAPS), "pre_action_mask_count"], errors="coerce").eq(1).all()),
            }
        )
    return pd.DataFrame(rows)


def build_learning_curve(metrics: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap_mask = daily_cap_and_mask_summary(metrics)
    cap_mask.to_csv(OUT / "evaluation" / "190_checkpoint_cap_mask_detail.csv", index=False, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINT_STEPS:
        frame = metrics[pd.to_numeric(metrics["checkpoint_step"], errors="coerce").eq(checkpoint)].copy()
        action = actions[pd.to_numeric(actions["checkpoint_step"], errors="coerce").eq(checkpoint)].copy()
        cap = cap_mask[pd.to_numeric(cap_mask.get("checkpoint_step", pd.Series(dtype=float)), errors="coerce").eq(checkpoint)].copy()
        non_original = action[action["weather_variant"].ne("original")].copy() if not action.empty else pd.DataFrame()
        action_sequences = frame.get("action_sequence", pd.Series(dtype=str)).astype(str)
        rows.append(
            {
                "checkpoint_step": checkpoint,
                "scenario_rows": int(len(frame)),
                "mean_final_grnwt": float(pd.to_numeric(frame.get("final_grnwt"), errors="coerce").mean()),
                "min_final_grnwt": float(pd.to_numeric(frame.get("final_grnwt"), errors="coerce").min()),
                "max_final_grnwt": float(pd.to_numeric(frame.get("final_grnwt"), errors="coerce").max()),
                "mean_profit_simple": float(pd.to_numeric(frame.get("profit_simple"), errors="coerce").mean()),
                "min_profit_simple": float(pd.to_numeric(frame.get("profit_simple"), errors="coerce").min()),
                "max_profit_simple": float(pd.to_numeric(frame.get("profit_simple"), errors="coerce").max()),
                "mean_total_irrigation": float(pd.to_numeric(frame.get("total_irrigation"), errors="coerce").mean()),
                "mean_total_n": float(pd.to_numeric(frame.get("total_n"), errors="coerce").mean()),
                "mean_PFP_N": float(pd.to_numeric(frame.get("PFP_N"), errors="coerce").mean()),
                "action_diversity_unique_full_season_sequences": int(action_sequences.nunique()),
                "action_diversity_fraction": float(action_sequences.nunique() / len(frame)) if len(frame) else np.nan,
                "mean_action_change_rate_after_dap1": float(pd.to_numeric(non_original.get("action_change_rate_after_dap1"), errors="coerce").mean()) if not non_original.empty else np.nan,
                "max_action_change_rate_after_dap1": float(pd.to_numeric(non_original.get("action_change_rate_after_dap1"), errors="coerce").max()) if not non_original.empty else np.nan,
                "weather_response_rate_nonoriginal_fraction_gt_0p01": float(pd.to_numeric(non_original.get("action_change_rate_after_dap1"), errors="coerce").gt(0.01).mean()) if not non_original.empty else np.nan,
                "all_scenarios_reached_i240": bool(cap["reached_i240"].all()) if not cap.empty else False,
                "mean_first_i240_dap": float(pd.to_numeric(cap.loc[cap["reached_i240"], "first_i240_dap"], errors="coerce").mean()) if not cap.empty else np.nan,
                "late_pre_action_noop_only_rate": float(pd.to_numeric(cap.get("late_pre_action_noop_only_rate"), errors="coerce").mean()) if not cap.empty else np.nan,
                "all_target_dap97_104_noop_only": bool(cap["target_dap97_104_noop_only"].all()) if not cap.empty else False,
            }
        )
    curve = pd.DataFrame(rows)
    curve.to_csv(OUT / "evaluation" / "190_learning_curve_by_checkpoint.csv", index=False, encoding="utf-8-sig")
    return curve, cap_mask


def conclusion_from_curve(status: str, curve: pd.DataFrame) -> str:
    if status != "completed" or curve.empty or int(curve["checkpoint_step"].max()) != 50000:
        return "D. 训练未完成：不能把中途 checkpoint 当作 50K 结论。"
    final = curve[curve["checkpoint_step"].eq(50000)].iloc[0]
    early = curve[curve["checkpoint_step"].eq(2000)].iloc[0]
    best = curve.loc[curve["mean_profit_simple"].idxmax()]
    # Do not label a few-unit numerical fluctuation as collapse.  A collapse
    # needs a material final loss (at least 2%) after the best checkpoint.
    material_loss = (float(best["mean_profit_simple"]) - float(final["mean_profit_simple"])) / max(abs(float(best["mean_profit_simple"])), 1.0)
    if int(best["checkpoint_step"]) < 50000 and material_loss >= 0.02:
        return "B. 支持训练中途策略坍缩：某个中间 checkpoint 的平均 profit 至少比50K高2%，需结合动作多样性和天气响应解释。"
    if float(final["mean_profit_simple"]) > float(early["mean_profit_simple"]) + 1e-6 and float(final["weather_response_rate_nonoriginal_fraction_gt_0p01"]) > float(early["weather_response_rate_nonoriginal_fraction_gt_0p01"]):
        return "A. 支持 2K 训练不足：50K 的平均收益和天气响应均优于2K；但若仍过早耗尽240 mm，则训练步数不是唯一问题。"
    return "C. 不支持“只是 2K 太少”：即使达到 50K，策略仍未同时表现出收益改善、资源时序改善和天气响应；下一步应回到 mask、观测、reward/time-credit 或训练流程审计。"


def write_record(result: dict[str, Any], curve: pd.DataFrame, cap_mask: pd.DataFrame, failure_note: str = "") -> None:
    checkpoints = curve[curve["checkpoint_step"].isin([1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000])] if not curve.empty else curve
    lines = [
        "# Task 190：SYA 平衡六天气情景 gap=1 PPO 50K 学习曲线记录",
        "",
        "## 1. 实验目的",
        "",
        "判断 Task188 的 2K PPO 是否仅仅因为训练不足，还是在更长训练中保持低质量固定策略或发生后期坍缩。",
        "",
        "## 2. 与 Task188 的唯一变化",
        "",
        "仅将训练步数从 2,000 增至 50,000，并把 checkpoint 扩展为 1K/2K/5K/10K/20K/30K/40K/50K。训练从零开始，不加载或续训 Task188 模型；天气、reward、observation、动作网格、gap=1、240 mm 上限、seed 和 PPO 参数保持不变。",
        "",
        "## 3. 代码、输入和命令",
        "",
        f"- prompt：`{rel(PROMPT)}`",
        f"- config：`{rel(DEFAULT_CONFIG)}`",
        f"- 主脚本：`{rel(Path(__file__))}`",
        f"- 复用运行链：`{rel(ROOT / 'src' / 'run_188_sya_balanced_irrigation_gap1_smoke2k.py')}`、`src/run_186_sya_balanced_scenario_diagnostic_smoke2k.py`。",
        f"- Task188 对照结果：`{rel(TASK188_RESULT)}`。",
        f"- 六天气情景 manifest：`{result.get('scenario_manifest', '')}`。",
        "",
        "```powershell",
        "docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python src/run_190_sya_balanced_irrigation_gap1_50k_learning_curve.py --config configs/190_sya_balanced_irrigation_gap1_50k_learning_curve.json",
        "```",
        "",
        "## 4. Preflight",
        "",
        f"- 状态：`{result.get('preflight_status', 'not_run')}`；物理天气响应 gate：`{result.get('physical_gate_passed', False)}`。",
        "- 已核对：50K、8 个 checkpoint、六情景固定顺序、seed=0、gap=1、N gap=7、季节灌溉上限240 mm，以及 Task188 PPO 参数一致。",
        "- Docker 语法检查由启动前的外部命令执行；本脚本在创建结果目录前再次执行配置、manifest、gate 和输出目录检查。",
        "",
        "## 5. 每 checkpoint 学习曲线",
        "",
        md_table(checkpoints),
        "",
        "字段说明：action diversity 是六个天气情景中不同完整管理序列的数目；weather response rate 是五个非 original 情景中 post-DAP1 action change rate 大于0.01的比例；late no-op rate 使用 action 后 mask 前移一日得到 pre-action mask，避免把下一状态误当决策状态。",
        "",
        "## 6. 灌溉上限与后期 mask 明细",
        "",
        md_table(cap_mask),
        "",
        "## 7. 运行状态、失败与修正",
        "",
        f"- 训练状态：`{result.get('status', 'not_started')}`；是否达到50K：`{result.get('reached_50k', False)}`。",
        f"- 失败/修正：{failure_note or '无；本任务只启动一次长训练，不进行自动重试。'}",
        "",
        "## 8. 结论和证据边界",
        "",
        f"{result.get('conclusion', 'D. 训练未完成：暂不能判断。')}",
        "学习曲线只判断该固定站点、固定品种、2005 派生六天气情景下的训练动力学；它不证明跨站点、跨品种或真实未来气候的普适最优性。",
        "",
        "## 9. 输出文件",
        "",
        f"- 结果摘要：`{rel(OUT / '190_result.json')}`",
        f"- checkpoint inventory：`{rel(OUT / 'evaluation' / '190_training_checkpoint_inventory.csv')}`",
        f"- 学习曲线：`{rel(OUT / 'evaluation' / '190_learning_curve_by_checkpoint.csv')}`",
        f"- 六情景指标：`{rel(OUT / 'evaluation' / '190_scenario_episode_metrics.csv')}`",
        f"- 天气响应与动作差异：`{rel(OUT / 'evaluation' / '190_action_change_summary.csv')}`",
        f"- 上限/mask 明细：`{rel(OUT / 'evaluation' / '190_checkpoint_cap_mask_detail.csv')}`",
        f"- 逐日回放：`{rel(OUT / 'daily_outputs' / 'SYA')}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    gate, manifest, split = preflight(cfg)
    OUT.mkdir(parents=True, exist_ok=False)
    for folder in ["configs", "evaluation", "logs", "models", "daily_outputs"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, OUT / "configs" / config_path.name)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    (OUT / "configs" / "190_training_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    execution = OUT / "logs" / "190_execution_status.json"
    execution.write_text(json.dumps({"status": "started", "total_timesteps": 50000, "checkpoint_steps": CHECKPOINT_STEPS}, ensure_ascii=False, indent=2), encoding="utf-8")

    task177 = task188.task186.task177
    base03222 = task177.task174.task173.task172.task168.base03222
    engine = task177.task174.task173.task172.task168.engine
    old_task177 = set_task177_globals(task177)
    old_random_env = base03222.RandomYearEnv
    old_callback = base03222.FixedStepCheckpointCallback
    task177_patch, original_patch, original_restore = task188.patch_runtime_gap1()
    old_runtime: Any = None
    status = "completed"
    error = ""
    metrics = pd.DataFrame()
    actions = pd.DataFrame()
    curve = pd.DataFrame()
    cap_mask = pd.DataFrame()
    try:
        old_runtime = task177_patch.patch_runtime_177(manifest)
        # The inherited chain initially installs 2K constants.  This is the only
        # runtime extension: reset them to Task190's 50K schedule, then restore
        # the already-patched Task188 config and Task177 dense-reward wrapper.
        runtime_load_config = base03222.load_config
        engine.patch_base_module(OUT, DOC, int(cfg["total_timesteps"]), list(cfg["checkpoint_steps"]))
        base03222.load_config = runtime_load_config
        base03222.base.StressAwareDiscreteWrapper = task177.DenseYieldCreditWrapper
        base03222.RandomYearEnv = task188.task186.BalancedScenarioEnv
        base03222._original_fixed_step_callback = old_callback
        base03222.FixedStepCheckpointCallback = task188.task186.CombinedCheckpointCallback
        print("[Task190 1/4] Starting the single from-scratch 50K PPO training process.", flush=True)
        base03222.main()
        print("[Task190 2/4] Replaying every saved checkpoint on all six scenarios.", flush=True)
        metrics, actions = task177.task174.task173.evaluate_and_audit_173(split)
        task177.build_increment_summary()
        rename_inherited_outputs()
        metrics = pd.read_csv(OUT / "evaluation" / "190_scenario_episode_metrics.csv", keep_default_na=False)
        actions = pd.read_csv(OUT / "evaluation" / "190_action_change_summary.csv", keep_default_na=False)
        curve, cap_mask = build_learning_curve(metrics, actions)
        print("[Task190 3/4] Wrote the checkpoint learning curve and action/mask audit.", flush=True)
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        try:
            delattr(base03222, "_original_fixed_step_callback")
        except AttributeError:
            pass
        base03222.FixedStepCheckpointCallback = old_callback
        base03222.RandomYearEnv = old_random_env
        if old_runtime is not None:
            task177_patch.restore_runtime_177(*old_runtime)
        task177.patch_runtime_177 = original_patch
        task177.restore_runtime_177 = original_restore
        restore_task177_globals(task177, old_task177)

    inventory_path = OUT / "evaluation" / "190_training_checkpoint_inventory.csv"
    inventory = pd.read_csv(inventory_path, keep_default_na=False) if inventory_path.exists() else pd.DataFrame()
    completed_steps = sorted(pd.to_numeric(inventory.get("checkpoint_step", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()) if not inventory.empty else []
    ok_steps = sorted(pd.to_numeric(inventory.loc[inventory.get("run_status", pd.Series(dtype=str)).astype(str).str.startswith("ok"), "checkpoint_step"], errors="coerce").dropna().astype(int).tolist()) if not inventory.empty else []
    reached = status == "completed" and ok_steps == CHECKPOINT_STEPS
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "training_executed": True,
        "from_scratch": True,
        "total_timesteps": int(cfg["total_timesteps"]),
        "checkpoint_steps_required": CHECKPOINT_STEPS,
        "checkpoint_steps_inventory": completed_steps,
        "checkpoint_steps_ok": ok_steps,
        "reached_50k": reached,
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "preflight_status": "passed",
        "scenario_manifest": cfg["scenario_manifest"],
        "scenario_sampling": cfg["scenario_sampling"],
        "scenario_order": cfg["scenario_order"],
        "seed": cfg["seed"],
        "action_safety": cfg["action_safety_override"],
        "ppo_parameters": {**cfg["fixed_ppo_parameters"], **cfg["ppo_overrides"]},
        "only_change_from_task188": "total_timesteps and checkpoint schedule",
        "conclusion": conclusion_from_curve(status if reached else "failed", curve),
        "error": error[-5000:] if error else "",
        "outputs": {
            "learning_curve": rel(OUT / "evaluation" / "190_learning_curve_by_checkpoint.csv"),
            "inventory": rel(inventory_path),
            "scenario_metrics": rel(OUT / "evaluation" / "190_scenario_episode_metrics.csv"),
            "action_summary": rel(OUT / "evaluation" / "190_action_change_summary.csv"),
        },
    }
    (OUT / "190_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    execution.write_text(json.dumps({"status": status, "reached_50k": reached, "error": result["error"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_record(result, curve, cap_mask, "未自动重试；若失败，完整 traceback 已保留在结果 JSON 和终端日志。" if error else "")
    if status != "completed":
        raise RuntimeError(error)
    print("[Task190 4/4] Task190 completed.", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    print(json.dumps(run(config_path.resolve()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
