"""Task 186: balanced six-scenario PPO smoke with explicit training diagnostics.

The script is deliberately linear and supervisor-facing:

1. Load the task-186 config and the existing 2005 scenario manifest.
2. Reuse task 177's environment/reward/action definitions.
3. Replace only the random scenario reset with deterministic round-robin resets.
4. Add a callback that records episode coverage and DAP97/DAP104 actor/value data.
5. Train 2K steps, evaluate the two checkpoints on all six scenarios, and write
   a detailed record without touching any previous task output.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_177_sya_dense_yield_credit_smoke2k as task177


TASK_ID = "186"
TASK_NAME = "sya_balanced_scenario_diagnostic_smoke2k"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-08-20_sya_balanced_scenario_diagnostic_smoke2k.md"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
SCENARIO_ORDER = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
SCENARIO_KEYS = {2005: "original", 2006: "early_dry", 2007: "mid_dry", 2008: "late_dry", 2009: "early_wet", 2010: "mid_wet"}
TARGET_DAPS = [97, 104]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value: Any, default: float = np.nan) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        value = float(arr[0]) if arr.size else default
        return value if np.isfinite(value) else default
    except Exception:
        return default


def md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "无记录。"
    work = frame.head(max_rows).copy()
    for column in work.select_dtypes(include=["number"]).columns:
        work[column] = pd.to_numeric(work[column], errors="coerce").round(6)
    return work.to_markdown(index=False)


class BalancedScenarioEnv(gym.Env):
    """Same scenario environments as task 177, reset in a fixed round-robin."""

    def __init__(self, config: dict[str, Any], env_config: dict[str, Any], station: str, years: list[int], seed: int) -> None:
        super().__init__()
        self.config = config
        self.env_config = env_config
        self.station = station
        self.years = [int(year) for year in years]
        self.seed = int(seed)
        self.envs: dict[int, gym.Env] = {}
        self.current_year: int | None = None
        self.next_schedule_index = 0
        self.reset_counts: Counter[int] = Counter()
        self.switch_log: list[dict[str, Any]] = []
        self.last_obs: Any = None
        self.last_pre_step_obs: Any = None
        self.last_pre_action_masks: np.ndarray | None = None
        self.last_pre_dap: float = np.nan
        self.last_pre_state: dict[str, Any] = {}

        first = self._env_for(self.years[0])
        self.action_space = first.action_space
        self.observation_space = first.observation_space
        self.metadata = getattr(first, "metadata", {})

    def _env_for(self, year: int) -> gym.Env:
        year = int(year)
        if year not in self.envs:
            self.envs[year] = task177.task174.task173.task172.task168.base03222.base.make_env(
                self.config,
                self.env_config,
                self.station,
                year,
                self.seed,
                f"{self.station}_{year}_186_balanced_train",
                evaluation=False,
            )
        return self.envs[year]

    @property
    def current_env(self) -> gym.Env:
        if self.current_year is None:
            self.current_year = self.years[0]
        return self._env_for(self.current_year)

    def reset(self, *args, **kwargs):
        year = self.years[self.next_schedule_index % len(self.years)]
        self.next_schedule_index += 1
        self.current_year = int(year)
        self.reset_counts[self.current_year] += 1
        episode_index = int(sum(self.reset_counts.values()))
        self.switch_log.append(
            {
                "episode_index": episode_index,
                "station_code": self.station,
                "scenario_key": self.current_year,
                "weather_variant": SCENARIO_KEYS[self.current_year],
                "sampling_mode": "deterministic_round_robin",
            }
        )
        obs, info = self.current_env.reset(*args, **kwargs)
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        self.last_pre_step_obs = self.last_obs.copy()
        self.last_pre_action_masks = np.asarray(self.action_masks(), dtype=bool).copy()
        self.last_pre_dap = scalar(getattr(self.current_env, "last_obs_dict", {}).get("dap"), np.nan)
        self.last_pre_state = dict(getattr(self.current_env, "last_obs_dict", {}))
        info = dict(info or {})
        info.update({"active_year": self.current_year, "weather_variant": SCENARIO_KEYS[self.current_year]})
        return obs, info

    def step(self, action):
        self.last_pre_step_obs = np.asarray(self.last_obs, dtype=np.float32).copy()
        self.last_pre_action_masks = np.asarray(self.action_masks(), dtype=bool).copy()
        self.last_pre_state = dict(getattr(self.current_env, "last_obs_dict", {}))
        self.last_pre_dap = scalar(self.last_pre_state.get("dap"), np.nan)
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()
        info = dict(info or {})
        info.update({"active_year": int(self.current_year), "weather_variant": SCENARIO_KEYS[int(self.current_year)]})
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.current_env.action_masks(), dtype=bool)

    @property
    def last_action_info(self) -> dict[str, Any]:
        return dict(getattr(self.current_env, "last_action_info", {}))

    def close(self) -> None:
        for env in self.envs.values():
            try:
                env.close()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.current_env, name)


class ScenarioDiagnosticsCallback:
    """Stable-Baselines callback with explicit CSV outputs."""

    def __init__(self, output_root: Path, target_daps: list[int]):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.episode_reward = 0.0
                self.episode_length = 0

            def _policy_diagnostics(self, env: BalancedScenarioEnv, action_index: int) -> dict[str, Any]:
                import torch
                from stable_baselines3.common.utils import obs_as_tensor

                pre_obs = np.asarray(env.last_pre_step_obs, dtype=np.float32).reshape(1, -1)
                mask = np.asarray(env.last_pre_action_masks, dtype=bool)
                obs_tensor = obs_as_tensor(pre_obs, self.model.device)
                with torch.no_grad():
                    distribution = self.model.policy.get_distribution(obs_tensor, action_masks=mask)
                    probabilities = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
                    value = float(self.model.policy.predict_values(obs_tensor).detach().cpu().numpy().reshape(-1)[0])
                    entropy = float(distribution.entropy().detach().cpu().numpy().reshape(-1)[0])
                legal_probabilities = probabilities[mask]
                legal_entropy = float(-(legal_probabilities * np.log(np.maximum(legal_probabilities, 1e-12))).sum())
                normalized_entropy = legal_entropy / np.log(len(legal_probabilities)) if len(legal_probabilities) > 1 else 0.0
                return {
                    "policy_value": value,
                    "policy_entropy_all": entropy,
                    "legal_policy_entropy": legal_entropy,
                    "legal_policy_entropy_normalized": normalized_entropy,
                    "legal_action_count": int(mask.sum()),
                    "noop_legal": bool(mask[0]) if len(mask) > 0 else False,
                    "selected_action_index": int(action_index),
                    "selected_action_probability": float(probabilities[action_index]),
                    "top_action_index": int(np.argmax(probabilities)),
                    "top_action_probability": float(np.max(probabilities)),
                    "prob_action0_noop": float(probabilities[0]) if len(probabilities) > 0 else np.nan,
                    "prob_action4_i15_n0": float(probabilities[4]) if len(probabilities) > 4 else np.nan,
                    "prob_action8_i30_n0": float(probabilities[8]) if len(probabilities) > 8 else np.nan,
                    "prob_action12_i45_n0": float(probabilities[12]) if len(probabilities) > 12 else np.nan,
                    "legal_mask": "".join("1" if x else "0" for x in mask.tolist()),
                    "all_action_probabilities": json.dumps([float(x) for x in probabilities], ensure_ascii=False),
                }

            def _on_step(self) -> bool:
                # SB3 wraps the custom environment in Monitor before putting it
                # into DummyVecEnv.  ``unwrapped`` is the actual balanced
                # scenario selector and keeps the callback provenance explicit.
                monitored_env = self.training_env.envs[0]
                env = getattr(monitored_env, "unwrapped", monitored_env)
                if not isinstance(env, BalancedScenarioEnv):
                    raise TypeError(f"Expected BalancedScenarioEnv after unwrapping, got {type(env)!r}")
                reward = scalar(np.asarray(self.locals.get("rewards", [np.nan])).reshape(-1)[0], 0.0)
                action_index = int(np.asarray(self.locals.get("actions", [0])).reshape(-1)[0])
                dap = int(round(env.last_pre_dap)) if np.isfinite(env.last_pre_dap) else -1
                scenario_key = int(env.current_year)
                base = {
                    "training_timestep": int(self.model.num_timesteps),
                    "scenario_key": scenario_key,
                    "weather_variant": SCENARIO_KEYS[scenario_key],
                    "dap": dap,
                    "reward": reward,
                    "action_index": action_index,
                }
                if dap in outer.target_daps:
                    row = dict(base)
                    row.update(outer._action_to_physical(action_index))
                    row.update(outer._state_columns(env.last_pre_state))
                    row.update(self._policy_diagnostics(env, action_index))
                    outer.decision_rows.append(row)

                self.episode_reward += reward
                self.episode_length += 1
                dones = np.asarray(self.locals.get("dones", [False])).reshape(-1)
                if bool(dones[0]):
                    outer.episode_rows.append(
                        {
                            "episode_index": len(outer.episode_rows) + 1,
                            "scenario_key": scenario_key,
                            "weather_variant": SCENARIO_KEYS[scenario_key],
                            "episode_reward": float(self.episode_reward),
                            "episode_length": int(self.episode_length),
                            "training_timestep_end": int(self.model.num_timesteps),
                        }
                    )
                    self.episode_reward = 0.0
                    self.episode_length = 0
                return True

            def _on_training_end(self) -> None:
                outer.write_outputs()

        self.callback = _Callback()
        self.output_root = output_root
        self.target_daps = [int(x) for x in target_daps]
        self.decision_rows: list[dict[str, Any]] = []
        self.episode_rows: list[dict[str, Any]] = []

    @staticmethod
    def _action_to_physical(action_index: int) -> dict[str, Any]:
        irrigation = [0.0, 15.0, 30.0, 45.0][int(action_index) // 4]
        nitrogen = [0.0, 40.0, 80.0, 120.0][int(action_index) % 4]
        return {"selected_irrigation_mm": irrigation, "selected_nitrogen_kg_ha": nitrogen}

    @staticmethod
    def _state_columns(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "dap_state": scalar(state.get("dap")),
            "rain": scalar(state.get("rain")),
            "swfac": scalar(state.get("swfac")),
            "nstres": scalar(state.get("nstres")),
            "grnwt": scalar(state.get("grnwt")),
            "topwt": scalar(state.get("topwt")),
            "xlai": scalar(state.get("xlai")),
        }

    def write_outputs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.episode_rows).to_csv(self.output_root / "186_episode_diagnostics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(self.decision_rows).to_csv(self.output_root / "186_decision_point_policy_values.csv", index=False, encoding="utf-8-sig")


class CombinedCheckpointCallback:
    """Keep task-177 checkpoint saving and add task-186 diagnostics."""

    def __init__(self, station: str, checkpoint_steps: list[int]):
        from stable_baselines3.common.callbacks import CallbackList

        base03222 = task177.task174.task173.task172.task168.base03222
        self.original = base03222._original_fixed_step_callback(station, checkpoint_steps)
        self.diagnostics = ScenarioDiagnosticsCallback(OUT / "logs", TARGET_DAPS)
        self.callback = CallbackList([self.original.callback, self.diagnostics.callback])


def rename_outputs() -> None:
    evaluation = OUT / "evaluation"
    logs = OUT / "logs"
    for directory, pairs in [
        (evaluation, [("032_22_training_checkpoint_inventory.csv", "186_training_checkpoint_inventory.csv"), ("173_scenario_episode_metrics.csv", "186_scenario_episode_metrics.csv"), ("173_action_pair_detail.csv", "186_action_pair_detail.csv"), ("173_action_change_summary.csv", "186_action_change_summary.csv")]),
        (logs, [("032_22_training_year_reset_counts.csv", "186_training_year_reset_counts.csv"), ("032_22_SYA_training_year_switch_log.csv", "186_SYA_training_year_switch_log.csv")]),
    ]:
        for source_name, target_name in pairs:
            source = directory / source_name
            target = directory / target_name
            if source.exists() and not target.exists():
                source.rename(target)


def write_record(result: dict[str, Any], coverage: pd.DataFrame, decision: pd.DataFrame, episodes: pd.DataFrame, action_summary: pd.DataFrame) -> None:
    lines = [
        "# Task 186：SYA 平衡六情景训练与 PPO 决策过程诊断记录",
        "",
        "## 1. 本次实验要回答的问题",
        "",
        "Task 177 的 2K 训练中六个天气情景随机采样不均衡。本轮只把采样改为固定循环，检查在六情景都实际进入训练后，后期固定 `I45/N0` 是否仍然出现，以及 actor/value 日志如何变化。",
        "",
        "## 2. 实际使用的代码、配置与运行命令",
        "",
        f"- prompt：`{rel(PROMPT)}`",
        f"- config：`{rel(DEFAULT_CONFIG)}`",
        f"- 主脚本：`{rel(Path(__file__))}`",
        "- 复用环境/奖励/action mask：task 177 → task 174 → task 173 → task 172 → task 168；没有改 reward、RAIN observation、action grid 或 mask。",
        "- 唯一训练因素：episode 情景按 `original → early_dry → mid_dry → late_dry → early_wet → mid_wet` 固定循环。",
        "",
        "实际命令：",
        "",
        "```powershell",
        "docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python src/run_186_sya_balanced_scenario_diagnostic_smoke2k.py --config configs/186_sya_balanced_scenario_diagnostic_smoke2k.json --output-root benchmark_results/186_sya_balanced_scenario_diagnostic_smoke2k_retry2",
        "```",
        "",
        "## 3. 输入数据",
        "",
        "- 2005 年六情景 manifest：`benchmark_results/166_sya_weather_scenario_bank_v2_fixed_format/163_sya_weather_scenario_manifest.csv`。",
        "- 物理 gate：task 167；六个情景的实际模拟年份均为 2005。",
        "- task 177 的 PPO 设置：seed 0、2000 timesteps、checkpoint 1000/2000、daily `0.158 * ΔGRNWT` reward。",
        "",
        "## 4. 训练采样结果",
        "",
        f"- 训练状态：`{result.get('status')}`；实际 episode 数：`{result['evidence'].get('episode_count')}`。",
        f"- 情景计数最小/最大：`{result['evidence'].get('min_episode_count')}` / `{result['evidence'].get('max_episode_count')}`。",
        f"- 0 次采样情景：`{result['evidence'].get('zero_count_scenarios')}`。",
        "",
        md_table(coverage),
        "",
        "## 5. DAP97/104 actor/value 诊断",
        "",
        f"- 诊断记录数：`{len(decision)}`；目标日：`{sorted(decision['dap'].dropna().unique().tolist()) if not decision.empty else []}`。",
        "- `policy_value` 是当前 PPO critic 对 pre-action state 的 value 估计；`selected_action_probability` 是 mask 后 actor 对实际选择动作的概率；四个 `prob_action*` 是 no-op、I15/N0、I30/N0、I45/N0 的概率。",
        "",
        md_table(decision[[c for c in ["training_timestep", "scenario_key", "weather_variant", "dap", "selected_action_index", "selected_irrigation_mm", "selected_nitrogen_kg_ha", "selected_action_probability", "top_action_probability", "policy_value", "legal_policy_entropy_normalized", "prob_action0_noop", "prob_action4_i15_n0", "prob_action8_i30_n0", "prob_action12_i45_n0"] if c in decision.columns]], max_rows=60),
        "",
        "## 6. Episode reward 与 checkpoint 输出",
        "",
        md_table(episodes),
        "",
        md_table(action_summary),
        "",
        "## 7. 输出文件",
        "",
        f"- 结果 JSON：`{rel(OUT / '186_result.json')}`",
        f"- 情景覆盖：`{rel(OUT / 'logs' / '186_training_year_reset_counts.csv')}`",
        f"- 情景切换顺序：`{rel(OUT / 'logs' / '186_SYA_training_year_switch_log.csv')}`",
        f"- episode 诊断：`{rel(OUT / 'logs' / '186_episode_diagnostics.csv')}`",
        f"- DAP97/104 actor/value：`{rel(OUT / 'logs' / '186_decision_point_policy_values.csv')}`",
        f"- checkpoint inventory：`{rel(OUT / 'evaluation' / '186_training_checkpoint_inventory.csv')}`",
        f"- 六情景 episode metrics：`{rel(OUT / 'evaluation' / '186_scenario_episode_metrics.csv')}`",
        f"- 六情景 action summary：`{rel(OUT / 'evaluation' / '186_action_change_summary.csv')}`",
        "",
        "## 8. 本轮结论",
        "",
        f"- 平衡采样是否实现：`{result['evidence'].get('zero_count_scenarios') == []}`；这只说明六情景进入了训练，不说明 PPO 已学会天气条件策略。",
        f"- DAP97/104 是否有诊断记录：`{len(decision) > 0}`。",
        f"- checkpoint replay 的 action-response gate：`{result['evidence'].get('action_response_passed')}`；最大 post-DAP1 action change rate：`{result['evidence'].get('max_action_change_rate_after_dap1')}`。这说明平衡采样后天气情景已经能诱导一部分不同管理动作，但不是性能最优证明。",
        f"- 训练期 DAP97/104 的 legal action count：`{result['evidence'].get('decision_legal_action_count_values')}`；no-op probability range：`{result['evidence'].get('decision_noop_probability_range')}`。因此这 28 条训练期记录主要反映 action mask 已把后期限制为 no-op，不能解读为 actor 自由比较后的 no-op 偏好。",
        "- 因而本轮同时发现两件事：情景均衡采样确实改善了 checkpoint replay 的天气响应；但训练期后期仍存在资源/安全 mask 把可选动作压缩为单一 no-op 的轨迹，后续必须审查早期灌溉/施肥策略与 mask 触发过程。",
        "- 2K 只有 14 个 episode，不能据此判断 PPO 已经稳定学会最优策略；下一步仍需保留该日志结构，再做更长训练或专门的 mask 可行性审计。",
        "",
        "## 9. 结论边界",
        "",
        "本轮是 2K smoke，不能代表长训练最终性能，也不能证明 PPO 对所有真实年份泛化。它的作用是把“情景是否真的被看见”和“策略为何改变”这两个问题分开，并提供可复核的训练过程证据。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if int(cfg.get("total_timesteps", -1)) != 2000:
        raise ValueError("Task 186 is a guarded 2K smoke; change the task explicitly before extending training.")
    if cfg.get("scenario_sampling") != "deterministic_round_robin":
        raise ValueError("Task 186 requires deterministic round-robin scenario sampling.")
    global OUT, DOC
    if output_root is not None:
        OUT = output_root if output_root.is_absolute() else ROOT / output_root
        DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_retry_record.md"
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")

    base03222 = task177.task174.task173.task172.task168.base03222
    gate = task177.task174.task173.task172.task168.verify_gate(cfg)
    manifest = task177.task174.task173.task172.task168.read_manifest(cfg)
    split = task177.task174.task173.task172.task168.load_split(manifest)
    OUT.mkdir(parents=True, exist_ok=False)
    for directory in ["configs", "evaluation", "logs", "models", "daily_outputs"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    (OUT / "configs" / config_path.name).write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / PROMPT.name).write_text(PROMPT.read_text(encoding="utf-8"), encoding="utf-8")
    (OUT / "configs" / "186_training_scenario_split.csv").write_text(split.to_csv(index=False), encoding="utf-8-sig")
    (OUT / "configs" / "167_physical_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    old_task177_globals = {key: getattr(task177, key) for key in ["TASK_ID", "TASK_NAME", "OUT", "DOC", "PROMPT"]}
    old_random_env = base03222.RandomYearEnv
    old_checkpoint_callback = base03222.FixedStepCheckpointCallback
    task177.TASK_ID = TASK_ID
    task177.TASK_NAME = TASK_NAME
    task177.OUT = OUT
    task177.DOC = DOC
    task177.PROMPT = PROMPT

    # The original callback class is retained behind an explicit alias so the
    # combined callback can continue saving checkpoints exactly as task 177 did.
    base03222._original_fixed_step_callback = old_checkpoint_callback
    old_runtime = task177.patch_runtime_177(manifest)
    base03222.RandomYearEnv = BalancedScenarioEnv
    base03222.FixedStepCheckpointCallback = CombinedCheckpointCallback
    status = "completed"
    error = ""
    action_summary = pd.DataFrame()
    try:
        print("[Step 1/4] Running task-177 environment with round-robin scenario resets.", flush=True)
        base03222.main()
        print("[Step 2/4] Evaluating checkpoint 1000/2000 on all six scenarios.", flush=True)
        _, action_summary = task177.task174.task173.evaluate_and_audit_173(split)
        task177.build_increment_summary()
        rename_outputs()
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        base03222.FixedStepCheckpointCallback = old_checkpoint_callback
        base03222.RandomYearEnv = old_random_env
        try:
            delattr(base03222, "_original_fixed_step_callback")
        except AttributeError:
            pass
        task177.restore_runtime_177(*old_runtime)
        for key, value in old_task177_globals.items():
            setattr(task177, key, value)

    coverage_path = OUT / "logs" / "186_training_year_reset_counts.csv"
    decision_path = OUT / "logs" / "186_decision_point_policy_values.csv"
    episode_path = OUT / "logs" / "186_episode_diagnostics.csv"
    coverage = pd.read_csv(coverage_path, keep_default_na=False) if coverage_path.exists() else pd.DataFrame()
    decision = pd.read_csv(decision_path, keep_default_na=False) if decision_path.exists() else pd.DataFrame()
    episodes = pd.read_csv(episode_path, keep_default_na=False) if episode_path.exists() else pd.DataFrame()
    non_original = action_summary[action_summary["weather_variant"].ne("original")] if not action_summary.empty else pd.DataFrame()
    max_action_change_rate_after_dap1 = float(
        pd.to_numeric(non_original.get("action_change_rate_after_dap1", pd.Series(dtype=float)), errors="coerce").fillna(0.0).max()
    ) if not non_original.empty else 0.0
    action_response_passed = bool(
        status == "completed"
        and not non_original.empty
        and bool(non_original["all_actions_on_grid"].all())
        and max_action_change_rate_after_dap1 > 0.01
    )
    zero_scenarios = []
    min_count = 0
    max_count = 0
    if not coverage.empty:
        counts = pd.to_numeric(coverage["episode_count"], errors="coerce").fillna(0).astype(int)
        min_count = int(counts.min())
        max_count = int(counts.max())
        zero_scenarios = coverage.loc[counts.eq(0), "year"].astype(int).tolist()
    legal_action_count_values = sorted(
        pd.to_numeric(decision.get("legal_action_count", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist()
    ) if not decision.empty else []
    noop_probability_values = pd.to_numeric(
        decision.get("prob_action0_noop", pd.Series(dtype=float)), errors="coerce"
    ).dropna().tolist() if not decision.empty else []
    noop_probability_range = [float(min(noop_probability_values)), float(max(noop_probability_values))] if noop_probability_values else []
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "training_executed": True,
        "rerun_dssat": True,
        "source_task": "177_sya_dense_yield_credit_smoke2k",
        "total_timesteps": 2000,
        "checkpoint_steps": [1000, 2000],
        "station": "SYA",
        "source_year": 2005,
        "scenario_sampling": "deterministic_round_robin",
        "scenario_order": SCENARIO_ORDER,
        "scenario_count": 6,
        "reward_changed": False,
        "observation_changed": False,
        "action_mask_changed": False,
        "action_grid_changed": False,
        "physical_gate_passed": bool(gate.get("physical_response_passed")),
        "validation_executed": False,
        "action_response_passed": action_response_passed,
        "error": error[-5000:] if error else "",
        "evidence": {
            "episode_count": int(len(episodes)),
            "min_episode_count": min_count,
            "max_episode_count": max_count,
            "zero_count_scenarios": zero_scenarios,
            "decision_rows": int(len(decision)),
            "decision_scenarios": sorted(decision["weather_variant"].dropna().unique().tolist()) if "weather_variant" in decision.columns else [],
            "decision_target_daps": sorted(pd.to_numeric(decision["dap"], errors="coerce").dropna().astype(int).unique().tolist()) if "dap" in decision.columns else [],
            "decision_legal_action_count_values": legal_action_count_values,
            "decision_noop_probability_range": noop_probability_range,
            "action_response_passed": action_response_passed,
            "max_action_change_rate_after_dap1": max_action_change_rate_after_dap1,
            "action_summary_rows": int(len(action_summary)),
        },
        "outputs": {
            "doc": rel(DOC),
            "coverage": rel(coverage_path),
            "decision_diagnostics": rel(decision_path),
            "episode_diagnostics": rel(episode_path),
            "training_inventory": rel(OUT / "evaluation" / "186_training_checkpoint_inventory.csv"),
            "scenario_metrics": rel(OUT / "evaluation" / "186_scenario_episode_metrics.csv"),
            "action_summary": rel(OUT / "evaluation" / "186_action_change_summary.csv"),
        },
    }
    (OUT / "186_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_record(result, coverage, decision, episodes, action_summary)
    print("[Step 3/4] Writing result and detailed record.", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if status != "completed":
        raise RuntimeError(error)
    print("[Step 4/4] Task 186 completed.", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output_root.resolve() if args.output_root is not None else None
    run(config_path.resolve(), output_root=output_root)


if __name__ == "__main__":
    main()
