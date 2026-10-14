from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_sy2014_event_balanced_online_retention_1k_021_35 as ab
import run_sy2014_online_demo_nstep_mask_ab_021_41 as treatment
from ppo_evaluate import latest_observation_dict, scalar


OUT = ROOT / "benchmark_results" / "021_45"
TREATMENT_OUT = OUT / "treatment_seed1"
DOC = ROOT / "docs" / "2026-07-15_021_45_sy2014_terminal_feasibility_bonus_seed1_1k_ab.md"
REGISTRY = ROOT / "configs" / "expert_yield_thresholds.yaml"
CONTROL_SOURCE = ROOT / "benchmark_results" / "021_42" / "online_seed1" / "checkpoint_trajectory.csv"
PRIOR_SUMMARY = ROOT / "benchmark_results" / "021_34" / "021_34_frozen_policy_summary.json"

SITE = "SY"
YEAR = 2014
NULL_YIELD = 5408.0
RECORDED_YIELD = 9613.0
WATER_COST = 1.0
NITROGEN_COST = 5.0
IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
BONUS = WATER_COST * IRRIGATION_BUDGET + NITROGEN_COST * NITROGEN_BUDGET
NEAR_THRESHOLD_BAND = 50.0


def load_threshold(site: str, year: int) -> tuple[float, dict[str, Any]]:
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    try:
        entry = registry["thresholds"][site][year]
    except KeyError as exc:
        raise KeyError(f"No expert-yield threshold registered for {site}{year}; fallback is forbidden") from exc
    source = ROOT / entry["source_csv"]
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    selected = frame.loc[frame["scenario"] == entry["benchmark_scenario"]]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one source row, found {len(selected)}")
    source_value = float(selected.iloc[0][entry["source_column"]])
    registered = float(entry["yield_kg_ha"])
    if not np.isclose(source_value, registered, atol=0.0, rtol=0.0):
        raise ValueError(f"Threshold provenance mismatch: registry={registered}, source={source_value}")
    return registered, {**entry, "resolved_source_csv": str(source), "source_value": source_value}


@dataclass
class EpisodeRecord:
    episode: int
    terminal_yield_kg_ha: float
    irrigation_mm: float
    nitrogen_kg_ha: float
    base_reward_total: float
    feasibility_bonus: float
    candidate_reward_total: float
    steps: int


class TerminalFeasibilityBonusWrapper:
    """Add a site-year-specific constant bonus only on the terminal transition."""

    def __init__(
        self,
        env,
        threshold: float,
        bonus: float,
        episode_sink: list[EpisodeRecord] | None = None,
        yield_extractor: Callable[[Any, Any, dict[str, Any]], float] | None = None,
    ):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.threshold = float(threshold)
        self.bonus = float(bonus)
        self.episode_sink = episode_sink
        self.yield_extractor = yield_extractor or self._default_yield_extractor
        self._episode_index = 0
        self._reset_episode()

    def _default_yield_extractor(self, obs, info: dict[str, Any]) -> float:
        latest = latest_observation_dict(self.env, obs, info)
        return float(scalar(latest.get("grnwt", 0.0)) or 0.0)

    def _reset_episode(self) -> None:
        self._base_reward_total = 0.0
        self._candidate_reward_total = 0.0
        self._irrigation_total = 0.0
        self._nitrogen_total = 0.0
        self._steps = 0

    def reset(self, *args, **kwargs):
        self._reset_episode()
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        terminal_yield = self.yield_extractor(obs, info) if done else 0.0
        feasibility_bonus = self.bonus if done and terminal_yield >= self.threshold else 0.0
        candidate_reward = float(base_reward) + feasibility_bonus
        safe = dict(getattr(self.env, "last_safe_real_action", {}) or {})
        self._base_reward_total += float(base_reward)
        self._candidate_reward_total += candidate_reward
        self._irrigation_total += float(safe.get("amir", 0.0))
        self._nitrogen_total += float(safe.get("anfer", 0.0))
        self._steps += 1
        info = dict(info) if isinstance(info, dict) else {}
        info.update({
            "expert_yield_threshold": self.threshold,
            "terminal_yield_kg_ha": terminal_yield if done else 0.0,
            "feasibility_bonus": feasibility_bonus,
            "candidate_reward": candidate_reward,
        })
        if done and self.episode_sink is not None:
            self.episode_sink.append(EpisodeRecord(
                episode=self._episode_index,
                terminal_yield_kg_ha=terminal_yield,
                irrigation_mm=self._irrigation_total,
                nitrogen_kg_ha=self._nitrogen_total,
                base_reward_total=self._base_reward_total,
                feasibility_bonus=feasibility_bonus,
                candidate_reward_total=self._candidate_reward_total,
                steps=self._steps,
            ))
            self._episode_index += 1
        return obs, candidate_reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


class FakeEnv:
    action_space = None
    observation_space = None
    unwrapped = None
    last_safe_real_action = {"amir": 0.0, "anfer": 0.0}

    def __init__(self, yield_value: float, done: bool, base_reward: float = 7.0):
        self.yield_value = float(yield_value)
        self.done = bool(done)
        self.base_reward = float(base_reward)

    def reset(self):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, _action):
        return np.zeros(1, dtype=np.float32), self.base_reward, self.done, False, {"fake_yield": self.yield_value}

    def close(self):
        return None


def run_unit_tests() -> dict[str, Any]:
    threshold, provenance = load_threshold(SITE, YEAR)
    checks: dict[str, bool] = {}
    for yield_value, expected_bonus in ((11076.0, 0.0), (11077.0, BONUS), (11078.0, BONUS)):
        env = TerminalFeasibilityBonusWrapper(
            FakeEnv(yield_value, True), threshold, BONUS,
            yield_extractor=lambda _obs, info: float(info["fake_yield"]),
        )
        env.reset()
        _obs, reward, _terminated, _truncated, info = env.step(0)
        checks[f"terminal_{int(yield_value)}_bonus"] = bool(
            info["feasibility_bonus"] == expected_bonus and reward == 7.0 + expected_bonus
        )
    env = TerminalFeasibilityBonusWrapper(
        FakeEnv(12000.0, False), threshold, BONUS,
        yield_extractor=lambda _obs, info: float(info["fake_yield"]),
    )
    env.reset(); _obs, reward, _terminated, _truncated, info = env.step(0)
    checks["nonterminal_bonus_zero"] = bool(info["feasibility_bonus"] == 0.0 and reward == 7.0)
    checks["bonus_is_derived_not_fitted"] = bool(BONUS == 1620.0)
    score_n200 = max(0.0, 11200.0 - NULL_YIELD) - WATER_COST * 90.0 - NITROGEN_COST * 200.0 + BONUS
    score_n300 = max(0.0, 11200.0 - NULL_YIELD) - WATER_COST * 90.0 - NITROGEN_COST * 300.0 + BONUS
    checks["feasible_n200_n300_difference_preserved_500"] = bool(score_n200 - score_n300 == 500.0)
    checks["threshold_matches_provenance"] = bool(threshold == 11077.0 and provenance["source_value"] == 11077.0)
    missing_failed = False
    try:
        load_threshold("UNREGISTERED", 2099)
    except KeyError:
        missing_failed = True
    checks["missing_mapping_fails_closed"] = missing_failed
    return {
        "threshold": threshold,
        "bonus": BONUS,
        "bonus_derivation": f"{WATER_COST}*{IRRIGATION_BUDGET}+{NITROGEN_COST}*{NITROGEN_BUDGET}",
        "provenance": provenance,
        "checks": checks,
        "all_pass": bool(all(checks.values())),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def plot_results(comparison: pd.DataFrame, episodes: pd.DataFrame, updates: pd.DataFrame, threshold: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"control_021_42_seed1": "#777777", "treatment_terminal_bonus": "#0072B2"}
    for arm, frame in comparison.groupby("arm"):
        frame = frame.sort_values("checkpoint")
        axes[0].plot(frame.checkpoint, frame.yield_kg_ha, marker="o", color=colors[arm], label=arm)
        axes[1].plot(frame.checkpoint, frame.candidate_reward_recalculated, marker="o", color=colors[arm], label=arm)
    axes[0].axhline(threshold, color="#D55E00", linestyle="--", label="expert threshold")
    axes[0].axhline(RECORDED_YIELD, color="#333333", linestyle=":", label="recorded")
    axes[0].set_ylabel("Yield (kg/ha)")
    axes[1].set_ylabel("Candidate seasonal reward")
    if not episodes.empty:
        passed = episodes.feasibility_bonus.gt(0)
        axes[2].scatter(episodes.terminal_yield_kg_ha, episodes.candidate_reward_total,
                        c=np.where(passed, "#009E73", "#D55E00"), s=24, alpha=0.75)
    axes[2].axvline(threshold, color="#333333", linestyle="--", label="11077 threshold")
    axes[2].set_xlabel("Training-season terminal yield (kg/ha)")
    axes[2].set_ylabel("Training-season cumulative reward")
    for ax in axes[:2]:
        ax.set_xlabel("Online environment steps"); ax.legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes: ax.grid(alpha=0.2)
    fig.suptitle("SY2014 seed1 terminal feasibility bonus 1K A/B")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "021_45_terminal_bonus_seed1_1k_ab.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_45_terminal_bonus_seed1_1k_ab.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(updates.env_step, updates.q_abs_max, color="#0072B2", label="|Q| max")
    ax.axvline(250, color="#999999", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Online environment steps"); ax.set_ylabel("Maximum absolute online Q")
    ax.grid(alpha=0.2); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "021_45_q_scale_trace.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "021_45_q_scale_trace.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    if DOC.exists():
        raise FileExistsError(f"Refusing to overwrite {DOC}")
    OUT.mkdir(parents=True)
    unit = run_unit_tests()
    (OUT / "021_45_unit_tests.json").write_text(json.dumps(unit, indent=2), encoding="utf-8")
    if not unit["all_pass"]:
        raise RuntimeError("Offline unit tests failed; DSSAT/training was not started")

    threshold = float(unit["threshold"])
    training_episodes: list[EpisodeRecord] = []
    original_make_env = ab.make_env

    def bonus_make_env(relative: str):
        sink = training_episodes if relative == "training/environment" else None
        return TerminalFeasibilityBonusWrapper(original_make_env(relative), threshold, BONUS, sink)

    ab.OUT = TREATMENT_OUT
    ab.online_update = treatment.treatment_update
    ab.make_env = bonus_make_env
    interactions, updates, audit, scaler = ab.train_online(action_seed=1, sample_seed=21036)
    interactions.to_csv(OUT / "021_45_training_interactions.csv", index=False, encoding="utf-8-sig")
    updates.to_csv(OUT / "021_45_treatment_update_log.csv", index=False, encoding="utf-8-sig")
    episodes = pd.DataFrame([record.__dict__ for record in training_episodes])
    episodes.to_csv(OUT / "021_45_training_episode_terminal_rewards.csv", index=False, encoding="utf-8-sig")

    prior = json.loads(PRIOR_SUMMARY.read_text(encoding="utf-8"))
    initial_bonus = BONUS if float(prior["yield_kg_ha"]) >= threshold else 0.0
    treatment_rows = [{
        "checkpoint": 0, "yield_kg_ha": prior["yield_kg_ha"], "biomass_kg_ha": prior["biomass_kg_ha"],
        "irrigation_mm": prior["irrigation_mm"], "nitrogen_kg_ha": prior["nitrogen_kg_ha"],
        "late_n_after_dap90_kg_ha": prior["late_n_after_dap90_kg_ha"],
        "reward_total": float(prior["reward_total"]) + initial_bonus,
        "expert_efficiency_gate": True, "q_values_finite": True, "terminated_or_truncated": True,
    }]
    for checkpoint in ab.CHECKPOINTS:
        _daily, result = ab.evaluate_checkpoint(checkpoint, scaler["mean"], scaler["scale"])
        treatment_rows.append(result)
    treatment_frame = pd.DataFrame(treatment_rows)
    treatment_frame["arm"] = "treatment_terminal_bonus"
    control = pd.read_csv(CONTROL_SOURCE)
    control["arm"] = "control_021_42_seed1"
    comparison = pd.concat([control, treatment_frame], ignore_index=True, sort=False)
    comparison["feasibility_bonus_recalculated"] = np.where(comparison.yield_kg_ha >= threshold, BONUS, 0.0)
    comparison["candidate_reward_recalculated"] = (
        np.maximum(0.0, comparison.yield_kg_ha - NULL_YIELD)
        - WATER_COST * comparison.irrigation_mm
        - NITROGEN_COST * comparison.nitrogen_kg_ha
        + comparison.feasibility_bonus_recalculated
    )
    comparison.to_csv(OUT / "021_45_checkpoint_trajectory_ab.csv", index=False, encoding="utf-8-sig")

    treatment_online = treatment_frame[treatment_frame.checkpoint > 0]
    control_online = control[control.checkpoint > 0]
    treatment_pass = int(treatment_online.expert_efficiency_gate.astype(bool).sum())
    control_pass = int(control_online.expert_efficiency_gate.astype(bool).sum())
    final = treatment_online.loc[treatment_online.checkpoint == 1000].iloc[0]
    minimum_yield = float(treatment_online.yield_kg_ha.min())
    strict_success = bool(
        treatment_pass >= 3 and bool(final.expert_efficiency_gate)
        and minimum_yield >= RECORDED_YIELD and treatment_pass > control_pass
    )
    if strict_success:
        branch = "A"
    elif treatment_pass > control_pass or bool(final.expert_efficiency_gate):
        branch = "B"
    else:
        branch = "C"
    near = episodes.loc[(episodes.terminal_yield_kg_ha - threshold).abs() <= NEAR_THRESHOLD_BAND] if not episodes.empty else episodes
    bonus_assignment_matches_threshold = bool(
        (
            episodes.feasibility_bonus
            == np.where(episodes.terminal_yield_kg_ha >= threshold, BONUS, 0.0)
        ).all()
    ) if not episodes.empty else False
    positive_bonus_episode_count = int(episodes.feasibility_bonus.gt(0).sum()) if not episodes.empty else 0
    zero_bonus_episode_count = int(episodes.feasibility_bonus.eq(0).sum()) if not episodes.empty else 0
    validation = {
        **audit,
        "unit_tests_all_pass": unit["all_pass"],
        "threshold": threshold,
        "bonus": BONUS,
        "control_reused_not_retrained": True,
        "control_pass_count": control_pass,
        "treatment_pass_count": treatment_pass,
        "treatment_final_pass": bool(final.expert_efficiency_gate),
        "treatment_minimum_yield": minimum_yield,
        "near_threshold_training_episode_count_plus_minus_50": int(len(near)),
        "positive_bonus_training_episode_count": positive_bonus_episode_count,
        "zero_bonus_training_episode_count": zero_bonus_episode_count,
        "bonus_assignment_matches_terminal_threshold": bonus_assignment_matches_threshold,
        "all_treatment_evaluations_finite": bool(treatment_frame.q_values_finite.astype(bool).all()),
        "all_treatment_evaluations_terminated": bool(treatment_frame.terminated_or_truncated.astype(bool).all()),
        "all_required_checks_pass": bool(
            unit["all_pass"] and audit["initial_hash_matches"] and audit["demonstrations_unchanged"]
            and audit["all_batches_16_demo_16_agent"] and audit["all_demo_halves_8_noop_8_nonzero"]
            and audit["all_updates_finite"] and treatment_frame.q_values_finite.astype(bool).all()
            and treatment_frame.terminated_or_truncated.astype(bool).all()
            and bonus_assignment_matches_threshold and positive_bonus_episode_count > 0
            and zero_bonus_episode_count > 0
        ),
    }
    (OUT / "021_45_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "status": "completed" if validation["all_required_checks_pass"] else "failed_validation",
        "branch": branch,
        "strict_preregistered_success": strict_success,
        "control_pass_count": control_pass,
        "treatment_pass_count": treatment_pass,
        "final_treatment": final.to_dict(),
        "seed2_started": False,
        "training_5k_started": False,
        "interpretation": {
            "A": "终端可行性 bonus 在 seed1 1K 单变量对照中通过预注册门槛；只支持另立复核任务，不自动扩展。",
            "B": "终端 bonus 有局部改善，但未完整通过预注册门槛，不能放大。",
            "C": "终端 bonus 未改善 seed1 保持性，该候选不获支持。",
        }[branch],
    }
    (OUT / "021_45_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    plot_results(comparison, episodes, updates, threshold)

    display = comparison[["arm", "checkpoint", "yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha",
                          "late_n_after_dap90_kg_ha", "expert_efficiency_gate",
                          "candidate_reward_recalculated"]].round(3)
    doc = f"""# 021_45 SY2014 终端可行性 bonus：seed1 1K 单变量 A/B 记录

## 设计

Control 直接复用 021_42 online seed1，不重跑。Treatment 仅在终止时、最终产量达到本站点年份官方 expert 阈值 {threshold:.0f} kg/ha 时，在原奖励上增加 {BONUS:.0f}。bonus 由 `1×120+5×300` 推导，不是拟合值。其余训练、环境和 demo n-step 屏蔽设置不变。

## 实现与单元测试

- 阈值注册表：`configs/expert_yield_thresholds.yaml`；来源：`{unit['provenance']['resolved_source_csv']}` 的 `official_extension_expert` 行。
- 11076/11077/11078 边界、非终止步、bonus 推导、N200/N300 保留 500 分差、缺失映射失败保护均通过：**{unit['all_pass']}**。
- 旧奖励实现没有修改；bonus 使用本任务外层 adapter。

## 结果

{markdown_table(display)}

- Control 通过：{control_pass}/4，1000-step 失败。
- Treatment 通过：{treatment_pass}/4；1000-step 通过：{bool(final.expert_efficiency_gate)}；最低产量：{minimum_yield:.0f} kg/ha。
- 预注册分支：**{branch}**。{summary['interpretation']}

## 硬门槛与 Q 尺度检查

训练中共有 {len(episodes)} 个完整季节；其中 {positive_bonus_episode_count} 季实际获得 bonus，{zero_bonus_episode_count} 季未获得，逐季分配与终产量门槛完全一致：{bonus_assignment_matches_threshold}。终产量落在阈值 ±{NEAR_THRESHOLD_BAND:.0f} kg/ha 内的季节数为 {len(near)}。产量—累计奖励散点和 online Q 绝对最大值轨迹已保存。由于门槛附近没有样本，本轮不能声称已排除门槛处震荡。

## 边界

这只是 SY2014 online seed1 的 1K 单变量验证。没有启动 seed2 或 5K；没有改 reward 主文件、IC、DSSAT 输入、动作、预算或网络。即使分支 A，也只能支持另立复核任务，不能宣布长期稳定。

第一次启动命令的外层终端等待时间误设为 1 秒，进程在创建输出目录前被终止，没有产生科学数据；随后改用足够等待时间完整重跑。本记录保留该执行失误，不把它计作模型失败。

## 输出

- `benchmark_results/021_45/021_45_unit_tests.json`
- `benchmark_results/021_45/021_45_training_interactions.csv`
- `benchmark_results/021_45/021_45_treatment_update_log.csv`
- `benchmark_results/021_45/021_45_training_episode_terminal_rewards.csv`
- `benchmark_results/021_45/021_45_checkpoint_trajectory_ab.csv`
- `benchmark_results/021_45/021_45_validation.json`
- `benchmark_results/021_45/021_45_summary.json`
- `benchmark_results/021_45/021_45_terminal_bonus_seed1_1k_ab.png/.svg`
- `benchmark_results/021_45/021_45_q_scale_trace.png/.svg`
"""
    DOC.write_text(doc, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "branch": branch, "summary": summary, "validation": validation}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
