from __future__ import annotations

import tempfile
from pathlib import Path

import gymnasium as gym
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "021_05"
HORIZON = 1000
SEGMENT = 100


class StopAt(BaseCallback):
    def __init__(self, stop_at: int):
        super().__init__()
        self.stop_at = int(stop_at)

    def _on_step(self) -> bool:
        return int(self.model.num_timesteps) < self.stop_at


def make_model() -> DQN:
    return DQN(
        "MlpPolicy",
        gym.make("CartPole-v1"),
        verbose=0,
        learning_starts=10_000,
        buffer_size=2_000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        seed=0,
    )


def state(method: str, segment: int, model: DQN) -> dict[str, float | int | str]:
    return {
        "method": method,
        "segment": segment,
        "num_timesteps": int(model.num_timesteps),
        "exploration_rate": float(model.exploration_rate),
        "sb3_internal_total_timesteps": int(model._total_timesteps),
        "dqn_n_calls": int(model._n_calls),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    old = make_model()
    old.learn(total_timesteps=SEGMENT, reset_num_timesteps=False)
    rows.append(state("old_increment", 1, old))
    old.learn(total_timesteps=SEGMENT, reset_num_timesteps=False)
    rows.append(state("old_increment", 2, old))
    old.env.close()

    full_each = make_model()
    full_each.learn(total_timesteps=HORIZON, reset_num_timesteps=False, callback=StopAt(SEGMENT))
    rows.append(state("wrong_full_each_call", 1, full_each))
    full_each.learn(total_timesteps=HORIZON, reset_num_timesteps=False, callback=StopAt(2 * SEGMENT))
    rows.append(state("wrong_full_each_call", 2, full_each))
    full_each.env.close()

    remaining = make_model()
    remaining.learn(total_timesteps=HORIZON, reset_num_timesteps=False, callback=StopAt(SEGMENT))
    rows.append(state("remaining_horizon", 1, remaining))
    remaining.learn(
        total_timesteps=HORIZON - remaining.num_timesteps,
        reset_num_timesteps=False,
        callback=StopAt(2 * SEGMENT),
    )
    rows.append(state("remaining_horizon", 2, remaining))
    remaining.env.close()

    continuous = make_model()
    continuous.learn(total_timesteps=HORIZON, reset_num_timesteps=False, callback=StopAt(2 * SEGMENT))
    rows.append(state("continuous_horizon", 2, continuous))
    continuous.env.close()

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "resume_model"
        resumed = make_model()
        resumed.learn(total_timesteps=HORIZON, reset_num_timesteps=False, callback=StopAt(SEGMENT))
        rows.append(state("resume_before_save", 1, resumed))
        resumed.save(str(model_path))
        resumed.env.close()
        resumed = DQN.load(str(model_path) + ".zip", env=gym.make("CartPole-v1"))
        rows.append(state("resume_after_load", 1, resumed))
        resumed.learn(
            total_timesteps=HORIZON - resumed.num_timesteps,
            reset_num_timesteps=False,
            callback=StopAt(2 * SEGMENT),
        )
        rows.append(state("resume_after_continue", 2, resumed))
        resumed.env.close()

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "021_05_segmented_exploration_unit_test.csv", index=False, encoding="utf-8-sig")

    expected_second = float(
        1.0 - ((2 * SEGMENT / HORIZON) / 0.35) * (1.0 - 0.05)
    )
    remaining_second = float(
        result.loc[
            (result["method"].eq("remaining_horizon")) & result["segment"].eq(2),
            "exploration_rate",
        ].iloc[0]
    )
    continuous_second = float(
        result.loc[
            (result["method"].eq("continuous_horizon")) & result["segment"].eq(2),
            "exploration_rate",
        ].iloc[0]
    )
    resume_second = float(
        result.loc[
            (result["method"].eq("resume_after_continue")) & result["segment"].eq(2),
            "exploration_rate",
        ].iloc[0]
    )
    assert abs(remaining_second - continuous_second) < 1e-12
    assert abs(resume_second - continuous_second) < 1e-12
    assert abs(continuous_second - expected_second) < 0.01
    assert int(
        result.loc[
            (result["method"].eq("remaining_horizon")) & result["segment"].eq(2),
            "sb3_internal_total_timesteps",
        ].iloc[0]
    ) == HORIZON
    print(result.to_string(index=False))
    print("021_05 segmented exploration unit test passed")


if __name__ == "__main__":
    main()
