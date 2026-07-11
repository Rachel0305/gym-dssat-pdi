from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONFIG_ID = "hla_nstep5_baseline_relative_v1_020_11"

ACTION_TABLE_9: dict[int, dict[str, float]] = {
    0: {"amir": 0.0, "anfer": 0.0},
    1: {"amir": 15.0, "anfer": 0.0},
    2: {"amir": 30.0, "anfer": 0.0},
    3: {"amir": 0.0, "anfer": 50.0},
    4: {"amir": 15.0, "anfer": 50.0},
    5: {"amir": 30.0, "anfer": 50.0},
    6: {"amir": 0.0, "anfer": 100.0},
    7: {"amir": 15.0, "anfer": 100.0},
    8: {"amir": 30.0, "anfer": 100.0},
}

WATER_COST = 1.0
NITROGEN_COST = 5.0
IRRIGATION_BUDGET = 120.0
NITROGEN_BUDGET = 300.0
DAILY_IRRIGATION_CAP = 30.0
DAILY_NITROGEN_CAP = 100.0
MIN_INTERVAL_DAYS = 7
IRRIGATION_WINDOWS = [(1, 120)]
NITROGEN_WINDOWS = [(1, 120)]

DQN_FIXED_KWARGS: dict[str, Any] = {
    "learning_rate": 1e-4,
    "buffer_size": 10_000,
    "learning_starts": 50,
    "batch_size": 32,
    "train_freq": 1,
    "gradient_steps": 1,
    "gamma": 0.99,
    "tau": 1.0,
    "target_update_interval": 10_000,
    "max_grad_norm": 10.0,
    "n_steps": 5,
    "exploration_fraction": 0.35,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
}

TRAINING_TIMESTEPS = 50_000
CHECKPOINT_INTERVAL = 5_000
CHECKPOINT_SELECTION = "maximum total_reward; earliest checkpoint on ties"
HLA_SELECTED_CHECKPOINTS = {0: 30_000, 1: 10_000, 2: 20_000}


def apply_environment_constants(yc_module: Any) -> None:
    """Apply the frozen reward/action/budget constants to the shared wrappers."""

    yc_module.WATER_COST = WATER_COST
    yc_module.NITROGEN_COST = NITROGEN_COST
    yc_module.IRRIGATION_BUDGET = IRRIGATION_BUDGET
    yc_module.NITROGEN_BUDGET = NITROGEN_BUDGET
    yc_module.DAILY_IRRIGATION_CAP = DAILY_IRRIGATION_CAP
    yc_module.DAILY_NITROGEN_CAP = DAILY_NITROGEN_CAP
    yc_module.MIN_INTERVAL_DAYS = MIN_INTERVAL_DAYS
    yc_module.ACTION_TABLE = {index: dict(action) for index, action in ACTION_TABLE_9.items()}


def dqn_kwargs(seed: int) -> dict[str, Any]:
    kwargs = dict(DQN_FIXED_KWARGS)
    kwargs["seed"] = int(seed)
    return kwargs


def manifest() -> dict[str, Any]:
    return {
        "config_id": CONFIG_ID,
        "algorithm": "stable_baselines3.DQN",
        "policy": "MlpPolicy",
        "policy_network": {
            "hidden_layers": [64, 64],
            "activation": "ReLU",
            "optimizer": "Adam",
            "source": "Stable-Baselines3 DQN MlpPolicy defaults (policy_kwargs=None)",
        },
        "reward": {
            "per_step": "-1.0 * irrigation_mm - 5.0 * nitrogen_kg_ha",
            "terminal": "+max(0, final_grain_kg_ha - local_site_year_null_yield)",
            "water_cost": WATER_COST,
            "nitrogen_cost": NITROGEN_COST,
            "null_baseline_scope": "local site-year; never shared across sites",
        },
        "action_table": ACTION_TABLE_9,
        "constraints": {
            "irrigation_budget_mm": IRRIGATION_BUDGET,
            "nitrogen_budget_kg_ha": NITROGEN_BUDGET,
            "single_irrigation_cap_mm": DAILY_IRRIGATION_CAP,
            "single_nitrogen_cap_kg_ha": DAILY_NITROGEN_CAP,
            "minimum_interval_days": MIN_INTERVAL_DAYS,
            "irrigation_windows_dap": IRRIGATION_WINDOWS,
            "nitrogen_windows_dap": NITROGEN_WINDOWS,
        },
        "dqn_kwargs": DQN_FIXED_KWARGS,
        "training_timesteps": TRAINING_TIMESTEPS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "deterministic_evaluation": True,
        "random_weather": False,
        "environment_seed": 0,
        "hla_selected_checkpoints": HLA_SELECTED_CHECKPOINTS,
        "frozen_scope": "framework constants only; site input and local null yield remain site-specific",
    }


def write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(), indent=2, ensure_ascii=False), encoding="utf-8")
