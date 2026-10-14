"""215HLA: HLA weather augmentation with the requested simple profit reward.

This runner reuses the validated 210HLA fixed-width weather bank and balanced
scenario sampler. It changes only the reward implementation relative to the
210HLA weather-augmented line; formal training is intentionally not authorized
by default, so the first invocation is a 2K smoke.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_210HLA_hla_lowIC_10y_six_weather_fixed_width_balanced_maskableppo as inherited
from run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04 import ProjectProfitDiscreteWrapper


base = inherited.base
engine = base.engine
base03222 = base.base03222

TASK_ID = "215HLA"
TASK_NAME = "hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo"
CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-09-03_hla_simple_profit_weather_augmentation.md"

base.TASK_ID = TASK_ID
base.TASK_NAME = TASK_NAME
base.DEFAULT_CONFIG = CONFIG
base.PROMPT = PROMPT

ORIGINAL_ENGINE_LOAD_CONFIG = engine.load_config
ORIGINAL_PATCH_BASE_MODULE = engine.patch_base_module
ORIGINAL_WRITE_RECORD = base.write_record
ORIGINAL_CODE_PROVENANCE = base.code_provenance


def simple_profit_config() -> dict:
    cfg = ORIGINAL_ENGINE_LOAD_CONFIG()
    reward = cfg.setdefault("reward", {})
    reward.update(
        {
            "reward_type": "harvest_project_simple_profit_scaled_0p001",
            "yield_coef": 1.0,
            "water_cost": 1.1,
            "nitrogen_cost": 1.58,
            "water_stress_relief_coef": 0.0,
            "nitrogen_stress_relief_coef": 0.0,
            "reward_scale": 0.001,
        }
    )
    return cfg


def patched_engine_load_config() -> dict:
    return simple_profit_config()


def patched_patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    ORIGINAL_PATCH_BASE_MODULE(out, doc, total_timesteps, checkpoint_steps)
    # 042/210 installs the stress-aware wrapper here; replace only that
    # wrapper with the terminal-yield-minus-input-cost implementation.
    base03222.base.StressAwareDiscreteWrapper = ProjectProfitDiscreteWrapper


def patched_code_provenance() -> dict[str, str]:
    result = ORIGINAL_CODE_PROVENANCE()
    result[base.rel(Path(__file__).resolve())] = base.sha256(Path(__file__).resolve())
    return result


def patched_write_record(
    cfg,
    smoke,
    preflight_result,
    effective,
    coverage,
    by_checkpoint,
    comparison,
    gate,
    status,
    error,
) -> None:
    ORIGINAL_WRITE_RECORD(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error)
    path = base.record_path(smoke)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "- PPO、reward、动作、mask、观测、seed、训练/验证年份和分层无放回采样器均保持不变。",
        "- PPO、动作、mask、观测、seed、训练/验证年份和分层无放回采样器保持不变；相对210HLA仅替换为简单产量-水氮成本奖励。",
    )
    text += (
        "\n## 215HLA 简单奖励定义\n\n"
        "- 终点奖励：`terminal_grain_yield - 1.1*irrigation_mm - 1.58*nitrogen_kg_ha`，统一乘 `0.001`。\n"
        "- 水分胁迫缓解奖励和氮胁迫缓解奖励均为 0，不修改动作网格或安全 mask。\n"
        "- 本次只完成 2K smoke；不因 smoke 通过而自动启动 100K。\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    old_load = engine.load_config
    old_patch = engine.patch_base_module
    old_write = base.write_record
    old_provenance = base.code_provenance
    try:
        engine.load_config = patched_engine_load_config
        engine.patch_base_module = patched_patch_base_module
        base.write_record = patched_write_record
        base.code_provenance = patched_code_provenance
        # Do not accept --formal in this first pass; the user asked to try the
        # site and the project safety rule requires a smoke gate first.
        if "--formal" in sys.argv:
            raise SystemExit("215HLA 当前只允许 2K smoke；请先检查 smoke 记录后再授权 formal 100K。")
        base.main()
    finally:
        engine.load_config = old_load
        engine.patch_base_module = old_patch
        base.write_record = old_write
        base.code_provenance = old_provenance


if __name__ == "__main__":
    main()
