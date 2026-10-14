"""216HLA: 500K long training for the HLA simple-profit weather line.

This is a new workline. It reuses the validated 210HLA fixed-width weather
bank and balanced sampler, changes no reward/action/observation/safety
definition relative to the 215HLA 2K smoke, and permits only the explicitly
authorized 500K formal run.
"""

from __future__ import annotations

import hashlib
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

TASK_ID = "216HLA"
TASK_NAME = "hla_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k"
CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-09-03_hla_simple_profit_weather_augmentation_500k.md"

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
    base03222.base.StressAwareDiscreteWrapper = ProjectProfitDiscreteWrapper


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patched_code_provenance() -> dict[str, str]:
    result = ORIGINAL_CODE_PROVENANCE()
    result[f"src/{Path(__file__).name}"] = sha256(Path(__file__).resolve())
    result[f"configs/{CONFIG.name}"] = sha256(CONFIG)
    result[f"prompts/{PROMPT.name}"] = sha256(PROMPT)
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
    ORIGINAL_WRITE_RECORD(
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
    )
    path = base.record_path(smoke)
    text = path.read_text(encoding="utf-8")
    text += (
        "\n## 216HLA 500K 运行边界\n\n"
        "- 本轮只授权 500K formal training；未授权 1000K。\n"
        "- checkpoint 为 2K、5K、10K、25K、50K、100K、150K、200K、300K、400K、500K。\n"
        "- 相对 215HLA 2K smoke，不改变奖励、动作网格、观测、动作安全 mask、天气情景和采样器。\n"
        "- 简单奖励：`terminal_grain_yield - 1.1*irrigation_mm - 1.58*nitrogen_kg_ha`，乘以 `0.001`；胁迫缓解项为 0。\n"
        "- 本轮结果用于诊断训练步数能否改变高频小灌和固定施肥，不自动替代冻结基线。\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if "--formal" not in sys.argv:
        raise SystemExit("216HLA 只允许按显式 --formal 启动 500K 长训练。")
    old_load = engine.load_config
    old_patch = engine.patch_base_module
    old_write = base.write_record
    old_provenance = base.code_provenance
    try:
        engine.load_config = patched_engine_load_config
        engine.patch_base_module = patched_patch_base_module
        base.write_record = patched_write_record
        base.code_provenance = patched_code_provenance
        base.main()
    finally:
        engine.load_config = old_load
        engine.patch_base_module = old_patch
        base.write_record = old_write
        base.code_provenance = old_provenance


if __name__ == "__main__":
    main()
