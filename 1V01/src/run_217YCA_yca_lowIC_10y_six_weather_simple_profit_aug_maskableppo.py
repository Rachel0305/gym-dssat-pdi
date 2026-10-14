"""217YCA: YC six-weather augmentation with the simple profit reward.

This is deliberately a smoke-only entry point.  It reuses the audited 204/210
balanced sampler and 203 training/evaluation implementation, parameterized for
YCA/YC, and changes the terminal reward to yield minus water and N costs.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo as inherited
from run_fqa2014_project_profit_reward_ppo_checkpoint_guardrail_035_04 import ProjectProfitDiscreteWrapper


base = inherited.base
engine = base.engine
base03222 = base.base03222
import ppo_safe_rendering


TASK_ID = "217YCA"
TASK_NAME = "yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo"
CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-09-04_yca_simple_profit_weather_augmentation.md"
PHYSICAL_GATE = ROOT / "benchmark_results" / "217YCA_yca_lowIC_weather_physical_gate" / "217YCA_result.json"

ORIGINAL_ENGINE_LOAD_CONFIG = engine.load_config
ORIGINAL_ENGINE_PATCH_BASE_MODULE = engine.patch_base_module
ORIGINAL_BASE_LOAD_CONFIG = base.load_config
ORIGINAL_BASE_PREFLIGHT = base.preflight
ORIGINAL_BASE_CODE_PROVENANCE = base.code_provenance
ORIGINAL_BASE_WRITE_RECORD = base.write_record


def simple_profit_engine_config() -> dict:
    cfg = ORIGINAL_ENGINE_LOAD_CONFIG()
    reward = cfg.setdefault("reward", {})
    reward.update({
        "reward_type": "harvest_project_simple_profit_scaled_0p001",
        "yield_coef": 1.0,
        "water_cost": 1.1,
        "nitrogen_cost": 1.58,
        "water_stress_relief_coef": 0.0,
        "nitrogen_stress_relief_coef": 0.0,
        "reward_scale": 0.001,
    })
    return cfg


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("task_id") != TASK_ID or cfg.get("task_name") != TASK_NAME:
        raise ValueError("217YCA task identity mismatch")
    if cfg.get("station_code") != "YCA" or cfg.get("site") != "YC":
        raise ValueError("217YCA is fixed to YCA/YC")
    if cfg.get("reference_run") != "055_00_yca_lowIC_expanded_action_maskableppo":
        raise ValueError("217YCA must pair with the frozen 055_00 YC line")
    if list(map(float, cfg["actions"]["irrigation_levels_mm"])) != [0.0, 15.0, 30.0, 45.0]:
        raise ValueError("217YCA keeps irrigation grid [0,15,30,45]")
    if list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"])) != [0.0, 40.0, 80.0, 120.0]:
        raise ValueError("217YCA keeps nitrogen grid [0,40,80,120]")
    if cfg.get("observation_contract") != {
        "base": "046_02_raw_observation", "normalization_enabled": False, "weather_forecast_enabled": False
    }:
        raise ValueError("217YCA observation contract differs from 055_00")
    return cfg


def ensure_baseline_alias(cfg: dict) -> Path:
    """Provide the inherited auditor's expected filename without editing 055_00."""
    source = ROOT / cfg["comparison"]["baseline_source"]
    alias = ROOT / cfg["comparison"]["baseline_run"] / "evaluation" / "054_00_checkpoint_validation_summary.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    if not alias.exists():
        alias.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, alias)
    return alias


def preflight(cfg: dict):
    alias = ensure_baseline_alias(cfg)
    result, manifest, split = ORIGINAL_BASE_PREFLIGHT(cfg)
    result["station_mapping"] = {"station_code": "YCA", "site": "YC", "short_input_dir": "YC"}
    result["paired_baseline_alias"] = str(alias.relative_to(ROOT)).replace("\\", "/")
    result["paired_baseline_source"] = cfg["comparison"]["baseline_source"]
    if not PHYSICAL_GATE.exists():
        result["issues"].append("217YCA physical gate result is missing")
    else:
        physical = json.loads(PHYSICAL_GATE.read_text(encoding="utf-8"))
        result["required_physical_gate"] = str(PHYSICAL_GATE.relative_to(ROOT)).replace("\\", "/")
        result["physical_gate"] = physical.get("gate", {})
        if not bool(result["physical_gate"].get("allow_2k_smoke")):
            result["issues"].append("217YCA physical gate did not allow 2K smoke")
    result["next_step_allowed"] = not result["issues"]
    return result, manifest, split


def patched_engine_patch_base_module(out: Path, doc: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
    ORIGINAL_ENGINE_PATCH_BASE_MODULE(out, doc, total_timesteps, checkpoint_steps)
    base03222.base.StressAwareDiscreteWrapper = ProjectProfitDiscreteWrapper


def code_provenance() -> dict[str, str]:
    result = ORIGINAL_BASE_CODE_PROVENANCE()
    for path in [
        ROOT / "src" / "build_217YCA_yca_lowIC_weather_scenario_bank_v1_fixed_width.py",
        ROOT / "src" / "audit_217YCA_yca_lowIC_weather_physical_gate.py",
        Path(__file__).resolve(),
    ]:
        result[base.rel(path)] = base.sha256(path)
    return result


def write_record(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error):
    ORIGINAL_BASE_WRITE_RECORD(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error)
    path = base.record_path(smoke)
    text = path.read_text(encoding="utf-8")
    text = text.replace("# 217YCA HLA lowIC", "# 217YCA YCA/YC lowIC")
    text = text.replace("与未增强 054_00 的相同步数配对比较", "与未增强 055_00 的相同步数配对比较")
    text = text.replace("相对 054_00 的变化", "相对 055_00 的变化")
    text = text.replace("相对 203HLA 的最小修正", "相对未增强 YC 线的最小修正")
    text += (
        "\n## 217YCA 简单奖励定义\n\n"
        "- 终点奖励：`grain_yield - 1.1*irrigation_mm - 1.58*nitrogen_kg_ha`，统一乘 `0.001`。\n"
        "- 水分胁迫奖励和氮胁迫奖励均为 0；天气增强只作用于训练集 RAIN。\n"
        "- 动作网格仍为 16 个联合动作；本轮只完成 2K smoke，不自动启动正式训练。\n"
    )
    path.write_text(text, encoding="utf-8")


base.TASK_ID = TASK_ID
base.TASK_NAME = TASK_NAME
base.DEFAULT_CONFIG = CONFIG
base.PROMPT = PROMPT
base.STATION = "YCA"
base.SITE = "YC"
base.EXPECTED_VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
base.EXPECTED_I = [0.0, 15.0, 30.0, 45.0]
base.EXPECTED_N = [0.0, 40.0, 80.0, 120.0]
base.load_config = load_config
base.preflight = preflight
base.code_provenance = code_provenance
base.write_record = write_record


def main() -> None:
    old = {
        "engine_load": engine.load_config,
        "engine_patch": engine.patch_base_module,
        "renderer_root": ppo_safe_rendering.MULTISITE_INPUT_ROOT,
    }
    try:
        engine.load_config = simple_profit_engine_config
        engine.patch_base_module = patched_engine_patch_base_module
        # The base runner sets this again during training; this assignment also
        # makes any renderer created during setup use the YC bank.
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "yca_lowIC_weather_scenario_bank_217_v1_fixed_width"
        if "--formal" in sys.argv:
            raise SystemExit("217YCA 当前只允许 2K smoke；请先检查 smoke 记录后再授权 formal 100K。")
        base.main()
    finally:
        engine.load_config = old["engine_load"]
        engine.patch_base_module = old["engine_patch"]
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["renderer_root"]


if __name__ == "__main__":
    main()
