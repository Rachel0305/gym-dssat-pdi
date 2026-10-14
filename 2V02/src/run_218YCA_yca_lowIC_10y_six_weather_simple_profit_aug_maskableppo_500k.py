"""218YCA: authorized 500K YC continuation after the 217YCA 2K smoke."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_217YCA_yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo as prior


base = prior.base
engine = prior.engine
ppo_safe_rendering = prior.ppo_safe_rendering
base03222 = prior.base03222

TASK_ID = "218YCA"
TASK_NAME = "yca_lowIC_10y_six_weather_simple_profit_aug_maskableppo_500k"
CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompts" / "2026-09-04_yca_simple_profit_weather_augmentation_500k.md"
PHYSICAL_GATE = ROOT / "benchmark_results" / "217YCA_yca_lowIC_weather_physical_gate" / "217YCA_result.json"

# Rebind the parameterized 217YCA helpers before base.main() resolves them.
prior.TASK_ID = TASK_ID
prior.TASK_NAME = TASK_NAME
prior.CONFIG = CONFIG
prior.PROMPT = PROMPT
prior.PHYSICAL_GATE = PHYSICAL_GATE
base.TASK_ID = TASK_ID
base.TASK_NAME = TASK_NAME
base.DEFAULT_CONFIG = CONFIG
base.PROMPT = PROMPT
base.STATION = "YCA"
base.SITE = "YC"
base.EXPECTED_VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
base.EXPECTED_I = [0.0, 15.0, 30.0, 45.0]
base.EXPECTED_N = [0.0, 40.0, 80.0, 120.0]
base.load_config = prior.load_config
base.preflight = prior.preflight
base.code_provenance = prior.code_provenance
base.write_record = prior.write_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ORIGINAL_PROVENANCE = base.code_provenance
ORIGINAL_WRITE_RECORD = base.write_record


def code_provenance() -> dict[str, str]:
    result = ORIGINAL_PROVENANCE()
    result[f"src/{Path(__file__).name}"] = sha256(Path(__file__).resolve())
    result[f"configs/{CONFIG.name}"] = sha256(CONFIG)
    result[f"prompts/{PROMPT.name}"] = sha256(PROMPT)
    return result


def write_record(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error):
    ORIGINAL_WRITE_RECORD(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error)
    path = base.record_path(smoke)
    text = path.read_text(encoding="utf-8")
    text = text.replace("## 217YCA 简单奖励定义", "## 218YCA 简单奖励定义")
    text += (
        "\n## 218YCA 500K 运行边界\n\n"
        "- 本轮为用户明确授权的 500K formal training；不自动启动 1000K。\n"
        "- checkpoint：2K、5K、10K、25K、50K、100K、150K、200K、300K、400K、500K。\n"
        "- 训练更新日志由 PPO 每个 rollout 输出；watchdog 每 10 分钟读取日志并输出一次汇总。\n"
        "- 内存 warning=1536 MB、stop=2048 MB；若 stop 触发，保留已保存 checkpoint 并将任务标为 partial/failed，不继续冒险。\n"
        "- 简单奖励：`terminal_grain_yield - 1.1*irrigation_mm - 1.58*nitrogen_kg_ha`，乘以 `0.001`；胁迫项为 0。\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if "--formal" not in sys.argv:
        raise SystemExit("218YCA 只允许按显式 --formal 启动 500K 长训练。")
    old_load = engine.load_config
    old_patch = engine.patch_base_module
    old_provenance = base.code_provenance
    old_write = base.write_record
    old_renderer = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    try:
        engine.load_config = prior.simple_profit_engine_config
        engine.patch_base_module = prior.patched_engine_patch_base_module
        base.code_provenance = code_provenance
        base.write_record = write_record
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "yca_lowIC_weather_scenario_bank_217_v1_fixed_width"
        base.main()
    finally:
        engine.load_config = old_load
        engine.patch_base_module = old_patch
        base.code_provenance = old_provenance
        base.write_record = old_write
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_renderer


if __name__ == "__main__":
    main()
