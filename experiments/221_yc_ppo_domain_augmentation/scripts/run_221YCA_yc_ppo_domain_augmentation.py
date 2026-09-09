"""221YCA: YC/YCA domain-diversity PPO, paired to frozen 055_00.

Only the training scenario pool changes. The inherited 203/204 runner supplies
the bounded cache, balanced sampler, instrumentation, validation and action
audit. This module deliberately does not patch the reward to simple-profit.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo as inherited


base = inherited.base
engine = base.engine
base03222 = base.base03222
import ppo_safe_rendering


TASK_ID = "221YCA"
TASK_NAME = "yc_ppo_domain_augmentation"
CONFIG = ROOT / "experiments" / "221_yc_ppo_domain_augmentation" / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT = ROOT / "prompt_01" / "002_codex_prompt_yc_ppo_domain_augmentation.md"
PHYSICAL_GATE = ROOT / "benchmark_results" / "217YCA_yca_lowIC_weather_physical_gate" / "217YCA_result.json"

EXPECTED_VARIANTS = ["original", "early_dry", "mid_dry", "late_dry", "early_wet", "mid_wet"]
EXPECTED_I = [0.0, 15.0, 30.0, 45.0]
EXPECTED_N = [0.0, 40.0, 80.0, 120.0]

ORIGINAL_PREFLIGHT = inherited.ORIGINAL_PREFLIGHT
ORIGINAL_CODE_PROVENANCE = inherited.ORIGINAL_CODE_PROVENANCE
ORIGINAL_WRITE_RECORD = inherited.ORIGINAL_WRITE_RECORD


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("task_id") != TASK_ID or cfg.get("task_name") != TASK_NAME:
        raise ValueError("221YCA task identity mismatch")
    if cfg.get("station_code") != "YCA" or cfg.get("site") != "YC":
        raise ValueError("221YCA is fixed to YCA/YC")
    if cfg.get("reference_run") != "055_00_yca_lowIC_expanded_action_maskableppo":
        raise ValueError("221YCA must pair with frozen 055_00")
    if list(map(float, cfg["actions"]["irrigation_levels_mm"])) != EXPECTED_I:
        raise ValueError("221YCA irrigation grid differs from 055_00")
    if list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"])) != EXPECTED_N:
        raise ValueError("221YCA nitrogen grid differs from 055_00")
    if cfg.get("observation_contract") != {
        "base": "046_02_raw_observation",
        "normalization_enabled": False,
        "weather_forecast_enabled": False,
    }:
        raise ValueError("221YCA observation contract differs from 055_00")
    return cfg


def ensure_baseline_alias(cfg: dict) -> Path:
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
    result, manifest, split = ORIGINAL_PREFLIGHT(cfg)
    result["station_mapping"] = {"station_code": "YCA", "site": "YC", "short_input_dir": "YC"}
    result["paired_baseline_alias"] = alias.relative_to(ROOT).as_posix()
    result["paired_baseline_source"] = cfg["comparison"]["baseline_source"]
    result["augmentation_factor"] = "training-domain diversity only"
    if not PHYSICAL_GATE.exists():
        result["issues"].append("217YCA physical gate result is missing")
    else:
        physical = json.loads(PHYSICAL_GATE.read_text(encoding="utf-8"))
        result["required_physical_gate"] = PHYSICAL_GATE.relative_to(ROOT).as_posix()
        result["physical_gate"] = physical.get("gate", {})
        if not bool(result["physical_gate"].get("allow_2k_smoke")):
            result["issues"].append("217YCA physical gate did not allow 2K smoke")
    result["ic_profile"] = "lowIC only; originIC not mixed"
    result["cultivar"] = "ZD0985 only; genotype randomization skipped"
    result["next_step_allowed"] = not result["issues"]
    return result, manifest, split


def code_provenance() -> dict[str, str]:
    result = ORIGINAL_CODE_PROVENANCE()
    paths = [
        Path(__file__).resolve(),
        CONFIG,
        PROMPT,
        ROOT / "benchmark_results" / "217YCA_yca_lowIC_weather_scenario_bank_v1_fixed_width" / "217YCA_weather_scenario_manifest.csv",
        PHYSICAL_GATE,
    ]
    for path in paths:
        if path.exists():
            result[base.rel(path)] = base.sha256(path)
    return result


def write_record(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error):
    ORIGINAL_WRITE_RECORD(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error)
    path = base.record_path(smoke)
    text = path.read_text(encoding="utf-8")
    text = text.replace(f"# {TASK_ID} HLA lowIC", f"# {TASK_ID} YCA/YC lowIC")
    text = text.replace("054_00", "055_00")
    text += (
        "\n## 221YCA 单因素控制\n\n"
        "- 唯一主动变化：训练域情景多样性；使用 10 个源年 × 6 个完整天气情景，采用分层无放回循环。\n"
        "- reward 保持 055_00 的 040_36/042_10 stress-aware + swfac guardrail；没有 simple-profit 替换。\n"
        "- IC 固定 lowIC，不混合 originIC；cultivar 固定 ZD0985；二者均不作为本轮增强因素。\n"
        "- WP_ET 不从 daily CSV 反推，若无 Summary.OUT/ETCP 精确回放则保持 pending。\n"
    )
    path.write_text(text, encoding="utf-8")


base.TASK_ID = TASK_ID
base.TASK_NAME = TASK_NAME
base.DEFAULT_CONFIG = CONFIG
base.PROMPT = PROMPT
base.STATION = "YCA"
base.SITE = "YC"
base.EXPECTED_VARIANTS = EXPECTED_VARIANTS
base.EXPECTED_I = EXPECTED_I
base.EXPECTED_N = EXPECTED_N
base.BoundedAuditedRandomYearEnv = inherited.BalancedAuditedRandomYearEnv
base.load_config = load_config
base.preflight = preflight
base.code_provenance = code_provenance
base.write_record = write_record

# Ensure the pre-run effective contract reports the expanded 16-action grid.
engine.BINARY_IRRIGATION_LEVELS = list(EXPECTED_I)
engine.BINARY_NITROGEN_LEVELS = list(EXPECTED_N)
engine.LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "yca_lowIC_weather_scenario_bank_217_v1_fixed_width"


def safe_add_04210_summaries(out: Path, suffix: str) -> None:
    """Avoid a secondary EmptyDataError if an upstream run leaves an empty CSV."""
    prefix = "042_10" if not suffix else f"042_10_{suffix}"
    path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        ORIGINAL_ADD_04210_SUMMARIES(out, suffix)
    except pd.errors.EmptyDataError:
        return


ORIGINAL_ADD_04210_SUMMARIES = engine.add_04210_summaries


def main() -> None:
    old_renderer_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_add = engine.add_04210_summaries
    try:
        engine.add_04210_summaries = safe_add_04210_summaries
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
        base.main()
    finally:
        engine.add_04210_summaries = old_add
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_renderer_root


if __name__ == "__main__":
    main()
