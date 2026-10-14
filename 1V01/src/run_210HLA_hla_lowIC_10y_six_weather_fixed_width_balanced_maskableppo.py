"""210HLA corrected fixed-width continuation of the balanced HLA workline."""

from __future__ import annotations

import json
from pathlib import Path

import run_204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo as prior


base = prior.base
ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "210HLA"
TASK_NAME = "hla_lowIC_10y_six_weather_fixed_width_balanced_maskableppo"
PHYSICAL_GATE = ROOT / "benchmark_results" / "209HLA_hla_lowIC_v2_weather_physical_gate" / "209HLA_result.json"

ORIGINAL_PREFLIGHT = base.preflight
ORIGINAL_CODE_PROVENANCE = base.code_provenance
ORIGINAL_WRITE_RECORD = base.write_record


def patched_preflight(cfg):
    result, manifest, split = ORIGINAL_PREFLIGHT(cfg)
    if not PHYSICAL_GATE.exists():
        result["issues"].append("209HLA physical gate result is missing")
    else:
        physical = json.loads(PHYSICAL_GATE.read_text(encoding="utf-8"))
        gate = dict(physical.get("gate", {}))
        result["required_physical_gate"] = base.rel(PHYSICAL_GATE)
        result["physical_gate"] = gate
        if not bool(gate.get("allow_2k_smoke")):
            result["issues"].append("209HLA physical gate did not allow 2K smoke")
    result["wth_data_line_contract"] = "DATE(7)+SRAD/TMAX/TMIN/RAIN(6 each)=31 chars"
    result["invalid_v1_run_is_diagnostic_only"] = "204HLA"
    result["next_step_allowed"] = not result["issues"]
    return result, manifest, split


def patched_code_provenance():
    result = ORIGINAL_CODE_PROVENANCE()
    for path in [
        ROOT / "src" / "build_208HLA_hla_lowIC_weather_scenario_bank_v2_fixed_width.py",
        ROOT / "src" / "audit_209HLA_hla_lowIC_v2_weather_physical_gate.py",
        ROOT / "src" / "finalize_209HLA_hla_lowIC_v2_weather_physical_gate.py",
        Path(__file__).resolve(),
    ]:
        result[base.rel(path)] = base.sha256(path)
    return result


def patched_write_record(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error):
    ORIGINAL_WRITE_RECORD(cfg, smoke, preflight_result, effective, coverage, by_checkpoint, comparison, gate, status, error)
    path = base.record_path(smoke)
    text = path.read_text(encoding="utf-8")
    text += (
        "\n## 210HLA v2 fixed-width 数据修复\n\n"
        "- 204HLA/v1 的 WTH 天气列宽为 7，破坏 DSSAT 固定列读取；该 100K 只作为负面诊断，不作为有效增强比较。\n"
        "- 210HLA 使用 208HLA v2：数据行严格为 31 字符，50 个增强情景映射到 1950--1999。\n"
        "- 209HLA 在训练前验证六情景均正常成熟、日志降雨与 WTH 相差不超过 2 mm，并产生非零物理产量响应。\n"
        "- PPO、reward、动作、mask、观测、seed、训练/验证年份和分层无放回采样器均保持不变。\n"
    )
    path.write_text(text, encoding="utf-8")


base.TASK_ID = TASK_ID
base.TASK_NAME = TASK_NAME
base.DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
base.PROMPT = ROOT / "prompts" / "2026-08-24_hla_v2_fixed_width_augmentation_100k.md"
base.BoundedAuditedRandomYearEnv = prior.BalancedAuditedRandomYearEnv
base.preflight = patched_preflight
base.code_provenance = patched_code_provenance
base.write_record = patched_write_record


if __name__ == "__main__":
    base.main()
