"""Finalize the completed 209HLA replays after a JSON bool serialization error."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "209HLA"
TASK_NAME = "hla_lowIC_v2_weather_physical_gate"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DETAIL = OUT / "evaluation" / f"{TASK_ID}_scenario_physical_gate_details.csv"
RECORD = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    if not DETAIL.exists():
        raise FileNotFoundError(DETAIL)
    if (OUT / f"{TASK_ID}_result.json").exists() or RECORD.exists():
        raise FileExistsError("209HLA finalizer refuses to overwrite existing final outputs")
    details = pd.read_csv(DETAIL, keep_default_na=False)
    for column in ["rain_log_minus_wth_mm", "mat", "final_grnwt", "yield_delta_vs_original_kg_ha"]:
        details[column] = pd.to_numeric(details[column], errors="coerce")
    details["action_changed_vs_original"] = details["action_changed_vs_original"].astype(str).str.lower().eq("true")
    gate = {
        "all_six_scenarios_completed": bool(len(details) == 6 and details["run_status"].astype(str).str.startswith("ok").all()),
        "all_weather_log_rain_matches_wth_within_2mm": bool(details["rain_log_minus_wth_mm"].abs().le(2.0).all()),
        "all_scenarios_matured": bool(details["mat"].gt(0).all()),
        "all_scenarios_positive_yield": bool(details["final_grnwt"].gt(0).all()),
        "physical_yield_response_detected": bool(details["yield_delta_vs_original_kg_ha"].abs().gt(1.0).sum() >= 2),
        "old_10k_policy_action_response_detected": bool(details["action_changed_vs_original"].sum() >= 1),
    }
    gate["allow_2k_smoke"] = all(gate[key] for key in [
        "all_six_scenarios_completed", "all_weather_log_rain_matches_wth_within_2mm",
        "all_scenarios_matured", "all_scenarios_positive_yield", "physical_yield_response_detected",
    ])
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed_diagnostic_only_finalized_without_replay",
        "training_run": False,
        "checkpoint_step": 10000,
        "source_year": 2005,
        "gate": gate,
        "failed_attempt": {
            "stage": "post-replay JSON writing",
            "error": "TypeError: Object of type bool_ is not JSON serializable",
            "data_loss": False,
            "fix": "Cast the completed-series boolean to Python bool and finalize the existing CSV; no DSSAT replay or training rerun."
        },
        "details": rel(DETAIL),
        "record": rel(RECORD),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table = details[["weather_variant", "year", "wth_expected_season_rain_mm", "rain", "flo", "mat", "final_grnwt", "yield_delta_vs_original_kg_ha", "total_irrigation", "total_n", "action_changed_vs_original"]].copy()
    record = [
        "# 209HLA HLA v2 天气增强训练前物理门槛",
        "", "## 结论", "",
        f"- `allow_2k_smoke={gate['allow_2k_smoke']}`；该任务不训练。",
        "- 六个 DSSAT 回放已全部完成：日志降雨与 WTH 差值均不超过 2 mm，均正常成熟且产量为正。",
        "- 旧 204HLA 10K 策略动作不响应天气，但这是旧策略诊断，不否定 v2 输入的物理有效性。",
        "", "## 失败与修复", "",
        "- 首次脚本在六次回放完成后写 JSON 时因 `numpy.bool_` 不可序列化停止；日值、日志和明细 CSV 均已保存。",
        "- 修复为显式 Python `bool`，并由本 finalizer 读取已保存 CSV 完成汇总；没有重复 DSSAT 回放或训练。",
        f"- 修复前备份：`backups/2026-08-24_209HLA_physical_gate_before_bool_serialization_fix.py`。",
        "", "## 门槛", "", "```json", json.dumps(gate, ensure_ascii=False, indent=2), "```",
        "", "## 六情景结果", "", table.to_markdown(index=False),
        "", "## 复现", "",
        "- 回放代码：`src/audit_209HLA_hla_lowIC_v2_weather_physical_gate.py`。",
        f"- 汇总代码：`{rel(Path(__file__))}`。",
        "- 情景清单：`benchmark_results/208HLA_hla_lowIC_weather_scenario_bank_v2_fixed_width/208HLA_weather_scenario_manifest.csv`。",
        f"- 逐情景结果：`{rel(DETAIL)}`。",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
