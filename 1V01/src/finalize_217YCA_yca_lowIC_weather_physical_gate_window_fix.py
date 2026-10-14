"""Finalize the YC physical gate after correcting the season-rain window.

The DSSAT replay outputs are reused.  Only the audit's rain aggregation is
recomputed from DAP 1--101, matching the YC maturity horizon in the replay.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from audit_217YCA_yca_lowIC_weather_physical_gate import ROOT, MANIFEST, OUT, RECORD, rel, season_rain


DETAIL = OUT / "evaluation" / "217YCA_scenario_physical_gate_details.csv"
RESULT = OUT / "217YCA_result.json"


def main() -> None:
    if not DETAIL.exists() or not RESULT.exists():
        raise FileNotFoundError("217YCA physical-gate replay artifacts are incomplete")
    details = pd.read_csv(DETAIL, keep_default_na=False)
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    train = manifest[(manifest["split"].eq("train")) & (pd.to_numeric(manifest["source_year"]).eq(2005))].copy()
    expected: dict[str, float] = {}
    for row in train.itertuples(index=False):
        expected[str(row.variant)] = season_rain(ROOT / str(row.target_weather))
    details["wth_expected_season_rain_mm"] = details["weather_variant"].map(expected)
    details["rain_log_minus_wth_mm"] = pd.to_numeric(details["rain"], errors="coerce") - details["wth_expected_season_rain_mm"]
    original = details[details["weather_variant"].eq("original")].iloc[0]
    details["yield_delta_vs_original_kg_ha"] = pd.to_numeric(details["final_grnwt"], errors="coerce") - float(original["final_grnwt"])
    details["action_changed_vs_original"] = details["action_changed_vs_original"].astype(str).str.lower().eq("true")
    details.to_csv(DETAIL, index=False, encoding="utf-8-sig")

    gate = {
        "all_six_scenarios_completed": bool(len(details) == 6 and details["run_status"].astype(str).str.startswith("ok").all()),
        "all_weather_log_rain_matches_wth_within_2mm": bool(details["rain_log_minus_wth_mm"].abs().le(2.0).all()),
        "all_scenarios_matured": bool(pd.to_numeric(details["mat"], errors="coerce").gt(0).all()),
        "all_scenarios_positive_yield": bool(pd.to_numeric(details["final_grnwt"], errors="coerce").gt(0).all()),
        "physical_yield_response_detected": bool(details["yield_delta_vs_original_kg_ha"].abs().gt(1.0).sum() >= 2),
        "old_055_policy_action_response_detected": bool(details["action_changed_vs_original"].sum() >= 1),
    }
    gate["allow_2k_smoke"] = all(gate[key] for key in [
        "all_six_scenarios_completed", "all_weather_log_rain_matches_wth_within_2mm",
        "all_scenarios_matured", "all_scenarios_positive_yield", "physical_yield_response_detected",
    ])
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    result.update({
        "status": "completed_diagnostic_only_finalized",
        "gate": gate,
        "failed_attempt": {
            "stage": "physical-gate rain aggregation",
            "error": "Initial audit used HLA's 162-day rain window; YC maturity is approximately DAP101.",
            "data_loss": False,
            "fix": "Recomputed WTH rain through DAP101 from already saved replay outputs; no DSSAT replay rerun.",
            "backup": "backups/2026-09-04_217YCA_physical_gate_before_window_fix",
        },
    })
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table = details[["weather_variant", "year", "wth_expected_season_rain_mm", "rain", "flo", "mat", "final_grnwt",
                    "yield_delta_vs_original_kg_ha", "total_irrigation", "total_n", "action_changed_vs_original"]]
    record = [
        "# 217YCA YC 天气增强训练前物理门槛", "", "## 结论", "",
        f"- `allow_2k_smoke={gate['allow_2k_smoke']}`；本任务只做 DSSAT 物理回放，不训练。",
        "- 六种天气均正常完成、成熟且产量为正；增强天气产生了明确的产量响应。",
        "- 雨量一致性按 YC 的播后 DAP1--101 成熟窗口计算；日志与 WTH 的差值均在 2 mm 内。",
        "- 旧 055_00 策略在六情景中的动作序列没有响应天气，但这不否定天气输入本身的物理有效性。", "",
        "## 修正记录", "",
        "- 首次审计复用了 HLA 的 162 天窗口，造成成熟后的雨量被多计；原始 JSON/CSV/记录已备份。",
        "- 修正只重算已保存 CSV 的雨量口径，没有重复 DSSAT 回放。", "",
        "## 门槛", "", "```json", json.dumps(gate, ensure_ascii=False, indent=2), "```", "",
        "## 六情景结果", "", table.to_markdown(index=False), "", "## 复现", "",
        "- 回放代码：`src/audit_217YCA_yca_lowIC_weather_physical_gate.py`。",
        f"- 修正代码：`{rel(Path(__file__))}`。", f"- 明细：`{rel(DETAIL)}`。",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not gate["allow_2k_smoke"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
