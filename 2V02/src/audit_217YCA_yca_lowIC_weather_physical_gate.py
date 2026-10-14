"""Audit that the YC six-weather bank is physically readable before PPO smoke."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


TASK_ID = "217YCA"
TASK_NAME = "yca_lowIC_weather_physical_gate"
BANK = ROOT / "benchmark_results" / "217YCA_yca_lowIC_weather_scenario_bank_v1_fixed_width"
MANIFEST = BANK / "217YCA_weather_scenario_manifest.csv"
INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "yca_lowIC_weather_scenario_bank_217_v1_fixed_width"
RUN_055 = ROOT / "benchmark_results" / "055_00_yca_lowIC_expanded_action_maskableppo"
INVENTORY = RUN_055 / "evaluation" / "055_00_training_checkpoint_inventory.csv"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
RECORD = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
LOG_ROW_RE = re.compile(
    r"^\s*1\s+MZ\s+1\s+(?P<flo>-?\d+)\s+(?P<mat>-?\d+)\s+"
    r"(?P<topwt>-?\d+)\s+(?P<harwt>-?\d+)\s+(?P<rain>-?\d+)\s+"
    r"(?P<tirr>-?\d+)\s+(?P<cet>-?\d+)\s+(?P<pesw>-?\d+)\s+"
    r"(?P<tnup>-?\d+)\s+(?P<tnlf>-?\d+)"
)
WTH_RE = re.compile(r"^\s*(\d{7})\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s+([-+]?\d*\.?\d+)\s*$")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_last_log_row(path: Path) -> dict[str, int]:
    rows = [m for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if (m := LOG_ROW_RE.match(line))]
    if not rows:
        raise RuntimeError(f"No DSSAT summary row in {rel(path)}")
    return {key: int(value) for key, value in rows[-1].groupdict().items()}


def season_rain(path: Path, planting_doy: int = 170, days: int = 101) -> float:
    end = planting_doy + days - 1
    total = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WTH_RE.match(line)
        if match and planting_doy <= int(match.group(1)[-3:]) <= end:
            total += float(match.group(2))
    return total


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite {rel(OUT)}")
    missing = [rel(p) for p in [MANIFEST, INPUT_ROOT, INVENTORY] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing YC physical-gate inputs: " + "; ".join(missing))
    for sub in ["configs", "evaluation", "logs", "daily_outputs", "rendered_inputs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    scenarios = manifest[(manifest["split"].eq("train")) & (pd.to_numeric(manifest["source_year"]).eq(2005))].copy()
    scenarios["year"] = pd.to_numeric(scenarios["pseudo_year"], errors="raise").astype(int)
    if len(scenarios) != 6 or scenarios["variant"].nunique() != 6:
        raise RuntimeError("217YCA requires six scenarios for source year 2005")
    scenarios.to_csv(OUT / "configs" / f"{TASK_ID}_evaluated_scenarios.csv", index=False, encoding="utf-8-sig")
    selection = pd.DataFrame([
        {"station_code": "YCA", "site": "YC", "station_name": "YC", "source_year": int(row.source_year),
         "year": int(row.year), "split": "train", "weather_variant": str(row.variant),
         "scenario_type": f"{TASK_ID}_{row.variant}", "weather_file": Path(str(row.target_weather)).name,
         "selected_for_train": True, "selected_for_eval": True, "selection_reason": "217YCA_physical_gate"}
        for row in scenarios.itertuples(index=False)
    ])

    inventory = pd.read_csv(INVENTORY, keep_default_na=False)
    inventory["checkpoint_step"] = pd.to_numeric(inventory["checkpoint_step"], errors="raise").astype(int)
    model_rows = inventory[inventory["checkpoint_step"].eq(100000) & inventory["run_status"].astype(str).str.startswith("ok")]
    if len(model_rows) != 1:
        raise RuntimeError("Expected one frozen 055_00 YCA 100K checkpoint")
    model_row = model_rows.iloc[0]

    old = {
        "lowic": engine.LOWIC_INPUT_ROOT, "i": list(engine.BINARY_IRRIGATION_LEVELS),
        "n": list(engine.BINARY_NITROGEN_LEVELS), "out": base03222.OUT,
        "seed": base03222.SEED, "sites": list(base03222.SITES),
        "site_names": dict(base03222.SITE_NAMES), "renderer": ppo_safe_rendering.MULTISITE_INPUT_ROOT,
    }
    rows: list[dict[str, Any]] = []
    try:
        engine.LOWIC_INPUT_ROOT = INPUT_ROOT
        engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        base03222.OUT = OUT
        base03222.SEED = 0
        base03222.SITES = ["YCA"]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update({"YCA": "YC"})
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        run_config = engine.load_config()
        run_config["paths"]["output_root"] = rel(OUT)
        env_config = direct_ppo.build_env_config(run_config, selection)
        for scenario in scenarios.itertuples(index=False):
            result = base03222.evaluate_checkpoint(run_config, env_config, model_row, int(scenario.year))
            log = OUT / "logs" / "YCA" / f"YCA_{scenario.year}_YCA_{scenario.year}_032_10_ckpt100000_eval.log"
            result.update(parse_last_log_row(log))
            weather = ROOT / str(scenario.target_weather)
            result.update({
                "source_year": int(scenario.source_year), "weather_variant": str(scenario.variant),
                "rain_multiplier": float(scenario.rain_multiplier), "perturbation_window": str(scenario.window),
                "wth_expected_season_rain_mm": season_rain(weather),
            })
            rows.append(result)
    finally:
        engine.LOWIC_INPUT_ROOT = old["lowic"]
        engine.BINARY_IRRIGATION_LEVELS = old["i"]
        engine.BINARY_NITROGEN_LEVELS = old["n"]
        base03222.OUT = old["out"]
        base03222.SEED = old["seed"]
        base03222.SITES = old["sites"]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update(old["site_names"])
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["renderer"]

    details = pd.DataFrame(rows).sort_values("weather_variant")
    details["rain_log_minus_wth_mm"] = pd.to_numeric(details["rain"], errors="coerce") - details["wth_expected_season_rain_mm"]
    original = details[details["weather_variant"].eq("original")].iloc[0]
    details["yield_delta_vs_original_kg_ha"] = pd.to_numeric(details["final_grnwt"], errors="coerce") - float(original["final_grnwt"])
    details["action_changed_vs_original"] = ~details["action_sequence"].astype(str).eq(str(original["action_sequence"]))
    detail_path = OUT / "evaluation" / f"{TASK_ID}_scenario_physical_gate_details.csv"
    details.to_csv(detail_path, index=False, encoding="utf-8-sig")
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
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}", "status": "completed_diagnostic_only", "training_run": False,
        "checkpoint_step": 100000, "source_year": 2005, "gate": gate,
        "details": rel(detail_path), "record": rel(RECORD),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["weather_variant", "year", "wth_expected_season_rain_mm", "rain", "flo", "mat", "final_grnwt",
               "yield_delta_vs_original_kg_ha", "total_irrigation", "total_n", "action_changed_vs_original"]
    record = [
        "# 217YCA YC 天气增强训练前物理门槛", "", "## 结论", "",
        f"- `allow_2k_smoke={gate['allow_2k_smoke']}`；本任务只做 DSSAT 物理回放，不训练。",
        "- 使用已有 YC 055_00 100K checkpoint，仅检查六种天气是否被正确读取并产生有效作物轨迹。",
        "- 策略动作是否响应天气单列报告，不作为天气输入物理有效性的必要门槛。", "",
        "## 门槛", "", "```json", json.dumps(gate, ensure_ascii=False, indent=2), "```", "",
        "## 六情景结果", "", details[columns].to_markdown(index=False), "", "## 复现", "",
        f"- 代码：`{rel(Path(__file__))}`。", f"- 情景清单：`{rel(MANIFEST)}`。",
        f"- 明细：`{rel(detail_path)}`。",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not gate["allow_2k_smoke"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
