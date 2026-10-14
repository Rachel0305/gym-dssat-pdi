"""209HLA: pre-training physical gate for the corrected 208HLA WTH bank."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for module_path in [ROOT, ROOT / "src"]:
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


TASK_ID = "209HLA"
TASK_NAME = "hla_lowIC_v2_weather_physical_gate"
BANK = ROOT / "benchmark_results" / "208HLA_hla_lowIC_weather_scenario_bank_v2_fixed_width"
MANIFEST = BANK / "208HLA_weather_scenario_manifest.csv"
INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "hla_lowIC_weather_scenario_bank_208_v2_fixed_width"
RUN_204 = ROOT / "benchmark_results" / "204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo"
INVENTORY = RUN_204 / "evaluation" / "204HLA_training_checkpoint_inventory.csv"
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
    rows = [match for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if (match := LOG_ROW_RE.match(line))]
    if not rows:
        raise RuntimeError(f"No DSSAT summary row in {rel(path)}")
    return {key: int(value) for key, value in rows[-1].groupdict().items()}


def season_rain(path: Path, planting_doy: int = 125, days: int = 162) -> float:
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
    required = [MANIFEST, INPUT_ROOT, INVENTORY]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing 209HLA inputs: " + "; ".join(missing))
    for sub in ["configs", "evaluation", "logs", "daily_outputs", "rendered_inputs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    scenarios = manifest[(manifest["split"].eq("train")) & (pd.to_numeric(manifest["source_year"]).eq(2005))].copy()
    scenarios["year"] = pd.to_numeric(scenarios["pseudo_year"], errors="raise").astype(int)
    if len(scenarios) != 6 or scenarios["variant"].nunique() != 6:
        raise RuntimeError("209HLA requires six v2 scenarios for source year 2005")
    scenarios.to_csv(OUT / "configs" / f"{TASK_ID}_evaluated_scenarios.csv", index=False, encoding="utf-8-sig")
    selection = pd.DataFrame(
        [
            {
                "station_code": "HLA", "year": int(row.year),
                "scenario_type": f"209HLA_{row.variant}",
                "weather_file": Path(str(row.target_weather)).name,
            }
            for row in scenarios.itertuples(index=False)
        ]
    )

    inventory = pd.read_csv(INVENTORY, keep_default_na=False)
    inventory["checkpoint_step"] = pd.to_numeric(inventory["checkpoint_step"], errors="raise").astype(int)
    model_rows = inventory[inventory["checkpoint_step"].eq(10000) & inventory["run_status"].astype(str).str.startswith("ok")]
    if len(model_rows) != 1:
        raise RuntimeError("Expected one 204HLA 10K checkpoint")
    model_row = model_rows.iloc[0]
    old: dict[str, Any] = {
        "engine_lowic": engine.LOWIC_INPUT_ROOT,
        "engine_i": list(engine.BINARY_IRRIGATION_LEVELS),
        "engine_n": list(engine.BINARY_NITROGEN_LEVELS),
        "base_out": base03222.OUT,
        "base_seed": base03222.SEED,
        "base_sites": list(base03222.SITES),
        "base_site_names": dict(base03222.SITE_NAMES),
        "renderer_root": ppo_safe_rendering.MULTISITE_INPUT_ROOT,
    }
    rows: list[dict[str, Any]] = []
    try:
        engine.LOWIC_INPUT_ROOT = INPUT_ROOT
        engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        base03222.OUT = OUT
        base03222.SEED = 0
        base03222.SITES = ["HLA"]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update({"HLA": "HLA"})
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = INPUT_ROOT
        run_config = engine.load_config()
        run_config["paths"]["output_root"] = rel(OUT)
        env_config = direct_ppo.build_env_config(run_config, selection)
        for scenario in scenarios.itertuples(index=False):
            result = base03222.evaluate_checkpoint(run_config, env_config, model_row, int(scenario.year))
            log = OUT / "logs" / "HLA" / f"HLA_{scenario.year}_HLA_{scenario.year}_032_10_ckpt10000_eval.log"
            result.update(parse_last_log_row(log))
            weather = ROOT / str(scenario.target_weather)
            result.update(
                {
                    "source_year": int(scenario.source_year),
                    "weather_variant": str(scenario.variant),
                    "rain_multiplier": float(scenario.rain_multiplier),
                    "perturbation_window": str(scenario.window),
                    "wth_expected_season_rain_mm": season_rain(weather),
                }
            )
            rows.append(result)
    finally:
        engine.LOWIC_INPUT_ROOT = old["engine_lowic"]
        engine.BINARY_IRRIGATION_LEVELS = old["engine_i"]
        engine.BINARY_NITROGEN_LEVELS = old["engine_n"]
        base03222.OUT = old["base_out"]
        base03222.SEED = old["base_seed"]
        base03222.SITES = old["base_sites"]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update(old["base_site_names"])
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["renderer_root"]

    details = pd.DataFrame(rows).sort_values("weather_variant")
    details["rain_log_minus_wth_mm"] = pd.to_numeric(details["rain"], errors="coerce") - details["wth_expected_season_rain_mm"]
    original = details[details["weather_variant"].eq("original")].iloc[0]
    details["yield_delta_vs_original_kg_ha"] = pd.to_numeric(details["final_grnwt"], errors="coerce") - float(original["final_grnwt"])
    details["action_changed_vs_original"] = ~details["action_sequence"].astype(str).eq(str(original["action_sequence"]))
    details.to_csv(OUT / "evaluation" / f"{TASK_ID}_scenario_physical_gate_details.csv", index=False, encoding="utf-8-sig")

    gate = {
        "all_six_scenarios_completed": bool(len(details) == 6 and details["run_status"].astype(str).str.startswith("ok").all()),
        "all_weather_log_rain_matches_wth_within_2mm": bool(details["rain_log_minus_wth_mm"].abs().le(2.0).all()),
        "all_scenarios_matured": bool(pd.to_numeric(details["mat"], errors="coerce").gt(0).all()),
        "all_scenarios_positive_yield": bool(pd.to_numeric(details["final_grnwt"], errors="coerce").gt(0).all()),
        "physical_yield_response_detected": bool(details["yield_delta_vs_original_kg_ha"].abs().gt(1.0).sum() >= 2),
        "old_10k_policy_action_response_detected": bool(details["action_changed_vs_original"].sum() >= 1),
    }
    gate["allow_2k_smoke"] = all(gate[key] for key in [
        "all_six_scenarios_completed", "all_weather_log_rain_matches_wth_within_2mm",
        "all_scenarios_matured", "all_scenarios_positive_yield", "physical_yield_response_detected",
    ])
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}", "status": "completed_diagnostic_only", "training_run": False,
        "checkpoint_step": 10000, "source_year": 2005, "gate": gate,
        "details": rel(OUT / "evaluation" / f"{TASK_ID}_scenario_physical_gate_details.csv"), "record": rel(RECORD),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    table = details[["weather_variant", "year", "wth_expected_season_rain_mm", "rain", "flo", "mat", "final_grnwt", "yield_delta_vs_original_kg_ha", "total_irrigation", "total_n", "action_changed_vs_original"]].copy()
    record = [
        "# 209HLA HLA v2 天气增强训练前物理门槛",
        "", "## 结论", "",
        f"- `allow_2k_smoke={gate['allow_2k_smoke']}`；该任务不训练。",
        "- 使用已冻结的 204HLA 10K checkpoint，只检查 208HLA 六情景是否被 DSSAT 正确读取并产生有效作物轨迹。",
        "- 策略动作响应单列报告，不作为物理输入有效性的必要门槛。",
        "", "## 门槛", "", "```json", json.dumps(gate, ensure_ascii=False, indent=2), "```",
        "", "## 六情景结果", "", table.to_markdown(index=False),
        "", "## 复现", "",
        f"- 代码：`{rel(Path(__file__))}`。", f"- 情景清单：`{rel(MANIFEST)}`。", f"- 逐情景结果：`{result['details']}`。",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
