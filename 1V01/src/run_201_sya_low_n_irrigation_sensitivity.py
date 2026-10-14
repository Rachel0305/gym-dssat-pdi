"""201: SYA irrigation sensitivity under zero and low nitrogen.

This task replays fixed DSSAT management schedules across the six weather
variants. It reuses task 200's corrected fixed-width Summary.OUT parser and
runner, while adding irrigation gradients at N=0 and N=100 kg/ha.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import run_164_sya_weather_v2_fixed_management_gate as fixed_gate
import run_200_sya_fixed_management_sensitivity as task200


TASK_ID = "201"
TASK_NAME = "sya_low_n_irrigation_sensitivity"
DEFAULT_CONFIG = ROOT / "configs" / "163_sya_weather_scenario_bank_v2.json"
MANIFEST_ROOT = ROOT / "benchmark_results" / "166_sya_weather_scenario_bank_v2_fixed_format"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
PREVIOUS = ROOT / "benchmark_results" / "200_sya_fixed_management_sensitivity_retry2"


N100 = {1: 25.0, 30: 25.0, 60: 25.0}
IRRIGATION = {
    "I60": {35: 30.0, 65: 30.0},
    "I120": {35: 50.0, 65: 50.0, 95: 20.0},
    "I240": {35: 50.0, 50: 50.0, 65: 50.0, 80: 50.0, 95: 40.0},
}

CASES: list[dict[str, Any]] = [
    {"management_id": "I60_N0", "irrigation": IRRIGATION["I60"], "nitrogen": {}},
    {"management_id": "I120_N0", "irrigation": IRRIGATION["I120"], "nitrogen": {}},
    {"management_id": "I240_N0", "irrigation": IRRIGATION["I240"], "nitrogen": {}},
    {"management_id": "I60_N100", "irrigation": IRRIGATION["I60"], "nitrogen": N100},
    {"management_id": "I120_N100", "irrigation": IRRIGATION["I120"], "nitrogen": N100},
    {"management_id": "I240_N100", "irrigation": IRRIGATION["I240"], "nitrogen": N100},
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def make_effect_table(summary: pd.DataFrame) -> pd.DataFrame:
    baseline_ids = {"0": "I0_N0", "100": "I0_N100", "200": "I0_N200", "300": "I0_N300"}
    out = summary.copy()
    out["nitrogen_level_kg_ha"] = pd.to_numeric(out["summary_nitrogen_total_kg_ha"], errors="coerce")
    out["irrigation_level_mm"] = pd.to_numeric(out["summary_irrigation_total_mm"], errors="coerce")
    out["nitrogen_level_label"] = out["nitrogen_level_kg_ha"].round().astype("Int64").astype(str)
    out["baseline_management_id_same_N"] = out["nitrogen_level_label"].map(baseline_ids)
    baseline = out.set_index(["weather_variant", "management_id"])
    metrics = [
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "gross_water_productivity_kg_m3",
        "PFP_N_kg_kg",
        "N_leaching_kg_ha",
        "ETCP_mm",
    ]
    for metric in metrics:
        values = []
        for row in out.itertuples(index=False):
            key = (row.weather_variant, row.baseline_management_id_same_N)
            current = getattr(row, metric)
            if key not in baseline.index:
                values.append(np.nan)
                continue
            reference = baseline.loc[key, metric]
            values.append(
                float(current - reference)
                if np.isfinite(current) and np.isfinite(reference)
                else np.nan
            )
        out[f"delta_{metric}_vs_same_N_I0"] = values
    return out


def write_mean_tables(summary: pd.DataFrame, out: Path) -> None:
    mean = (
        summary.groupby("management_id", as_index=False)
        .agg(
            nitrogen_kg_ha=("summary_nitrogen_total_kg_ha", "mean"),
            irrigation_mm=("summary_irrigation_total_mm", "mean"),
            grain_yield_kg_ha=("grain_yield_kg_ha", "mean"),
            WP_ET_kg_m3=("WP_ET_kg_m3", "mean"),
            gross_water_productivity_kg_m3=("gross_water_productivity_kg_m3", "mean"),
            PFP_N_kg_kg=("PFP_N_kg_kg", "mean"),
            N_uptake_kg_ha=("N_uptake_kg_ha", "mean"),
            N_leaching_kg_ha=("N_leaching_kg_ha", "mean"),
            max_WSPD=("max_WSPD", "mean"),
            max_NSTD=("max_NSTD", "mean"),
        )
        .sort_values(["nitrogen_kg_ha", "irrigation_mm", "management_id"])
    )
    mean.to_csv(out / "evaluation" / "mean_by_management.csv", index=False, encoding="utf-8-sig")


def write_diagnosis(summary: pd.DataFrame, effect: pd.DataFrame, out: Path) -> None:
    low = summary[summary["management_id"].isin([case["management_id"] for case in CASES])]
    mean = (
        low.groupby("management_id")[["grain_yield_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "N_leaching_kg_ha"]]
        .mean()
        .round(3)
    )
    effect_low = effect[effect["management_id"].isin([case["management_id"] for case in CASES])]
    mean_effect = (
        effect_low.groupby("management_id")[
            [
                "delta_grain_yield_kg_ha_vs_same_N_I0",
                "delta_WP_ET_kg_m3_vs_same_N_I0",
                "delta_PFP_N_kg_kg_vs_same_N_I0",
                "delta_N_leaching_kg_ha_vs_same_N_I0",
            ]
        ]
        .mean()
        .round(3)
    )
    lines = [
        "# 201 SYA 低氮/零氮灌溉敏感性实验",
        "",
        "本实验只改变固定管理措施，不训练 PPO；天气为同一组六种 2005 情景。",
        "新增组为 N=0 和 N=100 kg/ha 下的 I60/I120/I240，并分别以同一 N 水平的 I0 作为基线。",
        "",
        "## 新增组六情景均值",
        "",
        mean.to_markdown(),
        "",
        "## 相对同氮 I0 的灌溉效应",
        "",
        mean_effect.to_markdown(),
        "",
        "## 解释边界",
        "",
        "- 灌溉时间表与 task 200 相同，I120/I240 是拆分到多个不超过 50 mm 的事件，不代表所有可能的灌溉时机。",
        "- N=0 时 PFP_N 没有定义；NSTD 是氮胁迫指标，不能替代产量或氮收支。",
        "- N_uptake / applied N 不是严格的氮肥回收率，因为作物还可能利用初始土壤氮。",
        "- 结论首先适用于这组六种 2005 合成天气；跨年份仍需验证。",
        "",
        "## 文件",
        "",
        f"- 新增组汇总：`{rel(out / 'evaluation' / 'low_n_irrigation_summary.csv')}`",
        f"- 新增组灌溉效应：`{rel(out / 'evaluation' / 'low_n_irrigation_effect_vs_same_N_I0.csv')}`",
        f"- 完整管理矩阵：`{rel(out / 'evaluation' / 'combined_all_management_summary.csv')}`",
        f"- 逐日结果：`{rel(out / 'evaluation' / 'combined_all_management_daily.csv')}`",
    ]
    (out / "201_low_n_irrigation_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {output_root}")
    if not PREVIOUS.exists():
        raise FileNotFoundError(PREVIOUS)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    manifests = sorted(MANIFEST_ROOT.glob("*_manifest.csv"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"Expected one manifest, found {manifests}")
    manifest = pd.read_csv(manifests[0], keep_default_na=False)
    output_root.mkdir(parents=True, exist_ok=False)
    env_config = fixed_gate.make_env_config(output_root)
    summaries: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        variant = str(item.variant)
        scenario_root = ROOT / str(item.scenario_input_root)
        for case in CASES:
            try:
                daily, summary = task200.run_case(env_config, scenario_root, variant, case, output_root)
                daily_frames.append(daily)
                summaries.append(summary)
            except Exception as exc:
                failures.append({"weather_variant": variant, "management_id": case["management_id"], "error": repr(exc)})
    expected = len(manifest) * len(CASES)
    if len(summaries) != expected:
        raise RuntimeError(f"Incomplete new matrix: {len(summaries)}/{expected}; {failures}")

    new_summary = pd.DataFrame(summaries).sort_values(["weather_variant", "management_id"])
    new_daily = pd.concat(daily_frames, ignore_index=True)
    evaluation = output_root / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    new_summary.to_csv(evaluation / "low_n_irrigation_summary.csv", index=False, encoding="utf-8-sig")
    new_daily.to_csv(evaluation / "low_n_irrigation_daily.csv", index=False, encoding="utf-8-sig")

    previous_summary_path = PREVIOUS / "evaluation" / "six_weather_fixed_management_summary.csv"
    previous_daily_path = PREVIOUS / "evaluation" / "six_weather_fixed_management_daily.csv"
    previous_summary = pd.read_csv(previous_summary_path)
    previous_daily = pd.read_csv(previous_daily_path)
    combined_summary = pd.concat([previous_summary, new_summary], ignore_index=True)
    combined_daily = pd.concat([previous_daily, new_daily], ignore_index=True)
    combined_summary.to_csv(evaluation / "combined_all_management_summary.csv", index=False, encoding="utf-8-sig")
    combined_daily.to_csv(evaluation / "combined_all_management_daily.csv", index=False, encoding="utf-8-sig")

    effects = make_effect_table(combined_summary)
    effects.to_csv(evaluation / "combined_irrigation_effect_vs_same_N_I0.csv", index=False, encoding="utf-8-sig")
    new_effects = effects[effects["management_id"].isin([case["management_id"] for case in CASES])].copy()
    new_effects.to_csv(evaluation / "low_n_irrigation_effect_vs_same_N_I0.csv", index=False, encoding="utf-8-sig")
    write_mean_tables(combined_summary, output_root)
    write_diagnosis(combined_summary, effects, output_root)

    report = {
        "status": "completed",
        "task": f"{TASK_ID}_{TASK_NAME}",
        "training_executed": False,
        "new_episode_count": len(new_summary),
        "new_case_count": len(CASES),
        "weather_variant_count": len(manifest),
        "failures": failures,
        "previous_task_reused": rel(PREVIOUS),
        "outputs": {
            "new_summary": rel(evaluation / "low_n_irrigation_summary.csv"),
            "new_daily": rel(evaluation / "low_n_irrigation_daily.csv"),
            "new_effect": rel(evaluation / "low_n_irrigation_effect_vs_same_N_I0.csv"),
            "combined_summary": rel(evaluation / "combined_all_management_summary.csv"),
            "combined_daily": rel(evaluation / "combined_all_management_daily.csv"),
            "combined_effect": rel(evaluation / "combined_irrigation_effect_vs_same_N_I0.csv"),
            "mean": rel(evaluation / "mean_by_management.csv"),
            "diagnosis": rel(output_root / "201_low_n_irrigation_diagnosis.md"),
        },
    }
    (output_root / "201_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    print(json.dumps(run(config.resolve(), output_root), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
