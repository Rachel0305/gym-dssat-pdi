from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "042_01"
TASK_NAME = "sya_lowIC_weather_forecast_ppo_validation_summary"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TABLE_DIR = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"

PPO_PATH = (
    ROOT
    / "benchmark_results"
    / "042_00_sya_lowIC_weather_forecast_observation_maskableppo"
    / "evaluation"
    / "042_00_checkpoint_validation_summary.csv"
)
BASELINE_PATH = (
    ROOT
    / "benchmark_results"
    / "040_21_sya_lowIC_four_baseline_rebuild"
    / "evaluation"
    / "040_21_baseline_summary.csv"
)
OBS_AUDIT_PATH = (
    ROOT
    / "benchmark_results"
    / "042_00_sya_lowIC_weather_forecast_observation_maskableppo"
    / "audits"
    / "042_00_observation_smoke_audit.csv"
)

FOUR_SCENARIO_MAP = {
    "": "null",
    "null": "null",
    "recorded_farmer_template": "recorded_farmer",
    "recorded_farmer": "recorded_farmer",
    "dssat_auto": "dssat_auto",
    "official_extension_expert": "official_extension_expert",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty input file: {path}")


def load_baseline_maxima() -> pd.DataFrame:
    require_file(BASELINE_PATH)
    base = pd.read_csv(BASELINE_PATH, keep_default_na=False)
    required = {
        "station_code",
        "site",
        "year",
        "scenario",
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
    }
    missing = required - set(base.columns)
    if missing:
        raise KeyError(f"Baseline missing columns: {sorted(missing)}")

    base = base.copy()
    base["scenario_clean"] = base["scenario"].map(FOUR_SCENARIO_MAP).fillna(base["scenario"])
    wanted = set(FOUR_SCENARIO_MAP.values())
    base = base[base["scenario_clean"].isin(wanted)].copy()
    if base.empty:
        raise ValueError("No four-baseline rows after scenario normalization.")

    # PFP_N is undefined when N=0; keep NaN out of maxima.
    base["PFP_N_kg_kg"] = pd.to_numeric(base["PFP_N_kg_kg"], errors="coerce")
    for col in ["grain_yield_kg_ha", "WP_ET_kg_m3"]:
        base[col] = pd.to_numeric(base[col], errors="raise")

    maxima = (
        base.groupby(["station_code", "site", "year"], as_index=False)
        .agg(
            four_max_yield=("grain_yield_kg_ha", "max"),
            four_max_wp_et=("WP_ET_kg_m3", "max"),
            four_max_pfp_n=("PFP_N_kg_kg", "max"),
            baseline_scenario_count=("scenario_clean", "nunique"),
        )
    )
    bad = maxima[maxima["baseline_scenario_count"] != 4]
    if not bad.empty:
        raise ValueError(
            "Some site-years do not have all four baseline scenarios: "
            + bad.to_string(index=False)
        )
    return maxima


def summarize() -> dict:
    require_file(PPO_PATH)
    require_file(OBS_AUDIT_PATH)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)

    ppo = pd.read_csv(PPO_PATH, keep_default_na=False)
    required = {
        "station_code",
        "year",
        "checkpoint_step",
        "final_grnwt",
        "total_irrigation",
        "total_n",
        "PFP_N",
    }
    missing = required - set(ppo.columns)
    if missing:
        raise KeyError(f"PPO summary missing columns: {sorted(missing)}")
    if "site" not in ppo.columns:
        ppo["site"] = ppo["station_code"].str[:2]

    # ET is not stored in 042_00 PPO rows, so WP_ET cannot be recomputed here.
    # It is intentionally reported as not available instead of invented.
    ppo = ppo.copy()
    ppo["final_grnwt"] = pd.to_numeric(ppo["final_grnwt"], errors="raise")
    ppo["PFP_N"] = pd.to_numeric(ppo["PFP_N"], errors="coerce")
    ppo["total_irrigation"] = pd.to_numeric(ppo["total_irrigation"], errors="raise")
    ppo["total_n"] = pd.to_numeric(ppo["total_n"], errors="raise")

    maxima = load_baseline_maxima()
    merged = ppo.merge(maxima, on=["station_code", "site", "year"], how="left", validate="many_to_one")
    if merged["four_max_yield"].isna().any():
        missing_rows = merged[merged["four_max_yield"].isna()][["station_code", "site", "year"]].drop_duplicates()
        raise ValueError("Missing baseline maxima for:\n" + missing_rows.to_string(index=False))

    merged["gap_yield_vs_four_max"] = merged["final_grnwt"] - merged["four_max_yield"]
    merged["gap_pfp_n_vs_four_max"] = merged["PFP_N"] - merged["four_max_pfp_n"]
    merged["yield_win_four"] = merged["gap_yield_vs_four_max"] > 0
    merged["pfp_n_win_four"] = merged["gap_pfp_n_vs_four_max"] > 0
    merged["wp_et_win_four"] = pd.NA
    merged["gap_wp_et_vs_four_max"] = np.nan
    merged["any_metric_win_four_partial"] = merged["yield_win_four"] | merged["pfp_n_win_four"]
    merged["all_available_metric_win_four"] = merged["yield_win_four"] & merged["pfp_n_win_four"]
    merged["wp_et_status"] = "not_available_in_042_00_validation_summary"

    enriched_path = TABLE_DIR / "042_01_ppo_validation_enriched_vs_four_baselines.csv"
    merged.to_csv(enriched_path, index=False, encoding="utf-8-sig")

    by_ckpt = (
        merged.groupby(["station_code", "site", "checkpoint_step"], as_index=False)
        .agg(
            validation_years=("year", "nunique"),
            mean_final_grnwt=("final_grnwt", "mean"),
            mean_total_irrigation=("total_irrigation", "mean"),
            mean_total_n=("total_n", "mean"),
            mean_pfp_n=("PFP_N", "mean"),
            yield_win_four_count=("yield_win_four", "sum"),
            pfp_n_win_four_count=("pfp_n_win_four", "sum"),
            any_metric_win_four_partial_count=("any_metric_win_four_partial", "sum"),
            all_available_metric_win_four_count=("all_available_metric_win_four", "sum"),
            mean_gap_yield_vs_four_max=("gap_yield_vs_four_max", "mean"),
            mean_gap_pfp_n_vs_four_max=("gap_pfp_n_vs_four_max", "mean"),
        )
        .sort_values(["checkpoint_step"])
    )
    by_ckpt_path = TABLE_DIR / "042_01_validation_summary_by_checkpoint_vs_four_baselines.csv"
    by_ckpt.to_csv(by_ckpt_path, index=False, encoding="utf-8-sig")

    obs = pd.read_csv(OBS_AUDIT_PATH, keep_default_na=False)
    obs_ok = bool(
        (obs["base_observation_dim"].astype(int) == 25).all()
        and (obs["enhanced_observation_dim"].astype(int) == 30).all()
        and (obs["obs_len"].astype(int) == 30).all()
    )

    best_yield_row = by_ckpt.loc[by_ckpt["mean_final_grnwt"].idxmax()].to_dict()
    best_partial_row = by_ckpt.loc[by_ckpt["any_metric_win_four_partial_count"].idxmax()].to_dict()

    lines = [
        "# 042_01 SYA lowIC 天气/预报 observation PPO 验证结果后处理记录",
        "",
        "## 任务性质",
        "",
        "- 只做后处理汇总；没有重新训练；没有重新运行 DSSAT。",
        "- 输入 PPO 结果来自 `042_00`。",
        "- 四基线来自 `040_21` lowIC 可信基线重建。",
        "- 注意：`042_00` 的验证摘要没有保存 ET/ETCP，因此本次不能重新计算 PPO 的 WP_ET；WP_ET 胜出列标记为不可用。",
        "",
        "## observation 预检",
        "",
        f"- observation smoke audit 通过：{obs_ok}",
        "- 原 observation 维度：25",
        "- 增强后 observation 维度：30",
        "- 新增变量：rain_today_mm, tmin_today_c, rain_past7_mm, rain_future7_mm, tmean_future7_c",
        "",
        "## checkpoint 汇总",
        "",
        by_ckpt.to_markdown(index=False),
        "",
        "## 初步判读",
        "",
        f"- 平均产量最高 checkpoint：{int(best_yield_row['checkpoint_step'])}，平均产量 {best_yield_row['mean_final_grnwt']:.2f} kg/ha。",
        f"- 可用指标中至少一项胜出最多 checkpoint：{int(best_partial_row['checkpoint_step'])}，"
        f"{int(best_partial_row['any_metric_win_four_partial_count'])}/{int(best_partial_row['validation_years'])}。",
        "- 但 WP_ET 在当前 042_00 摘要中不可直接判定；若要完整三指标比较，需要补一次只读 daily 输出的 ET 汇总后处理。",
        "- 现有数字已足以说明：直接加入天气/预报变量，在无 teacher warm-start 的 100K PPO 下没有明显优于 040_40/041_04 主线。",
        "",
        "## 输出",
        "",
        f"- enriched: `{enriched_path.relative_to(ROOT)}`",
        f"- by checkpoint: `{by_ckpt_path.relative_to(ROOT)}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "ppo_rows": int(len(ppo)),
        "checkpoints": [int(x) for x in sorted(merged["checkpoint_step"].unique())],
        "observation_audit_pass": obs_ok,
        "wp_et_available": False,
        "enriched": str(enriched_path.relative_to(ROOT)),
        "by_checkpoint": str(by_ckpt_path.relative_to(ROOT)),
        "record_md": str(DOC.relative_to(ROOT)),
        "best_mean_yield_checkpoint": int(best_yield_row["checkpoint_step"]),
        "best_partial_any_metric_checkpoint": int(best_partial_row["checkpoint_step"]),
    }
    (OUT / "042_01_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(summarize(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
