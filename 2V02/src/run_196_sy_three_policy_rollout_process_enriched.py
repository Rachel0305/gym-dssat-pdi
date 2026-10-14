"""196: enrich task 195 rollout-process tables with argmax and weather context.

This is a read-only post-process of task 195.  It does not train PPO and does
not rerun DSSAT episodes.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "196"
TASK_NAME = "sy_three_policy_rollout_process_enriched"
INPUT_DIR = ROOT / "benchmark_results/195_sy_three_policy_rollout_process_full"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}_retry1"

ACTION_IRRIGATION = [0.0, 15.0, 30.0, 45.0]
ACTION_NITROGEN = [0.0, 40.0, 80.0, 120.0]
BASE_OBS_NAMES = [
    "cumsumfert",
    "dap",
    "dtt",
    "ep",
    "grnwt",
    "istage",
    "nstres",
    "rtdep",
    "srad",
    "sw",
    "swfac",
    "tmax",
    "topwt",
    "totir",
    "vstage",
    "wtdep",
    "xlai",
]

SCENARIO_RUNTIME_YEARS_191 = {
    "original": 2005,
    "early_dry": 2006,
    "mid_dry": 2007,
    "late_dry": 2008,
    "early_wet": 2009,
    "mid_wet": 2010,
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    return list(ast.literal_eval(text))


def parse_mask(value: Any) -> list[bool]:
    text = str(value).strip()
    if text.startswith("["):
        return [bool(item) for item in ast.literal_eval(text)]
    return [char == "1" for char in text]


def action_label(action_index: int) -> str:
    irrigation = ACTION_IRRIGATION[int(action_index) // 4]
    nitrogen = ACTION_NITROGEN[int(action_index) % 4]
    return f"I{int(irrigation)}/N{int(nitrogen)}"


def action_irrigation(action_index: int) -> float:
    return float(ACTION_IRRIGATION[int(action_index) // 4])


def action_nitrogen(action_index: int) -> float:
    return float(ACTION_NITROGEN[int(action_index) % 4])


def legal_argmax(row: pd.Series) -> dict[str, Any]:
    probabilities = np.asarray(parse_list(row["all_action_probabilities_before"]), dtype=float)
    mask = np.asarray(parse_mask(row["action_mask"]), dtype=bool)
    legal_indices = np.flatnonzero(mask)
    if legal_indices.size == 0:
        return {
            "legal_argmax_index": np.nan,
            "legal_argmax_action": "",
            "legal_argmax_probability": np.nan,
            "legal_argmax_irrigation": np.nan,
            "legal_argmax_nitrogen": np.nan,
        }
    best = int(legal_indices[np.nanargmax(probabilities[legal_indices])])
    return {
        "legal_argmax_index": best,
        "legal_argmax_action": action_label(best),
        "legal_argmax_probability": float(probabilities[best]),
        "legal_argmax_irrigation": action_irrigation(best),
        "legal_argmax_nitrogen": action_nitrogen(best),
    }


def policy_entropy(row: pd.Series) -> float:
    probabilities = np.asarray(parse_list(row["all_action_probabilities_before"]), dtype=float)
    valid = probabilities[np.isfinite(probabilities) & (probabilities > 0.0)]
    if valid.size == 0:
        return np.nan
    return float(-(valid * np.log(valid)).sum())


def selected_action_rank(row: pd.Series) -> float:
    probabilities = np.asarray(parse_list(row["all_action_probabilities_before"]), dtype=float)
    mask = np.asarray(parse_mask(row["action_mask"]), dtype=bool)
    action = int(row["action_index"])
    if action >= len(probabilities) or action >= len(mask) or not mask[action]:
        return np.nan
    legal_probabilities = probabilities[mask]
    return float(1 + np.sum(legal_probabilities > probabilities[action] + 1e-12))


def observation_value(row: pd.Series, name: str) -> float:
    values = parse_list(row["observation_vector"])
    if name not in BASE_OBS_NAMES:
        return np.nan
    index = BASE_OBS_NAMES.index(name)
    if index >= len(values):
        return np.nan
    try:
        return float(values[index])
    except Exception:
        return np.nan


def parse_wth(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*(\d{7})\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        yrdoy = int(match.group(1))
        year = yrdoy // 1000
        doy = yrdoy % 1000
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        rows.append(
            {
                "date": date,
                "year": year,
                "doy": doy,
                "srad": float(match.group(2)),
                "tmax": float(match.group(3)),
                "tmin": float(match.group(4)),
                "rain": float(match.group(5)),
            }
        )
    if not rows:
        raise ValueError(f"No daily WTH rows parsed from {path}")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def rain_windows(weather: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    def rain_sum(days: int) -> float:
        start = date + pd.Timedelta(days=1)
        end = date + pd.Timedelta(days=days)
        subset = weather[(weather["date"] >= start) & (weather["date"] <= end)]
        return float(pd.to_numeric(subset["rain"], errors="coerce").fillna(0.0).sum())

    today = weather[weather["date"].eq(date)]
    return {
        "rain_today": float(pd.to_numeric(today["rain"], errors="coerce").fillna(0.0).sum()) if len(today) else np.nan,
        "rain_next1": rain_sum(1),
        "rain_next3": rain_sum(3),
        "rain_next7": rain_sum(7),
    }


def build_weather_context() -> tuple[dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], pd.Timestamp], dict[tuple[str, str], int]]:
    weather: dict[tuple[str, str], pd.DataFrame] = {}
    planting: dict[tuple[str, str], pd.Timestamp] = {}
    runtime_year: dict[tuple[str, str], int] = {}

    cfg046 = json.loads((ROOT / "configs/046_10_sya_originIC_expanded_action_maskableppo.json").read_text(encoding="utf-8"))
    observed = cfg046.get("scope", {}).get("validation_years", [])
    for year in observed:
        key = ("SY_no_forecast_04610_ckpt100000", str(int(year)))
        wth = ROOT / "Leave_One_experiments" / "wth_generated_qc" / "SYA" / f"SYA{int(year)}.WTH"
        weather[key] = parse_wth(wth)
        planting[key] = pd.Timestamp(year=int(year), month=4, day=21 if int(year) != 2008 else 20)
        runtime_year[key] = int(year)

    cfg143 = json.loads((ROOT / "configs/143E2_sya_originIC_dual_branch_weather_reader_5k.json").read_text(encoding="utf-8"))
    for year in cfg143.get("scope", {}).get("validation_years", []):
        key = ("SY_weather_E2_143E2_e2_s0_ckpt5000", str(int(year)))
        wth = ROOT / "Leave_One_experiments" / "wth_generated_qc" / "SYA" / f"SYA{int(year)}.WTH"
        weather[key] = parse_wth(wth)
        planting[key] = pd.Timestamp(year=int(year), month=4, day=21 if int(year) != 2008 else 20)
        runtime_year[key] = int(year)

    manifest = pd.read_csv(ROOT / "benchmark_results/166_sya_weather_scenario_bank_v2_fixed_format/163_sya_weather_scenario_manifest.csv")
    manifest = manifest[manifest["station_code"].astype(str).eq("SYA")].copy()
    for _, row in manifest.iterrows():
        variant = str(row["variant"])
        if variant not in SCENARIO_RUNTIME_YEARS_191:
            continue
        key = ("SY_one_year_six_weather_191_ckpt100000", variant)
        weather_path = ROOT / str(row["target_weather"])
        weather[key] = parse_wth(weather_path)
        planting[key] = pd.Timestamp(year=2005, month=4, day=21)
        runtime_year[key] = SCENARIO_RUNTIME_YEARS_191[variant]

    return weather, planting, runtime_year


def scenario_lookup_key(row: pd.Series) -> tuple[str, str]:
    model_id = str(row["model_id"])
    if model_id == "SY_one_year_six_weather_191_ckpt100000":
        return model_id, str(row["weather_variant"])
    return model_id, str(row["scenario_key"])


def enrich() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}")
    (OUT / "evaluation").mkdir(parents=True, exist_ok=False)

    source = INPUT_DIR / "evaluation/rollout_transitions_all_models.csv"
    frame = pd.read_csv(source)
    weather, planting, runtime_year = build_weather_context()

    argmax_rows = frame.apply(legal_argmax, axis=1, result_type="expand")
    out = pd.concat([frame, argmax_rows], axis=1)
    out["policy_entropy"] = frame.apply(policy_entropy, axis=1)
    out["selected_action_probability_rank_legal"] = frame.apply(selected_action_rank, axis=1)
    out["soil_water"] = frame.apply(lambda row: observation_value(row, "sw"), axis=1)
    out["swfac"] = frame.apply(lambda row: observation_value(row, "swfac"), axis=1)
    out["nstres"] = frame.apply(lambda row: observation_value(row, "nstres"), axis=1)
    out["srad_obs"] = frame.apply(lambda row: observation_value(row, "srad"), axis=1)
    out["tmax_obs"] = frame.apply(lambda row: observation_value(row, "tmax"), axis=1)

    dates: list[str] = []
    runtime_years: list[int] = []
    rain_today: list[float] = []
    rain_next1: list[float] = []
    rain_next3: list[float] = []
    rain_next7: list[float] = []
    for _, row in out.iterrows():
        key = scenario_lookup_key(row)
        dap = float(row["dap"])
        base_date = planting[key]
        date = base_date + pd.Timedelta(days=max(int(round(dap)) - 1, 0))
        rain = rain_windows(weather[key], date)
        dates.append(date.strftime("%Y-%m-%d"))
        runtime_years.append(int(runtime_year[key]))
        rain_today.append(rain["rain_today"])
        rain_next1.append(rain["rain_next1"])
        rain_next3.append(rain["rain_next3"])
        rain_next7.append(rain["rain_next7"])

    insert_after = list(out.columns).index("dap") + 1
    out.insert(insert_after, "date", dates)
    out.insert(insert_after + 1, "runtime_year", runtime_years)
    out["rain_today"] = rain_today
    out["rain_next1"] = rain_next1
    out["rain_next3"] = rain_next3
    out["rain_next7"] = rain_next7

    all_path = OUT / "evaluation/rollout_transitions_all_models_enriched.csv"
    legal_path = OUT / "evaluation/legal_decision_rows_enriched.csv"
    summary_path = OUT / "evaluation/legal_argmax_summary.csv"
    out.to_csv(all_path, index=False, encoding="utf-8-sig")
    legal = out[pd.to_numeric(out["legal_count"], errors="coerce").ge(2)].copy()
    legal.to_csv(legal_path, index=False, encoding="utf-8-sig")
    summary = (
        legal.groupby(["model_id", "scenario_key", "weather_variant", "legal_argmax_action"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["model_id", "scenario_key", "weather_variant", "legal_argmax_action"])
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    notes = OUT / "field_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# 196 enriched rollout field notes",
                "",
                "- This task is a read-only post-process of task 195; it does not train PPO and does not rerun DSSAT episodes.",
                "- `legal_argmax_*` is computed from `all_action_probabilities_before` after applying `action_mask`.",
                "- `policy_entropy` is computed from the full 16-action probability vector.",
                "- `soil_water` is the raw observation variable `sw` at index 9. It is not DSSAT `SoilWat.OUT` `SWTD(mm)`.",
                "- `swfac`, `nstres`, `srad_obs`, and `tmax_obs` are also decoded from the base observation vector.",
                "- `rain_today`, `rain_next1`, `rain_next3`, and `rain_next7` are computed from the corresponding WTH/weather source by date. The windows exclude the current day and sum date+1 through date+N.",
                "- For 191, `runtime_year` records the pseudo-year used by the six weather variants; `year` remains the source-year label inherited from task 195.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "training_executed": False,
        "dssat_rollout_executed": False,
        "source_table": rel(source),
        "outputs": {
            "rollout_transitions_all_models_enriched": rel(all_path),
            "legal_decision_rows_enriched": rel(legal_path),
            "legal_argmax_summary": rel(summary_path),
            "field_notes": rel(notes),
        },
        "rows": int(len(out)),
        "legal_rows": int(len(legal)),
        "columns": list(out.columns),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(enrich(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
