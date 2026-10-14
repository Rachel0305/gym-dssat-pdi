"""Read-only HL/FQ training-weather source audit and NASA POWER candidate screen.

Only fitting-period source values are read. External data are retained as a
candidate source and are never allowed to replace a valid station value.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import weather_preprocess as wp  # noqa: E402


SITES = {
    "HLA": {"short": "HL", "start": 2004, "end": 2013, "lat": 47.450, "lon": 126.900},
    "FQA": {"short": "FQ", "start": 2005, "end": 2013, "lat": 35.020, "lon": 114.540},
}
PARAMETERS = {
    "SRAD": "ALLSKY_SFC_SW_DWN",
    "TMAX": "T2M_MAX",
    "TMIN": "T2M_MIN",
    "RAIN": "PRECTOTCORR",
}
FILL_VALUE = -999.0
RAIN_WET_THRESHOLD_MM = 0.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def read_excel_source(path: Path, fields: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    raw = wp.read_excel_all_sheets(path)
    raw.columns = [wp.normalize_text(c) for c in raw.columns]
    station_col = wp.pick_column(raw.columns, ["生态站代码"])
    year_col = wp.pick_column(raw.columns, ["年"])
    month_col = wp.pick_column(raw.columns, ["月"])
    day_col = wp.pick_column(raw.columns, ["日"])
    dates = pd.to_datetime(
        {
            "year": wp.coerce_numeric(raw[year_col]),
            "month": wp.coerce_numeric(raw[month_col]),
            "day": wp.coerce_numeric(raw[day_col]),
        },
        errors="coerce",
    )
    values = {
        name: wp.coerce_numeric(raw[wp.pick_column(raw.columns, keywords, all_keywords=len(keywords) > 1)])
        for name, keywords in fields.items()
    }
    frame = pd.DataFrame({"site": raw[station_col].astype(str).str.strip(), "date": dates, **values})
    frame = frame[frame.site.isin(SITES) & frame.date.notna()].copy()
    frame = frame.drop_duplicates(["site", "date"], keep="last").reset_index(drop=True)
    meta = {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_bytes(path.read_bytes()),
        "size_bytes": path.stat().st_size,
        "columns": list(fields),
        "rows_after_date_dedup": len(frame),
        "valid_dates": int(frame.date.notna().sum()),
        "duplicate_dates_after_dedup": int(frame.duplicated(["site", "date"]).sum()),
    }
    return frame, meta


def consecutive_true_runs(mask: pd.Series) -> tuple[int, str | None, str | None]:
    values = mask.fillna(False).astype(bool).tolist()
    dates = pd.Series(mask.index)
    best_len = current_len = 0
    best_start = best_end = current_start = None
    for i, value in enumerate(values):
        if value:
            if current_len == 0:
                current_start = i
            current_len += 1
            if current_len > best_len:
                best_len, best_start, best_end = current_len, current_start, i
        else:
            current_len = 0
    if best_start is None:
        return 0, None, None
    idx = pd.to_datetime(dates)
    return best_len, str(idx.iloc[best_start].date()), str(idx.iloc[best_end].date())


def daily_nasa_frame(payload: dict) -> pd.DataFrame:
    values = payload["properties"]["parameter"]
    frame = pd.DataFrame(values)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    frame = frame.rename(
        columns={
            "ALLSKY_SFC_SW_DWN": "SRAD",
            "T2M_MAX": "TMAX",
            "T2M_MIN": "TMIN",
            "PRECTOTCORR": "RAIN",
        }
    )
    frame = frame.replace(FILL_VALUE, pd.NA).apply(pd.to_numeric, errors="coerce")
    return frame


def safe_metrics(observed: pd.Series, candidate: pd.Series) -> dict:
    pairs = pd.concat([observed.rename("observed"), candidate.rename("candidate")], axis=1).dropna()
    if pairs.empty:
        return {"overlap_n": 0, "bias_candidate_minus_station": None, "corrected_mae": None, "corrected_rmse": None, "pearson_r": None}
    correction = float((pairs.observed - pairs.candidate).mean())
    corrected = pairs.candidate + correction
    residual = corrected - pairs.observed
    corr = pairs.observed.corr(corrected) if len(pairs) > 1 else math.nan
    return {
        "overlap_n": int(len(pairs)),
        "bias_candidate_minus_station": float((pairs.candidate - pairs.observed).mean()),
        "bias_correction_station_minus_candidate": correction,
        "corrected_mae": float(residual.abs().mean()),
        "corrected_rmse": float((residual.pow(2).mean()) ** 0.5),
        "pearson_r": None if pd.isna(corr) else float(corr),
        "correction_formula": "candidate + mean(station - candidate) over valid fitting-period overlap",
    }


def load_local_weather() -> tuple[pd.DataFrame, dict]:
    srad, m_srad = read_excel_source(ROOT / "my_data" / "D32.xls", {"SRAD": ["总辐射总量"]})
    temp, m_temp = read_excel_source(ROOT / "my_data" / "T2.xls", {"TMAX": ["日最大值"], "TMIN": ["日最小值"]})
    rain, m_rain = read_excel_source(ROOT / "my_data" / "HLLCYCFQ降雨数据.xls", {"RAIN": ["20-20", "合计"]})
    weather = srad.merge(temp, on=["site", "date"], how="outer", validate="one_to_one")
    weather = weather.merge(rain, on=["site", "date"], how="outer", validate="one_to_one")
    return weather.sort_values(["site", "date"]).reset_index(drop=True), {"D32.xls": m_srad, "T2.xls": m_temp, "HLLCYCFQ降雨数据.xls": m_rain}


def read_weather_header(site: str) -> dict:
    short = SITES[site]["short"]
    path = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual" / short / f"CN{short}0701.WTH"
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    header = next((line.split() for line in lines if line.strip().startswith(f"CN{short} ")), [])
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_bytes(path.read_bytes()),
        "station_code": header[0] if header else None,
        "latitude": float(header[1]) if len(header) > 1 else None,
        "longitude": float(header[2]) if len(header) > 2 else None,
        "elevation_m": float(header[3]) if len(header) > 3 else None,
    }


def screen_site(site: str, local: pd.DataFrame, source_files: dict) -> tuple[dict, list[dict], dict]:
    spec = SITES[site]
    start, end = spec["start"], spec["end"]
    begin, finish = pd.Timestamp(start, 1, 1), pd.Timestamp(end, 12, 31)
    obs = local[(local.site == site) & local.date.between(begin, finish)].set_index("date").sort_index()
    expected = int((finish - begin).days + 1)
    daily = pd.date_range(begin, finish, freq="D")
    obs = obs.reindex(daily)
    obs.index.name = "date"

    source_weather = {
        "files": source_files,
        "site": site,
        "fit_period": {"start": str(begin.date()), "end": str(finish.date()), "expected_days": expected},
        "weather_header": read_weather_header(site),
        "raw_variables": {},
    }
    rain = obs["RAIN"]
    longest_blank, blank_start, blank_end = consecutive_true_runs(rain.isna())
    source_weather["raw_rain_blank_convention"] = {
        "project_rule": "Treat blank daily rain as 0 mm, as stated in docs/experiment_records/weather_data_check_report.md.",
        "raw_blank_days": int(rain.isna().sum()),
        "explicit_numeric_days": int(rain.notna().sum()),
        "longest_consecutive_blank_run_days": longest_blank,
        "longest_blank_run_start": blank_start,
        "longest_blank_run_end": blank_end,
        "note": "This interpretation remains a provenance assumption and is independently screened against NASA POWER precipitation; NASA rain is not used to fill values in this audit.",
    }

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(PARAMETERS.values()),
        "community": "AG",
        "longitude": spec["lon"],
        "latitude": spec["lat"],
        "start": f"{start}0101",
        "end": f"{end}1231",
        "format": "JSON",
        "time-standard": "LST",
    }
    raw_path = OUT / f"nasa_power_{site}_{start}_{end}_raw.json"
    request_url = requests.Request("GET", url, params=params).prepare().url
    reused_response = raw_path.exists()
    if reused_response:
        response_bytes = raw_path.read_bytes()
        payload = json.loads(response_bytes)
        http_status = 200  # The file was written only after the earlier request passed raise_for_status().
    else:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        response_bytes = response.content
        payload = response.json()
        raw_path.write_bytes(response_bytes)
        http_status = response.status_code
    nasa = daily_nasa_frame(payload).reindex(daily)
    source_weather["nasa_power"] = {
        "request_url": request_url,
        "http_status": http_status,
        "response_path": raw_path.relative_to(ROOT).as_posix(),
        "response_sha256": sha256_bytes(response_bytes),
        "reused_prior_response": reused_response,
        "query_parameters": params,
        "api": payload.get("header", {}).get("api"),
        "sources": payload.get("header", {}).get("sources"),
        "time_standard": payload.get("header", {}).get("time_standard"),
        "units": {k: v.get("units") for k, v in payload.get("parameters", {}).items()},
        "fill_value": payload.get("header", {}).get("fill_value"),
        "returned_days": len(nasa),
        "valid_days_by_variable": {var: int(nasa[var].notna().sum()) for var in PARAMETERS},
    }

    gate_metrics: dict[str, dict] = {}
    gapfill_rows: list[dict] = []
    physical_invalid_masks = {
        "SRAD": (obs.SRAD.lt(0) | obs.SRAD.gt(60)),
        "TMAX": (obs.TMAX.lt(-60) | obs.TMAX.gt(50)),
        "TMIN": (obs.TMIN.lt(-60) | obs.TMIN.gt(50)),
    }
    limits = {"TMAX": {"mae_max": 3.0, "r_min": 0.85}, "TMIN": {"mae_max": 3.0, "r_min": 0.85}, "SRAD": {"mae_max": 4.0, "r_min": 0.70}}
    for var in ("TMAX", "TMIN", "SRAD"):
        source = obs[var]
        ext = nasa[var]
        valid = source.notna() & ext.notna() & ~physical_invalid_masks[var]
        metrics = safe_metrics(source[valid], ext[valid])
        data_coverage = int(ext.notna().sum()) == expected
        gate = data_coverage and metrics["overlap_n"] >= 3000 and metrics["corrected_mae"] is not None and metrics["corrected_mae"] <= limits[var]["mae_max"] and metrics["pearson_r"] is not None and metrics["pearson_r"] >= limits[var]["r_min"]
        invalid = physical_invalid_masks[var]
        fill_mask = (source.isna() | invalid) & ext.notna()
        corrected_fill = ext[fill_mask] + (metrics.get("bias_correction_station_minus_candidate") or 0.0)
        gapfill_rows.extend(
            {
                "site": site,
                "date": str(date.date()),
                "variable": var,
                "original_station_value": None if pd.isna(source.loc[date]) else float(source.loc[date]),
                "reason": "raw_missing" if pd.isna(source.loc[date]) else "physical_qc_invalid",
                "nasa_raw_value": float(ext.loc[date]),
                "bias_correction_station_minus_nasa": metrics.get("bias_correction_station_minus_candidate"),
                "candidate_value": float(corrected_fill.loc[date]),
                "candidate_source": "NASA_POWER_training_overlap_mean_bias_corrected",
                "source_gate_pass": bool(gate),
                "promoted_to_fitting_weather": False,
            }
            for date in source.index[fill_mask]
        )
        physical_after = corrected_fill.dropna()
        if var == "SRAD":
            physical_candidate = bool((physical_after >= 0).all())
        else:
            physical_candidate = bool((physical_after >= -60).all() and (physical_after <= 50).all())
        source_weather["raw_variables"][var] = {
            "raw_missing_days": int(source.isna().sum()),
            "raw_observed_days": int(source.notna().sum()),
            "raw_min": None if source.notna().sum() == 0 else float(source.min()),
            "raw_max": None if source.notna().sum() == 0 else float(source.max()),
            "raw_physical_invalid_days": int(invalid.sum()),
            "nasa_overlap_metrics": metrics,
            "nasa_gate_limits": limits[var],
            "nasa_candidate_gate": "PASS" if gate else "FAIL",
            "candidate_fill_or_replacement_rows": int(fill_mask.sum()),
            "candidate_afterfill_physical_values_pass": physical_candidate,
            "method": "Fill only missing or physical-QC-invalid station values; retain every other valid station value unchanged.",
        }
        gate_metrics[var] = {"gate": "PASS" if gate else "FAIL", **metrics, "fill_rows": int(fill_mask.sum()), "candidate_afterfill_physical_pass": physical_candidate}

    # External precipitation is only a source screen. No NASA rain is accepted here.
    station_rain = rain.fillna(0.0)
    external_rain = nasa["RAIN"]
    rain_pairs = pd.concat([station_rain.rename("station"), external_rain.rename("external")], axis=1).dropna()
    agreement = float(((rain_pairs.station > RAIN_WET_THRESHOLD_MM) == (rain_pairs.external > RAIN_WET_THRESHOLD_MM)).mean()) if not rain_pairs.empty else None
    station_total = float(station_rain.sum())
    external_total = float(external_rain.sum()) if external_rain.notna().all() else None
    rain_pass = agreement is not None and agreement >= 0.80
    source_weather["raw_variables"]["RAIN"] = {
        "raw_missing_values": int(rain.isna().sum()),
        "project_assigned_zero_days": int(rain.isna().sum()),
        "station_total_with_blank_as_zero_mm": station_total,
        "nasa_precipitation_screen": {
            "overlap_n": int(len(rain_pairs)),
            "wet_day_threshold_mm": RAIN_WET_THRESHOLD_MM,
            "event_agreement": agreement,
            "required_event_agreement": 0.80,
            "station_training_total_mm": station_total,
            "nasa_training_total_mm": external_total,
            "nasa_to_station_total_ratio": None if external_total in (None, 0) else external_total / station_total,
            "gate": "PASS" if rain_pass else "FAIL",
            "nasa_rain_used_for_filling": False,
        },
        "status": "blank_as_zero_rule_supported_by_external_screen" if rain_pass else "blank_as_zero_rule_not_independently_supported",
    }
    source_weather["site_status"] = "RECONSTRUCT_REQUIRED" if all(g["gate"] == "PASS" and g["candidate_afterfill_physical_pass"] for g in gate_metrics.values()) and rain_pass else "BLOCKED"
    source_weather["stop_before_wgen"] = source_weather["site_status"] == "BLOCKED"
    source_weather["wgen_started"] = False
    source_weather["cli_generated"] = False
    source_weather["ppo_started"] = False
    return source_weather, gapfill_rows, gate_metrics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weather, source_files = load_local_weather()
    payloads = {}
    all_gapfills = []
    all_metrics = {}
    for site in SITES:
        result, gapfills, metrics = screen_site(site, weather, source_files)
        payloads[site] = result
        all_gapfills.extend(gapfills)
        all_metrics[site] = metrics
    audit = {
        "task": "HL/FQ WGEN random-weather 8-seed data-source audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_audit_status": {site: payloads[site]["site_status"] for site in payloads},
        "sites": payloads,
        "validation_years_read_or_used": False,
        "original_workbooks_modified": False,
        "historical_wth_modified": False,
        "fitting_weather_materialized": False,
        "cli_generated": False,
        "wgen_started": False,
        "ppo_started": False,
        "decision": "STOP_BEFORE_WGEN" if any(item["stop_before_wgen"] for item in payloads.values()) else "RECONSTRUCT_TRAINING_WEATHER_CANDIDATES_NEXT",
    }
    (OUT / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(all_gapfills).to_csv(OUT / "nasa_candidate_gapfill.csv", index=False, encoding="utf-8-sig")
    rows = []
    for site, variables in all_metrics.items():
        for variable, values in variables.items():
            rows.append({"site": site, "variable": variable, **values})
    pd.DataFrame(rows).to_csv(OUT / "nasa_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"source_audit_status": audit["source_audit_status"], "decision": audit["decision"], "nasa_candidate_metrics": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
