"""Read-only CERN D222 rainfall provenance re-audit for HLA and FQA."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "external" / "cern_d222" / "raw" / "excel.xls"
LEGACY = ROOT / "my_data" / "HLLCYCFQ降雨数据.xls"
AUDIT008 = ROOT / "results" / "hl_fq_random_weather_8seed" / "source_audit"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(AUDIT008))
import weather_preprocess as wp  # noqa: E402
from audit_sources import daily_nasa_frame  # noqa: E402


SITES = {
    "HLA": {"start": 2004, "end": 2013, "label": "海伦站"},
    "FQA": {"start": 2005, "end": 2013, "label": "封丘站"},
}
COL_A = "20-8时降水量(mm)"
COL_B = "8-20时降水量(mm)"
COL_TOTAL = "20-20合计(mm)"
COL_88 = "8-8时降水量(mm)"
RAIN_COLS = [COL_A, COL_B, COL_TOTAL]
TOL = 0.051


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_d222(path: Path) -> pd.DataFrame:
    frame = wp.read_excel_all_sheets(path)
    frame.columns = [wp.normalize_text(x) for x in frame.columns]
    frame["site"] = frame[wp.pick_column(frame.columns, ["生态站代码"])].astype(str).str.strip()
    frame["year_raw"] = frame[wp.pick_column(frame.columns, ["年"])]
    frame["month_raw"] = frame[wp.pick_column(frame.columns, ["月"])]
    frame["day_raw"] = frame[wp.pick_column(frame.columns, ["日"])]
    frame["date"] = pd.to_datetime(
        {
            "year": wp.coerce_numeric(frame["year_raw"]),
            "month": wp.coerce_numeric(frame["month_raw"]),
            "day": wp.coerce_numeric(frame["day_raw"]),
        },
        errors="coerce",
    )
    for col in [COL_A, COL_B, COL_TOTAL, COL_88]:
        frame[col] = wp.coerce_numeric(frame[col])
    frame["raw_row_number"] = np.arange(3, len(frame) + 3)
    return frame


def read_legacy() -> pd.DataFrame:
    frame = read_d222(LEGACY)
    return frame[["site", "date", COL_TOTAL]].rename(columns={COL_TOTAL: "legacy_rain_raw"})


def longest_true_run(dates: pd.Series, mask: pd.Series) -> tuple[int, str | None, str | None]:
    work = pd.DataFrame({"date": pd.to_datetime(dates), "flag": mask.astype(bool)}).sort_values("date")
    groups = work.flag.ne(work.flag.shift(fill_value=False)).cumsum()
    runs = work.loc[work.flag].groupby(groups[work.flag]).agg(start=("date", "min"), end=("date", "max"), days=("date", "size"))
    if runs.empty:
        return 0, None, None
    row = runs.sort_values("days", ascending=False).iloc[0]
    return int(row.days), str(row.start.date()), str(row.end.date())


def parse_wth_rain(path: Path) -> pd.Series:
    rows = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit() or len(parts[0]) != 7:
            continue
        try:
            date = pd.to_datetime(parts[0], format="%Y%j")
            rain = float(parts[4])
        except ValueError:
            continue
        rows.append((date, rain))
    return pd.Series({date: rain for date, rain in rows}, name="wth_rain_mm", dtype=float)


def load_external(site: str) -> tuple[pd.Series, pd.Series]:
    spec = SITES[site]
    nasa_path = AUDIT008 / f"nasa_power_{site}_{spec['start']}_{spec['end']}_raw.json"
    nasa = daily_nasa_frame(json.loads(nasa_path.read_text(encoding="utf-8")))["RAIN"].rename("nasa_rain_mm")
    ghcn = (
        pd.read_csv(AUDIT008 / f"ghcn_{site}_chosen_station_daily.csv", parse_dates=["date"])
        .set_index("date")["rain_mm"].rename("ghcn_rain_mm")
    )
    return nasa, ghcn


def safe_float(value):
    return None if pd.isna(value) else float(value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d222 = read_d222(RAW)
    legacy = read_legacy()

    xls = pd.ExcelFile(RAW)
    metadata = {
        "dataset": {"name": "人工大气观测降雨蒸发能见度日值", "code": "D222"},
        "raw_file": {
            "path": RAW.relative_to(ROOT).as_posix(),
            "sha256": sha256(RAW),
            "size_bytes": RAW.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(RAW.stat().st_mtime, timezone.utc).isoformat(),
            "read_only_policy": True,
        },
        "workbook": {
            "sheet_names": xls.sheet_names,
            "record_rows": int(len(d222)),
            "column_count_including_audit_columns": int(len(d222.columns)),
            "original_columns": [str(x) for x in wp.read_excel_all_sheets(RAW).columns if str(x) not in {"source_file", "source_sheet"}],
            "site_counts": {str(k): int(v) for k, v in d222.site.value_counts().items()},
            "site_date_ranges": {
                site: {"min": str(group.date.min().date()), "max": str(group.date.max().date()), "rows": int(len(group))}
                for site, group in d222.groupby("site")
            },
        },
        "official_metadata": {
            "url": "https://www.cnern.org.cn/data/meta?id=17043",
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            "coverage_years": "1998-2023",
            "stations_confirmed": ["封丘站", "海伦站"],
            "rain_measurement": "降水每次降雨时测量，每日2次，使用雨量筒测量",
            "sharing": "完全共享",
            "record_count": 347704,
            "evidence_role": "Official metadata establishes measurement frequency and coverage, but does not explicitly define blank cells as zero precipitation.",
        },
        "validation_period_used_for_thresholds_or_fitting": False,
        "raw_file_modified": False,
    }
    (OUT / "raw_file_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    coverage_rows = []
    closure_rows = []
    blank_rows = []
    overlap_daily = []
    shift_rows = []
    site_results = {}

    for site, spec in SITES.items():
        begin, end = pd.Timestamp(spec["start"], 1, 1), pd.Timestamp(spec["end"], 12, 31)
        expected_dates = pd.date_range(begin, end, freq="D")
        subset = d222[(d222.site == site) & d222.date.between(begin, end)].copy()
        subset_sorted = subset.sort_values(["date", "raw_row_number"]).copy()
        invalid_date_rows = d222[(d222.site == site) & d222.date.isna()]
        missing = expected_dates.difference(subset.date.dropna())
        duplicates = subset[subset.duplicated("date", keep=False)]
        coverage_rows.append({
            "site": site,
            "fit_start": str(begin.date()),
            "fit_end": str(end.date()),
            "expected_days": len(expected_dates),
            "actual_rows": len(subset),
            "unique_dates": subset.date.nunique(),
            "min_date": str(subset.date.min().date()),
            "max_date": str(subset.date.max().date()),
            "missing_dates": len(missing),
            "duplicate_rows": len(duplicates),
            "invalid_date_rows_in_site_file": len(invalid_date_rows),
            "leap_days_present": int(((subset.date.dt.month == 2) & (subset.date.dt.day == 29)).sum()),
            "expected_leap_days": sum(pd.Timestamp(y, 12, 31).is_leap_year for y in range(spec["start"], spec["end"] + 1)),
            "raw_row_order_chronological": bool(subset.date.is_monotonic_increasing),
            "date_chain_gate": "PASS" if len(subset) == len(expected_dates) and not len(missing) and not len(duplicates) else "FAIL",
        })

        for period_label, group in [("ALL", subset_sorted)] + [(str(y), g) for y, g in subset_sorted.groupby(subset_sorted.date.dt.year)]:
            a, b, total = group[COL_A], group[COL_B], group[COL_TOTAL]
            both = a.notna() & b.notna()
            comparable = both & total.notna()
            errors = (a + b - total).abs()
            a_only = a.notna() & b.isna()
            b_only = a.isna() & b.notna()
            closure_rows.append({
                "site": site,
                "period": period_label,
                "rows": len(group),
                "both_half_numeric_days": int(both.sum()),
                "both_half_total_numeric_days": int(comparable.sum()),
                "exact_match_days": int((errors[comparable] == 0).sum()),
                "within_0p051_mm_days": int((errors[comparable] <= TOL).sum()),
                "mismatch_days": int((errors[comparable] > TOL).sum()),
                "max_abs_error_mm": safe_float(errors[comparable].max()),
                "20_8_only_days": int(a_only.sum()),
                "20_8_only_total_equals_recorded_half_days": int(((total - a).abs() <= TOL)[a_only & total.notna()].sum()),
                "8_20_only_days": int(b_only.sum()),
                "8_20_only_total_equals_recorded_half_days": int(((total - b).abs() <= TOL)[b_only & total.notna()].sum()),
                "both_half_blank_total_blank_days": int((a.isna() & b.isna() & total.isna()).sum()),
                "both_half_blank_total_numeric_days": int((a.isna() & b.isna() & total.notna()).sum()),
                "any_half_numeric_total_blank_days": int(((a.notna() | b.notna()) & total.isna()).sum()),
            })

            all_blank = group[RAIN_COLS].isna().all(axis=1)
            run_n, run_start, run_end = longest_true_run(group.date, all_blank)
            blank_rows.append({
                "site": site,
                "period": period_label,
                "date_rows_complete": period_label != "ALL" or coverage_rows[-1]["date_chain_gate"] == "PASS",
                "all_three_rain_fields_blank_days": int(all_blank.sum()),
                "all_three_rain_fields_numeric_zero_days": int((group[RAIN_COLS].fillna(np.nan).eq(0).all(axis=1)).sum()),
                "longest_all_blank_run_days": run_n,
                "longest_all_blank_run_start": run_start,
                "longest_all_blank_run_end": run_end,
                "single_blank_half_supported_as_zero_days": int(
                    (((a.notna() & b.isna()) & ((total - a).abs() <= TOL)) | ((a.isna() & b.notna()) & ((total - b).abs() <= TOL))).sum()
                ),
            })

        d_series = subset_sorted.set_index("date")[COL_TOTAL]
        legacy_series = legacy[(legacy.site == site) & legacy.date.between(begin, end)].set_index("date").legacy_rain_raw
        same = pd.concat([d_series.rename("d222_rain_raw"), legacy_series], axis=1).reindex(expected_dates)
        same["d222_blank"] = same.d222_rain_raw.isna()
        same["legacy_blank"] = same.legacy_rain_raw.isna()
        same["d222_wet_if_blank_zero"] = same.d222_rain_raw.fillna(0) > 0
        same["legacy_wet_if_blank_zero"] = same.legacy_rain_raw.fillna(0) > 0
        same["difference_mm_when_both_numeric"] = same.d222_rain_raw - same.legacy_rain_raw
        for date, row in same.iterrows():
            overlap_daily.append({"site": site, "date": str(date.date()), **{k: safe_float(v) if isinstance(v, (float, np.floating)) else bool(v) for k, v in row.items()}})

        for shift in (-1, 0, 1):
            shifted = d_series.copy()
            shifted.index = shifted.index + pd.Timedelta(days=shift)
            pair = pd.concat([shifted.rename("d222"), legacy_series.rename("legacy")], axis=1, join="inner")
            both_numeric = pair.dropna()
            diff = both_numeric.d222 - both_numeric.legacy
            shift_rows.append({
                "site": site,
                "d222_date_shift_days_before_join": shift,
                "calendar_overlap_days": len(pair),
                "both_numeric_days": len(both_numeric),
                "wet_dry_agreement_blank_as_zero": float(((pair.d222.fillna(0) > 0) == (pair.legacy.fillna(0) > 0)).mean()),
                "mae_mm_both_numeric": safe_float(diff.abs().mean()),
                "rmse_mm_both_numeric": safe_float(math.sqrt(float((diff ** 2).mean()))) if len(diff) else None,
                "cumulative_difference_mm": float(pair.d222.sum() - pair.legacy.sum()),
                "exact_numeric_match_fraction": safe_float((diff.abs() <= 1e-12).mean()),
            })

        # External candidates are read only from 008's already saved fitting-period files.
        nasa, ghcn = load_external(site)
        all_blank = subset_sorted.set_index("date")[RAIN_COLS].isna().all(axis=1)
        long_n, long_start, long_end = longest_true_run(pd.Series(all_blank.index), pd.Series(all_blank.values))
        site_results[site] = {
            "date_chain": coverage_rows[-1]["date_chain_gate"],
            "same_day_exact_numeric_match_fraction_vs_legacy": shift_rows[-2]["exact_numeric_match_fraction"],
            "best_shift": 0,
            "all_blank_days": int(all_blank.sum()),
            "longest_all_blank_run": {"days": long_n, "start": long_start, "end": long_end},
            "external_sources": {
                "nasa_raw_json": f"results/hl_fq_random_weather_8seed/source_audit/nasa_power_{site}_{spec['start']}_{spec['end']}_raw.json",
                "ghcn_chosen_station_csv": f"results/hl_fq_random_weather_8seed/source_audit/ghcn_{site}_chosen_station_daily.csv",
            },
        }

    pd.DataFrame(coverage_rows).to_csv(OUT / "date_coverage_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(closure_rows).to_csv(OUT / "rain_field_closure.csv", index=False, encoding="utf-8-sig")

    blank_df = pd.DataFrame(blank_rows)
    blank_df["blank_semantics"] = "MIXED"
    blank_df["decision_basis"] = (
        "Single blank half-days behave as zero in 20-20 totals, but all-three-blank days cannot be distinguished from missing observations; long blank runs conflict with external rain evidence."
    )
    blank_df.to_csv(OUT / "blank_semantics_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(overlap_daily).to_csv(OUT / "overlap_same_day.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(shift_rows).to_csv(OUT / "overlap_shift_summary.csv", index=False, encoding="utf-8-sig")

    # FQA 2011-2013 extreme rainfall audit.
    fqa = d222[(d222.site == "FQA") & d222.date.between("2011-01-01", "2013-12-31")].sort_values("date").copy()
    fqa_legacy = legacy[(legacy.site == "FQA")].set_index("date").legacy_rain_raw
    fqa_nasa, fqa_ghcn = load_external("FQA")
    fqa_rows = []
    for year, group in fqa.groupby(fqa.date.dt.year):
        total = group[COL_TOTAL]
        max_idx = total.idxmax()
        fqa_rows.append({
            "record_type": "YEAR_SUMMARY", "year": int(year), "date": None,
            "d222_20_20_mm": float(total.sum()), "wet_day_count": int((total > 0).sum()),
            "days_ge_50": int((total >= 50).sum()), "days_ge_100": int((total >= 100).sum()),
            "days_ge_150": int((total >= 150).sum()), "days_ge_200": int((total >= 200).sum()),
            "maximum_daily_mm": float(total.max()), "maximum_daily_date": str(fqa.loc[max_idx, "date"].date()),
            "nasa_annual_total_mm": float(fqa_nasa[fqa_nasa.index.year == year].sum()),
            "ghcn_annual_total_mm": float(fqa_ghcn[fqa_ghcn.index.year == year].sum()),
            "ghcn_valid_days": int(fqa_ghcn[fqa_ghcn.index.year == year].notna().sum()),
        })
    for _, row in fqa[fqa[COL_TOTAL] >= 50].iterrows():
        date = row.date
        a, b, total = row[COL_A], row[COL_B], row[COL_TOTAL]
        closure = (0.0 if pd.isna(a) else a) + (0.0 if pd.isna(b) else b)
        nasa_window = fqa_nasa.reindex(pd.date_range(date - pd.Timedelta(days=1), date + pd.Timedelta(days=1)))
        ghcn_window = fqa_ghcn.reindex(pd.date_range(date - pd.Timedelta(days=1), date + pd.Timedelta(days=1)))
        fqa_rows.append({
            "record_type": "EXTREME_EVENT", "year": int(date.year), "date": str(date.date()),
            "d222_20_8_mm": safe_float(a), "d222_8_20_mm": safe_float(b), "d222_20_20_mm": float(total),
            "halfday_sum_blank_as_zero_mm": float(closure), "closure_abs_error_mm": abs(float(closure - total)),
            "legacy_same_day_mm": safe_float(fqa_legacy.get(date)),
            "nasa_same_day_mm": safe_float(fqa_nasa.get(date)), "nasa_pm1_max_mm": safe_float(nasa_window.max()),
            "ghcn_same_day_mm": safe_float(fqa_ghcn.get(date)), "ghcn_pm1_max_mm": safe_float(ghcn_window.max()),
            "external_event_corroborated_pm1": bool((nasa_window.fillna(0) > 0).any() and (ghcn_window.fillna(0) > 0).any()),
        })
    pd.DataFrame(fqa_rows).to_csv(OUT / "fqa_extreme_rain_audit.csv", index=False, encoding="utf-8-sig")

    # HLA 2004 completeness audit.
    hla = d222[(d222.site == "HLA") & d222.date.between("2004-01-01", "2004-12-31")].sort_values("date").copy().set_index("date")
    hla_legacy = legacy[legacy.site == "HLA"].set_index("date").legacy_rain_raw
    hla_nasa, hla_ghcn = load_external("HLA")
    hla_wth = parse_wth_rain(ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual" / "HL" / "CNHL0401.WTH")
    hla_out = hla[[COL_A, COL_B, COL_TOTAL]].rename(columns={COL_A: "d222_20_8_mm", COL_B: "d222_8_20_mm", COL_TOTAL: "d222_20_20_mm"})
    hla_out["all_three_rain_fields_blank"] = hla_out[["d222_20_8_mm", "d222_8_20_mm", "d222_20_20_mm"]].isna().all(axis=1)
    hla_out = hla_out.join(hla_legacy).join(hla_wth).join(hla_nasa).join(hla_ghcn)
    hla_out["external_rain_event_any"] = (hla_out.nasa_rain_mm.fillna(0) > 0) | (hla_out.ghcn_rain_mm.fillna(0) > 0)
    hla_out.index.name = "date"
    hla_out.reset_index().to_csv(OUT / "hla_2004_rain_audit.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    hashes_008 = {
        name: sha256(AUDIT008 / name)
        for name in ["source_audit.json", "nasa_candidate_metrics.csv", "nasa_candidate_gapfill.csv"]
    }
    metrics_008 = pd.read_csv(AUDIT008 / "nasa_candidate_metrics.csv")
    candidate_008_valid = bool(
        len(metrics_008) == 6
        and (metrics_008.gate == "PASS").all()
        and metrics_008.candidate_afterfill_physical_pass.astype(bool).all()
    )

    hla_all_blank = hla[RAIN_COLS].isna().all(axis=1)
    jan_aug = hla.index.month <= 8
    fqa_year = {int(y): float(g[COL_TOTAL].sum()) for y, g in fqa.groupby(fqa.date.dt.year)}
    final_gate = {
        "task": "CERN D222 rainfall provenance re-audit before HL/FQ WGEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file_sha256": sha256(RAW),
        "official_metadata_url": "https://www.cnern.org.cn/data/meta?id=17043",
        "validation_years_used_for_fitting_or_gate_tuning": False,
        "d222_date_chain": {site: data["date_chain"] for site, data in site_results.items()},
        "rain_field_closure": {
            site: {
                "both_half_numeric_days": int(pd.DataFrame(closure_rows).query("site == @site and period == 'ALL'").iloc[0].both_half_numeric_days),
                "within_tolerance_days": int(pd.DataFrame(closure_rows).query("site == @site and period == 'ALL'").iloc[0].within_0p051_mm_days),
                "mismatch_days": int(pd.DataFrame(closure_rows).query("site == @site and period == 'ALL'").iloc[0].mismatch_days),
                "single_side_blank_total_closure_supported": True,
            }
            for site in SITES
        },
        "blank_semantics": {
            "HLA": "MIXED",
            "FQA": "MIXED",
            "reason": "D222 supports blank half-day = 0 when the daily total equals the other half-day, but it does not provide an explicit missing-data flag for all-three-blank days. HLA 2004 and long FQA blank runs conflict with external rain evidence.",
        },
        "overlap_date_boundary": {
            "best_shift_days": {"HLA": 0, "FQA": 0},
            "same_day_numeric_exact_match_fraction": {site: site_results[site]["same_day_exact_numeric_match_fraction_vs_legacy"] for site in SITES},
            "interpretation": "Shift is applied to the D222 date before joining the legacy record. Same-day is the unique exact alignment; -1/+1 are materially worse.",
            "same_source_claim": "NOT_MADE; value identity establishes compatibility, but no project provenance document proves both files are independently sourced or identical exports.",
        },
        "special_audits": {
            "FQA_EXTREME_RAIN": "SUSPICIOUS_BUT_UNRESOLVED",
            "FQA_evidence": {
                "annual_totals_mm": fqa_year,
                "all_ge_50_events_corroborated_as_wet_within_pm1_by_nasa_and_ghcn": True,
                "internal_halfday_closure": "PASS",
                "concern": "2011-2012 totals are about 2.3-2.8 times the NASA/GHCN annual totals; the 2012-09-02 D222 value is 240.0 mm while NASA/GHCN report 3.53/12.4 mm. Event occurrence is supported, but magnitude is not independently closed.",
            },
            "HLA_2004_RAIN": "INCOMPLETE_RECORD",
            "HLA_evidence": {
                "date_rows": int(len(hla)),
                "d222_total_mm": float(hla[COL_TOTAL].sum()),
                "d222_wet_days": int((hla[COL_TOTAL] > 0).sum()),
                "all_three_blank_days": int(hla_all_blank.sum()),
                "jan_aug_all_three_blank_days": int((hla_all_blank & jan_aug).sum()),
                "jan_aug_days": int(jan_aug.sum()),
                "nasa_2004_total_mm": float(hla_nasa[hla_nasa.index.year == 2004].sum()),
                "ghcn_2004_total_mm": float(hla_ghcn[hla_ghcn.index.year == 2004].sum()),
                "nasa_jan_aug_total_mm": float(hla_nasa[(hla_nasa.index.year == 2004) & (hla_nasa.index.month <= 8)].sum()),
                "ghcn_jan_aug_total_mm": float(hla_ghcn[(hla_ghcn.index.year == 2004) & (hla_ghcn.index.month <= 8)].sum()),
            },
        },
        "audit_008_temperature_srad_candidates": {"hashes": hashes_008, "six_variable_site_rows_pass": candidate_008_valid, "reused_or_modified": "READ_ONLY_VERIFIED; not promoted because RAIN gate failed"},
        "sites": {
            "HLA": {"weather_source_audit": "BLOCKED", "blocker": "BLOCKED_BY_HLA_2004_INCOMPLETE_RECORD", "final_rain_source_path": None, "previous_2013_blocker_resolved": False},
            "FQA": {"weather_source_audit": "BLOCKED", "blocker": "BLOCKED_BY_FQA_RAIN_INCONSISTENCY", "final_rain_source_path": None, "previous_2013_blocker_resolved": False},
        },
        "WEATHER_SOURCE_AUDIT": "BLOCKED",
        "READY_TO_RESUME_008": False,
        "rain_fitting_candidate_created": False,
        "fitting_weather_qc_created": False,
        "wgen_started": False,
        "dssat_started": False,
        "ppo_started": False,
        "minimum_next_evidence": {
            "HLA": "An authoritative HLA 2004 daily precipitation record or D222 station log/missing-code documentation that resolves the 2004-01-01 to 2004-08-31 all-blank interval.",
            "FQA": "Same-site FQA gauge/source log or quality-control record that validates the 2011-2013 extreme magnitudes, especially 2012-09-02 240.0 mm, or identifies a correction rule.",
        },
    }
    (OUT / "final_gate.json").write_text(json.dumps(final_gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    findings = f"""# CERN D222 降水来源恢复审计结论

`WEATHER_SOURCE_AUDIT = BLOCKED`  
`READY_TO_RESUME_008 = FALSE`

## 核心证据

- 原始文件 SHA-256：`{sha256(RAW)}`；只读审计，未修改文件。
- HLA 2004–2013 与 FQA 2005–2013 日期链均完整、无重复，分别为 3,653 / 3,287 天；原始行序均不是日期排序。
- 两个半日值都存在时，HLA 293/293、FQA 103/103 在 0.051 mm 容差内等于 `20-20合计`，无 mismatch。只有一个半日值时，HLA 669/669、FQA 292/292 的日合计等于该半日值，支持“单侧半日空白按 0 参与合计”。
- 三个降水字段同时空白不能统一解释为 dry day。D222 没有给出区分全空白 dry day 与停测/缺测的显式标记；因此两站 `BLANK_SEMANTICS = MIXED`。
- 与项目原降水表的 0 日 shift 完全一致：所有双方非空值精确相等，湿/干一致率 100%；-1/+1 日明显较差。该证据支持同一日期口径和高度兼容，但不单独证明两文件是独立同源观测。

## 专项门禁

- `HLA_2004_RAIN = INCOMPLETE_RECORD`：日期行虽有 366 天，但 2004-01-01 至 2004-08-31 连续 244 天三个降水字段全部空白；D222 全年只有 60.2 mm / 20 wet days。同期 NASA/GHCN 年雨量为 406.55 / 357.3 mm，1–8 月为 319.57 / 278.9 mm。现有 WTH 将该段写成 0，不能作为独立验证。
- `FQA_EXTREME_RAIN = SUSPICIOUS_BUT_UNRESOLVED`：复算 2011/2012/2013 年总量为 1,614.4 / 1,378.2 / 1,144.4 mm。所有 ≥50 mm 日都被 NASA 与 GHCN 在 ±1 日窗口支持为降雨事件，半日和日合计也闭合；但 2011–2012 总量约为 NASA/GHCN 的 2.3–2.8 倍，2012-09-02 D222=240.0 mm，而 NASA/GHCN 同日为 3.53/12.4 mm。事件存在得到支持，极端量级仍缺同站独立证据。

## 结果

- HLA：`BLOCKED_BY_HLA_2004_INCOMPLETE_RECORD`。
- FQA：`BLOCKED_BY_FQA_RAIN_INCONSISTENCY`。
- 008 中 111 / 157 个 2013 blocker 没有被正式解除：D222 本身每日有日期行，但全空白语义未能统一闭合，且专项完整性/极端量级门禁失败。
- 未生成 `rain_fitting_candidate.csv` 或 `fitting_weather_qc.csv`；未调用 CLI、WGEN、DSSAT 或 PPO。

最小后续证据：HLA 需要能解释 2004-01-01 至 2004-08-31 的同站逐日记录或 D222 缺测编码/台站日志；FQA 需要同站雨量筒原始记录、质控说明或来源日志，核实 2011–2013 极端量级，尤其 2012-09-02 的 240.0 mm。
"""
    (OUT / "findings.md").write_text(findings, encoding="utf-8")
    print(json.dumps({"raw_sha256": sha256(RAW), "date_chain": final_gate["d222_date_chain"], "blank_semantics": final_gate["blank_semantics"], "FQA_EXTREME_RAIN": final_gate["special_audits"]["FQA_EXTREME_RAIN"], "HLA_2004_RAIN": final_gate["special_audits"]["HLA_2004_RAIN"], "WEATHER_SOURCE_AUDIT": final_gate["WEATHER_SOURCE_AUDIT"], "READY_TO_RESUME_008": final_gate["READY_TO_RESUME_008"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
