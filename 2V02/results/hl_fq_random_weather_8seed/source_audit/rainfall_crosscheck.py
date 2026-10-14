"""Quantify how nearby GHCN rain candidates compare with recorded site rain."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from audit_sources import OUT, ROOT, SITES, read_excel_source


def longest_missing_run(series: pd.Series) -> tuple[int, str | None, str | None]:
    missing = series.isna()
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    runs = missing.groupby(groups).sum()
    if runs.empty or int(runs.max()) == 0:
        return 0, None, None
    group_id = runs.idxmax()
    dates = series.index[(groups == group_id) & missing]
    return int(runs.max()), str(dates[0].date()), str(dates[-1].date())


def main() -> None:
    local_raw, _ = read_excel_source(
        ROOT / "my_data" / "HLLCYCFQ降雨数据.xls", {"RAIN": ["20-20", "合计"]}
    )
    audit = json.loads((OUT / "ghcn_rain_station_audit.json").read_text(encoding="utf-8"))
    output = {"task": "Local rainfall semantics and GHCN candidate cross-check", "sites": {}}
    annual_rows: list[dict] = []
    for site, spec in SITES.items():
        begin = pd.Timestamp(spec["start"], 1, 1)
        finish = pd.Timestamp(spec["end"], 12, 31)
        days = pd.date_range(begin, finish, freq="D")
        local = (
            local_raw[(local_raw.site == site) & local_raw.date.between(begin, finish)]
            .set_index("date")["RAIN"].reindex(days)
        )
        selected = audit["sites"][site]["selected_station"]
        external = (
            pd.read_csv(OUT / f"ghcn_{site}_chosen_station_daily.csv")
            .assign(date=lambda x: pd.to_datetime(x.date))
            .set_index("date")["rain_mm"].reindex(days)
        )
        paired_recorded = local.notna() & external.notna()
        local_wet, external_wet = local.fillna(0.0) > 0.0, external > 0.0
        valid_external = external.notna()
        agreement_full = float((local_wet[valid_external] == external_wet[valid_external]).mean())
        agreement_recorded = float((local_wet[paired_recorded] == external_wet[paired_recorded]).mean())
        run_n, run_start, run_end = longest_missing_run(local)
        composite = local.combine_first(external)
        for year in range(spec["start"], spec["end"] + 1):
            mask = days.year == year
            local_total = float(local.loc[mask].fillna(0.0).sum())
            external_total = float(external.loc[mask].sum())
            composite_total = float(composite.loc[mask].sum())
            annual_rows.append(
                {"site": site, "year": year, "local_recorded_total_mm": local_total,
                 "external_total_mm": external_total, "candidate_gapfill_total_mm": composite_total,
                 "candidate_to_external_ratio": None if external_total == 0 else composite_total / external_total,
                 "local_recorded_days": int(local.loc[mask].notna().sum()),
                 "candidate_filled_days": int((local.loc[mask].isna() & external.loc[mask].notna()).sum())}
            )
        output["sites"][site] = {
            "selected_station": selected["station_id"],
            "distance_km": selected["distance_km"],
            "fit_days": int(len(days)),
            "local_nonmissing_days": int(local.notna().sum()),
            "local_explicit_zero_days": int((local.fillna(-1.0) == 0.0).sum()),
            "local_explicit_wet_days": int((local > 0.0).sum()),
            "local_missing_days": int(local.isna().sum()),
            "longest_local_missing_run_days": run_n,
            "longest_local_missing_run_start": run_start,
            "longest_local_missing_run_end": run_end,
            "external_valid_days": int(valid_external.sum()),
            "recorded_local_external_pairs": int(paired_recorded.sum()),
            "wet_dry_agreement_all_external_days_assuming_blank_is_dry": agreement_full,
            "wet_dry_agreement_only_recorded_local_days": agreement_recorded,
            "candidate_fill_days": int((local.isna() & external.notna()).sum()),
            "local_recorded_total_mm": float(local.fillna(0.0).sum()),
            "external_station_total_mm": float(external.sum()),
            "candidate_gapfill_total_mm": float(composite.sum()),
            "candidate_to_external_total_ratio": float(composite.sum() / external.sum()),
            "interpretation": (
                "Agreement on the full series assumes every blank local rain cell is a dry day. "
                "The recorded-only agreement is a sensitivity check because the workbook has sparse coverage."
            ),
        }
    pd.DataFrame(annual_rows).to_csv(OUT / "rainfall_candidate_annual_crosscheck.csv", index=False, encoding="utf-8-sig")
    (OUT / "rainfall_candidate_crosscheck.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["sites"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
