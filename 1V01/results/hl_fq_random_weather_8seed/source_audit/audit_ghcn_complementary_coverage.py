"""Check whether already downloaded passing GHCN stations cover local rain gaps."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from audit_ghcn_rain import OUT, SITES, parse_prcp_dly
from audit_sources import ROOT, load_local_weather


def main() -> None:
    audit = json.loads((OUT / "ghcn_rain_station_audit.json").read_text(encoding="utf-8"))
    local_raw, _ = load_local_weather()
    report = {"task": "Complementary official rain station coverage screen", "sites": {}}
    all_gap_rows = []

    for site, spec in SITES.items():
        begin, finish = pd.Timestamp(spec["start"], 1, 1), pd.Timestamp(spec["end"], 12, 31)
        days = pd.date_range(begin, finish, freq="D")
        local = (
            local_raw[(local_raw.site == site) & local_raw.date.between(begin, finish)]
            .set_index("date")["RAIN"].reindex(days)
        )
        qualified = [x for x in audit["sites"][site]["tested_candidates"] if x.get("gate") == "PASS"]
        ordered = sorted(qualified, key=lambda x: x["distance_km"])
        station_frames = {}
        for meta in ordered:
            raw_path = OUT / "ghcn_daily_raw" / f"{meta['station_id']}.dly"
            parsed = parse_prcp_dly(raw_path.read_bytes(), meta["station_id"], spec["start"], spec["end"])
            station_frames[meta["station_id"]] = parsed.set_index("date").reindex(days)

        selected = audit["sites"][site]["selected_station"]
        primary = station_frames[selected["station_id"]]["rain_mm"]
        fill_counts = Counter()
        uncovered = []
        for date in days[local.isna()]:
            source = next(
                (meta for meta in ordered if pd.notna(station_frames[meta["station_id"]].loc[date, "rain_mm"])),
                None,
            )
            if source is None:
                uncovered.append(str(date.date()))
                continue
            row = station_frames[source["station_id"]].loc[date]
            fill_counts[source["station_id"]] += 1
            all_gap_rows.append(
                {
                    "site": site,
                    "date": str(date.date()),
                    "variable": "RAIN",
                    "original_station_value": None,
                    "reason": "raw_missing",
                    "candidate_value_mm": float(row["rain_mm"]),
                    "candidate_station_id": source["station_id"],
                    "candidate_station_name": source["name"],
                    "candidate_distance_km": source["distance_km"],
                    "candidate_source_flag": str(row["source_flag"]),
                    "source_gate_pass": True,
                    "promoted_to_fitting_weather": False,
                }
            )

        report["sites"][site] = {
            "fit_period": f"{spec['start']}-01-01/{spec['end']}-12-31",
            "local_missing_days": int(local.isna().sum()),
            "primary_station": selected["station_id"],
            "primary_distance_km": selected["distance_km"],
            "primary_station_gap_days_after_local_records": int((local.isna() & primary.isna()).sum()),
            "qualified_stations_available": [
                {
                    "station_id": x["station_id"],
                    "name": x["name"],
                    "distance_km": x["distance_km"],
                    "paired_days": x["paired_days"],
                    "event_agreement": x["event_agreement"],
                    "cma_s_share": x["china_cma_source_flag_s_share"],
                }
                for x in ordered
            ],
            "gap_fill_rows_by_station": dict(fill_counts),
            "maximum_gap_fill_station_distance_km": max(
                (x["distance_km"] for x in ordered if fill_counts[x["station_id"]]), default=None
            ),
            "uncovered_local_rain_days": uncovered,
            "candidate_full_coverage": not uncovered,
            "candidate_gap_rows": int(sum(fill_counts.values())),
        }

    pd.DataFrame(all_gap_rows).to_csv(
        OUT / "ghcn_multi_station_candidate_rain_gapfill.csv", index=False, encoding="utf-8-sig"
    )
    (OUT / "ghcn_multi_station_coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact = {
        site: {
            "candidate_full_coverage": data["candidate_full_coverage"],
            "local_missing_days": data["local_missing_days"],
            "primary_station_gap_days": data["primary_station_gap_days_after_local_records"],
            "gap_fill_rows_by_station": data["gap_fill_rows_by_station"],
            "uncovered": len(data["uncovered_local_rain_days"]),
            "max_distance_km": data["maximum_gap_fill_station_distance_km"],
        }
        for site, data in report["sites"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
