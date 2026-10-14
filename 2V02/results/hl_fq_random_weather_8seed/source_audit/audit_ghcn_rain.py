"""Screen nearby official GHCN-Daily precipitation records for HL and FQ."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from audit_sources import ROOT, OUT, SITES, read_excel_source


INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
DAILY_ROOT = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all"
MAX_DISTANCE_KM = 300.0
MIN_OVERLAP_DAYS = 3000
MIN_EVENT_AGREEMENT = 0.80
MIN_CHINA_CMA_SOURCE_SHARE = 0.90
WET_THRESHOLD_MM = 0.0


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rad = math.pi / 180.0
    dlat, dlon = (lat2 - lat1) * rad, (lon2 - lon1) * rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def fetch_or_read(url: str, path: Path, *, persist: bool = True) -> tuple[bytes, int, str]:
    if path.exists():
        data = path.read_bytes()
        return data, 200, url
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.content
    if persist:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data, response.status_code, response.url


def nearby_stations(inventory: bytes) -> dict[str, list[dict]]:
    candidates = {site: [] for site in SITES}
    for line in inventory.decode("ascii", errors="replace").splitlines():
        try:
            station_id = line[0:11].strip()
            lat, lon, elevation = float(line[12:20]), float(line[21:30]), float(line[31:37])
            name = line[41:71].strip()
        except (ValueError, IndexError):
            continue
        for site, spec in SITES.items():
            distance = haversine_km(spec["lat"], spec["lon"], lat, lon)
            if distance <= MAX_DISTANCE_KM:
                candidates[site].append(
                    {"station_id": station_id, "name": name, "latitude": lat, "longitude": lon,
                     "elevation_m": elevation, "distance_km": round(distance, 2)}
                )
    for site in candidates:
        candidates[site].sort(key=lambda item: item["distance_km"])
    return candidates


def parse_prcp_dly(content: bytes, station_id: str, start: int, end: int) -> pd.DataFrame:
    rows = []
    for line in content.decode("ascii", errors="replace").splitlines():
        if len(line) < 21 or line[0:11].strip() != station_id or line[17:21] != "PRCP":
            continue
        year, month = int(line[11:15]), int(line[15:17])
        if year < start or year > end:
            continue
        for day in range(1, 32):
            offset = 21 + (day - 1) * 8
            field = line[offset : offset + 8]
            if len(field) < 8:
                continue
            try:
                value = int(field[0:5])
                date = pd.Timestamp(year, month, day)
            except (ValueError, TypeError):
                continue
            mflag, qflag, sflag = field[5], field[6], field[7]
            # GHCN-Daily flags occupy one-character columns; an absent flag is a space.
            if value == -9999 or qflag.strip() != "":
                continue
            rows.append(
                {"date": date, "rain_mm": value / 10.0, "measurement_flag": mflag,
                 "quality_flag": qflag, "source_flag": sflag}
            )
    if not rows:
        return pd.DataFrame(columns=["date", "rain_mm", "measurement_flag", "quality_flag", "source_flag"])
    frame = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
    return frame.reset_index(drop=True)


def evaluate_candidate(site: str, meta: dict, external: pd.DataFrame, local: pd.Series) -> dict:
    spec = SITES[site]
    start, end = spec["start"], spec["end"]
    begin, finish = pd.Timestamp(start, 1, 1), pd.Timestamp(end, 12, 31)
    daily = pd.date_range(begin, finish, freq="D")
    local_full = local.reindex(daily)
    ext_full = external.set_index("date").reindex(daily)
    valid = ext_full.rain_mm.notna()
    station = local_full.fillna(0.0)
    pairs = pd.DataFrame({"local": station[valid], "ghcn": ext_full.loc[valid, "rain_mm"], "source": ext_full.loc[valid, "source_flag"]})
    if pairs.empty:
        agreement = None
        official_share = 0.0
        wet_precision = wet_recall = None
    else:
        local_wet = pairs.local > WET_THRESHOLD_MM
        ghcn_wet = pairs.ghcn > WET_THRESHOLD_MM
        agreement = float((local_wet == ghcn_wet).mean())
        wet_precision = float((local_wet & ghcn_wet).sum() / max(1, ghcn_wet.sum()))
        wet_recall = float((local_wet & ghcn_wet).sum() / max(1, local_wet.sum()))
        official_share = float((pairs.source == "s").mean())

    month_rows = []
    for period, group in pairs.groupby(pairs.index.to_period("M")):
        expected = period.days_in_month
        if len(group) / expected >= 0.90 and group.ghcn.sum() > 0:
            month_rows.append({"period": str(period), "local_total_mm": float(group.local.sum()),
                               "ghcn_total_mm": float(group.ghcn.sum()),
                               "local_to_ghcn_ratio": float(group.local.sum() / group.ghcn.sum()),
                               "coverage": len(group) / expected})
    year_rows = []
    for year, group in pairs.groupby(pairs.index.year):
        expected = 366 if pd.Timestamp(year, 12, 31).dayofyear == 366 else 365
        if len(group) / expected >= 0.90 and group.ghcn.sum() > 0:
            year_rows.append({"year": int(year), "local_total_mm": float(group.local.sum()),
                              "ghcn_total_mm": float(group.ghcn.sum()),
                              "local_to_ghcn_ratio": float(group.local.sum() / group.ghcn.sum()),
                              "coverage": len(group) / expected})
    pass_gate = len(pairs) >= MIN_OVERLAP_DAYS and agreement is not None and agreement >= MIN_EVENT_AGREEMENT and official_share >= MIN_CHINA_CMA_SOURCE_SHARE
    return {
        **meta,
        "fit_period": f"{start}-01-01/{end}-12-31",
        "paired_days": int(len(pairs)),
        "valid_prcp_days": int(valid.sum()),
        "event_agreement": agreement,
        "wet_day_precision_local_vs_ghcn": wet_precision,
        "wet_day_recall_local_vs_ghcn": wet_recall,
        "china_cma_source_flag_s_share": official_share,
        "source_flag_counts": {str(k): int(v) for k, v in pairs.source.value_counts(dropna=False).items()},
        "monthly_total_ratio_summary": {
            "complete_months_n": len(month_rows),
            "median": None if not month_rows else float(pd.Series([x["local_to_ghcn_ratio"] for x in month_rows]).median()),
            "minimum": None if not month_rows else min(x["local_to_ghcn_ratio"] for x in month_rows),
            "maximum": None if not month_rows else max(x["local_to_ghcn_ratio"] for x in month_rows),
            "periods": month_rows,
        },
        "annual_total_ratio_summary": {
            "complete_years_n": len(year_rows),
            "median": None if not year_rows else float(pd.Series([x["local_to_ghcn_ratio"] for x in year_rows]).median()),
            "minimum": None if not year_rows else min(x["local_to_ghcn_ratio"] for x in year_rows),
            "maximum": None if not year_rows else max(x["local_to_ghcn_ratio"] for x in year_rows),
            "periods": year_rows,
        },
        "gate": "PASS" if pass_gate else "FAIL",
        "gate_definition": {
            "paired_days_minimum": MIN_OVERLAP_DAYS,
            "wet_dry_agreement_minimum": MIN_EVENT_AGREEMENT,
            "CMA_source_flag_s_share_minimum": MIN_CHINA_CMA_SOURCE_SHARE,
            "wet_threshold_mm": WET_THRESHOLD_MM,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory_path = OUT / "ghcnd-stations.txt"
    inventory, inventory_status, inventory_url = fetch_or_read(INVENTORY_URL, inventory_path, persist=False)
    candidates = nearby_stations(inventory)
    # Persist a compact station catalogue; the full 11 MB inventory is not copied into experiment outputs.
    (OUT / "ghcn_station_candidates.json").write_text(
        json.dumps({"source_url": inventory_url, "http_status": inventory_status,
                    "sha256": digest(inventory), "source_version": "GHCN-Daily v3 station inventory",
                    "search_radius_km": MAX_DISTANCE_KM, "candidates": candidates}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    local_rain_by_site = {}
    local_raw, _ = read_excel_source(ROOT / "my_data" / "HLLCYCFQ降雨数据.xls", {"RAIN": ["20-20", "合计"]})
    for site, spec in SITES.items():
        start, end = spec["start"], spec["end"]
        local = local_raw[(local_raw.site == site) & local_raw.date.between(pd.Timestamp(start, 1, 1), pd.Timestamp(end, 12, 31))].set_index("date").RAIN.sort_index()
        local_rain_by_site[site] = local

    result = {
        "task": "HL/FQ nearby official rainfall source audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": {"url": inventory_url, "sha256": digest(inventory), "size_bytes": len(inventory),
                      "station_count_within_radius": {site: len(candidates[site]) for site in candidates}},
        "official_format_source": "NOAA NCEI GHCN-Daily README; PRCP is 0.1 mm, -9999 missing, source flag s identifies China Meteorological Administration/NMIC.",
        "sites": {},
        "raw_input_modified": False,
    }
    all_candidate_rows = []
    for site, site_candidates in candidates.items():
        spec = SITES[site]
        outcomes = []
        chosen_frame = None
        chosen = None
        for meta in site_candidates:
            station_id = meta["station_id"]
            url = f"{DAILY_ROOT}/{station_id}.dly"
            raw_path = OUT / "ghcn_daily_raw" / f"{station_id}.dly"
            try:
                data, status, final_url = fetch_or_read(url, raw_path)
            except requests.RequestException as exc:
                outcomes.append({**meta, "gate": "FAIL", "fetch_error": str(exc)})
                continue
            frame = parse_prcp_dly(data, station_id, spec["start"], spec["end"])
            evaluated = evaluate_candidate(site, meta, frame, local_rain_by_site[site])
            evaluated.update({"source_url": final_url, "http_status": status, "raw_path": raw_path.relative_to(ROOT).as_posix(), "raw_sha256": digest(data), "raw_size_bytes": len(data)})
            outcomes.append(evaluated)
            all_candidate_rows.append({k: v for k, v in evaluated.items() if k not in ("monthly_total_ratio_summary", "annual_total_ratio_summary")})
            if evaluated["gate"] == "PASS" and chosen is None:
                chosen, chosen_frame = evaluated, frame

        accepted_rows = []
        if chosen is not None and chosen_frame is not None:
            spec = SITES[site]
            begin, finish = pd.Timestamp(spec["start"], 1, 1), pd.Timestamp(spec["end"], 12, 31)
            local = local_rain_by_site[site]
            ext = chosen_frame.set_index("date")
            for date, raw_value in local.items():
                if pd.isna(raw_value) and date in ext.index:
                    accepted_rows.append({"site": site, "date": str(date.date()), "variable": "RAIN",
                                          "original_station_value": None, "reason": "raw_missing",
                                          "candidate_value_mm": float(ext.loc[date, "rain_mm"]),
                                          "candidate_station_id": chosen["station_id"],
                                          "candidate_distance_km": chosen["distance_km"],
                                          "candidate_source_flag": ext.loc[date, "source_flag"],
                                          "source_gate_pass": True, "promoted_to_fitting_weather": False})
            pd.DataFrame(accepted_rows).to_csv(OUT / f"ghcn_{site}_candidate_rain_gapfill.csv", index=False, encoding="utf-8-sig")
            chosen_frame.to_csv(OUT / f"ghcn_{site}_chosen_station_daily.csv", index=False, encoding="utf-8-sig")

        result["sites"][site] = {
            "fit_period": f"{spec['start']}-01-01/{spec['end']}-12-31",
            "tested_candidates": outcomes,
            "selected_station": chosen,
            "candidate_rain_gapfill_rows": len(accepted_rows),
            "status": "RECONSTRUCT_REQUIRED" if chosen is not None else "BLOCKED",
            "next_gate": "CANDIDATE_GAPFILL_QC" if chosen is not None else "STOP_BEFORE_WGEN",
        }

    (OUT / "ghcn_rain_station_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(all_candidate_rows).to_csv(OUT / "ghcn_rain_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({site: {"status": data["status"], "selected_station": data["selected_station"],
                             "tested": [{"station_id": x.get("station_id"), "name": x.get("name"), "distance_km": x.get("distance_km"),
                                         "paired_days": x.get("paired_days"), "event_agreement": x.get("event_agreement"),
                                         "CMA_s_share": x.get("china_cma_source_flag_s_share"), "gate": x.get("gate")} for x in data["tested_candidates"]]}
                      for site, data in result["sites"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
