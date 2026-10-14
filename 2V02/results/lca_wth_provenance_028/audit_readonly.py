"""Read existing LCA source snapshots and WTHs; write only 028 audit artifacts."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/lca_wth_provenance_028"
D222 = ROOT / "results/five_site_d222_version_audit_022_rerun_20261007/current_d222_daily_normalized.csv"
CLEANED = ROOT / "weather_clean/LCA_weather_cleaned.csv"
BASE = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/LC"
LOWIC = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC"
QC = ROOT / "Leave_One_experiments/wth_generated_qc/LCA"
ARCHIVE = BASE / "CNLC9816.WTH"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_d222() -> dict[str, dict]:
    result = {}
    with D222.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["project_site"] == "LCA":
                result[row["date"]] = row
    return result


def load_cleaned() -> dict[str, float]:
    with CLEANED.open(newline="", encoding="utf-8-sig") as f:
        return {r["date"]: float(r["RAIN"]) for r in csv.DictReader(f)}


def load_wth(folder: Path) -> tuple[dict[str, dict], list[Path]]:
    result = {}
    files = []
    for p in sorted(folder.glob("CNLC??01.WTH")):
        year = 2000 + int(p.stem[4:6])
        if not 2000 <= year <= 2023:
            continue
        files.append(p)
        with p.open(encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                parts = line.split()
                if len(parts) < 5 or len(parts[0]) != 7 or not parts[0].isdigit():
                    continue
                date = dt.date(int(parts[0][:4]), 1, 1) + dt.timedelta(days=int(parts[0][4:]) - 1)
                if date.year != year:
                    continue
                key = date.isoformat()
                if key in result:
                    raise ValueError(f"duplicate WTH date {key}")
                result[key] = {"rain": float(parts[4]), "file": rel(p), "line": line_no}
    return result, files


def load_single_wth(path: Path) -> dict[str, float]:
    values = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5 or len(parts[0]) != 7 or not parts[0].isdigit():
                continue
            day = dt.date(int(parts[0][:4]), 1, 1) + dt.timedelta(days=int(parts[0][4:]) - 1)
            values[day.isoformat()] = float(parts[4])
    return values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_d222()
    cleaned = load_cleaned()
    base, base_files = load_wth(BASE)
    lowic, lowic_files = load_wth(LOWIC)
    archive = load_single_wth(ARCHIVE)
    qc = {}
    for year in range(2000, 2024):
        p = QC / f"LCA{year}.WTH"
        if p.exists():
            qc.update(load_single_wth(p))
    if set(base) != set(lowic):
        raise ValueError("base and lowIC WTH dates differ")
    changed = []
    blank_to_rain = []
    all_differences = []
    base_lowic_differences = []
    for date in sorted(base):
        if abs(base[date]["rain"] - lowic[date]["rain"]) > 1e-9:
            base_lowic_differences.append(date)
        r = raw.get(date)
        if r is None:
            continue
        wth = lowic[date]
        old = float(r["rain_mm"]) if r["rain_mm"] != "" else None
        common = {
            "date": date,
            "year": date[:4],
            "RAIN_d222_mm": "" if old is None else old,
            "RAIN_cleaned_mm": cleaned.get(date, ""),
            "RAIN_wth_mm": wth["rain"],
            "RAIN_archive_mm": archive.get(date, ""),
            "RAIN_early_qc_mm": qc.get(date, ""),
            "wth_file": wth["file"],
            "wth_line": wth["line"],
            "d222_status": r["value_status"],
            "d222_source_row": r["source_row"],
            "daily_replacement_source": "CNLC9816.WTH matching counterpart; copy direction and original downloaded file unresolved",
        }
        if old is not None and abs(old - wth["rain"]) > 1e-9:
            changed.append({**common, "delta_wth_minus_d222_mm": round(wth["rain"] - old, 6)})
        elif old is None and wth["rain"] > 0:
            blank_to_rain.append(common)
        if abs(cleaned[date] - wth["rain"]) > 1e-9:
            all_differences.append(date)
    write_csv("lca_changed_nonmissing_days.csv", changed, list(changed[0]))
    write_csv("lca_blank_to_rain_days.csv", blank_to_rain, list(blank_to_rain[0]))
    deltas = [r["delta_wth_minus_d222_mm"] for r in changed]
    both_hash = sum(digest(a) == digest(b) for a, b in zip(base_files, lowic_files))
    file_inventory = []
    for a, b in zip(base_files, lowic_files):
        if a.name != b.name:
            raise ValueError("base and lowIC annual filenames differ")
        year = 2000 + int(a.stem[4:6])
        q = QC / f"LCA{year}.WTH"
        ha, hb = digest(a), digest(b)
        file_inventory.append({
            "year": year,
            "base_wth_file": rel(a),
            "base_mtime_utc": dt.datetime.fromtimestamp(a.stat().st_mtime, dt.timezone.utc).isoformat(),
            "base_sha256": ha,
            "lowic_wth_file": rel(b),
            "lowic_mtime_utc": dt.datetime.fromtimestamp(b.stat().st_mtime, dt.timezone.utc).isoformat(),
            "lowic_sha256": hb,
            "base_lowic_same_sha256": ha == hb,
            "early_qc_wth_file": rel(q) if q.exists() else "",
            "early_qc_mtime_utc": dt.datetime.fromtimestamp(q.stat().st_mtime, dt.timezone.utc).isoformat() if q.exists() else "",
            "early_qc_sha256": digest(q) if q.exists() else "",
            "git_commit_for_wth": "NOT_FOUND_IN_CURRENT_GIT",
        })
    write_csv("lca_wth_file_inventory.csv", file_inventory, list(file_inventory[0]))
    qc_compare = {}
    for year in range(2000, 2024):
        q = QC / f"LCA{year}.WTH"
        if q.exists():
            annual_dates = [d for d in base if d.startswith(str(year))]
            # QC layout can differ; hash identity is the conservative comparison.
            qc_compare[year] = {"file": rel(q), "sha256": digest(q), "annual_date_count": len(annual_dates)}
    stats = {
        "d222_snapshot": rel(D222),
        "d222_source_original": "data/external/人工大气观测降雨蒸发能见度日值.xls",
        "base_wth_files": len(base_files),
        "lowic_wth_files": len(lowic_files),
        "wth_dates": len(lowic),
        "identical_base_lowic_files_sha256": both_hash,
        "base_lowic_rain_difference_dates": len(base_lowic_differences),
        "archive_rain_match_2000_2013": sum(date in archive and archive[date] == lowic[date]["rain"] for date in lowic if "2000" <= date[:4] <= "2013"),
        "archive_coverage_2000_2013": sum(date in archive for date in lowic if "2000" <= date[:4] <= "2013"),
        "changed_nonmissing_archive_match": sum(r["RAIN_archive_mm"] == r["RAIN_wth_mm"] for r in changed),
        "blank_to_rain_archive_match": sum(r["RAIN_archive_mm"] == r["RAIN_wth_mm"] for r in blank_to_rain),
        "changed_nonmissing_early_qc_match_d222": sum(abs(r["RAIN_early_qc_mm"] - r["RAIN_d222_mm"]) < 1e-9 for r in changed),
        "blank_to_rain_early_qc_zero": sum(r["RAIN_early_qc_mm"] == 0 for r in blank_to_rain),
        "changed_nonmissing_count": len(changed),
        "blank_to_rain_count": len(blank_to_rain),
        "all_cleaned_wth_rain_difference_count": len(all_differences),
        "changed_nonmissing_first_date": changed[0]["date"] if changed else None,
        "changed_nonmissing_last_date": changed[-1]["date"] if changed else None,
        "changed_nonmissing_delta_max_mm": max(deltas) if deltas else None,
        "changed_nonmissing_delta_min_mm": min(deltas) if deltas else None,
        "changed_nonmissing_abs_delta_max_mm": max(map(abs, deltas)) if deltas else None,
        "changed_nonmissing_delta_mean_mm": statistics.mean(deltas) if deltas else None,
        "changed_nonmissing_abs_delta_mean_mm": statistics.mean(map(abs, deltas)) if deltas else None,
        "changed_nonmissing_by_year": dict(sorted(Counter(r["year"] for r in changed).items())),
        "blank_to_rain_by_year": dict(sorted(Counter(r["year"] for r in blank_to_rain).items())),
        "blank_to_rain_first_date": blank_to_rain[0]["date"] if blank_to_rain else None,
        "blank_to_rain_last_date": blank_to_rain[-1]["date"] if blank_to_rain else None,
        "qc_wth_file_count": len(qc_compare),
    }
    (OUT / "difference_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
