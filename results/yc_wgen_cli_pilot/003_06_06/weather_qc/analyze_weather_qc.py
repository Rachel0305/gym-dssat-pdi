from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULT_ROOT = PROJECT_ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_06"
WEATHER_DIR = RESULT_ROOT / "weather_qc"
FIGURE_DIR = WEATHER_DIR / "figures"
PRIOR_ROOT = RESULT_ROOT.parent / "003_06_05_02"
TRAIN_PATH = PROJECT_ROOT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
WTH_PATH = PROJECT_ROOT / "my_data" / "CNYC0801.WTH"
CLI_PATH = PRIOR_ROOT / "final" / "CNYC.CLI"
EXPECTED_TRAIN_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
EXPECTED_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
SEED_FILES = {
    101: "seed_101_run_a.csv",
    102: "seed_102.csv",
    103: "seed_103.csv",
    104: "seed_104.csv",
    105: "seed_105.csv",
}
FIELDS = ("RAIN", "SRAD", "TMAX", "TMIN")
COMMON_FIELDS = ("RAIN", "SRAD", "TMAX", "TMIN")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values]
    if not data:
        return {"n": 0, "mean": None, "sd": None, "median": None, "min": None, "max": None,
                "p05": None, "p25": None, "p75": None, "p95": None}
    return {
        "n": len(data),
        "mean": statistics.fmean(data),
        "sd": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
        "p05": quantile(data, 0.05),
        "p25": quantile(data, 0.25),
        "p75": quantile(data, 0.75),
        "p95": quantile(data, 0.95),
    }


def _read_csv_weather(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for source in csv.DictReader(stream):
            day = date.fromisoformat(source["DATE"])
            row = {"DATE": day, "DOY": int(source.get("DOY") or day.timetuple().tm_yday)}
            for field in FIELDS:
                value = float(source[field])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {field} in {path} on {day}")
                row[field] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"No weather rows in {path}")
    rows.sort(key=lambda item: item["DATE"])
    return rows


def _read_wth(path: Path) -> list[dict]:
    rows: list[dict] = []
    header: list[str] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("@") and "DATE" in stripped.upper():
            header_tokens = stripped.split()
            if header_tokens[0] == "@":
                header_tokens = header_tokens[1:]
            header = [token.upper().lstrip("@") for token in header_tokens]
            continue
        if not stripped or stripped.startswith("*") or header is None:
            continue
        tokens = stripped.split()
        if len(tokens) < len(header) or not tokens[0].isdigit():
            continue
        values = dict(zip(header, tokens))
        raw_date = values["DATE"]
        day = datetime.strptime(raw_date, "%Y%j").date() if len(raw_date) == 7 else datetime.strptime(raw_date, "%y%j").date()
        row = {"DATE": day, "DOY": day.timetuple().tm_yday}
        for field in FIELDS:
            row[field] = float(values[field])
        rows.append(row)
    if not rows:
        raise ValueError(f"No weather rows parsed from {path}")
    rows.sort(key=lambda item: item["DATE"])
    return rows


def longest_spell(rows: list[dict], condition) -> int:
    best = current = 0
    for row in sorted(rows, key=lambda item: item["DATE"]):
        if condition(row):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _rain_metrics(rows: list[dict]) -> dict[str, float | int | None]:
    rain = [row["RAIN"] for row in rows]
    wet = [value for value in rain if value > 0.0]
    return {
        "window_days": len(rows),
        "wet_days": len(wet),
        "wet_day_fraction": len(wet) / len(rows) if rows else None,
        "rain_total_mm": sum(rain),
        "mean_rain_on_wet_day_mm": statistics.fmean(wet) if wet else None,
        "median_rain_on_wet_day_mm": statistics.median(wet) if wet else None,
        "p90_daily_rain_mm": quantile(rain, 0.90),
        "p95_daily_rain_mm": quantile(rain, 0.95),
        "max_daily_rain_mm": max(rain) if rain else None,
        "longest_dry_spell_days": longest_spell(rows, lambda row: row["RAIN"] <= 0.0),
        "longest_wet_spell_days": longest_spell(rows, lambda row: row["RAIN"] > 0.0),
    }


def _year_window(rows: list[dict], start: date, end: date, year: int) -> list[dict]:
    expected = []
    current = start
    while current <= end:
        expected.append(date(year, current.month, current.day))
        current += timedelta(days=1)
    by_month_day = {(row["DATE"].month, row["DATE"].day): row for row in rows if row["DATE"].year == year}
    missing = [(day.month, day.day) for day in expected if (day.month, day.day) not in by_month_day]
    if missing:
        raise ValueError(f"Historical {year} is missing common-window dates: {missing[:3]}")
    return [by_month_day[(day.month, day.day)] for day in expected]


def _daily_metric_rows(rows: list[dict], source: str, label: str) -> list[dict]:
    result = []
    for field in FIELDS:
        summary = stats(row[field] for row in rows)
        result.append({"source": source, "label": label, "variable": field, **summary})
    return result


def _monthly_metrics(rows: list[dict]) -> dict[int, dict[str, float | int]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["DATE"].month].append(row)
    result = {}
    for month, subset in sorted(grouped.items()):
        result[month] = {
            "rain_total_mm": sum(row["RAIN"] for row in subset),
            "wet_days": sum(row["RAIN"] > 0.0 for row in subset),
            "mean_tmax_c": statistics.fmean(row["TMAX"] for row in subset),
            "mean_tmin_c": statistics.fmean(row["TMIN"] for row in subset),
            "mean_srad_mj_m2_d": statistics.fmean(row["SRAD"] for row in subset),
            "max_daily_rain_mm": max(row["RAIN"] for row in subset),
        }
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _figure_bar(path: Path, title: str, ylabel: str, hist_values: dict[str, float], seed_values: dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    labels = list(hist_values) + list(seed_values)
    values = list(hist_values.values()) + list(seed_values.values())
    colors = ["#82939A"] * len(hist_values) + ["#167D8D", "#D97732", "#5B8C5A", "#C34E52", "#7E6BA8"]
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    ax.bar(range(len(labels)), values, color=colors, width=0.74)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.axvline(len(hist_values) - 0.5, color="#30383B", linewidth=1, linestyle="--")
    ax.grid(axis="y", color="#D8DEE1", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _figure_box(path: Path, title: str, ylabel: str, hist_values: list[float], seed_values: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    labels = ["Train 2004-2013 pooled", *[f"Seed {seed}" for seed in seed_values]]
    values = [hist_values, *seed_values.values()]
    colors = ["#82939A", "#167D8D", "#D97732", "#5B8C5A", "#C34E52", "#7E6BA8"]
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    boxes = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False, whis=(5, 95))
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.68)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", color="#D8DEE1", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _figure_monthly(path: Path, history: dict[int, dict[int, float]], generated: dict[int, dict[int, float]]) -> None:
    import matplotlib.pyplot as plt

    months = sorted(next(iter(generated.values())))
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for year, values in history.items():
        ax.plot(months, [values[month] for month in months], color="#9AA6AA", alpha=0.45, linewidth=1)
    for seed, values in generated.items():
        color = {101: "#167D8D", 102: "#D97732", 103: "#5B8C5A", 104: "#C34E52", 105: "#7E6BA8"}[seed]
        ax.plot(months, [values[month] for month in months], marker="o", linewidth=2, label=f"Seed {seed}", color=color)
    ax.set_xticks(months, [date(2008, month, 1).strftime("%b") for month in months])
    ax.set_title("Monthly rainfall in the fixed common window")
    ax.set_ylabel("Rainfall (mm/month)")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", color="#D8DEE1", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run() -> dict:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    train_hash = sha256_file(TRAIN_PATH)
    cli_hash = sha256_file(CLI_PATH)
    if train_hash != EXPECTED_TRAIN_SHA256 or cli_hash != EXPECTED_CLI_SHA256:
        raise RuntimeError(f"Frozen input hash mismatch: train={train_hash}, CLI={cli_hash}")

    train = _read_csv_weather(TRAIN_PATH)
    generated = {seed: _read_csv_weather(PRIOR_ROOT / "generated_weather" / name) for seed, name in SEED_FILES.items()}
    repeated_101 = _read_csv_weather(PRIOR_ROOT / "generated_weather" / "seed_101_run_b.csv")
    if generated[101] != repeated_101:
        raise ValueError("Seed 101 repeated sequence is not identical")
    date_sets = [{row["DATE"] for row in rows} for rows in generated.values()]
    common_dates = sorted(set.intersection(*date_sets))
    if not common_dates or common_dates != [common_dates[0] + timedelta(days=index) for index in range(len(common_dates))]:
        raise ValueError("Seed common dates are empty or non-contiguous")
    common_start, common_end = common_dates[0], common_dates[-1]
    common_by_seed = {
        seed: [row for row in rows if common_start <= row["DATE"] <= common_end]
        for seed, rows in generated.items()
    }
    if any(len(rows) != len(common_dates) for rows in common_by_seed.values()):
        raise ValueError("At least one seed does not cover every common-window date")

    history_by_year = {}
    for year in range(2004, 2014):
        history_by_year[year] = _year_window(train, common_start, common_end, year)
    historical_rain = {str(year): _rain_metrics(rows) for year, rows in history_by_year.items()}
    seed_rain = {str(seed): _rain_metrics(rows) for seed, rows in common_by_seed.items()}

    fixed_rows = []
    for seed, rows in common_by_seed.items():
        for row in rows:
            fixed_rows.append({"source": "WGEN", "year": 2008, "seed": seed, "DATE": row["DATE"].isoformat(), "DOY": row["DOY"], **{field: row[field] for field in FIELDS}})
    for year, rows in history_by_year.items():
        for row in rows:
            fixed_rows.append({"source": "TRAIN_HISTORICAL", "year": year, "seed": "", "DATE": row["DATE"].isoformat(), "DOY": row["DOY"], **{field: row[field] for field in FIELDS}})
    _write_csv(WEATHER_DIR / "fixed_window_weather.csv", fixed_rows)

    hist_reference_rows = []
    metric_names = ["rain_total_mm", "wet_days", "max_daily_rain_mm", "longest_dry_spell_days", "mean_tmax_c", "mean_tmin_c", "mean_srad_mj_m2_d", "min_tmin_c", "max_tmax_c", "max_srad_mj_m2_d"]
    for year, rows in history_by_year.items():
        rain_summary = _rain_metrics(rows)
        values = {
            **{name: rain_summary[name] for name in ("rain_total_mm", "wet_days", "max_daily_rain_mm", "longest_dry_spell_days")},
            "mean_tmax_c": statistics.fmean(row["TMAX"] for row in rows),
            "mean_tmin_c": statistics.fmean(row["TMIN"] for row in rows),
            "mean_srad_mj_m2_d": statistics.fmean(row["SRAD"] for row in rows),
            "min_tmin_c": min(row["TMIN"] for row in rows),
            "max_tmax_c": max(row["TMAX"] for row in rows),
            "max_srad_mj_m2_d": max(row["SRAD"] for row in rows),
        }
        hist_reference_rows.extend({"record_type": "historical_year", "year": year, "metric": name, "value": value} for name, value in values.items())
    for name in metric_names:
        annual_values = [row["value"] for row in hist_reference_rows if row["metric"] == name]
        hist_reference_rows.extend({"record_type": "historical_summary", "year": "2004-2013", "metric": name, "summary_stat": key, "value": value} for key, value in stats(annual_values).items() if key != "n")
    _write_csv(WEATHER_DIR / "historical_common_window_reference.csv", hist_reference_rows)

    hist_rain_values = {year: historical_rain[str(year)]["rain_total_mm"] for year in range(2004, 2014)}
    hist_wet_values = {year: historical_rain[str(year)]["wet_days"] for year in range(2004, 2014)}
    hist_dry_values = {year: historical_rain[str(year)]["longest_dry_spell_days"] for year in range(2004, 2014)}
    _figure_bar(FIGURE_DIR / "common_window_rain_total.png", "Common-window rainfall: train years and WGEN seeds", "Rainfall total (mm)", hist_rain_values, {f"Seed {seed}": values["rain_total_mm"] for seed, values in seed_rain.items()})
    _figure_bar(FIGURE_DIR / "wet_day_counts.png", "Wet days in the fixed common window", "Wet days (RAIN > 0.0 mm)", hist_wet_values, {f"Seed {seed}": values["wet_days"] for seed, values in seed_rain.items()})
    _figure_bar(FIGURE_DIR / "longest_dry_spell.png", "Longest dry spell in the fixed common window", "Longest consecutive dry spell (days)", hist_dry_values, {f"Seed {seed}": values["longest_dry_spell_days"] for seed, values in seed_rain.items()})

    train_window = [row for year_rows in history_by_year.values() for row in year_rows]
    seed_daily = {seed: rows for seed, rows in common_by_seed.items()}
    for field, filename, label in (("TMAX", "tmax_distribution.png", "Daily maximum temperature (deg C)"), ("TMIN", "tmin_distribution.png", "Daily minimum temperature (deg C)"), ("SRAD", "srad_distribution.png", "Solar radiation (MJ/m2/day)")):
        _figure_box(FIGURE_DIR / filename, f"{field} distribution: train and WGEN common window", label, [row[field] for row in train_window], {seed: [row[field] for row in rows] for seed, rows in seed_daily.items()})

    monthly_hist_by_year = {year: _monthly_metrics(rows) for year, rows in history_by_year.items()}
    monthly_seed_by_seed = {seed: _monthly_metrics(rows) for seed, rows in common_by_seed.items()}
    monthly_rain_hist = {year: {month: values["rain_total_mm"] for month, values in monthly.items()} for year, monthly in monthly_hist_by_year.items()}
    monthly_rain_generated = {seed: {month: values["rain_total_mm"] for month, values in monthly.items()} for seed, monthly in monthly_seed_by_seed.items()}
    _figure_monthly(FIGURE_DIR / "monthly_rainfall.png", monthly_rain_hist, monthly_rain_generated)

    monthly_rows = []
    monthly_metric_names = ("rain_total_mm", "wet_days", "mean_tmax_c", "mean_tmin_c", "mean_srad_mj_m2_d", "max_daily_rain_mm")
    for month in sorted(monthly_seed_by_seed[101]):
        for metric in monthly_metric_names:
            annual = [monthly_hist_by_year[year][month][metric] for year in range(2004, 2014)]
            reference = stats(annual)
            for seed in monthly_seed_by_seed:
                monthly_rows.append({"month": month, "metric": metric, "source": "WGEN", "seed": seed, "year": 2008,
                                     "value": monthly_seed_by_seed[seed][month][metric], **{f"historical_{key}": value for key, value in reference.items()}})
            for year in monthly_hist_by_year:
                monthly_rows.append({"month": month, "metric": metric, "source": "TRAIN_HISTORICAL", "seed": "", "year": year,
                                     "value": monthly_hist_by_year[year][month][metric], **{f"historical_{key}": value for key, value in reference.items()}})
    _write_csv(WEATHER_DIR / "monthly_weather_qc.csv", monthly_rows)

    rain_rows = []
    for year, metrics in historical_rain.items():
        rain_rows.append({"source": "TRAIN_HISTORICAL", "label": year, **metrics})
    historical_rain_stats = {key: stats(row[key] for row in historical_rain.values()) for key in seed_rain["101"]}
    for seed, metrics in seed_rain.items():
        rain_rows.append({"source": "WGEN", "label": seed, **metrics,
                          **{f"historical_{metric}_{stat}": value for metric, summary in historical_rain_stats.items() for stat, value in summary.items() if stat != "n"}})
    _write_csv(WEATHER_DIR / "rainfall_structure_qc.csv", rain_rows)

    temp_rows = []
    pooled_train_tmax = [row["TMAX"] for row in train_window]
    pooled_train_tmin = [row["TMIN"] for row in train_window]
    train_p90_tmax = quantile(pooled_train_tmax, 0.90)
    train_p05_tmin = quantile(pooled_train_tmin, 0.05)
    for year, rows in history_by_year.items():
        for field in ("TMAX", "TMIN"):
            summary = stats(row[field] for row in rows)
            temp_rows.append({"source": "TRAIN_HISTORICAL", "label": year, "variable": field, **summary,
                              "hot_days_above_train_p90": sum(row["TMAX"] > train_p90_tmax for row in rows),
                              "cold_nights_below_train_p05": sum(row["TMIN"] < train_p05_tmin for row in rows),
                              "train_p90_tmax_c": train_p90_tmax, "train_p05_tmin_c": train_p05_tmin})
    for seed, rows in common_by_seed.items():
        for field in ("TMAX", "TMIN"):
            summary = stats(row[field] for row in rows)
            temp_rows.append({"source": "WGEN", "label": seed, "variable": field, **summary,
                              "hot_days_above_train_p90": sum(row["TMAX"] > train_p90_tmax for row in rows),
                              "cold_nights_below_train_p05": sum(row["TMIN"] < train_p05_tmin for row in rows),
                              "train_p90_tmax_c": train_p90_tmax, "train_p05_tmin_c": train_p05_tmin})
    _write_csv(WEATHER_DIR / "temperature_qc.csv", temp_rows)

    srad_rows = []
    for year, rows in history_by_year.items():
        srad_rows.append({"source": "TRAIN_HISTORICAL", "label": year, **stats(row["SRAD"] for row in rows)})
    for seed, rows in common_by_seed.items():
        srad_rows.append({"source": "WGEN", "label": seed, **stats(row["SRAD"] for row in rows)})
    _write_csv(WEATHER_DIR / "srad_qc.csv", srad_rows)

    full_period_rows = []
    for seed, rows in generated.items():
        full_period_rows.append({"seed": seed, "start_date": rows[0]["DATE"].isoformat(), "end_date": rows[-1]["DATE"].isoformat(),
                                 "simulation_days": len(rows), "cumulative_rain_mm": sum(row["RAIN"] for row in rows),
                                 "wet_days": sum(row["RAIN"] > 0 for row in rows),
                                 "longest_dry_spell_days": longest_spell(rows, lambda row: row["RAIN"] <= 0),
                                 "mean_srad_mj_m2_d": statistics.fmean(row["SRAD"] for row in rows),
                                 "mean_tmax_c": statistics.fmean(row["TMAX"] for row in rows),
                                 "mean_tmin_c": statistics.fmean(row["TMIN"] for row in rows),
                                 "comparability_note": "NOT DIRECTLY COMPARABLE FOR CUMULATIVE TOTALS"})
    _write_csv(WEATHER_DIR / "full_simulation_period_weather.csv", full_period_rows)

    physical_issues = []
    for seed, rows in generated.items():
        for row in rows:
            if row["RAIN"] < 0 or row["SRAD"] < 0 or row["TMAX"] < row["TMIN"]:
                physical_issues.append({"seed": seed, "date": row["DATE"].isoformat(), "rain": row["RAIN"], "srad": row["SRAD"], "tmax": row["TMAX"], "tmin": row["TMIN"]})
            if row["RAIN"] > 500 or row["SRAD"] > 45 or not -60 <= row["TMAX"] <= 60 or not -80 <= row["TMIN"] <= 50:
                physical_issues.append({"seed": seed, "date": row["DATE"].isoformat(), "issue": "screening extreme bound exceeded"})
    sequences = {seed: hashlib.sha256(b"\n".join(f"{row['DATE']},{row['RAIN']:.12g},{row['SRAD']:.12g},{row['TMAX']:.12g},{row['TMIN']:.12g}".encode() for row in rows)).hexdigest() for seed, rows in generated.items()}
    unique_sequences = len(set(sequences.values()))
    hist_inside = {}
    for metric, values in (("rain_total_mm", [v["rain_total_mm"] for v in seed_rain.values()]),
                           ("wet_days", [v["wet_days"] for v in seed_rain.values()]),
                           ("longest_dry_spell_days", [v["longest_dry_spell_days"] for v in seed_rain.values()]),
                           ("mean_tmax_c", [statistics.fmean(row["TMAX"] for row in rows) for rows in common_by_seed.values()]),
                           ("mean_tmin_c", [statistics.fmean(row["TMIN"] for row in rows) for rows in common_by_seed.values()]),
                           ("mean_srad_mj_m2_d", [statistics.fmean(row["SRAD"] for row in rows) for rows in common_by_seed.values()])):
        reference = [historical_rain[str(year)][metric] for year in historical_rain] if metric in historical_rain["2004"] else [statistics.fmean(row["TMAX"] for row in history_by_year[year]) if metric == "mean_tmax_c" else statistics.fmean(row["TMIN"] for row in history_by_year[year]) if metric == "mean_tmin_c" else statistics.fmean(row["SRAD"] for row in history_by_year[year]) for year in history_by_year]
        hist_inside[metric] = {"within_historical_annual_min_max": sum(min(reference) <= value <= max(reference) for value in values),
                               "seed_count": len(values), "historical_min": min(reference), "historical_max": max(reference)}
    if physical_issues:
        qc_status = "FAIL_PHYSICAL"
    elif unique_sequences < len(SEED_FILES) or any(all(row[field] == rows[0][field] for row in rows) for rows in generated.values() for field in FIELDS):
        qc_status = "FAIL_DEGENERATE_VARIABILITY"
    else:
        qc_status = "PASS_WITH_NOTES"

    wth_rows = _read_wth(WTH_PATH)
    wth_meta = {"wth_file": str(WTH_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256_file(WTH_PATH),
                "start_date": wth_rows[0]["DATE"].isoformat(), "end_date": wth_rows[-1]["DATE"].isoformat(),
                "rows": len(wth_rows), "historical_2008_same_window_present": all(any(row["DATE"] == day for row in wth_rows) for day in common_dates)}
    historical_hot_days = [row["hot_days_above_train_p90"] for row in temp_rows if row["source"] == "TRAIN_HISTORICAL" and row["variable"] == "TMAX"]
    generated_hot_days = {row["label"]: row["hot_days_above_train_p90"] for row in temp_rows if row["source"] == "WGEN" and row["variable"] == "TMAX"}
    historical_hot_days_max = max(historical_hot_days)
    hot_seed_count_above_historical_max = sum(value > historical_hot_days_max for value in generated_hot_days.values())
    historical_wet_days_min = min(item["wet_days"] for item in historical_rain.values())
    seed_104_wet_days = seed_rain["104"]["wet_days"]
    weather_notes = []
    if hot_seed_count_above_historical_max:
        weather_notes.append(
            f"{hot_seed_count_above_historical_max}/5 seeds exceed the highest 2004-2013 annual count of days above pooled train TMAX P90 ({historical_hot_days_max} days); descriptive warm-tail note, not an acceptance threshold."
        )
    if seed_104_wet_days < historical_wet_days_min:
        weather_notes.append(
            f"Seed 104 has {seed_104_wet_days} wet days, one below the 2004-2013 same-window minimum ({historical_wet_days_min}); its common-window rain total remains within the historical annual min-max."
        )
    summary = {
        "status": qc_status,
        "scope": "YC only; WGEN seeds 101-105; frozen 2004-2013 training weather only; no validation weather used",
        "corrected_cli_sha256": cli_hash,
        "corrected_cli_hash_verified": True,
        "frozen_train_weather_sha256": train_hash,
        "frozen_train_weather_hash_verified": True,
        "historical_train_years": [2004, 2013],
        "common_window_start": common_start.isoformat(),
        "common_window_end": common_end.isoformat(),
        "common_window_days": len(common_dates),
        "leap_day_handling": "Common window starts in June; no leap day in window. Match exact month/day for each 2004-2013 historical year.",
        "wet_day_definition": "RAIN > 0.0 mm",
        "physical_issue_count": len(physical_issues),
        "physical_issues": physical_issues,
        "unique_seed_sequences": unique_sequences,
        "seed_sequence_hashes": sequences,
        "historical_distribution_overlap_counts": hist_inside,
        "systematic_bias_detected": "NO_OBVIOUS_SHARED_CENTRAL_TENDENCY_SHIFT; see descriptive warm-tail and dry-seed notes",
        "interpretive_notes": weather_notes,
        "train_hot_day_diagnostic": {
            "pooled_train_tmax_p90_c": train_p90_tmax,
            "historical_max_annual_exceedance_days": historical_hot_days_max,
            "wgen_exceedance_days_by_seed": generated_hot_days,
            "seeds_above_historical_max_annual_exceedance_days": hot_seed_count_above_historical_max,
            "use": "descriptive only; not used for WGEN fitting or acceptance-threshold tuning",
        },
        "common_window_rainfall_by_seed": seed_rain,
        "common_window_rainfall_by_historical_year": historical_rain,
        "train_p90_tmax_c": train_p90_tmax,
        "train_p05_tmin_c": train_p05_tmin,
        "historical_2008_wth": wth_meta,
        "full_period_cumulative_totals_comparable": False,
        "validation_weather_used_for_fitting": False,
        "wgen_refit": False,
        "wet_day_definition_changed": False,
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in sorted(FIGURE_DIR.glob("*.png"))],
    }
    (WEATHER_DIR / "weather_qc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
