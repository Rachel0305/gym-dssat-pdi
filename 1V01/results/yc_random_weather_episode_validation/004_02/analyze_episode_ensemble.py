from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
EPISODES = ROOT / "episodes"
REPLAYS = ROOT / "reproducibility"
VALIDATION_WTH = PROJECT / "DSSAT_auto_validation" / "run_CNYC0802_DSSAT480_IC0_null_2000_2023"
TRAIN_CSV = PROJECT / "results" / "yc_weather_gapfill_finalize" / "yc_wgen_fitting_weather_2004_2013.csv"
CLI = PROJECT / "results" / "yc_wgen_cli_pilot" / "003_06_05_02" / "final" / "CNYC.CLI"
EXPECTED_CLI_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"
EXPECTED_TRAIN_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
SEEDS = list(range(1001, 1101))
REPLAY_SEEDS = [1001, 1025, 1050, 1075, 1100]
WEATHER_FIELDS = ("rain", "srad", "tmax", "tmin")
WET_THRESHOLD_MM = 0.0
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_daily(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {key: raw[key] for key in raw}
            row["date"] = date.fromisoformat(raw["date"])
            for field in WEATHER_FIELDS:
                row[field] = float(raw[field])
            rows.append(row)
    return rows


def read_training() -> dict[int, list[dict]]:
    output: dict[int, list[dict]] = defaultdict(list)
    with TRAIN_CSV.open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            day = date.fromisoformat(raw["DATE"])
            output[day.year].append({
                "date": day,
                "rain": float(raw["RAIN"]),
                "srad": float(raw["SRAD"]),
                "tmax": float(raw["TMAX"]),
                "tmin": float(raw["TMIN"]),
            })
    return dict(output)


def read_validation_year(year: int) -> list[dict]:
    path = VALIDATION_WTH / f"CNYC{year % 100:02d}01.WTH"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="ascii", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5 or not parts[0].isdigit() or len(parts[0]) != 7:
            continue
        year_day = int(parts[0])
        item_year, doy = divmod(year_day, 1000)
        day = date(item_year, 1, 1) + timedelta(days=doy - 1)
        rows.append({
            "date": day,
            "srad": float(parts[1]),
            "tmax": float(parts[2]),
            "tmin": float(parts[3]),
            "rain": float(parts[4]),
        })
    if not rows or any(row["date"].year != year for row in rows):
        raise ValueError(f"Invalid or empty validation WTH: {path}")
    return rows


def quantile(values: list[float], p: float) -> float | None:
    data = sorted(values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return data[low] + (data[high] - data[low]) * (position - low)


def describe(values: list[float | None]) -> dict:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not data:
        return {"n": 0, "mean": None, "sd": None, "cv_pct": None, "median": None,
                "min": None, "max": None, **{f"p{p:02d}": None for p in PERCENTILES}}
    mean = statistics.fmean(data)
    sd = statistics.stdev(data) if len(data) > 1 else 0.0
    return {
        "n": len(data), "mean": mean, "sd": sd,
        "cv_pct": (100.0 * sd / abs(mean)) if mean else None,
        "median": quantile(data, 50), "min": min(data), "max": max(data),
        **{f"p{p:02d}": quantile(data, p) for p in PERCENTILES if p != 50},
    }


def longest_run(flags: list[bool], target: bool) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag is target else 0
        best = max(best, current)
    return best


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    if not sx or not sy:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def lag1(values: list[float]) -> float | None:
    return pearson(values[:-1], values[1:]) if len(values) > 2 else None


def validate_daily(rows: list[dict]) -> list[str]:
    errors = []
    dates = [row["date"] for row in rows]
    if len(dates) != len(set(dates)):
        errors.append("duplicate_dates")
    if any(right - left != timedelta(days=1) for left, right in zip(dates, dates[1:])):
        errors.append("non_continuous_dates")
    for row in rows:
        for field in WEATHER_FIELDS:
            if not math.isfinite(row[field]):
                errors.append(f"non_finite_{field}")
        if row["rain"] < 0:
            errors.append("negative_rain")
        if row["srad"] < 0:
            errors.append("negative_srad")
        if row["tmax"] < row["tmin"]:
            errors.append("tmax_below_tmin")
    return sorted(set(errors))


def metrics_for(rows: list[dict]) -> dict:
    rain = [row["rain"] for row in rows]
    wet = [value > WET_THRESHOLD_MM for value in rain]
    wet_amounts = [value for value, is_wet in zip(rain, wet) if is_wet]
    return {
        "days": len(rows), "rain_total_mm": sum(rain), "wet_days": sum(wet),
        "rain_per_wet_day_mean_mm": statistics.fmean(wet_amounts) if wet_amounts else None,
        "rain_per_wet_day_median_mm": quantile(wet_amounts, 50),
        "max_daily_rain_mm": max(rain) if rain else None,
        "longest_dry_spell_days": longest_run(wet, False),
        "longest_wet_spell_days": longest_run(wet, True),
        "tmax_mean_c": statistics.fmean(row["tmax"] for row in rows) if rows else None,
        "tmin_mean_c": statistics.fmean(row["tmin"] for row in rows) if rows else None,
        "srad_mean_mj_m2_day": statistics.fmean(row["srad"] for row in rows) if rows else None,
    }


def group_summary(rows: list[dict], group_field: str, metrics: list[str], output_field: str = "group") -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[group_field]].append(row)
    result = []
    for group, items in grouped.items():
        for metric in metrics:
            stats = describe([item.get(metric) for item in items])
            result.append({output_field: group, "metric": metric, **stats})
    return result


def daily_metric_summary(groups: dict[str, list[list[dict]]], fields: list[str]) -> list[dict]:
    output = []
    for group, sequences in groups.items():
        for field in fields:
            series_stats: dict[str, list[float]] = defaultdict(list)
            for sequence in sequences:
                values = [row[field] for row in sequence]
                series_stats["mean"].append(statistics.fmean(values))
                series_stats["sd"].append(statistics.stdev(values) if len(values) > 1 else 0.0)
                series_stats["min"].append(min(values))
                series_stats["max"].append(max(values))
                for p in (5, 50, 95):
                    series_stats[f"p{p:02d}"].append(quantile(values, p))
                if field == "tmax":
                    series_stats["hot_tail_frequency_pct"].append(100.0 * sum(v > FIT_HOT_THRESHOLD for v in values) / len(values))
                elif field == "tmin":
                    series_stats["cold_tail_frequency_pct"].append(100.0 * sum(v < FIT_COLD_THRESHOLD for v in values) / len(values))
            for metric, values in series_stats.items():
                output.append({"group": group, "variable": field, "metric": metric, **describe(values)})
    return output


def normalized_weather_hash(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        values = [row["date"].isoformat()] + [format(row[field], ".17g") for field in WEATHER_FIELDS]
        digest.update((",".join(values) + "\n").encode("ascii"))
    return digest.hexdigest().upper()


def in_window(rows: list[dict], start: date, end: date) -> list[dict]:
    selected = [row for row in rows if start <= row["date"] <= end]
    expected = (end - start).days + 1
    if len(selected) != expected or selected[0]["date"] != start or selected[-1]["date"] != end:
        raise ValueError(f"Incomplete fixed-window coverage {start}..{end}: got {len(selected)}, expected {expected}")
    if validate_daily(selected):
        raise ValueError(f"Physical/date QC failed in fixed window: {validate_daily(selected)}")
    return selected


def observed_window(rows: list[dict], year: int, start: date, end: date) -> list[dict]:
    local_start = date(year, start.month, start.day)
    local_end = date(year, end.month, end.day)
    selected = [row for row in rows if local_start <= row["date"] <= local_end]
    expected = (local_end - local_start).days + 1
    if len(selected) != expected or selected[0]["date"] != local_start or selected[-1]["date"] != local_end:
        raise ValueError(f"Observed year {year} does not cover {local_start}..{local_end}")
    if validate_daily(selected):
        raise ValueError(f"Observed fixed-window date/physical QC failed in {year}: {validate_daily(selected)}")
    return selected


def self_tests() -> None:
    assert quantile([0.0, 10.0], 25) == 2.5
    assert longest_run([False, False, True, False], False) == 2
    sample = [{"date": date(2008, 6, day), "rain": float(day % 2), "srad": 10.0,
               "tmax": 30.0, "tmin": 20.0} for day in range(1, 5)]
    assert not validate_daily(sample)
    assert metrics_for(sample)["rain_total_mm"] == 2.0
    assert metrics_for(sample)["wet_days"] == 2
    assert math.isclose(pearson([1, 2, 3], [3, 2, 1]), -1.0, abs_tol=1e-12)
    toy_start, toy_end = date(2008, 6, 1), date(2008, 6, 4)
    assert len(in_window(sample, toy_start, toy_end)) == 4
    bad = sample + [sample[-1]]
    assert "duplicate_dates" in validate_daily(bad)
    monthly_total = sum(metrics_for([row for row in sample if row["date"].month == month])["rain_total_mm"] for month in (6,))
    assert monthly_total == metrics_for(sample)["rain_total_mm"]
    assert all(2004 <= year <= 2013 for year in range(2004, 2014))
    assert not (set(range(2004, 2014)) & set(range(2014, 2024)))


def main() -> None:
    global FIT_HOT_THRESHOLD, FIT_COLD_THRESHOLD
    self_tests()
    if not CLI.is_file():
        raise FileNotFoundError(CLI)
    cli_hash = hashlib.sha256(CLI.read_bytes()).hexdigest().upper()
    if cli_hash != EXPECTED_CLI_SHA256:
        raise RuntimeError("Frozen CNYC.CLI hash changed; refusing analysis")
    train_hash = hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest().upper()
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Frozen fitting weather hash changed; refusing analysis")

    episode_rows: dict[int, list[dict]] = {}
    episode_status: dict[int, dict] = {}
    generation_qc = []
    for seed in SEEDS:
        data_path = EPISODES / f"seed_{seed}.csv"
        status_path = EPISODES / f"seed_{seed}" / "crop_smoke_status.json"
        rows = read_daily(data_path)
        status = read_json(status_path)
        errors = validate_daily(rows)
        dates = [row["date"] for row in rows]
        expected_seed = status.get("weather_seed") == seed
        runtime_ok = status.get("runtime_status") == "PASS"
        wgen_ok = status.get("random_weather") is True and status.get("runtime_wther") == "W"
        cli_ok = status.get("runtime_cli_sha256") == EXPECTED_CLI_SHA256
        if len(rows) != status.get("runtime_steps"):
            errors.append("row_count_steps_mismatch")
        if not expected_seed:
            errors.append("weather_seed_mismatch")
        if not runtime_ok:
            errors.append("runtime_not_pass")
        if not wgen_ok:
            errors.append("wgen_not_confirmed")
        if not cli_ok:
            errors.append("cli_hash_mismatch")
        episode_rows[seed] = rows
        episode_status[seed] = status
        generation_qc.append({
            "seed": seed, "runtime_status": status.get("runtime_status"), "weather_seed": status.get("weather_seed"),
            "random_weather": status.get("random_weather"), "runtime_wther": status.get("runtime_wther"),
            "cli_sha256": status.get("runtime_cli_sha256"), "n_rows": len(rows),
            "simulation_start": dates[0].isoformat() if dates else "", "simulation_end": dates[-1].isoformat() if dates else "",
            "episode_days": len(rows), "planting_date": status.get("planting_date"),
            "anthesis_date": status.get("anthesis_date"), "maturity_date": status.get("maturity_date"),
            "finite_physical_continuous_qc": "PASS" if not errors else "FAIL",
            "qc_errors": ";".join(sorted(set(errors))),
        })
    write_csv(ROOT / "episode_generation_qc.csv", generation_qc)

    expected_start = date(2008, 6, 1)
    preferred_end = date(2008, 9, 24)
    earliest_end = min(rows[-1]["date"] for rows in episode_rows.values())
    common_end = min(preferred_end, earliest_end)
    common_start = expected_start
    common_days = (common_end - common_start).days + 1
    if common_days < 1:
        raise RuntimeError("No common fixed window among 100 weather episodes")
    synthetic_fixed = {seed: in_window(rows, common_start, common_end) for seed, rows in episode_rows.items()}

    training_all = read_training()
    if sorted(training_all) != list(range(2004, 2014)):
        raise RuntimeError(f"Unexpected fitting years in frozen source: {sorted(training_all)}")
    validation_all = {}
    validation_manifest = []
    for year in range(2014, 2024):
        source_path = VALIDATION_WTH / f"CNYC{year % 100:02d}01.WTH"
        validation_all[year] = read_validation_year(year)
        validation_manifest.append({"year": year, "source_file": str(source_path.relative_to(PROJECT)),
                                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest().upper(),
                                    "rows": len(validation_all[year]),
                                    "first_date": validation_all[year][0]["date"].isoformat(),
                                    "last_date": validation_all[year][-1]["date"].isoformat(),
                                    "usage": "comparison_only_not_used_for_WGEN_fitting_or_seed_selection"})
    (ROOT / "validation").mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / "validation" / "observed_wth_source_manifest.csv", validation_manifest)
    if set(training_all) & set(validation_all):
        raise RuntimeError("Fitting and independent-comparison year sets overlap")
    generation_provenance = read_json(ROOT / "generation_provenance.json")
    leakage_check_pass = (generation_provenance.get("validation_weather_used_for_generation") is False
                          and generation_provenance.get("wgen_refit") is False
                          and set(training_all) == set(range(2004, 2014))
                          and set(validation_all) == set(range(2014, 2024)))
    if not leakage_check_pass:
        raise RuntimeError("Fitting/validation separation provenance check failed")
    observed_fixed: dict[str, dict[int, list[dict]]] = {"fitting_observed": {}, "independent_observed": {}}
    for year, yearly in training_all.items():
        observed_fixed["fitting_observed"][year] = observed_window(yearly, year, common_start, common_end)
    for year, yearly in validation_all.items():
        observed_fixed["independent_observed"][year] = observed_window(yearly, year, common_start, common_end)

    groups: dict[str, list[list[dict]]] = {
        "fitting_observed": list(observed_fixed["fitting_observed"].values()),
        "independent_observed": list(observed_fixed["independent_observed"].values()),
        "synthetic_ensemble": [synthetic_fixed[seed] for seed in SEEDS],
    }
    fit_daily = [row for sequence in groups["fitting_observed"] for row in sequence]
    FIT_HOT_THRESHOLD = quantile([row["tmax"] for row in fit_daily], 95)
    FIT_COLD_THRESHOLD = quantile([row["tmin"] for row in fit_daily], 5)

    fixed_by_seed = []
    for seed in SEEDS:
        fixed_by_seed.append({"seed": seed, "window_start": common_start.isoformat(), "window_end": common_end.isoformat(),
                              "window_days": common_days, **metrics_for(synthetic_fixed[seed])})
    write_csv(ROOT / "fixed_window_weather_by_seed.csv", fixed_by_seed)

    full_season = []
    for seed in SEEDS:
        rows = episode_rows[seed]
        status = episode_status[seed]
        full_season.append({
            "seed": seed, "simulation_start": rows[0]["date"].isoformat(), "simulation_end": rows[-1]["date"].isoformat(),
            "episode_days": len(rows), "planting_date": status.get("planting_date"), "anthesis_date": status.get("anthesis_date"),
            "maturity_date": status.get("maturity_date"), "season_total_rain_mm": sum(row["rain"] for row in rows),
            "season_wet_days": sum(row["rain"] > WET_THRESHOLD_MM for row in rows),
            "season_tmax_mean_c": statistics.fmean(row["tmax"] for row in rows),
            "season_tmin_mean_c": statistics.fmean(row["tmin"] for row in rows),
            "season_srad_mean_mj_m2_day": statistics.fmean(row["srad"] for row in rows),
        })
    write_csv(ROOT / "full_crop_season_weather_by_seed.csv", full_season)

    rain_metrics = ["rain_total_mm", "wet_days", "rain_per_wet_day_mean_mm", "rain_per_wet_day_median_mm",
                    "max_daily_rain_mm", "longest_dry_spell_days", "longest_wet_spell_days"]
    rain_units = []
    for group, by_year in observed_fixed.items():
        for year, rows in by_year.items():
            rain_units.append({"group": group, "unit": str(year), **metrics_for(rows)})
    for row in fixed_by_seed:
        rain_units.append({"group": "synthetic_ensemble", "unit": str(row["seed"]), **row})
    for group_name, sequences in groups.items():
        for sequence in sequences:
            total = metrics_for(sequence)["rain_total_mm"]
            monthly_total = sum(sum(row["rain"] for row in sequence if row["date"].month == month)
                                for month in (6, 7, 8, 9))
            if not math.isclose(total, monthly_total, rel_tol=0.0, abs_tol=1e-9):
                raise AssertionError(f"Monthly rainfall does not aggregate to fixed-window total for {group_name}")
    rainfall_summary = group_summary(rain_units, "group", rain_metrics)
    write_csv(ROOT / "rainfall_validation_summary.csv", rainfall_summary)

    monthly_rows = []
    for group, sequences in groups.items():
        for month in (6, 7, 8, 9):
            month_metrics = []
            for index, sequence in enumerate(sequences):
                subset = [row for row in sequence if row["date"].month == month]
                if not subset:
                    continue
                values = metrics_for(subset)
                month_metrics.append({"month": month, **values,
                                      "rain_intensity_mm_per_wet_day": values["rain_per_wet_day_mean_mm"]})
            for metric in ("rain_total_mm", "wet_days", "rain_intensity_mm_per_wet_day", "max_daily_rain_mm"):
                monthly_rows.append({"group": group, "month": month, "metric": metric,
                                     **describe([item.get(metric) for item in month_metrics])})
    write_csv(ROOT / "monthly_rainfall_validation.csv", monthly_rows)

    temperature_rows = daily_metric_summary(groups, ["tmax", "tmin"])
    write_csv(ROOT / "temperature_validation_summary.csv", temperature_rows)
    srad_rows = [{**row, "month": ""} for row in daily_metric_summary(groups, ["srad"])]
    for group, sequences in groups.items():
        for month in (6, 7, 8, 9):
            subsets = [[row for row in sequence if row["date"].month == month] for sequence in sequences]
            for metric in ("daily_mean", "daily_sd", "daily_p05", "daily_p50", "daily_p95", "daily_min", "daily_max"):
                values = []
                for subset in subsets:
                    daily = [row["srad"] for row in subset]
                    if not daily:
                        continue
                    if metric == "daily_mean":
                        values.append(statistics.fmean(daily))
                    elif metric == "daily_sd":
                        values.append(statistics.stdev(daily) if len(daily) > 1 else 0.0)
                    elif metric == "daily_min":
                        values.append(min(daily))
                    elif metric == "daily_max":
                        values.append(max(daily))
                    else:
                        values.append(quantile(daily, int(metric[-2:])))
                srad_rows.append({"group": group, "variable": "srad_monthly", "metric": metric,
                                  "month": month, **describe(values)})
    write_csv(ROOT / "srad_validation_summary.csv", srad_rows)

    cross_pairs = [
        ("rain_occurrence", "srad", "RAIN_occurrence_vs_SRAD"),
        ("rain_occurrence", "tmax", "RAIN_occurrence_vs_TMAX"),
        ("tmax", "tmin", "TMAX_vs_TMIN"),
        ("tmax", "srad", "TMAX_vs_SRAD"),
    ]
    cross_rows = []
    serial_rows = []
    for group, sequences in groups.items():
        for left, right, label in cross_pairs:
            values = []
            for sequence in sequences:
                x = [float(row["rain"] > WET_THRESHOLD_MM) if left == "rain_occurrence" else row[left] for row in sequence]
                y = [float(row["rain"] > WET_THRESHOLD_MM) if right == "rain_occurrence" else row[right] for row in sequence]
                corr = pearson(x, y)
                if corr is not None:
                    values.append(corr)
            cross_rows.append({"group": group, "relationship": label, **describe(values)})
        for field, label in (("tmax", "TMAX"), ("tmin", "TMIN"), ("srad", "SRAD"), ("rain", "wet_dry_occurrence")):
            values = []
            for sequence in sequences:
                series = [float(row["rain"] > WET_THRESHOLD_MM) if field == "rain" else row[field] for row in sequence]
                value = lag1(series)
                if value is not None:
                    values.append(value)
            serial_rows.append({"group": group, "variable": label, "lag": 1, **describe(values)})
    write_csv(ROOT / "cross_correlation_validation.csv", cross_rows)
    write_csv(ROOT / "serial_correlation_validation.csv", serial_rows)

    repro_rows = []
    for seed in REPLAY_SEEDS:
        original = episode_rows[seed]
        replay_path = REPLAYS / f"seed_{seed}_repeat.csv"
        replay = read_daily(replay_path)
        replay_status = read_json(REPLAYS / "runs" / f"seed_{seed}_repeat" / "crop_smoke_status.json")
        original_signature = [(row["date"], *(row[field] for field in WEATHER_FIELDS)) for row in original]
        replay_signature = [(row["date"], *(row[field] for field in WEATHER_FIELDS)) for row in replay]
        mismatch = sum(left != right for left, right in zip(original_signature, replay_signature)) + abs(len(original_signature) - len(replay_signature))
        repro_rows.append({"seed": seed, "original_days": len(original), "replay_days": len(replay),
                           "original_weather_sha256": normalized_weather_hash(original),
                           "replay_weather_sha256": normalized_weather_hash(replay),
                           "daily_weather_mismatch_count": mismatch,
                           "replay_runtime_status": replay_status.get("runtime_status"),
                           "replay_wther": replay_status.get("runtime_wther"),
                           "replay_weather_seed": replay_status.get("weather_seed"),
                           "same_seed_identical": mismatch == 0 and len(original_signature) == len(replay_signature)
                               and replay_status.get("runtime_status") == "PASS"
                               and replay_status.get("runtime_wther") == "W"
                               and replay_status.get("weather_seed") == seed})
    write_csv(ROOT / "weather_seed_reproducibility.csv", repro_rows)

    first_seed_by_hash: dict[str, int] = {}
    diversity_rows = []
    for seed in SEEDS:
        sequence_hash = normalized_weather_hash(episode_rows[seed])
        duplicate_of = first_seed_by_hash.get(sequence_hash)
        if duplicate_of is None:
            first_seed_by_hash[sequence_hash] = seed
        diversity_rows.append({"seed": seed, "episode_days": len(episode_rows[seed]),
                               "weather_sequence_sha256": sequence_hash,
                               "unique_full_sequence": duplicate_of is None,
                               "duplicate_of_seed": duplicate_of or ""})
    write_csv(ROOT / "weather_seed_diversity.csv", diversity_rows)

    monthly_fit = {row["month"]: row["mean"] for row in monthly_rows if row["group"] == "fitting_observed" and row["metric"] == "rain_total_mm"}
    monthly_syn = {row["month"]: row["mean"] for row in monthly_rows if row["group"] == "synthetic_ensemble" and row["metric"] == "rain_total_mm"}
    month_corr = pearson([monthly_fit[month] for month in (6, 7, 8, 9)], [monthly_syn[month] for month in (6, 7, 8, 9)])
    rain_mean = {row["metric"]: row["mean"] for row in rainfall_summary if row["group"] == "fitting_observed"}
    syn_rain_mean = {row["metric"]: row["mean"] for row in rainfall_summary if row["group"] == "synthetic_ensemble"}
    fit_rain_sd = {row["metric"]: row["sd"] for row in rainfall_summary if row["group"] == "fitting_observed"}
    rain_z = {metric: ((syn_rain_mean[metric] - rain_mean[metric]) / fit_rain_sd[metric]
                       if fit_rain_sd.get(metric) not in (None, 0) else None)
              for metric in ("rain_total_mm", "wet_days")}

    def summary_lookup(rows: list[dict], group: str, variable: str, metric: str) -> dict:
        return next(row for row in rows if row["group"] == group and row.get("variable") == variable and row["metric"] == metric)

    temp_diagnostics = {}
    for variable in ("tmax", "tmin"):
        fit = summary_lookup(temperature_rows, "fitting_observed", variable, "mean")
        syn = summary_lookup(temperature_rows, "synthetic_ensemble", variable, "mean")
        fit_sd = summary_lookup(temperature_rows, "fitting_observed", variable, "sd")
        syn_sd = summary_lookup(temperature_rows, "synthetic_ensemble", variable, "sd")
        temp_diagnostics[variable] = {
            "fitting_mean_c": fit["mean"], "synthetic_mean_c": syn["mean"],
            "mean_bias_c": syn["mean"] - fit["mean"],
            "mean_bias_in_fitting_year_sd": (syn["mean"] - fit["mean"]) / fit["sd"] if fit["sd"] else None,
            "synthetic_to_fitting_daily_sd_ratio": syn_sd["mean"] / fit_sd["mean"] if fit_sd["mean"] else None,
        }
    fit_srad = summary_lookup(srad_rows, "fitting_observed", "srad", "mean")
    syn_srad = summary_lookup(srad_rows, "synthetic_ensemble", "srad", "mean")
    fit_srad_sd = summary_lookup(srad_rows, "fitting_observed", "srad", "sd")
    syn_srad_sd = summary_lookup(srad_rows, "synthetic_ensemble", "srad", "sd")
    srad_bias_z = (syn_srad["mean"] - fit_srad["mean"]) / fit_srad["sd"] if fit_srad["sd"] else None
    srad_sd_ratio = syn_srad_sd["mean"] / fit_srad_sd["mean"] if fit_srad_sd["mean"] else None

    dependence_diffs = []
    for label in (item[2] for item in cross_pairs):
        fit = next(row for row in cross_rows if row["group"] == "fitting_observed" and row["relationship"] == label)
        syn = next(row for row in cross_rows if row["group"] == "synthetic_ensemble" and row["relationship"] == label)
        if fit["mean"] is not None and syn["mean"] is not None:
            dependence_diffs.append(abs(fit["mean"] - syn["mean"]))
    serial_diffs = []
    for variable in ("TMAX", "TMIN", "SRAD", "wet_dry_occurrence"):
        fit = next(row for row in serial_rows if row["group"] == "fitting_observed" and row["variable"] == variable)
        syn = next(row for row in serial_rows if row["group"] == "synthetic_ensemble" and row["variable"] == variable)
        if fit["mean"] is not None and syn["mean"] is not None:
            serial_diffs.append(abs(fit["mean"] - syn["mean"]))

    generation_pass = len(generation_qc) == 100 and all(row["finite_physical_continuous_qc"] == "PASS" for row in generation_qc)
    reproducibility_pass = len(repro_rows) == 5 and all(row["same_seed_identical"] for row in repro_rows)
    diversity_pass = len(first_seed_by_hash) == 100
    rain_severe = any(value is None or abs(value) > 2.0 for value in rain_z.values()) or month_corr is None or month_corr < 0.50
    rain_notes = any(value is None or abs(value) > 1.0 for value in rain_z.values()) or month_corr is None or month_corr < 0.80
    temp_biases = [abs(item["mean_bias_in_fitting_year_sd"]) for item in temp_diagnostics.values() if item["mean_bias_in_fitting_year_sd"] is not None]
    temp_ratios = [item["synthetic_to_fitting_daily_sd_ratio"] for item in temp_diagnostics.values() if item["synthetic_to_fitting_daily_sd_ratio"] is not None]
    temp_severe = len(temp_biases) < 2 or max(temp_biases, default=99) > 2.0 or any(not 0.35 <= ratio <= 2.0 for ratio in temp_ratios)
    temp_notes = max(temp_biases, default=99) > 1.0 or any(not 0.60 <= ratio <= 1.60 for ratio in temp_ratios)
    srad_severe = srad_bias_z is None or abs(srad_bias_z) > 2.0 or srad_sd_ratio is None or not 0.35 <= srad_sd_ratio <= 2.0
    srad_notes = srad_bias_z is None or abs(srad_bias_z) > 1.0 or srad_sd_ratio is None or not 0.60 <= srad_sd_ratio <= 1.60
    dependence_severe = not dependence_diffs or max(dependence_diffs) > 0.60 or not serial_diffs or max(serial_diffs) > 0.60
    dependence_notes = not dependence_diffs or max(dependence_diffs) > 0.30 or not serial_diffs or max(serial_diffs) > 0.30

    def status(severe: bool, notes: bool, hard_pass: bool = True) -> str:
        if not hard_pass or severe:
            return "FAIL"
        return "PASS_WITH_NOTES" if notes else "PASS"

    gates = {
        "generation_status": status(False, False, generation_pass),
        "reproducibility_status": status(False, False, reproducibility_pass),
        "diversity_status": status(False, False, diversity_pass),
        "rainfall_status": status(rain_severe, rain_notes),
        "temperature_status": status(temp_severe, temp_notes),
        "srad_status": status(srad_severe, srad_notes),
        "dependence_structure_status": status(dependence_severe, dependence_notes),
    }
    final_fail = any(value == "FAIL" for value in gates.values())
    final_notes = any(value == "PASS_WITH_NOTES" for value in gates.values())
    overall = "FAIL" if final_fail else ("PASS_WITH_NOTES" if final_notes else "PASS")

    fitting_totals = [row["rain_total_mm"] for row in rain_units if row["group"] == "fitting_observed"]
    validation_totals = [row["rain_total_mm"] for row in rain_units if row["group"] == "independent_observed"]
    synthetic_totals = [row["rain_total_mm"] for row in rain_units if row["group"] == "synthetic_ensemble"]
    bias_detected = bool(rain_severe or temp_severe or srad_severe or dependence_severe)
    summary = {
        "reference_method": "Wang_2025_style_episode_weather",
        "fitting_period": "2004-2013",
        "independent_validation_period": "2014-2023 (comparison only; prior project audit did not certify a globally pristine holdout)",
        "cnyc_cli_sha256": cli_hash,
        "synthetic_episode_count": len(SEEDS), "weather_seed_range": "1001-1100",
        "fixed_common_window": {"start": common_start.isoformat(), "end": common_end.isoformat(), "days": common_days,
                                 "selection_reason": "earliest end among all 100 episodes, capped at the preferred 2008-09-24"},
        "generation_status": gates["generation_status"],
        "episodes_passed": sum(row["finite_physical_continuous_qc"] == "PASS" for row in generation_qc),
        "episodes_failed": sum(row["finite_physical_continuous_qc"] != "PASS" for row in generation_qc),
        "same_seed_reproducibility": {"status": gates["reproducibility_status"], "passed": sum(row["same_seed_identical"] for row in repro_rows), "total": len(repro_rows)},
        "different_seed_diversity": {"status": gates["diversity_status"], "unique_full_sequences": len(first_seed_by_hash), "total": len(SEEDS)},
        "rainfall_means_mm": {"fitting": statistics.fmean(fitting_totals), "independent_validation": statistics.fmean(validation_totals),
                              "synthetic": statistics.fmean(synthetic_totals)},
        "rainfall_ranges_mm": {"fitting": [min(fitting_totals), max(fitting_totals)],
                               "independent_validation": [min(validation_totals), max(validation_totals)],
                               "synthetic": [min(synthetic_totals), max(synthetic_totals)]},
        "rain_mean_bias_in_fitting_interannual_sd": rain_z,
        "monthly_rainfall_profile_pearson": month_corr,
        "temperature_diagnostics": temp_diagnostics,
        "temperature_tail_thresholds_c": {"hot_tail_tmax_gt_fitting_daily_p95": FIT_HOT_THRESHOLD,
                                           "cold_tail_tmin_lt_fitting_daily_p05": FIT_COLD_THRESHOLD},
        "srad_diagnostics": {"fitting_mean": fit_srad["mean"], "synthetic_mean": syn_srad["mean"],
                             "mean_bias_in_fitting_year_sd": srad_bias_z, "synthetic_to_fitting_daily_sd_ratio": srad_sd_ratio},
        "systematic_bias_detected": bias_detected,
        "gate_status": gates,
        "random_weather_episode_quality_status": overall,
        "yc_random_weather_ready_for_ppo_pilot": "YES" if overall != "FAIL" else "NO",
        "full_year_wgen_validation": "DEFERRED_OPTIONAL_ADDITIONAL_VALIDATION",
        "full_year_wgen_is_ppo_blocker": "NO",
        "ppo_run": "NO", "runtime_modified": "NO", "cnyc_cli_modified": "NO", "wgen_refit": "NO",
        "validation_data_used_for_fitting": "NO", "ppt_created": "NO",
        "validation_data_note": "2014-2023 comparison was not used by the WGEN fitting/generation path; existing project audit did not certify these files as globally untouched holdout data.",
        "tests_status": {"self_tests": "PASS", "fixed_window_extraction": "PASS", "physical_qc": "PASS" if generation_pass else "FAIL",
                         "aggregation_consistency": "PASS", "validation_leakage_check": "PASS" if leakage_check_pass else "FAIL"},
        "decision_rules": {
            "rainfall": "descriptive screen: annual rain-total/wet-day synthetic mean shift <=2 fitting interannual SD; monthly profile correlation >=0.50",
            "temperature": "descriptive screen: seasonal mean shift <=2 fitting year SD; daily SD ratio 0.35-2.0",
            "srad": "descriptive screen: seasonal mean shift <=2 fitting year SD; daily SD ratio 0.35-2.0",
            "dependence": "descriptive screen: maximum group mean correlation/lag-1 difference <=0.60",
            "interpretation": "heuristic audit flags, not hypothesis tests or tuned generator thresholds; PASS_WITH_NOTES used for moderate departures",
        },
    }
    write_json(ROOT / "summary.json", summary)

    make_figures(groups, full_season, monthly_rows)
    make_report(summary, rainfall_summary, monthly_rows, temperature_rows, srad_rows, cross_rows, serial_rows, full_season)
    print(json.dumps({"summary": summary, "self_tests": "PASS", "output_directory": str(ROOT)}, ensure_ascii=False, indent=2))


def make_figures(groups: dict[str, list[list[dict]]], full_season: list[dict], monthly_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"fitting_observed": "#53636E", "independent_observed": "#198F83", "synthetic_ensemble": "#D9822B"}
    labels = {"fitting_observed": "Fitting observed (2004-2013)",
              "independent_observed": "Comparison observed (2014-2023)", "synthetic_ensemble": "WGEN synthetic (100 episodes)"}

    def boxplot_metric(name: str, metric: str, ylabel: str) -> None:
        arrays = []
        for group in groups:
            if metric in ("rain_total_mm", "wet_days", "longest_dry_spell_days"):
                vals = [metrics_for(seq)[metric] for seq in groups[group]]
            else:
                vals = [row[metric] for seq in groups[group] for row in seq]
            arrays.append(vals)
        fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        boxes = ax.boxplot(arrays, patch_artist=True, showfliers=False, tick_labels=[labels[group].replace(" (", "\n(") for group in groups])
        for patch, group in zip(boxes["boxes"], groups):
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.78)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.24)
        fig.savefig(figure_dir / f"{name.lower().replace(' ', '_')}.png", dpi=180)
        plt.close(fig)

    boxplot_metric("Fixed-window rainfall", "rain_total_mm", "Rainfall (mm per window)")
    boxplot_metric("Wet days", "wet_days", "Days with RAIN > 0 mm")
    boxplot_metric("Longest dry spell", "longest_dry_spell_days", "Consecutive dry days")

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for group in groups:
        means, sds = [], []
        for month in (6, 7, 8, 9):
            row = next(item for item in monthly_rows if item["group"] == group and item["month"] == month and item["metric"] == "rain_total_mm")
            means.append(row["mean"])
            sds.append(row["sd"])
        ax.errorbar((6, 7, 8, 9), means, yerr=sds, marker="o", capsize=3, color=colors[group], label=labels[group])
    ax.set_xticks((6, 7, 8, 9), ("June", "July", "August", "September*"))
    ax.set_ylabel("Monthly rainfall (mm; mean +/- SD across years/episodes)")
    ax.set_title("Monthly rainfall profile")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.01, -0.19, "*September is truncated at the shared window end.", transform=ax.transAxes, fontsize=8)
    fig.savefig(figure_dir / "monthly_rainfall_climatology.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    boxplot_metric("TMAX distribution", "tmax", "Daily TMAX (C)")
    boxplot_metric("TMIN distribution", "tmin", "Daily TMIN (C)")
    boxplot_metric("SRAD distribution", "srad", "Daily SRAD (MJ m-2 d-1)")

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    lengths = [row["episode_days"] for row in full_season]
    ax.hist(lengths, bins=range(min(lengths), max(lengths) + 2), color=colors["synthetic_ensemble"], edgecolor="white", align="left")
    ax.axvline(statistics.fmean(lengths), color="#198F83", linestyle="--", label=f"Mean = {statistics.fmean(lengths):.1f} d")
    ax.set_xlabel("Full crop-season episode length (days)")
    ax.set_ylabel("Episode count")
    ax.set_title("WGEN episode lengths")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.24)
    fig.savefig(figure_dir / "episode_length_distribution.png", dpi=180)
    plt.close(fig)


def make_report(summary: dict, rainfall: list[dict], monthly: list[dict], temperature: list[dict],
                srad: list[dict], cross: list[dict], serial: list[dict], full_season: list[dict]) -> None:
    def find(rows: list[dict], group: str, metric: str, **keys) -> dict:
        return next(row for row in rows if row.get("group") == group and row.get("metric") == metric and
                    all(row.get(key) == value for key, value in keys.items()))

    def fmt(value, digits=2):
        return "NA" if value is None else f"{value:.{digits}f}"

    gates = summary["gate_status"]
    rain_fit = find(rainfall, "fitting_observed", "rain_total_mm")
    rain_val = find(rainfall, "independent_observed", "rain_total_mm")
    rain_syn = find(rainfall, "synthetic_ensemble", "rain_total_mm")
    wet_fit = find(rainfall, "fitting_observed", "wet_days")
    wet_syn = find(rainfall, "synthetic_ensemble", "wet_days")
    mean_seasons = statistics.fmean(row["episode_days"] for row in full_season)
    min_season = min(row["episode_days"] for row in full_season)
    max_season = max(row["episode_days"] for row in full_season)
    temp_text = "；".join(f"{name.upper()}偏差 {fmt(data['mean_bias_c'])} C，按 fitting 年际 SD 标准化 {fmt(data['mean_bias_in_fitting_year_sd'])}，日 SD 比值 {fmt(data['synthetic_to_fitting_daily_sd_ratio'])}"
                        for name, data in summary["temperature_diagnostics"].items())
    srad_diag = summary["srad_diagnostics"]
    cross_text = "；".join(f"{row['relationship']}：{fmt(row['mean'])} (n={row['n']})"
                         for row in cross if row["group"] in ("fitting_observed", "synthetic_ensemble") and row["group"] == "synthetic_ensemble")
    serial_text = "；".join(f"{row['variable']}：{fmt(row['mean'])} (n={row['n']})"
                          for row in serial if row["group"] == "synthetic_ensemble")
    monthly_fit = [find(monthly, "fitting_observed", "rain_total_mm", month=m)["mean"] for m in (6, 7, 8, 9)]
    monthly_syn = [find(monthly, "synthetic_ensemble", "rain_total_mm", month=m)["mean"] for m in (6, 7, 8, 9)]
    report = f"""# YC WGEN 随机生长季天气 ensemble 验证

## 1. 为什么验证 episode-level 天气

本轮检验 Gym-DSSAT episode 中 DSSAT WGEN 实际提供给策略的随机生长季天气，不把未验证的 365/366 天导出路径作为前置门槛。episode 在作物成熟时终止，故不同 seed 的季长不同；气候主比较统一使用 **{summary['fixed_common_window']['start']} 至 {summary['fixed_common_window']['end']}（{summary['fixed_common_window']['days']} 天）**，完整作物季只作暴露量辅助描述。

## 2. 与 Wang et al. 2025 的关系

采用“每个 RL episode 经 DSSAT WGEN 产生随机天气 realization”的方法学思路。Wang 等的论文使用不同地点、策略和实验设定，本工作不是逐项复现。参考：[Wang et al. (2025), AgriEngineering 7, 252](https://doi.org/10.3390/agriengineering7080252)。

## 3. 冻结的 WGEN 输入

- YC fitting weather：`results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv`，固定 hash 为 `4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`。
- 正式 CNYC.CLI：`results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI`，SHA256 `{summary['cnyc_cli_sha256']}`。
- 同一 YC FileX treatment 1、土壤、品种与管理；WTHER=W；random_weather=True；降雨湿日阈值为 RAIN > 0.0 mm。
- 未修改运行时、DSSAT binary、reward/action/observation、CLI 或拟合参数；未重新拟合 WGEN，也未运行 PPO。

## 4. Seed 设计

按预先设定运行完整 seed 1001–1100，共 100 个 episode；全部保留。预指定 seed 1001、1025、1050、1075、1100 各额外重跑一次。ppo_seed 不适用。

## 5. 固定共同窗口

优先窗口为 2008-06-01 至 2008-09-24，但 seed 中最早结束的是 {min(row['episode_days'] for row in full_season)} 天的 episode（{min(row['simulation_end'] for row in full_season)}）；因此按“100 个 synthetic episode 全部覆盖”的规则将窗口终点收窄为 **{summary['fixed_common_window']['end']}**。三组观察与 synthetic 均裁取逐日完整覆盖的相同月日。9 月为截断月份，不外推未覆盖日期。

## 6. Generation QC

结果：**{summary['episodes_passed']}/100 PASS，{summary['episodes_failed']} FAIL**。逐集检查 DSSAT 成功、WGEN/random-weather 标志、weather_seed 注入、固定 CLI hash、状态行数、日期连续无重复、天气有限值、RAIN/SRAD 非负、TMAX >= TMIN。明细见 `episode_generation_qc.csv`。

## 7. 可复现性

固定 seed 重跑结果：**{summary['same_seed_reproducibility']['passed']}/{summary['same_seed_reproducibility']['total']} 逐日天气完全一致**。比较日期和 RAIN/SRAD/TMAX/TMIN 四列；校验表 `weather_seed_reproducibility.csv`。

## 8. 多样性

全生长季天气序列哈希共有 **{summary['different_seed_diversity']['unique_full_sequences']}/100 个唯一序列**。所有 seed 均计入，未基于天气观感筛选。逐 seed 哈希见 `weather_seed_diversity.csv`。

## 9. 降雨验证

固定窗口年/episode 累积降雨均值：fitting **{fmt(rain_fit['mean'])} mm**（10 年 SD {fmt(rain_fit['sd'])}，范围 {fmt(rain_fit['min'])}–{fmt(rain_fit['max'])}）；independent comparison **{fmt(rain_val['mean'])} mm**（10 年，范围 {fmt(rain_val['min'])}–{fmt(rain_val['max'])}）；synthetic **{fmt(rain_syn['mean'])} mm**（100 集，范围 {fmt(rain_syn['min'])}–{fmt(rain_syn['max'])}）。湿日均值 fitting/synthetic 分别为 {fmt(wet_fit['mean'])}/{fmt(wet_syn['mean'])} 天。降雨 gate：**{gates['rainfall_status']}**。该 gate 同时参考 rain total/wet day 的 fitting 年际 SD 标准化偏差及月降雨型相关，规则写入 `summary.json`。

## 10. 温度验证

{temp_text}。热尾阈值为 fitting 日值 TMAX P95={fmt(summary['temperature_tail_thresholds_c']['hot_tail_tmax_gt_fitting_daily_p95'])} C，冷尾阈值为 fitting 日值 TMIN P05={fmt(summary['temperature_tail_thresholds_c']['cold_tail_tmin_lt_fitting_daily_p05'])} C；阈值只作描述，不参与调参（频率及分位数见 `temperature_validation_summary.csv`）。温度 gate：**{gates['temperature_status']}**。

## 11. SRAD 验证

fitting 与 synthetic 的生长季日平均 SRAD 分别为 {fmt(srad_diag['fitting_mean'])} 与 {fmt(srad_diag['synthetic_mean'])} MJ m-2 d-1；偏差为 fitting 年际 SD 的 {fmt(srad_diag['mean_bias_in_fitting_year_sd'])} 倍，日 SD 比值 {fmt(srad_diag['synthetic_to_fitting_daily_sd_ratio'])}。按月与分位数明细见 `srad_validation_summary.csv`。SRAD gate：**{gates['srad_status']}**。

## 12. 相关结构

逐年/逐 episode 内分别计算 RAIN occurrence-SRAD、RAIN occurrence-TMAX、TMAX-TMIN、TMAX-SRAD 相关；lag-1 序列相关分别覆盖 TMAX、TMIN、SRAD 和 wet/dry 指示。Synthetic 相关：{cross_text}。Synthetic lag-1：{serial_text}。dependence gate：**{gates['dependence_structure_status']}**。观察年只有 10 年，结果是描述性比较，不以单一 p 值定性。

## 13. 完整 crop-season 辅助统计

完整 episode 长度为 {min_season}–{max_season} 天，均值 {mean_seasons:.1f} 天；每集另记录各自季节总雨量、湿日数、TMAX/TMIN/SRAD 均值及播种、开花、成熟日期，见 `full_crop_season_weather_by_seed.csv`。这些长度不一的累计量不用于 seed 间主气候比较。

## 14. Fitting 与 independent observed 对照

Fitting 数据仅为 2004–2013，来自冻结的参数拟合天气文件。2014–2023 WTH 只用于本轮独立于 WGEN fitting 的比较和报告，不用于拟合、阈值调节、seed 筛选或删除 episode。需限定：既有项目审计未将这些 2014–2023 文件认证为项目全局范围内完全未触碰的 pristine holdout；因此这里称“independent comparison”，不扩大声称。

## 15. 局限

历史样本每组只有 10 个年份，不能据此证明完整气候分布、年际尾部或未来气候已被充分覆盖；synthetic episode 来自同一 WGEN 参数化，100 个 realization 不是 100 个独立历史年份。比较只覆盖共同生长季窗口，不代表全年气候验证。WGEN 合理产生偏干、偏湿、偏热或偏凉 episode；单个 realization 偏离观察范围不自动判失败。

## 16. 最终判定

总体 episode weather quality：**{summary['random_weather_episode_quality_status']}**。系统性偏差筛查：**{'检测到需阻断问题' if summary['systematic_bias_detected'] else '未发现按本轮预设描述性筛查规则需阻断的系统性偏差'}**。各 gate：generation={gates['generation_status']}；reproducibility={gates['reproducibility_status']}；diversity={gates['diversity_status']}；rainfall={gates['rainfall_status']}；temperature={gates['temperature_status']}；SRAD={gates['srad_status']}；dependence={gates['dependence_structure_status']}。

## 17. PPO readiness

`yc_random_weather_ready_for_ppo_pilot`：**{summary['yc_random_weather_ready_for_ppo_pilot']}**。通过只表示 WGEN episode-level 天气在本轮共同生长季描述性检验下足以进入受控 pilot，不意味着完美复现气候。

## 18. 下一步

若 gate 允许，下一任务为 **004_03 YC random-weather PPO pilot**。本轮没有启动 PPO。完整日历年 WGEN 输出路径仍为未验证状态，但 **full-year climatology validation: DEFERRED / OPTIONAL ADDITIONAL VALIDATION；full-year WGEN is not an episode-level random-weather PPO blocker**。

## 19. Git 状态

本报告、汇总表/图/脚本和逐 seed 天气 CSV 作为本任务产物进行本地提交，未推送到 GitHub。重复的逐运行状态、日志和 runtime snapshots 保留在本地工作区但不提交，以控制仓库体积；本任务目录的 `.gitignore` 仅忽略这些副本。最终提交号见任务终端摘要，未改动无关工作区文件。

## 产物导航

- 机器摘要：`summary.json`
- 逐集 QC：`episode_generation_qc.csv`
- 固定窗逐 seed：`fixed_window_weather_by_seed.csv`
- 完整季逐 seed：`full_crop_season_weather_by_seed.csv`
- 其余表：`rainfall_validation_summary.csv`、`monthly_rainfall_validation.csv`、`temperature_validation_summary.csv`、`srad_validation_summary.csv`、`cross_correlation_validation.csv`、`serial_correlation_validation.csv`、`weather_seed_reproducibility.csv`、`weather_seed_diversity.csv`
- 图片：`figures/`

图表：

![固定窗降雨分布](../results/yc_random_weather_episode_validation/004_02/figures/fixed-window_rainfall.png)
![湿日数分布](../results/yc_random_weather_episode_validation/004_02/figures/wet_days.png)
![最长干旱连日](../results/yc_random_weather_episode_validation/004_02/figures/longest_dry_spell.png)
![月降雨气候型](../results/yc_random_weather_episode_validation/004_02/figures/monthly_rainfall_climatology.png)
![TMAX 分布](../results/yc_random_weather_episode_validation/004_02/figures/tmax_distribution.png)
![TMIN 分布](../results/yc_random_weather_episode_validation/004_02/figures/tmin_distribution.png)
![SRAD 分布](../results/yc_random_weather_episode_validation/004_02/figures/srad_distribution.png)
![生长季长度](../results/yc_random_weather_episode_validation/004_02/figures/episode_length_distribution.png)
"""
    report = report.replace("`temperature_validation_summary.csv` 的频率与分位数", "`temperature_validation_summary.csv` 的频率与分位数")
    target = PROJECT / "docs" / "yc_random_weather_episode_ensemble_validation.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
