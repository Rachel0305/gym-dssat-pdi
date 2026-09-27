from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_random_weather_ppo/004_16_yc_wgen_weather_reconstruction"
EXP = ROOT / "results/yc_random_weather_ppo/004_05"
SEED5 = EXP / "training/random_weather/ppo_seed_5"
REPORT = ROOT / "docs/yc_random_weather_004_16_wgen_weather_reconstruction_and_coverage.md"
EXPECTED = {
    "fitting_weather": "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34",
    "frozen_cli": "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929",
}
DRY_RAIN_MM = 0.0
HOT_TMAX_C = 32.0
LOW_RAIN_MM = 1.0


def io_path(path: Path) -> Path:
    value = os.path.abspath(str(path))
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value.lstrip("\\") if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def relative(path: Path) -> str:
    value = str(path)
    if value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value).resolve().relative_to(ROOT.resolve()).as_posix()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with io_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with io_path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(io_path(path).read_text(encoding="utf-8-sig"))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def canonical_weather_hash(rows: list[dict]) -> str:
    lines = ["DATE,SRAD,TMAX,TMIN,RAIN"]
    for row in rows:
        lines.append(f"{row['date']:%Y%j},{row['srad']:.6f},{row['tmax']:.6f},{row['tmin']:.6f},{row['rain']:.6f}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest().upper()


def parse_wth(path: Path) -> list[dict]:
    rows, started = [], False
    for line in io_path(path).read_text(encoding="ascii", errors="replace").splitlines():
        if re.match(r"^\s*@\s*DATE\s+SRAD\s+TMAX\s+TMIN\s+RAIN", line, re.I):
            started = True
            continue
        parts = line.split()
        if not started or len(parts) < 5 or not re.fullmatch(r"\d{7}", parts[0]):
            continue
        year, doy = divmod(int(parts[0]), 1000)
        if not 1 <= doy <= date(year, 12, 31).timetuple().tm_yday:
            raise ValueError(f"Invalid date {parts[0]} in {path}")
        rows.append({"date": date(year, 1, 1) + timedelta(days=doy - 1), "srad": float(parts[1]),
                     "tmax": float(parts[2]), "tmin": float(parts[3]), "rain": float(parts[4])})
    if not rows:
        raise ValueError(f"No daily weather rows in {path}")
    return rows


def pdate_from_filex(path: Path, year: int) -> date:
    lines = io_path(path).read_text(encoding="ascii", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("@P PDATE"):
            for record in lines[i + 1:i + 5]:
                parts = record.split()
                if len(parts) >= 2 and parts[0].isdigit() and re.fullmatch(r"\d{5}", parts[1]):
                    return date(year, 1, 1) + timedelta(days=int(parts[1][-3:]) - 1)
            break
    raise ValueError(f"PDATE not found in {path}")


def quantile(values: list[float], q: float):
    if not values:
        return None
    a = sorted(values)
    pos = (len(a) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (pos - lo)


def longest_run(rows: list[dict], predicate) -> int:
    best = run = 0
    prior = None
    for row in rows:
        if prior is not None and (row["date"] - prior).days != 1:
            run = 0
        run = run + 1 if predicate(row) else 0
        best = max(best, run)
        prior = row["date"]
    return best


def rx5(rows: list[dict]):
    best = None
    for i in range(max(0, len(rows) - 4)):
        span = rows[i:i + 5]
        if len(span) == 5 and all((span[j]["date"] - span[j - 1]["date"]).days == 1 for j in range(1, 5)):
            total = sum(x["rain"] for x in span)
            best = total if best is None else max(best, total)
    return best


def weather_metrics(rows: list[dict]) -> dict:
    rain = [x["rain"] for x in rows]
    tmax, tmin, srad = [x["tmax"] for x in rows], [x["tmin"] for x in rows], [x["srad"] for x in rows]
    wet = [x for x in rain if x > 0]
    return {
        "day_count": len(rows), "total_rain_mm": sum(rain), "rainfall_days_gt0": len(wet),
        "mean_wet_day_intensity_mm": mean(wet) if wet else None,
        "longest_dry_spell_days_rain_le0": longest_run(rows, lambda x: x["rain"] <= DRY_RAIN_MM),
        "rx1day_mm": max(rain) if rain else None, "rx5day_mm": rx5(rows),
        "tmax_mean_c": mean(tmax) if tmax else None, "tmax_max_c": max(tmax) if tmax else None,
        "hot_days_gt30c": sum(x > 30 for x in tmax), "hot_days_gt32c": sum(x > 32 for x in tmax),
        "hot_days_gt35c": sum(x > 35 for x in tmax), "tmin_mean_c": mean(tmin) if tmin else None,
        "tmin_min_c": min(tmin) if tmin else None, "srad_mean_mj_m2_day": mean(srad) if srad else None,
        "srad_min_mj_m2_day": min(srad) if srad else None, "srad_max_mj_m2_day": max(srad) if srad else None,
        "srad_p10_mj_m2_day": quantile(srad, .1), "srad_p90_mj_m2_day": quantile(srad, .9),
    }


def main() -> None:
    if REPORT.exists() or (OUT.exists() and any(OUT.rglob("*"))):
        raise SystemExit("Refusing to overwrite existing 004_16 output files.")
    for name in ("reconstructed_weather/training", "reconstructed_weather/heldout", "provenance", "coverage", "figures", "logs"):
        (OUT / name).mkdir(parents=True, exist_ok=True)

    inputs = {
        "fitting_weather": ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv",
        "frozen_cli": ROOT / "results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI",
        "cli_metadata": ROOT / "results/yc_wgen_cli_pilot/003_06_05_02/final/cli_generation_metadata.json",
        "pilot_provenance": ROOT / "results/yc_wgen_cli_pilot/003_06_05_02/runtime/formal_seed_pilot_provenance.json",
        "path_decision": ROOT / "results/yc_wgen_cli_pilot/004_01/full_year_path_audit/path_decision.json",
        "source_trace": ROOT / "results/yc_wgen_cli_pilot/004_01/full_year_path_audit/source_trace.md",
        "runtime_audit": ROOT / "results/yc_wgen_cli_pilot/003_06_02/audit_summary.json",
        "runtime_environment": ROOT / "results/yc_wgen_cli_pilot/003_06_02/runtime_environment.txt",
        "design": EXP / "config/experiment_design.json",
        "schedule": EXP / "config/training_weather_schedule_random.csv",
        "train_episodes": SEED5 / "training_episode_summary.csv",
        "heldout_episodes": EXP / "evaluation/runs/heldout_wgen/random_weather/ppo_seed_5/evaluation_episode_level.csv",
        "004_13_manifest": ROOT / "results/yc_random_weather_ppo/004_13_yc_105mm_early_cap_paired_retraining/attempt_05/formal_manifest.json",
    }
    input_rows, input_hashes = [], {}
    for key, path in inputs.items():
        exists = path.is_file()
        actual = file_hash(path) if exists else ""
        expected = EXPECTED.get(key, "")
        status = "MISSING" if not exists else ("PASS" if not expected or actual == expected else "HASH_MISMATCH")
        input_hashes[key] = actual
        input_rows.append({"artifact": key, "path": relative(path), "exists": exists,
                           "expected_sha256": expected, "actual_sha256": actual, "status": status})
    write_csv(OUT / "provenance/input_hash_inventory.csv",
              ["artifact", "path", "exists", "expected_sha256", "actual_sha256", "status"], input_rows)

    schedule = read_csv(inputs["schedule"])
    design, path_decision, runtime_audit = (read_json(inputs[k]) for k in ("design", "path_decision", "runtime_audit"))
    train_eps, held_eps = read_csv(inputs["train_episodes"]), read_csv(inputs["heldout_episodes"])
    schedule_counts, contexts = defaultdict(int), defaultdict(set)
    for row in schedule:
        seed = int(row["weather_seed"])
        schedule_counts[seed] += 1
        contexts[seed].add(int(row["historical_year_context"]))
    train_hashes = {r["runtime_weather_sha256"].upper() for r in train_eps if r.get("runtime_weather_sha256")}
    held_hashes = {r["runtime_weather_sha256"].upper() for r in held_eps if r.get("runtime_weather_sha256")}

    manifest = []
    for seed in range(1001, 1101):
        cohort = "training" if seed <= 1080 else "heldout"
        manifest.append({"cohort": cohort, "weather_seed": seed,
                         "scheduled_crop_year_contexts": ";".join(map(str, sorted(contexts[seed]))) if cohort == "training" else "2008",
                         "expected_role": "004_05_training" if cohort == "training" else "004_05_heldout",
                         "generated_file": "", "generated_raw_sha256": "", "canonical_daily_sha256": "",
                         "status": "BLOCKED_PROVENANCE_GATE", "note": "No historical daily realization materialized."})
    write_csv(OUT / "weather_seed_file_manifest.csv",
              list(manifest[0]), manifest)

    hashes = []
    for cohort, rows, seed_col, context_col, source in (
        ("training", train_eps, "training_weather_seed", "historical_year", relative(inputs["train_episodes"])),
        ("heldout", held_eps, "actual_rseed1_", "crop_year_context", relative(inputs["heldout_episodes"])),
    ):
        for row in rows:
            old = row.get("runtime_weather_sha256", "")
            if old:
                hashes.append({"cohort": cohort, "seed": row.get(seed_col, ""),
                    "crop_year_context": row.get(context_col, ""), "generated_file": "",
                    "generated_raw_sha256": "", "generated_canonical_daily_sha256": "",
                    "historical_runtime_hash": old, "hash_match": "NOT_TESTED_NO_REGENERATION",
                    "historical_source": source, "context_match": "NO_REGENERATED_ARTIFACT",
                    "notes": "The project does not define this hash payload/serialization."})
    write_csv(OUT / "hash_verification.csv", list(hashes[0]) if hashes else ["cohort", "seed", "crop_year_context"], hashes)

    episodes = read_csv(EXP / "all_evaluation_episode_level_0_7.csv")
    by_year = {int(r["evaluation_weather_year"]): r for r in episodes
        if r.get("training_regime") == "RANDOM_WEATHER_WGEN" and r.get("evaluation_weather_type") == "observed_weather"
        and r.get("ppo_seed") == "5" and r.get("evaluation_weather_year")}
    observed_base = EXP / "evaluation/runs/observed_weather/random_weather/ppo_seed_5/rendered_inputs/YCA"
    observed, stages, compounds, inventory, key_profile = [], [], [], [], []
    for year in range(2014, 2024):
        year_dir = observed_base / str(year)
        wths, filexs = list(year_dir.rglob("*.WTH")), list(year_dir.rglob("*.jinja2"))
        if not wths or not filexs or year not in by_year:
            inventory.append({"year": year, "status": "INCOMPLETE_INPUTS"})
            continue
        wth, filex = wths[0], filexs[0]
        rows = parse_wth(wth)
        start = pdate_from_filex(filex, year)
        n_days = int(float(by_year[year]["episode_days"]))
        season = [x for x in rows if 0 <= (x["date"] - start).days < n_days]
        raw, canonical = file_hash(wth), canonical_weather_hash(rows)
        inventory.append({"year": year, "source_file": relative(wth), "raw_sha256": raw,
                          "canonical_daily_sha256": canonical, "record_count": len(rows), "status": "AVAILABLE"})
        observed.append({"cohort": "observed", "realization_id": f"observed_{year}", "year": year,
                         "weather_source_file": relative(wth), "weather_raw_sha256": raw,
                         "canonical_daily_sha256": canonical, "planting_date": start.isoformat(),
                         "season_days_from_existing_episode": n_days, **weather_metrics(season),
                         "status": "DESCRIPTIVE_OBSERVED_ONLY"})
        windows = {"CROP_SEASON": (0, n_days - 1), "DAP_00_30": (0, 30), "DAP_31_60": (31, 60),
                   "DAP_61_90": (61, 90), "DAP_GT90": (91, n_days - 1)}
        for name, (lo, hi) in windows.items():
            subset = [x for x in season if lo <= (x["date"] - start).days <= hi]
            met = weather_metrics(subset)
            stages.append({"cohort": "observed", "realization_id": f"observed_{year}", "year": year,
                           "stage": name, "dap_start": lo, "dap_end": hi, "planting_date": start.isoformat(),
                           **met, "status": "DESCRIPTIVE_OBSERVED_ONLY"})
            hot_dry = sum(x["tmax"] > HOT_TMAX_C and x["rain"] < LOW_RAIN_MM for x in subset)
            compounds.append({"cohort": "observed", "realization_id": f"observed_{year}", "year": year,
                "period": name, "hot_tmax_threshold_c": HOT_TMAX_C, "low_rain_threshold_mm": LOW_RAIN_MM,
                "dry_spell_threshold_mm": DRY_RAIN_MM, "hot_dry_days": hot_dry,
                "hot_dry_fraction": hot_dry / len(subset) if subset else None,
                "longest_hot_spell_days": longest_run(subset, lambda x: x["tmax"] > HOT_TMAX_C),
                "longest_hot_dry_spell_days": longest_run(subset, lambda x: x["tmax"] > HOT_TMAX_C and x["rain"] < LOW_RAIN_MM),
                "status": "DESCRIPTIVE_OBSERVED_ONLY"})
            if year in (2014, 2019):
                key_profile.append({"year": year, "period": name, **met, "source_file": relative(wth)})

    wgen_metric_fields = ["cohort", "realization_id", "weather_seed", "crop_year_context", "weather_source_file",
        "weather_raw_sha256", "canonical_daily_sha256", "day_count", "total_rain_mm", "rainfall_days_gt0",
        "mean_wet_day_intensity_mm", "longest_dry_spell_days_rain_le0", "rx1day_mm", "rx5day_mm", "tmax_mean_c",
        "tmax_max_c", "hot_days_gt30c", "hot_days_gt32c", "hot_days_gt35c", "tmin_mean_c", "tmin_min_c",
        "srad_mean_mj_m2_day", "srad_min_mj_m2_day", "srad_max_mj_m2_day", "srad_p10_mj_m2_day", "srad_p90_mj_m2_day", "status"]
    write_csv(OUT / "weather_metrics_by_realization.csv", wgen_metric_fields, [])
    write_csv(OUT / "observed_weather_metrics.csv", list(observed[0]) if observed else ["year", "status"], observed)
    write_csv(OUT / "observed_training_percentiles.csv", ["observed_year", "metric", "training_empirical_percentile",
        "training_min", "training_max", "inside_training_range", "inside_training_p05_p95", "tail_flag", "status", "note"], [])
    write_csv(OUT / "stage_weather_metrics.csv", list(stages[0]) if stages else ["year", "stage"], stages)
    write_csv(OUT / "compound_extreme_metrics.csv", list(compounds[0]) if compounds else ["year", "period"], compounds)
    write_csv(OUT / "heldout_vs_training_distribution.csv", ["metric", "training_n", "heldout_n", "training_mean",
        "heldout_mean", "standardized_mean_difference", "status", "note"], [])
    write_csv(OUT / "observed_wth_inventory.csv", list(inventory[0]) if inventory else ["year", "status"], inventory)
    write_csv(OUT / "seed5_key_year_weather_profile.csv", list(key_profile[0]) if key_profile else ["year", "period"], key_profile)

    snapshots = []
    for cohort, folder, old_hashes in (("training", SEED5, train_hashes),
        ("heldout", EXP / "evaluation/runs/heldout_wgen/random_weather/ppo_seed_5", held_hashes)):
        for path in folder.rglob("*.WTH"):
            try:
                daily = parse_wth(path)
                raw = file_hash(path)
                snapshots.append({"cohort": cohort, "path": relative(path), "raw_sha256": raw,
                    "canonical_daily_sha256": canonical_weather_hash(daily), "record_count": len(daily),
                    "raw_hash_matches_runtime_hash": raw in old_hashes})
            except Exception as exc:
                snapshots.append({"cohort": cohort, "path": relative(path), "parse_error": str(exc)})

    chain = [
        {"item": "frozen fitting CSV/hash", "status": "VERIFIED" if input_hashes["fitting_weather"] == EXPECTED["fitting_weather"] else "UNKNOWN"},
        {"item": "frozen CLI/hash", "status": "VERIFIED" if input_hashes["frozen_cli"] == EXPECTED["frozen_cli"] else "UNKNOWN"},
        {"item": "historical DSSAT version", "status": "INFERRED", "detail": "Old provenance records 4.8.0.024; live binary identity unknown."},
        {"item": "current WeatherMan/standalone WGEN version and API", "status": "UNKNOWN"},
        {"item": "seed schedule and RSEED1 transport", "status": "VERIFIED"},
        {"item": "seed semantic reproducibility and full RNG state", "status": "UNKNOWN"},
        {"item": "training crop-year context 2004-2013", "status": "VERIFIED"},
        {"item": "held-out crop-year context 2008", "status": "VERIFIED"},
        {"item": "WTH template/format chain", "status": "UNKNOWN"},
        {"item": "runtime_weather_sha256 payload definition", "status": "UNKNOWN"},
        {"item": "allowed weather-only materialization", "status": "UNKNOWN"},
    ]
    provenance = {
        "task": "004_16", "weather_reconstruction_status": "FAILED", "gate": "FAILED_BEFORE_MATERIALIZATION",
        "generation_attempted": False, "scope": "read-only audit; no WGEN, WeatherMan, DSSAT, PPO, checkpoint execution",
        "frozen_fitting_weather_sha256": input_hashes["fitting_weather"], "frozen_cli_sha256": input_hashes["frozen_cli"],
        "schedule_sha256": file_hash(inputs["schedule"]), "schedule_rows": len(schedule),
        "training_seed_count": len(schedule_counts), "seed_occurrences_min": min(schedule_counts.values()),
        "seed_occurrences_max": max(schedule_counts.values()),
        "training_contexts_by_seed": {str(s): sorted(contexts[s]) for s in range(1001, 1081)},
        "training_runtime_hash_count": len(train_hashes), "heldout_runtime_hash_count": len(held_hashes),
        "runtime_weather_hash_semantics": "UNKNOWN", "chain_assessment": chain,
        "retained_wth_snapshots": snapshots, "input_hash_inventory": input_rows,
        "prior_full_year_path_decision": path_decision, "runtime_audit": runtime_audit,
        "blockers": ["No verified standalone WeatherMan/WGEN materializer compatible with the historical CLI/runtime.",
            "The evidenced DSSAT WGEN path is inside the crop-simulation daily loop, prohibited by this task.",
            "Current WeatherMan and DSSAT binary identities are unknown.",
            "The project does not define the runtime_weather_sha256 hashed object/serialization."],
    }
    write_json(OUT / "generation_provenance.json", provenance)
    write_json(OUT / "decision.json", {
        "task": "004_16", "weather_reconstruction_status": "FAILED", "observed_training_coverage": "INSUFFICIENT",
        "tail_gap_type": "INSUFFICIENT", "weather_pool_expansion_recommendation": "INSUFFICIENT_EVIDENCE",
        "formal_coverage_analysis_performed": False, "observed_years_with_descriptive_metrics": len(observed),
        "wgen_materialization_attempted": False, "ppo_training_or_evaluation": False, "dssat_crop_simulation": False,
        "canonical_or_historical_files_modified": False, "git_commit": None, "git_push": None,
        "reason": "Provenance gate failed before any weather materialization.",
        "next_required_evidence": ["Verified weather-only generator/runtime and frozen CLI import path.",
            "Documented seed, RNG, crop-year and date context sufficient to replay daily values.",
            "Definition of runtime_weather_sha256 payload or seed/context-mapped raw WTH."],
    })
    (OUT / "logs/provenance_gate.log").write_text(
        "Gate failed before weather materialization. No generator, crop simulation, PPO, checkpoint load, or evaluation was invoked.\n"
        "Only archived observed WTH files were parsed for descriptive metrics.\n", encoding="utf-8")
    for name, text in (("reconstructed_weather/training/README.md", "No files generated; provenance gate failed."),
        ("reconstructed_weather/heldout/README.md", "No files generated; held-out remains isolated."),
        ("coverage/README.md", "Formal WGEN coverage tables are empty because daily realizations are unavailable."),
        ("figures/README.md", "Coverage figures are blocked to avoid unsupported historical claims.")):
        (OUT / name).write_text(text + "\n", encoding="utf-8")

    y14 = next((r for r in observed if r["year"] == 2014), {})
    y19 = next((r for r in observed if r["year"] == 2019), {})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# 004_16 YC WGEN 天气重建与气候覆盖审计

## 结论

任务在provenance gate失败后停止，未物化seeds 1001–1100；未运行PPO、加载checkpoint、执行evaluation、调用WGEN/WeatherMan或DSSAT crop simulation。只对已有observed WTH计算描述统计。

weather_reconstruction_status = FAILED  
observed_training_coverage = INSUFFICIENT  
tail_gap_type = INSUFFICIENT  
weather_pool_expansion_recommendation = INSUFFICIENT_EVIDENCE

FAILED表示重建gate未通过且物化未启动，不表示WGEN生成器运行失败，也不否定004_05作物结果。

## 生成链

Frozen fitting CSV SHA256：{input_hashes['fitting_weather']}  
Frozen CNYC.CLI SHA256：{input_hashes['frozen_cli']}  
训练schedule：{len(schedule)}行，{len(schedule_counts)}个seed，每seed出现{min(schedule_counts.values())}–{max(schedule_counts.values())}次；training context为2004–2013，held-out context为2008。seed调度及RSEED1传递为VERIFIED。旧provenance记录DSSAT 4.8.0.024，但当前runtime/binary未核实，版本状态为INFERRED。

当前WeatherMan/standalone WGEN版本和API、额外RNG state、WTH模板及格式化步骤、runtime_weather_sha256的被哈希对象均UNKNOWN。既有full-year path audit没有验证独立天气导出路径；可证明的DSSAT WGEN位于逐日作物模拟循环内，本任务禁止该模拟。seed编号与CLI一致不能证明日序列就是004_05原天气。

004_05 seed5训练日志有{len(train_hashes)}个distinct runtime weather hash，held-out有{len(held_hashes)}个。runtime hash没有重建文件对应，hash_match标为NOT_TESTED_NO_REGENERATION；不可把未知语义的runtime hash直接当作raw WTH SHA。细节见generation_provenance.json、provenance/input_hash_inventory.csv、weather_seed_file_manifest.csv、hash_verification.csv。

## Observed描述统计

从已存档004_05 seed5 observed WTH读取天气；PDATE取对应FileX，season length取既有evaluation episode。无雨段定义RAIN<=0 mm连续日；hot-dry day定义TMAX>32°C且RAIN<1 mm。结果只描述observed，不代表training WGEN coverage。

2014 season rainfall={y14.get('total_rain_mm', 'N/A')} mm，最长无雨段={y14.get('longest_dry_spell_days_rain_le0', 'N/A')}天，mean Tmax={y14.get('tmax_mean_c', 'N/A')}°C，>32°C天数={y14.get('hot_days_gt32c', 'N/A')}。  
2019 season rainfall={y19.get('total_rain_mm', 'N/A')} mm，最长无雨段={y19.get('longest_dry_spell_days_rain_le0', 'N/A')}天，mean Tmax={y19.get('tmax_mean_c', 'N/A')}°C，>32°C天数={y19.get('hot_days_gt32c', 'N/A')}。

2014–2023逐年指标见observed_weather_metrics.csv，DAP阶段见stage_weather_metrics.csv，复合极端见compound_extreme_metrics.csv，2014/2019分期见seed5_key_year_weather_profile.csv；raw与canonical hash见observed_wth_inventory.csv。

## 阻塞输出与下一步

weather_metrics_by_realization.csv、observed_training_percentiles.csv、heldout_vs_training_distribution.csv为空表；未生成WGEN对observed覆盖图/heatmap，避免把未恢复的序列画成历史证据。现有observed WTH文件本身完成10年描述统计。

本轮没有证据支持KEEP_80、随机扩容、定向补尾或修改WGEN分布。需先取得可验证weather-only工具/runtime、完整seed/RNG/context定义，以及runtime hash规范或seed/context映射的raw WTH。未启动80→160 PPO训练；未commit、未push。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "FAILED_GATE", "observed_years": len(observed),
        "training_runtime_hashes": len(train_hashes), "heldout_runtime_hashes": len(held_hashes),
        "result_files": sum(p.is_file() for p in OUT.rglob("*")), "report_written": REPORT.is_file()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
