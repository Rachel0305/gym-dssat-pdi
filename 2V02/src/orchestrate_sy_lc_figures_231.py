"""Sequentially render the fourteen completed SY/LC seed packages and tables."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from plot_sy_lc_seed_extension_231 import BASE, FROZEN, PREFIX, REPORT, normalize_summary, source_tables

ROOT = Path(__file__).resolve().parents[1]
EVENTS = BASE / "figure_generation_events.jsonl"
LOGS = BASE / "figure_generation_logs"
RUNNER = ROOT / "src/plot_sy_lc_seed_extension_231.py"
FIELDS = ["grain_yield_kg_ha", "actual_irrigation_mm", "actual_nitrogen_kg_ha", "WP_ET_kg_m3", "PFP_N_kg_kg", "unified_reward"]


def emit(event: str, **fields: object) -> None:
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}) + "\n")


def package_ok(site: str, seed: int) -> bool:
    root = REPORT / site / f"seed_{seed:02d}"
    audit_path = root / "audit.json"
    if not audit_path.exists():
        if root.exists():
            raise FileExistsError(f"Incomplete output retained for review: {root}")
        return False
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("seed") != seed or audit.get("site") != site or audit.get("figure_count") != 12 or audit.get("summary_rows") != 50:
        raise RuntimeError(f"Package audit mismatch: {root}")
    if len(list((root / "figures").glob("*.png"))) != 12:
        raise RuntimeError(f"Figure count mismatch: {root}")
    table = root / "tables" / f"{PREFIX[site]}_five_scenario_season_summary.csv"
    frame = pd.read_csv(table, keep_default_na=False)
    if len(frame) != 50 or len(frame[frame.scenario.eq("rl_candidate")]) != 10:
        raise RuntimeError(f"Table coverage mismatch: {table}")
    return True


def render_all() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    for site in ("SY", "LC"):
        for seed in range(1, 8):
            if package_ok(site, seed):
                emit("already_complete", site=site, seed=seed)
                continue
            log = LOGS / f"{site.lower()}_seed{seed}.log"
            if log.exists():
                raise FileExistsError(f"Existing log requires review: {log}")
            command = [sys.executable, str(RUNNER), "--site", site, "--seed", str(seed)]
            emit("started", site=site, seed=seed)
            with log.open("x", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                emit("failed", site=site, seed=seed, returncode=result.returncode)
                raise RuntimeError(f"Figure generation failed: {site} seed{seed}; see {log}")
            if not package_ok(site, seed):
                emit("failed_audit", site=site, seed=seed)
                raise RuntimeError(f"Figure package incomplete: {site} seed{seed}")
            emit("complete_seed", site=site, seed=seed)


def summarize() -> None:
    for site in ("SY", "LC"):
        yearly = []
        frozen = normalize_summary(site, pd.read_csv(source_tables(site)[1], keep_default_na=False))
        for seed in range(8):
            if seed == 0:
                source = frozen
                provenance = FROZEN[site].relative_to(ROOT).as_posix()
            else:
                path = REPORT / site / f"seed_{seed:02d}/tables/{PREFIX[site]}_five_scenario_season_summary.csv"
                source = normalize_summary(site, pd.read_csv(path, keep_default_na=False))
                provenance = path.relative_to(ROOT).as_posix()
            ppo = source[source.scenario.eq("rl_candidate")].copy()
            expert = source[source.scenario.eq("official_extension_expert")].copy()
            if len(ppo) != len(expert) or len(ppo) != 10:
                raise RuntimeError(f"Seed/year coverage mismatch: {site} seed{seed}")
            ppo["seed"] = seed
            ppo["source"] = provenance
            yearly.append(ppo)
        all_yearly = pd.concat(yearly, ignore_index=True, sort=False)
        all_yearly = all_yearly.sort_values(["seed", "year"]).reset_index(drop=True)
        for field in FIELDS:
            all_yearly[field] = pd.to_numeric(all_yearly[field], errors="coerce")
        summary = all_yearly.groupby("seed", as_index=False)[FIELDS].mean(numeric_only=True)
        summary.insert(0, "site", site)
        summary.insert(2, "validation_years", 10)
        report = REPORT / site
        annual_path = report / "eight_seed_annual_ppo_metrics.csv"
        means_path = report / "eight_seed_ten_year_means.csv"
        if annual_path.exists() or means_path.exists():
            raise FileExistsError(f"Cohort summary exists: {report}")
        all_yearly.to_csv(annual_path, index=False, encoding="utf-8-sig")
        summary.to_csv(means_path, index=False, encoding="utf-8-sig")
        emit("cohort_summary_complete", site=site, seeds=8, annual_rows=len(all_yearly))


if __name__ == "__main__":
    try:
        render_all()
        summarize()
        emit("complete", seed_packages=14)
    except Exception as exc:
        emit("stopped", error=str(exc))
        raise
