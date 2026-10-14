#!/usr/bin/env python3
"""Read-only DAP-aware action trajectory audit for existing LC seed outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REQUIRED_COLUMNS = {"year", "dap", "safe_action_amir", "safe_action_anfer"}


def _as_float(value: str) -> float:
    return float(value)


def _pair(value_i: float, value_n: float) -> str:
    return f"I{value_i:g}/N{value_n:g}"


def _entropy(items: Iterable[str]) -> Tuple[float, float]:
    counts = Counter(items)
    n = sum(counts.values())
    if not n:
        return 0.0, 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    h_norm = h / math.log(len(counts)) if len(counts) > 1 else 0.0
    return h, h_norm


def _trajectory_hash(events: List[Tuple[int, float, float]]) -> str:
    raw = "|".join(f"{dap}:{i:g}:{n:g}" for dap, i, n in events).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _jaccard(a: List[Tuple[int, float, float]], b: List[Tuple[int, float, float]]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _daily_path(root: Path, year: int, seed: str, checkpoint: int) -> Path:
    station = "LCA"
    candidates = [
        root / "daily_outputs" / station / f"{station}_{year}_seed{seed}_ckpt{checkpoint}_daily.csv",
        root / "daily_outputs" / station / f"{station}_{year}_seed{seed.replace('_smoke', '')}_ckpt{checkpoint}_daily.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def inspect_daily(path: Path, expected_year: int) -> Dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "error": "missing"}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - fields)
        rows = list(reader)
    if missing:
        return {"path": str(path), "exists": True, "error": "missing_columns", "missing_columns": missing}

    events: List[Tuple[int, float, float]] = []
    pairs: List[str] = []
    post_dap1 = 0
    mismatch = {k: 0 for k in ("off_grid_rows", "requested_to_safe_mismatch_rows", "raw_to_safe_mismatch_rows", "safe_to_dssat_mismatch_rows")}
    for row in rows:
        dap = int(float(row["dap"]))
        i = _as_float(row["safe_action_amir"])
        n = _as_float(row["safe_action_anfer"])
        if abs(i) > 1e-9 or abs(n) > 1e-9:
            event = (dap, i, n)
            events.append(event)
            pairs.append(_pair(i, n))
            if dap > 1:
                post_dap1 += 1
        for key in mismatch:
            if key in row and row[key] not in ("", "0", "0.0", "False", "false"):
                try:
                    mismatch[key] += int(float(row[key]))
                except ValueError:
                    mismatch[key] += 1
    entropy, entropy_norm = _entropy(pairs)
    signature = ";".join(sorted(set(pairs)))
    return {
        "path": str(path),
        "exists": True,
        "error": None,
        "year": expected_year,
        "rows": len(rows),
        "positive_action_rows": len(events),
        "positive_actions_after_dap1": post_dap1,
        "unique_nonzero_action_pairs": len(set(pairs)),
        "nonzero_action_pairs": signature,
        "pair_entropy_nats": entropy,
        "pair_entropy_normalized": entropy_norm,
        "trajectory_signature": _trajectory_hash(events),
        "trajectory_events": events,
        **mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/071_lc_seed_trajectory_diagnostic.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    out = Path(cfg["output_root"])
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "config": str(config_path), "output_root": str(out)}, indent=2))
        return 0
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing non-empty output root: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    year_events: Dict[Tuple[str, int], Dict[int, List[Tuple[int, float, float]]]] = {}
    for seed, spec in cfg["seeds"].items():
        root = Path(spec["root"])
        for checkpoint in spec["checkpoints"]:
            for year in cfg["validation_years"]:
                result = inspect_daily(_daily_path(root, year, seed, checkpoint), year)
                events = result.pop("trajectory_events", [])
                result.update({"seed": int(seed) if str(seed).isdigit() else str(seed), "checkpoint": checkpoint})
                rows.append(result)
                year_events.setdefault((seed, checkpoint), {})[year] = events

    detail_path = out / "lc_seed_trajectory_detail.csv"
    detail_fields = sorted({key for row in rows for key in row if key != "trajectory_events"})
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summaries: List[Dict[str, object]] = []
    for (seed, checkpoint), by_year in year_events.items():
        pair_sigs = {r["nonzero_action_pairs"] for r in rows if str(r.get("seed")) == seed and r.get("checkpoint") == checkpoint and r.get("error") is None}
        traj_sigs = {_trajectory_hash(events) for events in by_year.values()}
        jaccards = []
        years = sorted(by_year)
        for idx, year_a in enumerate(years):
            for year_b in years[idx + 1 :]:
                jaccards.append(_jaccard(by_year[year_a], by_year[year_b]))
        summaries.append({
            "seed": int(seed) if str(seed).isdigit() else str(seed),
            "checkpoint": checkpoint,
            "years_with_daily": len(by_year),
            "pair_set_signature_count": len(pair_sigs),
            "dap_aware_trajectory_signature_count": len(traj_sigs),
            "mean_pairwise_trajectory_jaccard": sum(jaccards) / len(jaccards) if jaccards else None,
            "min_pairwise_trajectory_jaccard": min(jaccards) if jaccards else None,
            "max_pairwise_trajectory_jaccard": max(jaccards) if jaccards else None,
        })

    summary_path = out / "lc_seed_trajectory_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else ["seed", "checkpoint"])
        writer.writeheader()
        writer.writerows(summaries)

    manifest = {
        "task": cfg["task"],
        "status": "completed_read_only",
        "no_training": True,
        "no_dssat": True,
        "no_replay": True,
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_path),
        "rows": len(rows),
        "missing_or_error_rows": sum(1 for row in rows if row.get("error")),
        "decision_note": "DAP-aware trajectory evidence only; no cross-seed performance claim",
    }
    (out / "071_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
