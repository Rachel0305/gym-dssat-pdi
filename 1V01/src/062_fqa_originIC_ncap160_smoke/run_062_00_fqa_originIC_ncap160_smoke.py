"""Isolated FQA 2K smoke: only season N cap changes from 250 to 160 kg/ha."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
CONFIG = ROOT / "configs" / "062_00_fqa_originIC_expanded_action_maskableppo_ncap160_smoke.json"
BASELINE_CAP = 250.0
NCAP = 160.0


def load_051_runner():
    path = ROOT / "src" / "051_fqa_originIC_site_transfer" / "run_051_00_fqa_originIC_expanded_action_maskableppo.py"
    spec = importlib.util.spec_from_file_location("fqa_051_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load 051 runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_only_n_cap(runner):
    original = runner.engine.base04036.load_config

    def capped_config():
        cfg = original()
        safety = cfg.get("action_safety", {})
        actual = float(safety.get("season_n_soft_limit", float("nan")))
        if not np.isclose(actual, BASELINE_CAP):
            raise RuntimeError(f"Unexpected inherited season_n_soft_limit={actual}; expected {BASELINE_CAP}")
        safety["season_n_soft_limit"] = NCAP
        cfg["action_safety"] = safety
        return cfg

    runner.engine.base04036.load_config = capped_config
    return original


def gate_report(out: Path) -> dict:
    summary_path = out / "evaluation" / "062_00_checkpoint_validation_summary.csv"
    audit_path = out / "audits" / "062_00_smoke_action_audit.csv"
    summary = pd.read_csv(summary_path)
    summary = summary[pd.to_numeric(summary["checkpoint_step"], errors="coerce").eq(2000)].copy()
    audit = pd.read_csv(audit_path)
    rows = []
    all_pairs: set[str] = set()
    signatures: set[str] = set()
    max_cumulative_n = 0.0
    request_safe_mismatch = 0
    safe_dssat_mismatch = 0
    for rec in summary.itertuples(index=False):
        daily = pd.read_csv(ROOT / str(rec.daily_csv_path))
        raw_i = pd.to_numeric(daily["raw_action_amir"], errors="coerce").fillna(0.0)
        raw_n = pd.to_numeric(daily["raw_action_anfer"], errors="coerce").fillna(0.0)
        safe_i = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
        safe_n = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
        previous_i = pd.to_numeric(daily["previous_cumulative_irrigation"], errors="coerce")
        previous_n = pd.to_numeric(daily["previous_cumulative_n"], errors="coerce")
        cumulative_i = pd.to_numeric(daily["season_cumulative_irrigation"], errors="coerce")
        cumulative_n = pd.to_numeric(daily["season_cumulative_n"], errors="coerce")
        dap = pd.to_numeric(daily["dap"], errors="coerce")
        positive = (safe_i > 1e-9) | (safe_n > 1e-9)
        request_safe_mismatch += int(((~np.isclose(raw_i, safe_i, atol=1e-6)) | (~np.isclose(raw_n, safe_n, atol=1e-6))).sum())
        safe_dssat_mismatch += int((positive & ((~np.isclose(cumulative_i - previous_i, safe_i, atol=1e-6)) | (~np.isclose(cumulative_n - previous_n, safe_n, atol=1e-6)))).sum())
        max_cumulative_n = max(max_cumulative_n, float(cumulative_n.max()))
        pairs = sorted({f"I{i:g}/N{n:g}" for i, n in zip(safe_i[positive], safe_n[positive])})
        all_pairs.update(pairs)
        signature = "; ".join(pairs)
        signatures.add(signature)
        rows.append({
            "year": int(rec.year), "final_grnwt": float(rec.final_grnwt),
            "total_irrigation": float(rec.total_irrigation), "total_n": float(rec.total_n),
            "PFP_N": float(rec.PFP_N) if pd.notna(rec.PFP_N) else None,
            "positive_actions_after_dap1": int((positive & (dap > 1)).sum()),
            "nonzero_action_pairs": signature,
            "max_cumulative_n": float(cumulative_n.max()),
        })
    final = audit[audit["checkpoint_step"].eq(2000)]
    return {
        "checkpoint": 2000,
        "validation_rows": len(final),
        "max_cumulative_n_kg_ha": max_cumulative_n,
        "request_safe_difference_rows": request_safe_mismatch,
        "safe_dssat_transmission_mismatch_rows": safe_dssat_mismatch,
        "nonzero_action_pair_count": len(all_pairs),
        "nonzero_action_pairs": sorted(all_pairs),
        "positive_action_rows_after_dap1": int(final["positive_action_rows_after_dap1"].sum()),
        "cross_year_distinct_signatures": len(signatures),
        "per_year": rows,
        "mean_final_grnwt": float(summary["final_grnwt"].mean()),
        "mean_total_irrigation": float(summary["total_irrigation"].mean()),
        "mean_total_n": float(summary["total_n"].mean()),
        "mean_pfp_n": float(summary["PFP_N"].mean()),
        "wp_et": "not_exported_by_2k_checkpoint_evaluator",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    if args.summarize_existing:
        out = ROOT / "benchmark_results" / "062_00_fqa_originIC_expanded_action_maskableppo_ncap160_smoke_smoke2k"
        report = gate_report(out)
        report["cap_compliant"] = report["max_cumulative_n_kg_ha"] <= NCAP + 1e-6
        report["action_diversity_pass"] = report["nonzero_action_pair_count"] >= 3
        report["post_dap1_pass"] = report["positive_action_rows_after_dap1"] > 0
        report["cross_year_signature_pass"] = report["cross_year_distinct_signatures"] > 1
        report["transmission_pass"] = report["safe_dssat_transmission_mismatch_rows"] == 0
        report["next_step_allowed"] = bool(report["cap_compliant"] and report["action_diversity_pass"] and report["post_dap1_pass"] and report["cross_year_signature_pass"] and report["transmission_pass"])
        path = out / "062_00_ncap160_gate.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    runner = load_051_runner()
    original = patch_only_n_cap(runner)
    try:
        result = runner.run(CONFIG, dry_run=args.dry_run, smoke=True, formal=False)
    finally:
        runner.engine.base04036.load_config = original
    if args.dry_run:
        result["single_factor_diff"] = json.loads(CONFIG.read_text(encoding="utf-8"))["single_factor_diff"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    out = ROOT / result["output_root"]
    report = gate_report(out)
    report["cap_compliant"] = report["max_cumulative_n_kg_ha"] <= NCAP + 1e-6
    report["action_diversity_pass"] = report["nonzero_action_pair_count"] >= 3
    report["post_dap1_pass"] = report["positive_action_rows_after_dap1"] > 0
    report["cross_year_signature_pass"] = report["cross_year_distinct_signatures"] > 1
    report["transmission_pass"] = report["safe_dssat_transmission_mismatch_rows"] == 0
    report["next_step_allowed"] = bool(report["cap_compliant"] and report["action_diversity_pass"] and report["post_dap1_pass"] and report["cross_year_signature_pass"] and report["transmission_pass"])
    path = out / "062_00_ncap160_gate.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**result, "ncap160_gate": str(path.relative_to(ROOT)), "next_step_allowed": report["next_step_allowed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
