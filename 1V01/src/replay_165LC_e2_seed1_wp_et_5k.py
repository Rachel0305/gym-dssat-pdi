"""165LC: serial WP_ET/PFP-N replay for the corrected LC-E2 seed1 5K checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import replay_151_e2_noforecast_wp_et_5k as replay
import run_155LC_lca_lowIC_dual_branch_weather_reader_smoke2k  # noqa: F401 - register LC extractor


TASK = "165LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_seed1_replay"
OUT = ROOT / "benchmark_results" / TASK
STATION = "LCA"
YEARS = list(range(2014, 2024))
CFG = ROOT / "configs/163LC_lca_lowIC_dual_branch_weather_reader_5k_seed1.json"
INVENTORY = ROOT / "benchmark_results/163LC_lca_lowIC_dual_branch_weather_reader_5k_seed1/evaluation/163LC_checkpoint_validation_summary.csv"


def main() -> int:
    replay.TASK = TASK
    replay.OUT = OUT
    replay.STATION = STATION
    replay.YEARS = YEARS
    replay.POLICIES = {"e2_s1": {"kind": "forecast", "seed": 1, "cfg": CFG, "inventory": INVENTORY}}
    OUT.mkdir(parents=True, exist_ok=True)
    replay.run(["e2_s1"], YEARS, resume=True)
    by_year = OUT / "151_wp_et_replay_by_policy_year.csv"
    summary = OUT / "151_wp_et_replay_summary_by_policy.csv"
    if by_year.exists():
        pd.read_csv(by_year).to_csv(OUT / "165LC_wp_et_replay_by_policy_year.csv", index=False, encoding="utf-8-sig")
    if summary.exists():
        pd.read_csv(summary).to_csv(OUT / "165LC_wp_et_replay_summary_by_policy.csv", index=False, encoding="utf-8-sig")
    record = {
        "task": TASK, "station_code": STATION, "site": "LC", "policy": "163LC_lca_lowIC_dual_branch_weather_reader_5k_seed1",
        "checkpoint_step": 5000, "seed": 1, "years": YEARS, "no_training": True,
        "source_config": CFG.relative_to(ROOT).as_posix(), "source_inventory": INVENTORY.relative_to(ROOT).as_posix(),
        "replay_output_root": OUT.relative_to(ROOT).as_posix(),
        "by_year_csv": (OUT / "165LC_wp_et_replay_by_policy_year.csv").relative_to(ROOT).as_posix(),
        "summary_csv": (OUT / "165LC_wp_et_replay_summary_by_policy.csv").relative_to(ROOT).as_posix(),
        "metric_definition": "WP_ET_kg_m3 = grain yield / (10 * ETCP); PFP_N_kg_kg = grain yield / total N; ETCP in mm",
    }
    (OUT / "165LC_replay_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "docs/165LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_seed1_replay_record.md").write_text(
        "# 165LC LC-E2 seed1 5K WP_ET/PFP-N replay\n\n"
        "- 已有 163LC 真实 seed1 5K checkpoint 的只读串行回放；本轮不训练。\n"
        "- 验证年：2014–2023；策略：LC-E2 dual-branch weather reader。\n"
        "- WP_ET = grain yield / (10 × ETCP)；PFP-N = grain yield / total N。\n"
        f"- 逐年表：`{record['by_year_csv']}`。\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
