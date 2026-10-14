"""157LC: serial WP_ET replay for the existing LC-E2 5K checkpoint.

This is a thin LC adapter around the already-audited serial replay engine.
It performs no training and writes LC-specific aliases and provenance files
after the replay completes.
"""

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
import run_155LC_lca_lowIC_dual_branch_weather_reader_smoke2k  # noqa: F401 - register LC extractor for pickle loading


TASK = "157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay"
OUT = ROOT / "benchmark_results" / TASK
STATION = "LCA"
YEARS = list(range(2014, 2024))
CFG = ROOT / "configs/156LC_lca_lowIC_dual_branch_weather_reader_5k_seed0.json"
INVENTORY = ROOT / "benchmark_results/156LC_lca_lowIC_dual_branch_weather_reader_5k_seed0/evaluation/156LC_checkpoint_validation_summary.csv"


def main() -> int:
    replay.TASK = TASK
    replay.OUT = OUT
    replay.STATION = STATION
    replay.YEARS = YEARS
    replay.POLICIES = {
        "e2_s0": {
            "kind": "forecast",
            "seed": 0,
            "cfg": CFG,
            "inventory": INVENTORY,
        }
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result = replay.run(["e2_s0"], YEARS, resume=True)

    # The imported engine keeps its historical 151 filenames.  Add LC-scoped
    # aliases and a clean report without altering those raw replay artifacts.
    by_year = OUT / "151_wp_et_replay_by_policy_year.csv"
    summary = OUT / "151_wp_et_replay_summary_by_policy.csv"
    if by_year.exists():
        pd.read_csv(by_year).to_csv(OUT / "157LC_wp_et_replay_by_policy_year.csv", index=False, encoding="utf-8-sig")
    if summary.exists():
        pd.read_csv(summary).to_csv(OUT / "157LC_wp_et_replay_summary_by_policy.csv", index=False, encoding="utf-8-sig")
    record = {
        "task": TASK,
        "station_code": STATION,
        "site": "LC",
        "policy": "156LC_lca_lowIC_dual_branch_weather_reader_5k_seed0",
        "checkpoint_step": 5000,
        "years": YEARS,
        "no_training": True,
        "source_config": str(CFG.relative_to(ROOT).as_posix()),
        "source_inventory": str(INVENTORY.relative_to(ROOT).as_posix()),
        "replay_output_root": str(OUT.relative_to(ROOT).as_posix()),
        "raw_replay_manifest": str((OUT / "151_wp_et_replay_manifest.json").relative_to(ROOT).as_posix()),
        "by_year_csv": str((OUT / "157LC_wp_et_replay_by_policy_year.csv").relative_to(ROOT).as_posix()),
        "summary_csv": str((OUT / "157LC_wp_et_replay_summary_by_policy.csv").relative_to(ROOT).as_posix()),
        "metric_definition": "WP_ET_kg_m3 = YPEM*0.1 when valid, otherwise HWAM/(ETCP*10); ETCP in mm",
    }
    (OUT / "157LC_replay_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "docs/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay_record.md").write_text(
        "# 157LC LC-E2 5K WP_ET replay\n\n"
        "- 已有 156LC 5K seed0 checkpoint 的只读串行回放；本轮不训练。\n"
        "- 验证年：2014–2023；策略：LC-E2 dual-branch weather reader。\n"
        "- WP_ET = grain yield / (10 × ETCP)，PFP-N 对零施氮情景保持 N/A。\n"
        "- 原始回放清单：`benchmark_results/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay/151_wp_et_replay_manifest.json`。\n"
        "- LC 专用逐年表：`benchmark_results/157LC_lca_lowIC_dual_branch_weather_reader_wp_et_5k_replay/157LC_wp_et_replay_by_policy_year.csv`。\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
