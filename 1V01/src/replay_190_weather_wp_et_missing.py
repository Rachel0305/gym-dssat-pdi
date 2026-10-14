"""Read-only WP_ET replay for weather-transfer policies needed by the PPT figures.

The existing 178 replay intentionally covered only FQ 2K/5K and YC 2K.  The
figure package needs the frozen 5K HLA and YC pairs as well, so this wrapper
reuses the audited 178 replay engine with a four-policy, 40-season scope.
No training is performed and the original experiment folders are untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import replay_178_multisite_wp_et as replay


replay.TASK = "190_weather_wp_et_missing_replay"
replay.OUT = ROOT / "benchmark_results" / replay.TASK
replay.REPORT_PATH = ROOT / "docs" / "190_weather_wp_et_missing_replay_record.md"
replay.POLICIES = {
    "hla_forecast_5k": {
        "station": "HLA", "site": "HL", "seed": 1, "checkpoint_step": 5000,
        "kind": "forecast",
        "cfg": ROOT / "configs/170HLA_hla_lowIC_dual_branch_weather_reader_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/170HLA_hla_lowIC_dual_branch_weather_reader_5k_seed1/evaluation/170HLA_checkpoint_validation_summary.csv",
    },
    "hla_noforecast_5k": {
        "station": "HLA", "site": "HL", "seed": 1, "checkpoint_step": 5000,
        "kind": "noforecast",
        "cfg": ROOT / "configs/171HLA_hla_lowIC_dual_branch_noforecast_control_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/171HLA_hla_lowIC_dual_branch_noforecast_control_5k_seed1/evaluation/171HLA_checkpoint_validation_summary.csv",
    },
    "yca_forecast_5k": {
        "station": "YCA", "site": "YC", "seed": 1, "checkpoint_step": 5000,
        "kind": "forecast",
        "cfg": ROOT / "configs/172YCA_yca_lowIC_dual_branch_weather_reader_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/172YCA_yca_lowIC_dual_branch_weather_reader_5k_seed1/evaluation/172YCA_checkpoint_validation_summary.csv",
    },
    "yca_noforecast_5k": {
        "station": "YCA", "site": "YC", "seed": 1, "checkpoint_step": 5000,
        "kind": "noforecast",
        "cfg": ROOT / "configs/173YCA_yca_lowIC_dual_branch_noforecast_control_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/173YCA_yca_lowIC_dual_branch_noforecast_control_5k_seed1/evaluation/173YCA_checkpoint_validation_summary.csv",
    },
}


if __name__ == "__main__":
    print(json.dumps(replay.run(), ensure_ascii=False, indent=2))
