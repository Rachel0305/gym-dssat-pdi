"""Run the 216HLA irrigation-frequency counterfactual on all 2014--2023 years."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import audit_216HLA_irrigation_frequency_counterfactual_3y as audit


audit.AUDIT_RUN_NAME = "216HLA_irrigation_frequency_counterfactual_10y_ckpt200k"
audit.OUT = ROOT / "benchmark_results" / audit.AUDIT_RUN_NAME
audit.RECORD = audit.OUT / "216HLA_irrigation_frequency_counterfactual_10y_record.md"
audit.YEARS = list(range(2014, 2024))
audit.YEAR_LABELS = {
    2014: "validation",
    2015: "dry",
    2016: "dry",
    2017: "moderately_dry",
    2018: "wet",
    2019: "normal",
    2020: "wet",
    2021: "normal",
    2022: "dry",
    2023: "normal",
}


if __name__ == "__main__":
    audit.main()
