from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_yc_wgen_parameters_004_01 import (  # noqa: E402
    FITTING_SHA256,
    FROZEN_CLI_SHA256,
    PARAMETERS,
    run_audit,
)


def test_independent_168_parameter_month_values_match_and_cli_rounding_is_explained(tmp_path):
    summary = run_audit(ROOT, tmp_path)

    assert summary["parameter_method_status"] == "PASS_WITH_NOTE"
    assert summary["parameters_checked"] == 14
    assert summary["parameter_month_values_crosschecked"] == 168
    assert summary["raw_values_exactly_matching"] == 168
    assert summary["cli_serialized_values_explained_by_rounding"] == 168
    assert summary["raw_mismatch_count"] == 0
    assert summary["unexplained_cli_difference_count"] == 0
    assert summary["fitting_weather_sha256"] == FITTING_SHA256
    assert summary["frozen_cli_sha256"] == FROZEN_CLI_SHA256
    assert summary["positive_rain_below_0_254_mm_days"] == 48

    crosscheck = (tmp_path / "parameter_independent_crosscheck.csv").read_text(encoding="utf-8-sig")
    assert len(crosscheck.splitlines()) == 169
    assert len(PARAMETERS) == 14


def test_audit_records_threshold_note_without_changing_the_rule(tmp_path):
    summary = run_audit(ROOT, tmp_path)

    assert summary["wet_day_rule"] == "RAIN > 0.0 mm; unchanged"
    assert summary["validation_data_used_for_fitting"] is False
    assert summary["cli_or_input_modified"] is False
    audit_csv = (tmp_path / "wgen_parameter_audit.csv").read_text(encoding="utf-8-sig")
    assert "PASS_WITH_NOTE" in audit_csv
    assert "0.01 inch" in audit_csv
    assert "RTOT" in audit_csv
    assert "No method difference identified." in audit_csv
