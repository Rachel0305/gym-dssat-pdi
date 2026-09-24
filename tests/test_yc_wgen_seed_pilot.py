from datetime import date

from scripts.run_yc_wgen_seed_pilot import (
    canonical_weather_bytes,
    classify_runtime_error,
    compare_weather_rows,
    daily_weather_from_states,
    pdi_seed_configured,
    physical_sanity,
    prepare_wgen_filex,
    runtime_filex_mode,
    summarize_weather,
)


def sample_filex() -> str:
    return """*TREATMENTS
@N R O C TNAME
 1 1 1 0 Sim2008
 2 1 1 0 Sim2014
*FIELDS
@L ID WSTA
 1 CNYC0801
 2 CNYC1401
*SIMULATION CONTROLS
@N METHODS WTHER INCON LIGHT
 1 ME M M E
 2 ME M M E
*CLIMATE:CNYC
@ MTH TMAX
 1 30
 2 31
"""


def test_isolated_filex_keeps_only_yc_2008_treatment_and_templates_wther():
    text = prepare_wgen_filex(sample_filex())
    assert "{{ wther }}" in text
    assert "Sim2008" in text
    assert "Sim2014" not in text
    assert "CNYC1401" not in text
    assert " 2 31" in text
    rendered = text.replace("{{ wther }}", "W")
    assert runtime_filex_mode(rendered) == "W"


def test_runtime_filex_mode_reads_weather_method_column():
    assert runtime_filex_mode("@N METHODS WTHER INCON\n 1 ME W M\n") == "W"
    assert runtime_filex_mode("@N METHODS WTHER INCON\n 1 ME M M\n") == "M"


def test_pdi_seed_value_is_distinctly_recorded_as_rseed1():
    assert pdi_seed_configured("rseed1_ = 101\n", 101)
    assert not pdi_seed_configured("rseed1_ = 101\n", 102)


def test_daily_state_capture_maps_weather_names_and_constructs_dates():
    states = [
        {"rain": 0.0, "srad": 14.0, "tmax": 30.0, "tmin": 20.0},
        {"RAIN": 2.5, "SRAD": 15.0, "TMAX": 31.0, "TMIN": 19.0},
    ]
    rows = daily_weather_from_states(states, date(2008, 6, 1))
    assert rows[0]["DATE"] == "2008-06-01"
    assert rows[1]["DATE"] == "2008-06-02"
    assert rows[1]["RAIN"] == 2.5


def test_daily_state_capture_prefers_explicit_doy_and_date():
    rows = daily_weather_from_states(
        [{"date": "2008-07-01", "doy": 183, "rain": 1, "srad": 10, "tmax": 25, "tmin": 15}]
    )
    assert rows[0]["DATE"] == "2008-07-01"
    assert rows[0]["DOY"] == 183


def test_canonical_weather_bytes_are_order_independent():
    a = {"DATE": "2008-06-01", "DOY": 153, "RAIN": 0.0, "SRAD": 10.0, "TMAX": 20.0, "TMIN": 8.0}
    b = {"DATE": "2008-06-02", "DOY": 154, "RAIN": 1.25, "SRAD": 11.0, "TMAX": 21.0, "TMIN": 9.0}
    assert canonical_weather_bytes([a, b]) == canonical_weather_bytes([b, a])


def test_same_seed_weather_comparison_detects_exact_identity():
    rows = [
        {"DATE": "2008-06-01", "RAIN": 0.0, "SRAD": 10.0, "TMAX": 20.0, "TMIN": 8.0},
        {"DATE": "2008-06-02", "RAIN": 1.0, "SRAD": 11.0, "TMAX": 21.0, "TMIN": 9.0},
    ]
    result = compare_weather_rows(rows, [dict(row) for row in rows])
    assert result["dates_aligned"]
    assert not result["different_weather_values"]
    assert all(value["different_values"] == 0 for value in result["variables"].values())


def test_different_seed_weather_comparison_reports_variable_differences():
    left = [{"DATE": "2008-06-01", "RAIN": 0.0, "SRAD": 10.0, "TMAX": 20.0, "TMIN": 8.0}]
    right = [{"DATE": "2008-06-01", "RAIN": 2.0, "SRAD": 12.0, "TMAX": 22.0, "TMIN": 9.0}]
    result = compare_weather_rows(left, right)
    assert result["different_weather_values"]
    assert result["variables"]["RAIN"]["different_values"] == 1
    assert result["variables"]["TMIN"]["mean_absolute_difference"] == 1.0


def test_summary_uses_simulation_season_statistics():
    rows = [
        {"DATE": "2008-06-01", "RAIN": 0.0, "SRAD": 10.0, "TMAX": 20.0, "TMIN": 8.0},
        {"DATE": "2008-06-02", "RAIN": 2.0, "SRAD": 12.0, "TMAX": 22.0, "TMIN": 9.0},
        {"DATE": "2008-06-03", "RAIN": 0.0, "SRAD": 11.0, "TMAX": 21.0, "TMIN": 7.0},
    ]
    result = summarize_weather(rows)
    assert result["window_type"] == "simulation-season statistics"
    assert result["season_rainfall_mm"] == 2.0
    assert result["rainy_days_gt_0mm"] == 1
    assert result["longest_dry_spell_days"] == 1


def test_physical_sanity_detects_negative_weather_and_temperature_inversion():
    rows = [{"DATE": "2008-06-01", "RAIN": -1.0, "SRAD": 0.0, "TMAX": 10.0, "TMIN": 11.0}]
    result = physical_sanity(rows)
    assert result["status"] == "FAIL_OR_INCOMPLETE"
    assert any("negative RAIN" in issue for issue in result["issues"])
    assert any("TMAX below TMIN" in issue for issue in result["issues"])


def test_physical_sanity_marks_missing_weather_as_not_run():
    assert physical_sanity([])["status"] == "NOT_RUN_NO_WEATHER_OUTPUT"


def test_runtime_errors_are_classified_specifically():
    assert classify_runtime_error("CNYC.CLI not found in working directory") == "FAIL_CLI_LOOKUP"
    assert classify_runtime_error("parse error on CLI field ALPHA") == "FAIL_CLI_PARSE"
    assert classify_runtime_error("Unknown ERROR 5010; file CNYC.CLI; Error key: WGENIN") == "FAIL_CLI_PARSE"
    assert classify_runtime_error("WGEN runtime rejected climate file") == "FAIL_WGEN_RUNTIME"
