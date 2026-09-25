import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "results" / "yc_wgen_cli_pilot" / "003_06_05_02" / "analyze_seed_pilot.py"
SPEC = importlib.util.spec_from_file_location("yc_wgen_final_seed_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def row(day, doy, rain=0.0, srad=10.0, tmax=20.0, tmin=8.0):
    return {"DATE": day, "DOY": doy, "RAIN": rain, "SRAD": srad, "TMAX": tmax, "TMIN": tmin}


def test_first_difference_includes_date_doy_weather_and_sequence_length():
    a = [row("2008-06-01", 153), row("2008-06-02", 154)]
    assert analysis.first_difference(a, [dict(value) for value in a]) is None
    assert analysis.first_difference(a, [row("2008-06-01", 153), row("2008-06-02", 154, rain=1)])["field"] == "RAIN"
    assert analysis.first_difference(a, a[:1])["field"] == "ROW_COUNT"


def test_pairwise_comparison_counts_weather_field_differences_on_common_dates():
    result = analysis.compare_sequences(
        [row("2008-06-01", 153, rain=0), row("2008-06-02", 154, rain=1)],
        [row("2008-06-01", 153, rain=2), row("2008-06-03", 155, rain=3)],
    )
    assert result["left_only_dates"] == ["2008-06-02"]
    assert result["right_only_dates"] == ["2008-06-03"]
    assert result["field_differences"]["RAIN"]["differing_rows_on_common_dates"] == 1
    assert not result["identical_full_sequence_including_date_doy_and_weather"]


def test_weather_qc_checks_contiguous_dates_doy_and_runtime_boundaries():
    text = "CSM     YEAR DOY = 2008 153\nENDRUN  YEAR DOY = 2008 154\n"
    result = analysis.physical_check(
        [row("2008-06-01", 153), row("2008-06-02", 154, rain=1)], [], text
    )
    assert result["status"] == "PASS"
    assert result["checks"]["sequence_dates_match_runtime_info_out_start_and_end"]


def test_weather_qc_fails_on_temperature_inversion_and_duplicate_date():
    rows = [row("2008-06-01", 153, tmax=5, tmin=8), row("2008-06-01", 153)]
    result = analysis.physical_check(rows, [], "")
    assert result["status"] == "FAIL_OR_INCOMPLETE"
    assert not result["checks"]["no_duplicate_dates_or_rows"]
    assert not result["checks"]["tmax_greater_than_or_equal_to_tmin"]
