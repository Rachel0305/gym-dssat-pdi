from pathlib import Path

from scripts.debug_yc_coordinate_propagation import (
    EXPECTED,
    coordinate_gate,
    parse_dssat_field_coordinates,
    parse_filex_coordinates,
    render_filex_coordinates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILEX = PROJECT_ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013/YC/CNYC0801.MZX"
OLD_SEED101 = PROJECT_ROOT / "results/yc_wgen_cli_pilot/003_06_06/crop_smoke/seed_101/runtime_snapshot"


def test_filex_coordinate_parser_maps_xcrd_to_longitude_and_ycrd_to_latitude():
    sample = "*FIELDS\n@L ...........XCRD ...........YCRD .....ELEV .............AREA\n 1       116.57000        36.83000      22.0               -99\n"
    assert parse_filex_coordinates(sample) == {"LONG": 116.57, "LAT": 36.83, "ELEV": 22.0}


def test_filex_fixed_width_coordinate_rendering_preserves_record_key_and_roundtrips():
    source = SOURCE_FILEX.read_text(encoding="utf-8", errors="replace")
    rendered = render_filex_coordinates(source, EXPECTED)
    assert parse_filex_coordinates(rendered) == {"LONG": 116.57, "LAT": 36.83, "ELEV": 22.0}


def test_coordinate_gate_requires_all_three_expected_runtime_values():
    assert coordinate_gate({"LAT": 36.83, "LONG": 116.57, "ELEV": 22.0})
    assert not coordinate_gate({"LAT": 36.83, "LONG": 116.57, "ELEV": None})
    assert not coordinate_gate({"LAT": 36.83, "LONG": -99.0, "ELEV": 22.0})


def test_dssat48_inp_field_coordinate_substitution_gate_detects_retained_placeholders():
    inp = (OLD_SEED101 / "DSSAT48.INP").read_text(encoding="utf-8", errors="replace")
    actual = parse_dssat_field_coordinates(inp)
    assert actual == {"LONG": -999.0, "LAT": -99.0, "ELEV": -99.0}
    assert not coordinate_gate(actual)


def test_dssat48_inp_gate_reads_field_record_not_soil_site_metadata():
    inp = (OLD_SEED101 / "DSSAT48.INP").read_text(encoding="utf-8", errors="replace")
    assert "36.830 116.570" in inp
    assert not coordinate_gate(parse_dssat_field_coordinates(inp))


def test_dssat48_inp_field_row_accepts_expected_values_after_substitution_in_fixture():
    corrected_fixture = "*FIELDS\n CNYC2008 CNYC.CLI\n 116.57000 36.83000 22.00 1.0\n*INITIAL CONDITIONS\n"
    assert coordinate_gate(parse_dssat_field_coordinates(corrected_fixture))


def test_dssat48_inh_companion_field_record_also_retains_placeholders():
    inh = (OLD_SEED101 / "DSSAT48.INH").read_text(encoding="utf-8", errors="replace")
    assert parse_dssat_field_coordinates(inh) == {"LONG": -999.0, "LAT": -99.0, "ELEV": -99.0}


def test_all_three_retained_runtime_filex_files_preserve_lat_long_elevation():
    for run_id in ("historical_control_retry4", "seed_101", "seed_104"):
        path = PROJECT_ROOT / "results/yc_wgen_cli_pilot/003_06_06/crop_smoke" / run_id / "runtime_snapshot/fileX.MZX"
        assert parse_filex_coordinates(path.read_text(encoding="utf-8", errors="replace")) == {
            "LONG": 116.57,
            "LAT": 36.83,
            "ELEV": 22.0,
        }
