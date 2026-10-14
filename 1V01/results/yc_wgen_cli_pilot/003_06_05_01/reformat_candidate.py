from __future__ import annotations

import difflib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dssat_cli as cli


RESULTS = Path(__file__).resolve().parent
SOURCE = ROOT / "results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI"
FIXED = RESULTS / "candidate/CNYC_parsefix_01.CLI"
SAMPLE = ROOT / "benchmark_results/026_05/ppo_runs/2012/seed0/transfer_frozen_ppo_seed0/input/CNSY.CLI"
EXPECTED_SOURCE_HASH = "5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0"
SOURCE_V480 = "https://github.com/DSSAT/dssat-csm-os/blob/v4.8.0.24/Weather/WGEN.for#L408-L414"
SOURCE_V485 = "https://github.com/DSSAT/dssat-csm-os/blob/v4.8.5.0/Weather/WGEN.for#L412-L418"


def read_text_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="ascii", newline="") as stream:
        return stream.read()


def write_text_preserving_newlines(path: Path, text: str) -> None:
    with path.open("w", encoding="ascii", newline="") as stream:
        stream.write(text)


def wgen_lines(text: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    active = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("*WGEN PARAMETERS"):
            active = True
            continue
        if active and line.startswith("*"):
            break
        if active and line.strip() and not line.lstrip().startswith("@"):
            rows.append((line_number, line))
    return rows


def wgen_header(text: str) -> str:
    active = False
    for line in text.splitlines():
        if line.startswith("*WGEN PARAMETERS"):
            active = True
            continue
        if active and line.startswith("*"):
            break
        if active and line.startswith("@"):
            return line
    raise ValueError("WGEN header not found")


def visible(line: str) -> str:
    return line.replace(" ", "·").replace("\t", "→")


def token_positions(line: str) -> list[str]:
    import re

    return [f"{m.group()}[{m.start()}-{m.end() - 1}]" for m in re.finditer(r"\S+", line)]


def schema_rows(source_first_row: str, fixed_first_row: str) -> list[dict[str, object]]:
    tokens = source_first_row.split()
    fixed = cli.parse_wgen_parameter_row(fixed_first_row)
    schema: list[dict[str, object]] = [
        {
            "field_position": "columns 1-6",
            "field_name": "MTH",
            "expected_type": "INTEGER",
            "expected_width": 6,
            "expected_format": "I6",
            "source_evidence": f"WGENIN READ {SOURCE_V480}; same READ in {SOURCE_V485}",
            "current_value": int(tokens[0]),
            "current_rendering": repr(fixed_first_row[:6]),
            "compatible_yes_no": True,
        }
    ]
    for index, field in enumerate(cli.WGEN_PARAMETER_FIELDS):
        separator = 6 + index * 6
        start = separator + 1
        end = start + 5
        schema.append(
            {
                "field_position": f"separator column {separator + 1}; value columns {start + 1}-{end}",
                "field_name": field,
                "expected_type": "REAL",
                "expected_width": 5,
                "expected_format": "1X,F5.0",
                "source_evidence": f"WGENIN READ {SOURCE_V480}; same READ in {SOURCE_V485}",
                "current_value": float(tokens[index + 1]),
                "current_rendering": repr(fixed_first_row[start:end]),
                "compatible_yes_no": math.isclose(
                    float(tokens[index + 1]), float(fixed[field]), rel_tol=0.0, abs_tol=0.0
                ),
            }
        )
    return schema


def write_report(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content.rstrip() + "\n")


def main() -> None:
    source_hash_before = cli.sha256_file(SOURCE)
    if source_hash_before != EXPECTED_SOURCE_HASH:
        raise SystemExit(f"Frozen source CLI hash mismatch: {source_hash_before}")

    original = read_text_preserving_newlines(SOURCE)
    original_rows = wgen_lines(original)
    if len(original_rows) != 12:
        raise SystemExit(f"Expected 12 source WGEN rows, got {len(original_rows)}")
    fixed_text = cli.reformat_wgen_fixed_width(original)
    fixed_rows = wgen_lines(fixed_text)
    if len(fixed_rows) != len(original_rows) or any(
        old_row.split() != new_row.split()
        for (_, old_row), (_, new_row) in zip(original_rows, fixed_rows)
    ):
        raise SystemExit("WGEN numeric tokens changed during serialization")

    original_lines = original.splitlines(keepends=True)
    fixed_lines = fixed_text.splitlines(keepends=True)
    changed = [i for i, pair in enumerate(zip(original_lines, fixed_lines)) if pair[0] != pair[1]]
    expected_changed = [line_number - 1 for line_number, _ in original_rows]
    if changed != expected_changed:
        raise SystemExit(f"Unexpected changed lines: {changed}; expected only {expected_changed}")

    FIXED.parent.mkdir(parents=True, exist_ok=True)
    write_text_preserving_newlines(FIXED, fixed_text)
    fixed_hash = cli.sha256_file(FIXED)
    if cli.sha256_file(SOURCE) != EXPECTED_SOURCE_HASH:
        raise SystemExit("Frozen source CLI changed during candidate generation")

    old_number, old_first = original_rows[0]
    second_number, old_second = original_rows[1]
    fixed_number, fixed_first = fixed_rows[0]
    header_line = original_lines[old_number - 2].rstrip("\r\n")
    diagnostic_lines = [
        "YC CNYC.CLI WGEN line 27 layout diagnosis",
        f"source_cli={SOURCE.relative_to(ROOT)}",
        f"source_sha256={source_hash_before}",
        f"WGEN section begins at line {old_number - 2}; header line {old_number - 1}",
        "",
        f"line {old_number - 1} header raw={header_line!r}",
        f"line {old_number - 1} header length={len(header_line)} visible={visible(header_line)}",
        "header token positions: " + " | ".join(token_positions(header_line)),
        "",
        f"line {old_number} raw={old_first!r}",
        f"line {old_number} length={len(old_first)} visible={visible(old_first)}",
        "tokenized fields: " + " | ".join(token_positions(old_first)),
        f"parsed source tokens: MTH={int(old_first.split()[0])}; "
        + ", ".join(
            f"{name}={float(value)}"
            for name, value in zip(cli.WGEN_PARAMETER_FIELDS, old_first.split()[1:])
        ),
        f"line {second_number} raw={old_second!r}",
        f"line {second_number} length={len(old_second)} visible={visible(old_second)}",
        "line 28 tokenized fields: " + " | ".join(token_positions(old_second)),
        f"line 28 parsed source tokens: MTH={int(old_second.split()[0])}; "
        + ", ".join(
            f"{name}={float(value)}"
            for name, value in zip(cli.WGEN_PARAMETER_FIELDS, old_second.split()[1:])
        ),
        "fixed-format slices from original row (Fortran skips each separator column):",
        f"line {fixed_number} fixed raw={fixed_first!r}",
        f"line {fixed_number} fixed length={len(fixed_first)} visible={visible(fixed_first)}",
        "",
        "WGENIN contract: (I6,14(1X,F5.0)); total width 6 + 14*(1+5) = 90 columns.",
        "MTH is one integer column group; the remaining 14 statistical values are REAL fields.",
        "",
        "Original row field slices under WGENIN's fixed positions:",
    ]
    original_tokens = old_first.split()
    diagnostic_lines.append(f"MTH I6 columns 1-6: {old_first[:6]!r}")
    for index, field in enumerate(cli.WGEN_PARAMETER_FIELDS):
        separator = 6 + index * 6
        raw_slot = old_first[separator : separator + 6]
        raw_value = old_first[separator + 1 : separator + 6]
        try:
            parsed_value = float(raw_value.strip())
            parse_result = f"parsed={parsed_value}"
            matches_value = parsed_value == float(original_tokens[index + 1])
        except ValueError:
            parse_result = "decimal-token parser failure"
            matches_value = False
        diagnostic_lines.append(
            f"original {field}: slot cols {separator + 1}-{separator + 6}={raw_slot!r}; "
            f"F5={raw_value!r}; token={original_tokens[index + 1]!r}; "
            f"{parse_result}; value_matches={matches_value}"
        )
    diagnostic_lines.append("First F5 slice that fails decimal-token parsing: XDSD, after the XDMN width spill.")
    diagnostic_lines.append("This slice analysis follows the declared columns; runtime IOSTAT behavior is determined by the one-shot DSSAT smoke.")
    diagnostic_lines.append("")
    diagnostic_lines.append("Field schema from the first YC month row after format-only serialization:")
    for item in schema_rows(old_first, fixed_first):
        diagnostic_lines.append(
            f"{item['field_name']}: {item['field_position']}; value={item['current_value']}; "
            f"rendering={item['current_rendering']}; format={item['expected_format']}; "
            f"compatible={item['compatible_yes_no']}"
        )
    write_report(RESULTS / "diagnostics/original_line27_layout.txt", "\n".join(diagnostic_lines))

    sample_text = read_text_preserving_newlines(SAMPLE)
    sample_rows = wgen_lines(sample_text)
    sample_header = wgen_header(sample_text)
    source_header = wgen_header(original)
    comparison = [
        "WGEN row schema comparison: YC candidate vs repository CNSY.CLI",
        f"YC source: {SOURCE.relative_to(ROOT)} (SHA256 {source_hash_before})",
        f"CNSY structural reference: {SAMPLE.relative_to(ROOT)}",
        "CNSY numerical parameter values are intentionally omitted; only layout is compared.",
        "",
        f"header_identical={sample_header == source_header}",
        f"header={sample_header}",
        f"YC first row: 15 tokens including MTH, {len(old_first)} characters, decimals retained from candidate.",
        f"CNSY first row: 15 tokens including MTH, {len(sample_rows[0][1])} characters; values redacted.",
        f"fixed candidate first row: 15 tokens including MTH, {len(fixed_first)} characters.",
        f"both DSSAT source versions read: {cli.WGEN_READ_FORMAT}",
        "column order: MTH, SDMN, SDSD, SWMN, SWSD, XDMN, XDSD, XWMN, XWSD, NAMN, NASD, ALPHA, RTOT, PDW, RNUM.",
        "decimal rendering: source READ accepts explicit decimal points inside each F5.0 input field; generator retains its existing 1- or 3-decimal token precision.",
        "",
        "Widths by field group:",
        "MTH: I6 (6 columns); 14 statistical values: each 1X + F5.0 (6 columns per value).",
        "YC source row width=93; CNSY reference row width=90; fixed YC candidate row width=90.",
        "Only the original YC row serialization is changed. No CNSY climate parameter values are copied.",
        f"4.8.0.24 source: {SOURCE_V480}",
        f"4.8.5.0 source: {SOURCE_V485}",
    ]
    write_report(RESULTS / "diagnostics/wgen_row_schema_comparison.txt", "\n".join(comparison))

    diff = difflib.unified_diff(
        original.splitlines(), fixed_text.splitlines(),
        fromfile=str(SOURCE.relative_to(ROOT)),
        tofile=str(FIXED.relative_to(ROOT)),
        lineterm="",
    )
    write_report(RESULTS / "candidate/parsefix_diff.txt", "\n".join(diff))

    value_preserved: list[bool] = []
    for (_, old_row), (_, new_row) in zip(original_rows, fixed_rows):
        old_tokens = old_row.split()
        parsed = cli.parse_wgen_parameter_row(new_row)
        value_preserved.append(
            parsed["MTH"] == int(old_tokens[0])
            and all(
                parsed[name] == float(token)
                for name, token in zip(cli.WGEN_PARAMETER_FIELDS, old_tokens[1:])
            )
        )
    schema = cli.check_cli_schema(fixed_text)
    static = {
        "status": "PASS" if schema["passed"] and all(value_preserved) else "FAIL",
        "source_cli": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash_before,
        "fixed_cli": str(FIXED.relative_to(ROOT)),
        "fixed_sha256": fixed_hash,
        "read_format": cli.WGEN_READ_FORMAT,
        "statistical_value_count": 14,
        "total_columns_including_month": 15,
        "expected_record_width": cli.WGEN_ROW_WIDTH,
        "months_checked": [int(row.split()[0]) for _, row in fixed_rows],
        "all_12_rows_exact_width": all(len(row) == cli.WGEN_ROW_WIDTH for _, row in fixed_rows),
        "all_fields_finite_and_parseable": not schema["wgen_fixed_width_malformed"],
        "no_nan_inf_or_width_overflow": not schema["nonfinite_tokens"] and all(value_preserved),
        "original_line27_first_fixed_field_failure": "XDSD",
        "parameter_values_preserved_exactly_as_source_tokens": all(value_preserved),
        "parameter_values_changed": False,
        "format_only_change": True,
        "changed_source_lines_1_based": [index + 1 for index in changed],
        "changed_field_widths": {"XDMN": "7 to 6", "XWMN": "7 to 6", "NAMN": "7 to 6"},
        "schema": schema_rows(old_first, fixed_first),
        "schema_check": schema,
    }
    (RESULTS / "validation/static_parse_check.json").write_text(
        json.dumps(static, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    metadata = {
        "source_cli": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash_before,
        "fixed_cli": str(FIXED.relative_to(ROOT)),
        "fixed_sha256": fixed_hash,
        "changed_lines": [index + 1 for index in changed],
        "changed_fields": ["XDMN", "XWMN", "NAMN"],
        "parameter_values_changed": "NO",
        "format_only_change": "YES",
        "root_cause": "Three fields were serialized with width 7 instead of the WGENIN 1X,F5.0 six-column group, shifting subsequent columns and producing 93-character rows instead of 90.",
        "evidence": [SOURCE_V480, SOURCE_V485, str((RESULTS / "diagnostics/original_line27_layout.txt").relative_to(ROOT)), str((RESULTS / "diagnostics/wgen_row_schema_comparison.txt").relative_to(ROOT))],
    }
    (RESULTS / "candidate/parsefix_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if static["status"] != "PASS":
        raise SystemExit("Static WGEN fixed-width parse check failed")
    print(json.dumps({"fixed_cli": str(FIXED), "fixed_sha256": fixed_hash, "static_parse_check": static["status"], "source_cli_unchanged": cli.sha256_file(SOURCE) == EXPECTED_SOURCE_HASH}, ensure_ascii=False))


if __name__ == "__main__":
    main()
