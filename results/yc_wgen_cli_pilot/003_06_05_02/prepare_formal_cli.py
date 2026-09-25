from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent
WEATHER = ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"
OLD_CLI = ROOT / "results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI"
PARSEFIX_CLI = ROOT / "results/yc_wgen_cli_pilot/003_06_05_01/candidate/CNYC_parsefix_01.CLI"
FINAL_CLI = RESULTS / "final/CNYC.CLI"
WEATHER_SHA256 = "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34"
OLD_CLI_SHA256 = "5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0"
PARSEFIX_SHA256 = "65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929"

sys.path.insert(0, str(ROOT / "scripts"))
import build_dssat_cli as generator  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def wgen_rows(text: str) -> list[str]:
    rows: list[str] = []
    in_wgen = False
    for line in text.splitlines():
        if line.startswith("*WGEN PARAMETERS"):
            in_wgen = True
            continue
        if in_wgen and line.startswith("*"):
            in_wgen = False
        if in_wgen and line.strip() and not line.lstrip().startswith("@"):
            rows.append(line)
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    if FINAL_CLI.exists():
        raise SystemExit(f"Refusing to overwrite formal corrected CLI: {FINAL_CLI}")
    if sha256(WEATHER) != WEATHER_SHA256:
        raise SystemExit("Frozen training weather SHA256 mismatch; Gate A stopped")
    if sha256(OLD_CLI) != OLD_CLI_SHA256:
        raise SystemExit("Historical broken CLI hash changed; refusing to continue")
    if sha256(PARSEFIX_CLI) != PARSEFIX_SHA256:
        raise SystemExit("Parse-fixed candidate hash changed; refusing to continue")

    records, input_hash = generator.read_weather(WEATHER, WEATHER_SHA256)
    if len(records) != 3653 or records[0].day.isoformat() != "2004-01-01" or records[-1].day.isoformat() != "2013-12-31":
        raise SystemExit("Frozen train-weather row count/date range failed preflight")

    metadata = generator.build(WEATHER, FINAL_CLI, expected_sha256=WEATHER_SHA256)
    formal_hash = sha256(FINAL_CLI)
    formal_text = FINAL_CLI.read_text(encoding="ascii")
    parsefix_text = PARSEFIX_CLI.read_text(encoding="ascii")
    formal_rows = wgen_rows(formal_text)
    parsefix_rows = wgen_rows(parsefix_text)
    schema = generator.check_cli_schema(formal_text)
    value_equal = (
        len(formal_rows) == len(parsefix_rows) == 12
        and [generator.parse_wgen_parameter_row(row) for row in formal_rows]
        == [generator.parse_wgen_parameter_row(row) for row in parsefix_rows]
    )
    serialization_equal = formal_rows == parsefix_rows
    line_changes = [
        {"line": number, "parsefix": before, "formal": after}
        for number, (before, after) in enumerate(
            zip(parsefix_text.splitlines(), formal_text.splitlines()), start=1
        )
        if before != after
    ]
    if len(parsefix_text.splitlines()) != len(formal_text.splitlines()):
        line_changes.append({
            "line_count": {
                "parsefix": len(parsefix_text.splitlines()),
                "formal": len(formal_text.splitlines()),
            }
        })

    byte_equal = sha256(PARSEFIX_CLI) == formal_hash
    comparison = {
        "formal_cli": str(FINAL_CLI.relative_to(ROOT)),
        "formal_sha256": formal_hash,
        "parsefix_candidate": str(PARSEFIX_CLI.relative_to(ROOT)),
        "parsefix_sha256": sha256(PARSEFIX_CLI),
        "formal_cli_matches_parsefix_candidate": byte_equal,
        "wgen_month_count_formal": len(formal_rows),
        "wgen_month_count_parsefix": len(parsefix_rows),
        "wgen_parameter_values_identical": value_equal,
        "wgen_serialization_equivalent": serialization_equal,
        "schema_status": "PASS" if schema["passed"] else "FAIL",
        "differences_by_text_line": line_changes,
        "difference_explanation": (
            "No byte or line differences; formal generation reproduces the parse-fixed candidate exactly."
            if byte_equal
            else "CLI hashes differ; inspect differences_by_text_line. WGEN values and row serialization must still match exactly."
        ),
        "frozen_weather_sha256": input_hash,
        "frozen_weather_row_count": len(records),
        "frozen_weather_date_range": [records[0].day.isoformat(), records[-1].day.isoformat()],
    }

    status_ready = bool(schema["passed"] and value_equal and serialization_equal)
    formal_status = "CORRECTED_CLI_READY_FOR_SEED_PILOT" if status_ready else "FAIL_FORMAL_CLI_REGENERATION"
    metadata.update({
        "official_corrected_cli_path": str(FINAL_CLI.relative_to(ROOT)),
        "official_corrected_cli_sha256": formal_hash,
        "formal_cli_matches_parsefix_candidate": "YES" if byte_equal else "NO",
        "parameter_values_identical_to_parsefix": "YES" if value_equal else "NO",
        "wgen_serialization_equivalent_to_parsefix": "YES" if serialization_equal else "NO",
        "formal_cli_status": formal_status,
        "validation_data_used_for_fitting": False,
        "weather_candidate_modified": False,
    })
    schema["formal_cli_status"] = formal_status
    schema["formal_cli_sha256"] = formal_hash
    write_json(RESULTS / "final/final_cli_metadata.json", metadata)
    write_json(RESULTS / "final/final_cli_schema_check.json", schema)
    write_json(RESULTS / "validation/formal_cli_comparison.json", comparison)

    generation_log = (RESULTS / "final/cli_generation_log.txt").read_text(encoding="utf-8")
    generation_log += (
        f"\nformal_cli_status={formal_status}"
        f"\nformal_cli_matches_parsefix_candidate={'YES' if byte_equal else 'NO'}"
        f"\nwgen_parameter_values_identical={'YES' if value_equal else 'NO'}"
        f"\nwgen_serialization_equivalent={'YES' if serialization_equal else 'NO'}"
        f"\nold_broken_cli_sha256_before={OLD_CLI_SHA256}"
        f"\nold_broken_cli_sha256_after={sha256(OLD_CLI)}"
        f"\nfrozen_weather_sha256_after={sha256(WEATHER)}\n"
    )
    (RESULTS / "final/final_cli_generation_log.txt").write_text(generation_log, encoding="utf-8", newline="\n")
    print(json.dumps({
        "formal_cli_status": formal_status,
        "formal_cli_sha256": formal_hash,
        "matches_parsefix_candidate": byte_equal,
        "parameter_values_identical": value_equal,
        "wgen_serialization_equivalent": serialization_equal,
        "schema_pass": schema["passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if status_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
