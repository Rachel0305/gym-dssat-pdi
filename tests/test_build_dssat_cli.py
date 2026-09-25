from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_dssat_cli as cli


WEATHER = ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv"


class BuildDssatCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records, cls.input_hash = cli.read_weather(WEATHER)
        cls.wgen = cli.fit_monthly_statistics(cls.records)
        cls.monthly = cli.monthly_weather_summary(cls.records)

    def test_frozen_hash_is_required(self) -> None:
        self.assertEqual(cls_hash := self.input_hash, cli.EXPECTED_SHA256)
        with self.assertRaisesRegex(ValueError, "SHA256 不匹配"):
            cli.read_weather(WEATHER, "0" * 64)

    def test_month_partition_includes_leap_days(self) -> None:
        february = [record for record in self.records if record.day.month == 2]
        self.assertEqual(len(february), 283)
        self.assertEqual(sum(int(row["day_count"]) for row in self.wgen), 3653)
        self.assertEqual([int(row["month"]) for row in self.wgen], list(range(1, 13)))

    def test_wet_day_classification_and_monthly_counts(self) -> None:
        wet_total = sum(record.rain > 0.0 for record in self.records)
        dry_total = sum(record.rain <= 0.0 for record in self.records)
        self.assertEqual(wet_total, 587)
        self.assertEqual(dry_total, 3066)
        self.assertEqual(sum(int(row["wet_day_count"]) for row in self.wgen), wet_total)

    def test_transition_counts_assign_cross_year_pair_to_current_month(self) -> None:
        records = [
            cli.WeatherDay(date(2004, 12, 31), 1.0, 2.0, 0.0, 0.0),
            cli.WeatherDay(date(2005, 1, 1), 1.0, 2.0, 0.0, 0.2),
        ]
        transitions = cli.count_monthly_transitions(records, initial_previous_wet=False)
        self.assertEqual(transitions[12]["dry_to_dry"], 1)
        self.assertEqual(transitions[1]["dry_to_wet"], 1)

    def test_temperature_radiation_and_gamma_parameters_crosscheck(self) -> None:
        result = cli.crosscheck_statistics(self.records, self.wgen)
        self.assertTrue(result["passed"])
        self.assertEqual(result["months_checked"], 12)
        self.assertTrue(all(not item["mismatches"] for item in result["month_results"]))
        self.assertTrue(all(0 < float(row["ALPHA"]) <= 0.998 for row in self.wgen))

    def test_leap_day_is_retained_in_february(self) -> None:
        leap_dates = {record.day for record in self.records if record.day.month == 2 and record.day.day == 29}
        self.assertEqual(leap_dates, {date(2004, 2, 29), date(2008, 2, 29), date(2012, 2, 29)})

    def test_rejects_tmax_below_tmin(self) -> None:
        with self.assertRaisesRegex(ValueError, "TMAX < TMIN"):
            self._read_single_row("2004-01-01", "1", "0", "1", "0")

    def test_rejects_negative_rain_or_srad(self) -> None:
        with self.assertRaisesRegex(ValueError, "RAIN < 0"):
            self._read_single_row("2004-01-01", "1", "2", "0", "-0.1")
        with self.assertRaisesRegex(ValueError, "SRAD < 0"):
            self._read_single_row("2004-01-01", "-0.1", "2", "0", "0")

    def test_cli_output_is_deterministic_and_has_twelve_months(self) -> None:
        first = cli._cli_text(self.records, "CNYC", 36.830, 116.570, 22, self.wgen, self.monthly)
        second = cli._cli_text(self.records, "CNYC", 36.830, 116.570, 22, self.wgen, self.monthly)
        self.assertEqual(first.encode("ascii"), second.encode("ascii"))
        schema = cli.check_cli_schema(first)
        self.assertTrue(schema["passed"])
        self.assertEqual(schema["monthly_averages_months"], list(range(1, 13)))
        self.assertEqual(schema["wgen_parameters_months"], list(range(1, 13)))
        self.assertEqual(schema["wgen_expected_read_format"], "(I6,14(1X,F5.0))")
        self.assertEqual(schema["wgen_statistical_field_count"], 14)
        self.assertEqual(schema["wgen_total_column_count"], 15)
        self.assertEqual(schema["wgen_fixed_width_malformed"], [])

    def test_wgen_fixed_width_repair_preserves_frozen_parameter_values(self) -> None:
        candidate = ROOT / "results/yc_wgen_cli_pilot/003_06_04/generated/CNYC.CLI"
        self.assertEqual(
            cli.sha256_file(candidate),
            "5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0",
        )
        original = candidate.read_text(encoding="ascii")
        original_schema = cli.check_cli_schema(original)
        self.assertFalse(original_schema["passed"])
        self.assertEqual(len(original_schema["wgen_fixed_width_malformed"]), 12)
        self.assertEqual(original_schema["wgen_fixed_width_malformed"][0]["error"], "WGEN row width must be 90, got 93")

        fixed = cli.reformat_wgen_fixed_width(original)
        fixed_schema = cli.check_cli_schema(fixed)
        self.assertTrue(fixed_schema["passed"])
        self.assertEqual(fixed_schema["wgen_fixed_width_malformed"], [])
        original_lines = original.splitlines()
        fixed_lines = fixed.splitlines()
        changed_indices = [index for index, pair in enumerate(zip(original_lines, fixed_lines)) if pair[0] != pair[1]]
        self.assertEqual(changed_indices, list(range(26, 38)))

        fixed_rows = [fixed_lines[index] for index in changed_indices]
        self.assertEqual([len(line) for line in fixed_rows], [90] * 12)
        for original_index, fixed_line in zip(changed_indices, fixed_rows):
            source_tokens = original_lines[original_index].split()
            parsed = cli.parse_wgen_parameter_row(fixed_line)
            self.assertEqual(parsed["MTH"], int(source_tokens[0]))
            self.assertEqual(
                [parsed[name] for name in cli.WGEN_PARAMETER_FIELDS],
                [float(value) for value in source_tokens[1:]],
            )

    def test_wgen_formatter_rejects_numeric_width_overflow(self) -> None:
        values = ["1.0"] * len(cli.WGEN_PARAMETER_FIELDS)
        values[4] = "-100.0"
        with self.assertRaisesRegex(ValueError, "does not fit F5"):
            cli.format_wgen_parameter_values("1", values)

    def _read_single_row(self, day: str, srad: str, tmax: str, tmin: str, rain: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["DATE", "SRAD", "TMAX", "TMIN", "RAIN"])
                writer.writerow([day, srad, tmax, tmin, rain])
            cli.read_weather(path, cli.sha256_file(path))


if __name__ == "__main__":
    unittest.main()
