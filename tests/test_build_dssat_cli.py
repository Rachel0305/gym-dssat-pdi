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
