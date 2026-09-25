import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "results" / "yc_wgen_cli_pilot" / "003_06_07_02"
PATCHED_SOURCE = ROOT / "tests" / "fixtures" / "dssat_v48024_ipfld_coordinates.for"
PATCH_FILE = TASK_DIR / "patch" / "ipfld_coordinate_fix.patch"


def read_fixed_decimal(token, sentinel):
    """Small fixture model of READ(..., IOSTAT=ERRNUM) and its error branch."""
    if not token.strip():
        return sentinel, 1
    try:
        return float(token), 0
    except ValueError:
        return sentinel, 1


def coordinates_transfer(x, y, x_text, y_text, last_iostat):
    """Mirror the patched coordinate gate; last_iostat is intentionally ignored."""
    del last_iostat
    return (
        -90.0 <= y <= 90.0
        and -180.0 <= x <= 180.0
        and bool(y_text.strip())
        and bool(x_text.strip())
        and (abs(y) > 1e-15 or abs(x) > 1e-15)
    )


class CoordinatePatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = PATCHED_SOURCE.read_text(encoding="utf-8")
        cls.patch = PATCH_FILE.read_text(encoding="utf-8")

    def test_only_the_three_coordinate_iostat_branches_are_fixed(self):
        expected = (
            ("CXCRD", "XCRD", "-999.0"),
            ("CYCRD", "YCRD", "-99.0"),
            ("CELEV", "ELEV", "-99.0"),
        )
        for token, value, sentinel in expected:
            match = re.search(
                rf"READ\({token},.*?\n\s*IF\(ERRNUM \.NE\. 0\) THEN"
                rf"(?P<body>.*?)\n\s*ENDIF",
                self.source,
                re.DOTALL,
            )
            self.assertIsNotNone(match, token)
            self.assertIn(f"{value} = {sentinel}", match.group("body"))
            self.assertIn("CALL WARNING", match.group("body"))

        self.assertEqual(self.patch.count("+      IF(ERRNUM .NE. 0) THEN"), 3)
        self.assertEqual(self.patch.count("-      IF(ERRNUM .EQ. 0) THEN"), 3)
        self.assertNotIn("PMWD", self.patch)
        self.assertNotIn("PMALB", self.patch)

    def test_coordinate_gate_is_not_coupled_to_last_read_status(self):
        gate = self.source.split("IF(YCRD", 1)[1].split("!     Transfer data", 1)[0]
        self.assertNotIn("ERRNUM", gate)
        self.assertIn("LEN_TRIM(CYCRD)", gate)
        self.assertIn("LEN_TRIM(CXCRD)", gate)

        x, _ = read_fixed_decimal("116.570", -999.0)
        y, _ = read_fixed_decimal("36.830", -99.0)
        self.assertTrue(coordinates_transfer(x, y, "116.570", "36.830", 0))

    def test_valid_yc_and_non_yc_fixtures_are_not_hard_coded(self):
        fixtures = (
            ("116.570", "36.830", "22"),
            ("-93.65", "42.03", "310"),
        )
        for x_text, y_text, elev_text in fixtures:
            x, x_iostat = read_fixed_decimal(x_text, -999.0)
            y, y_iostat = read_fixed_decimal(y_text, -99.0)
            elev, elev_iostat = read_fixed_decimal(elev_text, -99.0)
            self.assertEqual((x_iostat, y_iostat, elev_iostat), (0, 0, 0))
            self.assertTrue(
                coordinates_transfer(x, y, x_text, y_text, elev_iostat)
            )

    def test_malformed_blank_zero_and_out_of_range_coordinates_fail_gate(self):
        x, x_iostat = read_fixed_decimal("malformed", -999.0)
        y, _ = read_fixed_decimal("36.830", -99.0)
        self.assertNotEqual(x_iostat, 0)
        self.assertFalse(coordinates_transfer(x, y, "malformed", "36.830", 0))

        blank_x, _ = read_fixed_decimal("", -999.0)
        self.assertFalse(coordinates_transfer(blank_x, y, "", "36.830", 0))
        self.assertFalse(coordinates_transfer(0.0, 0.0, "0", "0", 0))
        self.assertFalse(coordinates_transfer(181.0, 36.0, "181", "36", 0))


if __name__ == "__main__":
    unittest.main()
