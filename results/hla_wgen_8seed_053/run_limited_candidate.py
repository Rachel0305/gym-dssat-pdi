"""Capture one isolated 2007 HLA WGEN season for a frozen CLI track."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCE = ROOT / "results/hla_runtime_wgen_smoke_032/run_probe.py"
SOURCE_FILEX = ROOT / "results/hla_runtime_wgen_smoke_032/inputs/fileX_after_W.jinja2"
SCENARIOS = {
    "gpcc_raw": "AF957CD7C28D427E6B1319CF95D37B1EA8963770795C70B826F71FF2CB08F29F",
    "cpc_raw": "907558F5399677A5A6492222919F239F88034984689E34389A6CF29117442CD8",
    "gpcc_biascorr": "6318FD282A213E6AAA76C54C22E129AE25EC9351D0044C9FC7D15EEAAE053A1E",
    "cpc_biascorr": "E38E776C178BF287AAB0721082F4A29894B25D3B972C92A3352C0ACAD2EF1DC9",
    "ensemble_biascorr": "62C9BF8B18D7D924E28E110B627247F849FBC6C13DA4FA5BDC63CA974FF40FAF",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    args = parser.parse_args()
    scenario = args.scenario
    cli = ROOT / "results/hla_weather_enhancement_029/weather_fitting" / scenario / "CNHL.CLI"
    target = BASE / "limited_candidates" / scenario / "2007_seed101"
    if target.exists():
        raise FileExistsError(f"Candidate exists; refusing overwrite: {target}")
    if sha(cli) != SCENARIOS[scenario]:
        raise RuntimeError(f"Frozen CLI SHA mismatch: {scenario}")
    if not SOURCE_FILEX.is_file() or not SOURCE.is_file():
        raise FileNotFoundError("032 frozen runtime fixture missing")

    target.mkdir(parents=True)
    (target / "inputs").mkdir()
    fixture = target / "inputs/fileX_after_W.jinja2"
    shutil.copyfile(SOURCE_FILEX, fixture)
    provenance = {
        "task": "053_limited_wgen_candidate",
        "site": "HLA",
        "scenario": scenario.upper(),
        "crop_year": 2007,
        "weather_seed": 101,
        "formal_training": False,
        "cli_sha256": sha(cli),
        "fixture_sha256": sha(fixture),
        "source_probe_sha256": sha(SOURCE),
        "known_coordinate_limit": "native FIELD coordinates remain placeholders in 032/033",
        "file_hashes": [
            {"path": str(cli.relative_to(ROOT)), "sha256": sha(cli)},
            {"path": str(fixture.relative_to(ROOT)), "sha256": sha(fixture)},
        ],
    }
    (target / "input_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    spec = importlib.util.spec_from_file_location("hla_032_probe_for_053", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = target
    module.CLI = cli
    module.SEED = 101
    sys.argv = [str(SOURCE), "--attempt", "attempt_01"]
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
