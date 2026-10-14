"""Isolated HLA coordinate-only fixture and one weather probe using 032 logic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/hla_coordinate_audit_033"
BASE = ROOT / "results/hla_runtime_wgen_smoke_032/inputs/fileX_after_W.jinja2"
ORIGINAL = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/HL/CNHL0701_corrected_IC123.MZX"
CLI = ROOT / "results/hla_weather_enhancement_029/weather_fitting/gpcc_raw/CNHL.CLI"
HL = ORIGINAL.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def prepare() -> None:
    before = OUT / "inputs/fileX_before.MZX"
    after = OUT / "inputs/fileX_after_W.jinja2"
    if before.exists() or after.exists():
        raise FileExistsError("033 coordinate fixture already exists")
    if sha(BASE) != "2240DED7C82CD52102FA0C0C8080BA1CB773C676649B50D66A2C005C25EFE856":
        raise RuntimeError("032 FileX baseline hash changed")
    if sha(ORIGINAL) != "384937C36487217BCA7A33BF5E64D1DE3D9E8A841408F638E77CAD71B06C0A04":
        raise RuntimeError("Original HLA FileX hash changed")
    lines = BASE.read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(" 1             -99             -99       -99"))
    old = lines[index]
    assert len(old) == 92
    new = old[:3] + f"{126.9:15.5f}" + " " + f"{47.45:15.5f}" + " " + f"{234.0:9.1f}" + old[44:]
    assert len(new) == len(old)
    lines[index] = new
    before.parent.mkdir(parents=True)
    shutil.copyfile(BASE, before)
    after.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    paths = [ORIGINAL, BASE, before, after, CLI, HL / "MZCER048.CUL", HL / "SOIL.SOL"]
    meta = {
        "task": "033", "purpose": "ISOLATED_COORDINATE_ONLY_VALIDATION",
        "site": "HLA", "year": 2007, "treatment": 1, "rseed1": 101,
        "scenario": "GPCC_RAW", "coordinate_source": "HLA CNHL.CLI and observed WTH header",
        "changed_line_1_based": index + 1, "before_field_row": old, "after_field_row": new,
        "changed_columns": "XCRD 4-18; YCRD 20-34; ELEV 36-44; all other rows identical",
        "file_hashes": [{"path": rel(p), "sha256": sha(p)} for p in paths],
        "original_modified": False, "wrapper_modified": False, "ppo_executed": False,
    }
    (OUT / "input_provenance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": True, "line": index + 1, "before": old, "after": new}, ensure_ascii=False))


def run() -> int:
    spec = importlib.util.spec_from_file_location("hla032_probe", ROOT / "results/hla_runtime_wgen_smoke_032/run_probe.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("032 probe import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = OUT
    sys.argv = [sys.argv[0], "--attempt", "attempt_01"]
    return module.main()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--prepare":
        prepare()
    elif len(sys.argv) == 2 and sys.argv[1] == "--run":
        raise SystemExit(run())
    else:
        raise SystemExit("Usage: --prepare or --run")
