"""Low-cost HLA 2004 forward check with the newly adjusted HY0006 cultivar.

This script intentionally does NOT train PPO, does NOT run a parameter search,
and does NOT generate calibration scatterplots. It only reruns three HLA 2004
forward scenarios under the candidate IC=1 setup with the new cultivar line.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_hla2004_candidate_ic_strict_four_scenario_010_14 as base


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "new_cultivar_hla2004_forward_check"
NEW_CUL = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "cultivar_calibration_HLA2004_480"
    / "input_corrected_package"
    / "MZCER048.CUL"
)
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-28_hla2004_new_cultivar_forward_check.md"


SCENARIOS = {
    "candidate_null": base.SCENARIOS["candidate_null"],
    "candidate_recorded_expert_replay": base.SCENARIOS["candidate_recorded_expert_replay"],
    "candidate_dssat_auto": base.SCENARIOS["candidate_dssat_auto"],
}
SCENARIO_ORDER = list(SCENARIOS)


def configure_base_module() -> None:
    base.OUT_DIR = OUT_DIR
    base.SCENARIOS = SCENARIOS
    base.SCENARIO_ORDER = SCENARIO_ORDER
    base.DOC_PATH = DOC_PATH


def read_hy0006_line() -> str:
    for line in NEW_CUL.read_text(encoding="latin-1", errors="ignore").splitlines():
        if line.startswith("HY0006 "):
            return line
    raise RuntimeError(f"HY0006 not found in {NEW_CUL}")


def assert_ic1(filex: Path) -> None:
    text = filex.read_text(encoding="latin-1", errors="ignore")
    if "*INITIAL CONDITIONS" not in text:
        raise RuntimeError(f"Missing INITIAL CONDITIONS block in {filex}")
    treatment_line = next(
        (line for line in text.splitlines() if line.startswith(" 1 1 1 0 Sim2004")),
        "",
    )
    # Treatment columns: CU FL SA IC MP MI MF ...
    if "  1  1  0  1  " not in treatment_line:
        raise RuntimeError(f"Treatment line does not appear to use IC=1 in {filex}: {treatment_line}")


def prepare_run_with_new_cultivar(scenario_key: str) -> Path:
    run_dir = base.prepare_run(scenario_key)
    input_dir = run_dir / "input"
    filex = sorted(input_dir.glob("*.MZX"))[0]
    assert_ic1(filex)
    shutil.copyfile(NEW_CUL, input_dir / "MZCER048.CUL")
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["new_cultivar_source"] = str(NEW_CUL)
    metadata["hy0006_line"] = read_hy0006_line()
    metadata["ic_check"] = "IC=1 confirmed from treatment line before run"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def parent_run(rerun: bool = False) -> None:
    configure_base_module()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for key in SCENARIO_ORDER:
        run_dir = prepare_run_with_new_cultivar(key)
        plantgro = run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT"
        if rerun or not plantgro.exists():
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child-run-dir",
                str(run_dir),
            ]
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=160, capture_output=True, text=True)
            (run_dir / "child_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="ignore")
            (run_dir / "child_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="ignore")
            statuses.append({"scenario_key": key, "returncode": proc.returncode, "stderr_tail": proc.stderr[-1000:]})
            if proc.returncode != 0:
                pd.DataFrame(statuses).to_csv(OUT_DIR / "run_status.csv", index=False, encoding="utf-8-sig")
                raise RuntimeError(f"{key} failed; see {run_dir / 'child_stderr.txt'}")
        else:
            statuses.append({"scenario_key": key, "returncode": 0, "stderr_tail": "skipped existing"})
    pd.DataFrame(statuses).to_csv(OUT_DIR / "run_status.csv", index=False, encoding="utf-8-sig")

    daily, events, summary = base.collect_outputs()
    # Rename outputs to make the new-cultivar scope explicit.
    daily.to_csv(OUT_DIR / "hla2004_new_cultivar_forward_check_daily_values.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT_DIR / "hla2004_new_cultivar_forward_check_management_events.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "hla2004_new_cultivar_forward_check_summary.csv", index=False, encoding="utf-8-sig")
    write_report(summary, events)
    print(summary.to_string(index=False))


def write_report(summary: pd.DataFrame, events: pd.DataFrame) -> None:
    hy_line = read_hy0006_line()
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "| {scenario_label} | {grain_yield_gwad:.0f} | {biomass_cwad:.0f} | {summary_ircm:.1f} | {summary_nicm:.1f} | {max_wspd:.3f} | {max_nstd:.3f} |".format(
                **row.to_dict()
            )
        )
    auto = events[events["scenario_key"].eq("candidate_dssat_auto")] if not events.empty else pd.DataFrame()
    auto_ir = auto[auto["operation"].str.contains("Irrigation", case=False, na=False)] if not auto.empty else pd.DataFrame()
    auto_n = auto[auto["operation"].str.contains("Fertil|Nitrogen", case=False, na=False)] if not auto.empty else pd.DataFrame()
    md = [
        "# HLA 2004 new HY0006 cultivar forward check",
        "",
        "## Scope",
        "",
        "- Runtime: PDI/gym-DSSAT forward simulation.",
        "- No PPO training.",
        "- No parameter search.",
        "- All scenarios keep the candidate `IC=1` setup.",
        "- The only cultivar source used is the adjusted `MZCER048.CUL` from the calibration input package.",
        "",
        "## HY0006 line",
        "",
        "```text",
        hy_line,
        "```",
        "",
        "## Summary",
        "",
        "| Scenario | GWAD/HWAM proxy (kg/ha) | CWAD/CWAM proxy (kg/ha) | IRCM (mm) | NICM (kg/ha) | max WSPD | max NSTD |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Native DSSAT automatic-management trigger check",
        "",
        f"- Auto-irrigation events: {len(auto_ir)}; total amount = {auto_ir['amount'].sum() if not auto_ir.empty else 0:.2f} mm.",
        f"- Auto-fertilizer events: {len(auto_n)}; total amount = {auto_n['amount'].sum() if not auto_n.empty else 0:.2f}.",
        "",
        "## Files",
        "",
        "- `hla2004_new_cultivar_forward_check_summary.csv`",
        "- `hla2004_new_cultivar_forward_check_daily_values.csv`",
        "- `hla2004_new_cultivar_forward_check_management_events.csv`",
        "- `runs/*/input/`",
        "- `runs/*/pdi_tmp_snapshot/`",
    ]
    DOC_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--child-run-dir", type=Path)
    args = parser.parse_args()
    configure_base_module()
    if args.child_run_dir:
        base.child_run(args.child_run_dir)
    else:
        parent_run(rerun=args.rerun)


if __name__ == "__main__":
    main()
