"""Low-cost diagnostic for DSSAT native automatic fertilizer triggering.

This script reuses the HLA 2004 candidate-IC automatic-management setup from
010_13 and changes only the automatic nitrogen threshold (NMTHR). It does not
train PPO. The purpose is to test whether FERTI=A fails because the threshold is
not reached, or because the native auto-fertilizer trigger is not active under
this setup.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_OUT = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "auto_fertilizer_nmthr_diagnosis_010_15"
)
MODULE_PATH = PROJECT_ROOT / "src" / "run_hla2004_candidate_ic_dssat_auto_management_010_13.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("auto01013", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["auto01013"] = module
    spec.loader.exec_module(module)
    return module


def patch_nmthr(mzx_path: Path, nmthr: int) -> None:
    lines = mzx_path.read_text(encoding="latin-1", errors="ignore").splitlines()
    patched: list[str] = []
    in_nitrogen_table = False
    changed = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@N NITROGEN") and "NMDEP" in stripped and "NMTHR" in stripped:
            in_nitrogen_table = True
            patched.append(line)
            continue
        if in_nitrogen_table and re.match(r"^\s*1\s+NI\b", line):
            # Preserve the exact field choices except NMTHR.
            patched.append(f" 1 NI             30 {nmthr:5d}    25 FE001 GS000")
            changed = True
            in_nitrogen_table = False
            continue
        if in_nitrogen_table and (stripped.startswith("@") or stripped.startswith("*")):
            in_nitrogen_table = False
        patched.append(line)
    if not changed:
        raise RuntimeError(f"Could not patch NMTHR in {mzx_path}")
    mzx_path.write_text("\n".join(patched) + "\n", encoding="latin-1", errors="ignore")


def read_summary_csv(out_dir: Path) -> dict:
    path = out_dir / "hla2004_candidate_ic_dssat_auto_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if not df.empty else {}


def read_event_counts(out_dir: Path) -> dict:
    path = out_dir / "hla2004_candidate_ic_dssat_auto_management_events.csv"
    if not path.exists():
        return {"irrig_events": 0, "fert_events": 0, "fert_amount": 0.0}
    df = pd.read_csv(path)
    if df.empty:
        return {"irrig_events": 0, "fert_events": 0, "fert_amount": 0.0}
    ops = df["operation"].astype(str)
    fert = df[ops.str.contains("Fertil", case=False, na=False)]
    irrig = df[ops.str.contains("Irrig", case=False, na=False)]
    return {
        "irrig_events": int(len(irrig)),
        "fert_events": int(len(fert)),
        "fert_amount": float(fert.get("amount", pd.Series(dtype=float)).sum()) if not fert.empty else 0.0,
    }


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    mod = load_base_module()
    rows = []
    for nmthr in [1, 10, 25, 50, 75, 90, 99]:
        out_dir = BASE_OUT / f"nmthr_{nmthr:02d}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        mod.OUT_DIR = out_dir
        mod.RUN_DIR = out_dir / "run"
        mod.prepare_run()
        mzx_files = sorted((mod.RUN_DIR / "input").glob("*.MZX"))
        if not mzx_files:
            raise RuntimeError(f"No MZX generated for NMTHR={nmthr}")
        patch_nmthr(mzx_files[0], nmthr)
        # Do not call parent_run(), because it calls prepare_run() again and
        # would overwrite the patched NMTHR. Run the forward simulation and
        # post-processing directly under the patched module globals.
        mod.child_run()
        daily = mod.standardize_plantgro()
        events = mod.parse_management_events()
        summary = mod.parse_summary()
        pd.DataFrame(
            [{**summary, "final_gwad_from_plantgro": daily.sort_values("dap").iloc[-1]["gwad"]}]
        ).to_csv(out_dir / "hla2004_candidate_ic_dssat_auto_summary.csv", index=False, encoding="utf-8-sig")
        mod.plot_daily(daily, events)
        mod.write_report(daily, events, summary)
        summary = read_summary_csv(out_dir)
        events = read_event_counts(out_dir)
        rows.append(
            {
                "NMTHR": nmthr,
                "HWAM": summary.get("HWAM"),
                "CWAM": summary.get("CWAM"),
                "MDAT": summary.get("MDAT"),
                "IR#M": summary.get("IR#M"),
                "IRCM": summary.get("IRCM"),
                "NI#M": summary.get("NI#M"),
                "NICM": summary.get("NICM"),
                **events,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(BASE_OUT / "auto_fertilizer_nmthr_diagnosis_summary.csv", index=False, encoding="utf-8-sig")
    report = [
        "# 010_15 HLA 2004 DSSAT native automatic fertilizer NMTHR diagnostic",
        "",
        "Purpose: reuse candidate IC 0.55 + 0.25N and DSSAT native `IRRIG=A, FERTI=A`, changing only `NMTHR`.",
        "",
        markdown_table(result),
        "",
        "Interpretation rule:",
        "",
        "- If any NMTHR value produces `NI#M>0`, then the original `NMTHR=50` likely missed the internal trigger.",
        "- If all NMTHR values keep `NI#M=0`, then the issue is not simply the numeric threshold; native `FERTI=A` is not triggering in this setup despite being parsed.",
    ]
    (PROJECT_ROOT / "docs" / "2026-06-27_hla2004_auto_fertilizer_nmthr_diagnosis_010_15.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(f"Wrote {BASE_OUT}")


if __name__ == "__main__":
    main()
