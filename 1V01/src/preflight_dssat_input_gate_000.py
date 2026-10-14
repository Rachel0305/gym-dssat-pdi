"""DSSAT input preflight gate.

Read-only quality-control script to run before any training/evaluation task.
It records source input hashes and checks that IC and management modes are
actually enabled in MZX templates.

Example:
    python src/preflight_dssat_input_gate_000.py \
      --input-root DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual \
      --stations SYA \
      --expected-irrig L \
      --expected-ferti L \
      --task 040_next
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "benchmark_results" / "000_dssat_input_preflight_gate"
DOCS = ROOT / "docs"

STATION_DIR_ALIASES = {
    "FQA": "FQ",
    "HLA": "HL",
    "LCA": "LC",
    "SYA": "SY",
    "YCA": "YC",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_initial_condition_ids(lines: list[str]) -> set[str]:
    ids: set[str] = set()
    in_ic = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.upper().startswith("*INITIAL CONDITIONS"):
            in_ic = True
            continue
        if in_ic and stripped.startswith("*") and not stripped.upper().startswith("*INITIAL CONDITIONS"):
            in_ic = False
            continue
        if not in_ic:
            continue
        if re.match(r"^\s*\d+\s+MZ\s+", raw):
            ids.add(raw.split()[0])
    return ids


def parse_treatment_ic_values(lines: list[str]) -> list[dict]:
    rows: list[dict] = []
    header_cols: list[str] | None = None
    in_treatments = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.upper().startswith("*TREATMENTS"):
            in_treatments = True
            continue
        if in_treatments and stripped.startswith("*") and not stripped.upper().startswith("*TREATMENTS"):
            in_treatments = False
            header_cols = None
            continue
        if not in_treatments:
            continue
        if stripped.startswith("@N"):
            header_cols = stripped.split()
            continue
        if header_cols and re.match(r"^\s*\d+\s+", raw):
            parts = raw.split()
            values = {col: parts[i] if i < len(parts) else "" for i, col in enumerate(header_cols)}
            rows.append(
                {
                    "treatment_n": values.get("@N", parts[0]),
                    "tname": values.get("TNAME", ""),
                    "ic": values.get("IC", ""),
                }
            )
    return rows


def parse_management_modes(lines: list[str]) -> dict:
    modes = {"IRRIG": "", "FERTI": ""}
    in_sim = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.upper().startswith("*SIMULATION CONTROLS"):
            in_sim = True
            continue
        if in_sim and stripped.startswith("*") and not stripped.upper().startswith("*SIMULATION CONTROLS"):
            in_sim = False
            continue
        if not in_sim:
            continue
        if stripped.startswith("@N") and "IRRIG" in stripped and "FERTI" in stripped:
            header = stripped.split()
            continue
        if re.match(r"^\s*\d+\s+", raw) and "header" in locals():
            parts = raw.split()
            vals = {col: parts[i] if i < len(parts) else "" for i, col in enumerate(header)}
            modes["IRRIG"] = vals.get("IRRIG", "")
            modes["FERTI"] = vals.get("FERTI", "")
            break
    return modes


@dataclass
class FileManifest:
    station: str
    station_dir: str
    kind: str
    file: str
    sha256: str
    bytes: int
    mtime: str


def station_dir_name(station: str) -> str:
    return STATION_DIR_ALIASES.get(station, station)


def audit_station(
    input_root: Path,
    station: str,
    expected_irrig: str | None,
    expected_ferti: str | None,
    mzx_name: str | None,
) -> tuple[list[FileManifest], list[dict], list[str]]:
    station_dir = station_dir_name(station)
    root = input_root / station_dir
    issues: list[str] = []
    files: list[FileManifest] = []
    checks: list[dict] = []

    if not root.exists():
        return files, checks, [f"{station}: station directory missing: {rel(root)}"]

    all_mzx_files = sorted(root.glob("*.MZX"))
    mzx_files = [p for p in all_mzx_files if mzx_name is None or p.name == mzx_name]
    wth_files = sorted(root.glob("*.WTH"))
    sol_files = sorted(root.glob("*.SOL"))
    if not mzx_files:
        if mzx_name:
            issues.append(f"{station}: selected MZX {mzx_name} not found under {rel(root)}")
        else:
            issues.append(f"{station}: no direct MZX file under {rel(root)}")

    for kind, paths in [("MZX", all_mzx_files), ("WTH", wth_files), ("SOL", sol_files)]:
        for path in paths:
            files.append(
                FileManifest(
                    station=station,
                    station_dir=station_dir,
                    kind=kind,
                    file=rel(path),
                    sha256=sha256(path),
                    bytes=path.stat().st_size,
                    mtime=datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                )
            )

    for mzx in mzx_files:
        lines = mzx.read_text(errors="replace").splitlines()
        ic_ids = parse_initial_condition_ids(lines)
        treatments = parse_treatment_ic_values(lines)
        modes = parse_management_modes(lines)
        mode_ok = True
        if expected_irrig is not None and modes["IRRIG"] != expected_irrig:
            mode_ok = False
            issues.append(f"{station}: {mzx.name} IRRIG={modes['IRRIG']} expected {expected_irrig}")
        if expected_ferti is not None and modes["FERTI"] != expected_ferti:
            mode_ok = False
            issues.append(f"{station}: {mzx.name} FERTI={modes['FERTI']} expected {expected_ferti}")
        if not treatments:
            issues.append(f"{station}: {mzx.name} no treatment rows parsed")
        for tr in treatments:
            ic = tr["ic"]
            ic_ok = ic not in ("", "0", "-99") and ic in ic_ids
            if not ic_ok:
                issues.append(
                    f"{station}: {mzx.name} treatment {tr['treatment_n']} IC={ic} not enabled or missing IC block"
                )
            checks.append(
                {
                    "station": station,
                    "station_dir": station_dir,
                    "mzx_file": rel(mzx),
                    "selected_mzx_name": mzx_name or "",
                    "treatment_n": tr["treatment_n"],
                    "tname": tr["tname"],
                    "ic": ic,
                    "ic_block_ids": ";".join(sorted(ic_ids)),
                    "ic_enabled_and_valid": ic_ok,
                    "irrig_mode": modes["IRRIG"],
                    "ferti_mode": modes["FERTI"],
                    "management_mode_ok": mode_ok,
                }
            )

    return files, checks, issues


def read_manifest(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {(row["station"], row["file"]): row["sha256"] for row in reader}


def compare_manifest(current: list[FileManifest], old_path: Path | None) -> list[dict]:
    if old_path is None:
        return []
    old = read_manifest(old_path)
    rows = []
    cur_map = {(r.station, r.file): r.sha256 for r in current}
    for key in sorted(set(old) | set(cur_map)):
        rows.append(
            {
                "station": key[0],
                "file": key[1],
                "old_sha256": old.get(key, ""),
                "current_sha256": cur_map.get(key, ""),
                "status": "same"
                if old.get(key, "") == cur_map.get(key, "")
                else ("new" if key not in old else "missing" if key not in cur_map else "changed"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--stations", required=True, help="Comma-separated station codes, e.g. SYA,FQA")
    parser.add_argument("--expected-irrig", default=None)
    parser.add_argument("--expected-ferti", default=None)
    parser.add_argument("--mzx-name", default=None, help="Optional authoritative MZX filename to validate")
    parser.add_argument("--task", default="manual_preflight")
    parser.add_argument("--compare-manifest", default=None)
    args = parser.parse_args()

    input_root = (ROOT / args.input_root).resolve() if not Path(args.input_root).is_absolute() else Path(args.input_root)
    task_dir = OUT_BASE / args.task
    table_dir = task_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    all_files: list[FileManifest] = []
    all_checks: list[dict] = []
    all_issues: list[str] = []

    if not input_root.exists():
        all_issues.append(f"input root missing: {rel(input_root)}")
    else:
        for station in stations:
            files, checks, issues = audit_station(
                input_root=input_root,
                station=station,
                expected_irrig=args.expected_irrig,
                expected_ferti=args.expected_ferti,
                mzx_name=args.mzx_name,
            )
            all_files.extend(files)
            all_checks.extend(checks)
            all_issues.extend(issues)

    manifest_rows = [asdict(r) for r in all_files]
    compare_rows = compare_manifest(
        all_files, Path(args.compare_manifest) if args.compare_manifest else None
    )
    if any(row["status"] != "same" for row in compare_rows):
        all_issues.append("manifest hash drift detected")

    manifest_csv = table_dir / "input_manifest.csv"
    checks_csv = table_dir / "mzx_ic_management_checks.csv"
    issues_csv = table_dir / "preflight_issues.csv"
    compare_csv = table_dir / "manifest_compare.csv"
    result_json = task_dir / "preflight_result.json"
    record_md = DOCS / f"{args.task}_dssat_input_preflight_record.md"

    write_csv(manifest_csv, manifest_rows)
    write_csv(checks_csv, all_checks)
    write_csv(issues_csv, [{"issue": x} for x in all_issues])
    write_csv(compare_csv, compare_rows)

    passed = len(all_issues) == 0
    result = {
        "task": args.task,
        "input_root": rel(input_root),
        "stations": stations,
        "expected_irrig": args.expected_irrig,
        "expected_ferti": args.expected_ferti,
        "mzx_name": args.mzx_name,
        "passed": passed,
        "issue_count": len(all_issues),
        "manifest_file_count": len(manifest_rows),
        "outputs": {
            "manifest_csv": rel(manifest_csv),
            "checks_csv": rel(checks_csv),
            "issues_csv": rel(issues_csv),
            "compare_csv": rel(compare_csv),
            "record_md": rel(record_md),
        },
    }
    result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    record_md.write_text(
        "\n".join(
            [
                f"# {args.task} DSSAT 输入预检查记录",
                "",
                f"- 输入根目录：`{rel(input_root)}`",
                f"- 站点：`{', '.join(stations)}`",
                f"- 预期 IRRIG：`{args.expected_irrig}`",
                f"- 预期 FERTI：`{args.expected_ferti}`",
                f"- 指定权威 MZX：`{args.mzx_name}`",
                f"- 是否通过：`{passed}`",
                f"- 问题数：`{len(all_issues)}`",
                "",
                "## 输出",
                "",
                f"- 输入 manifest：`{rel(manifest_csv)}`",
                f"- IC/管理模式检查：`{rel(checks_csv)}`",
                f"- 问题清单：`{rel(issues_csv)}`",
                f"- manifest 对比：`{rel(compare_csv)}`",
                f"- JSON 摘要：`{rel(result_json)}`",
                "",
                "## 问题",
                "",
                *(f"- {issue}" for issue in all_issues),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
