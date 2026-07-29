"""039_00 audit for manually lowered initial soil water and nitrogen inputs.

This script is intentionally read-only for DSSAT input data.  It compares the
original multisite input directory with a manually edited low-initial-condition
copy, then writes CSV/JSON/Markdown audit artifacts.

It does not run DSSAT and does not train RL models.  Its only purpose is to
answer: is the derived lowIC directory a clean, comparable input source before
we spend compute on stress smoke tests?
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
LOWIC_DIR = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
OUT_DIR = ROOT / "benchmark_results" / "039_00_low_initial_soil_water_nitrogen_input_audit"
TABLE_DIR = OUT_DIR / "tables"
DOC_PATH = OUT_DIR / "039_00_low_initial_soil_water_nitrogen_input_audit_record.md"
JSON_PATH = OUT_DIR / "039_00_result.json"

AUTHORITATIVE_TEMPLATES = {
    # Keep this aligned with src/ppo_safe_rendering.py SITE_INFO.
    "FQA": ("FQ", "CNFQ0801.MZX"),
    "HLA": ("HL", "CNHL0701_corrected_IC123.MZX"),
    "LCA": ("LC", "CNLC0801.MZX"),
    "SYA": ("SY", "CNSY1201.MZX"),
    "YCA": ("YC", "CNYC0801.MZX"),
}


@dataclass
class IcRow:
    file: str
    line: int
    c: str
    icbl: float
    sh2o: float
    snh4: float
    sno3: float


@dataclass
class CompareRow:
    station: str
    file: str
    row_index: int | str
    status: str
    orig_line: int | str = ""
    low_line: int | str = ""
    icbl_orig: float | str = ""
    icbl_low: float | str = ""
    sh2o_orig: float | str = ""
    sh2o_low: float | str = ""
    sh2o_ratio: float | str = ""
    snh4_orig: float | str = ""
    snh4_low: float | str = ""
    snh4_ratio: float | str = ""
    sno3_orig: float | str = ""
    sno3_low: float | str = ""
    sno3_ratio: float | str = ""
    issue: str = ""


def parse_initial_conditions(path: Path) -> list[IcRow]:
    lines = path.read_text(errors="ignore").splitlines()
    rows: list[IcRow] = []
    in_ic = False
    in_profile = False
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("*INITIAL CONDITIONS"):
            in_ic = True
            in_profile = False
            continue
        if in_ic and stripped.startswith("*") and not upper.startswith("*INITIAL CONDITIONS"):
            in_ic = False
            in_profile = False
        if not in_ic:
            continue
        tokens = upper.split()
        if stripped.startswith("@C") and {"ICBL", "SH2O", "SNH4", "SNO3"}.issubset(tokens):
            in_profile = True
            continue
        if in_profile and stripped and not stripped.startswith("@") and not stripped.startswith("!"):
            parts = stripped.split()
            if len(parts) < 5:
                continue
            try:
                rows.append(
                    IcRow(
                        file=str(path.relative_to(ROOT)),
                        line=lineno,
                        c=parts[0],
                        icbl=float(parts[1]),
                        sh2o=float(parts[2]),
                        snh4=float(parts[3]),
                        sno3=float(parts[4]),
                    )
                )
            except ValueError:
                continue
    return rows


def parse_treatment_ic_values(path: Path) -> list[dict[str, str]]:
    """Return treatment lines and IC column values from the treatments table."""
    lines = path.read_text(errors="ignore").splitlines()
    out: list[dict[str, str]] = []
    in_treatments = False
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.upper().startswith("*TREATMENTS"):
            in_treatments = True
            continue
        if in_treatments and stripped.startswith("*"):
            in_treatments = False
        if not in_treatments:
            continue
        if stripped.startswith("@N"):
            continue
        parts = stripped.split()
        # DSSAT treatment table columns start: N R O C TNAME CU FL SA IC ...
        if len(parts) >= 9 and parts[0].isdigit():
            out.append({"line": str(lineno), "treatment": parts[0], "ic": parts[8], "raw": stripped})
    return out


def direct_mzx_files(root: Path, station: str) -> list[Path]:
    station_dir = root / station
    if not station_dir.exists():
        return []
    return sorted(p for p in station_dir.glob("*.MZX") if p.is_file())


def ratio(orig: float, low: float) -> float | str:
    if orig == 0:
        return ""
    return low / orig


def issue_for(o: IcRow, l: IcRow, tolerance: float = 0.06) -> str:
    issues: list[str] = []
    if o.icbl != l.icbl:
        issues.append("ICBL_changed")
    for field in ("sh2o", "snh4", "sno3"):
        r = ratio(getattr(o, field), getattr(l, field))
        if r != "" and abs(float(r) - 0.5) > tolerance:
            issues.append(f"{field.upper()}_not_half")
    return ";".join(issues)


def compare_station_file(station: str, orig_path: Path, low_path: Path | None) -> list[CompareRow]:
    if low_path is None or not low_path.exists():
        return [CompareRow(station=station, file=orig_path.name, row_index="", status="missing_lowIC_file", issue="missing_lowIC_file")]

    orig_rows = parse_initial_conditions(orig_path)
    low_rows = parse_initial_conditions(low_path)
    rows: list[CompareRow] = []
    if len(orig_rows) != len(low_rows):
        rows.append(
            CompareRow(
                station=station,
                file=orig_path.name,
                row_index="",
                status="row_count_mismatch",
                issue=f"original_rows={len(orig_rows)};lowIC_rows={len(low_rows)}",
            )
        )
    for idx, (o, l) in enumerate(zip(orig_rows, low_rows), 1):
        issue = issue_for(o, l)
        rows.append(
            CompareRow(
                station=station,
                file=orig_path.name,
                row_index=idx,
                status="ok" if not issue else "issue",
                orig_line=o.line,
                low_line=l.line,
                icbl_orig=o.icbl,
                icbl_low=l.icbl,
                sh2o_orig=o.sh2o,
                sh2o_low=l.sh2o,
                sh2o_ratio=ratio(o.sh2o, l.sh2o),
                snh4_orig=o.snh4,
                snh4_low=l.snh4,
                snh4_ratio=ratio(o.snh4, l.snh4),
                sno3_orig=o.sno3,
                sno3_low=l.sno3,
                sno3_ratio=ratio(o.sno3, l.sno3),
                issue=issue,
            )
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    stations = sorted(p.name for p in ORIGINAL_DIR.iterdir() if p.is_dir())
    compare_rows: list[CompareRow] = []
    authoritative_rows: list[CompareRow] = []
    direct_file_rows: list[dict[str, str | int]] = []
    treatment_rows: list[dict[str, str]] = []

    for station in stations:
        orig_files = direct_mzx_files(ORIGINAL_DIR, station)
        low_files = direct_mzx_files(LOWIC_DIR, station)
        direct_file_rows.append(
            {
                "station": station,
                "original_direct_mzx_count": len(orig_files),
                "original_direct_mzx": ";".join(p.name for p in orig_files),
                "lowIC_direct_mzx_count": len(low_files),
                "lowIC_direct_mzx": ";".join(p.name for p in low_files),
            }
        )
        low_by_name = {p.name: p for p in low_files}
        for orig_file in orig_files:
            compare_rows.extend(compare_station_file(station, orig_file, low_by_name.get(orig_file.name)))
        for low_file in low_files:
            for row in parse_treatment_ic_values(low_file):
                treatment_rows.append({"station": station, "file": low_file.name, **row})

    for station_code, (station_dir, template_name) in AUTHORITATIVE_TEMPLATES.items():
        authoritative_rows.extend(
            compare_station_file(
                station=station_code,
                orig_path=ORIGINAL_DIR / station_dir / template_name,
                low_path=LOWIC_DIR / station_dir / template_name,
            )
        )

    recursive_low_mzx = sorted(str(p.relative_to(ROOT)) for p in LOWIC_DIR.rglob("*.MZX"))
    recursive_rows = [{"path": p} for p in recursive_low_mzx]

    compare_dicts = [asdict(r) for r in compare_rows]
    authoritative_dicts = [asdict(r) for r in authoritative_rows]
    issue_rows = [r for r in compare_dicts if r["status"] != "ok"]
    authoritative_issue_rows = [r for r in authoritative_dicts if r["status"] != "ok"]
    write_csv(TABLE_DIR / "039_00_ic_profile_comparison.csv", compare_dicts)
    write_csv(TABLE_DIR / "039_00_ic_profile_issues.csv", issue_rows)
    write_csv(TABLE_DIR / "039_00_authoritative_template_comparison.csv", authoritative_dicts)
    write_csv(TABLE_DIR / "039_00_authoritative_template_issues.csv", authoritative_issue_rows)
    write_csv(TABLE_DIR / "039_00_direct_mzx_file_audit.csv", direct_file_rows)
    write_csv(TABLE_DIR / "039_00_lowIC_recursive_mzx_inventory.csv", recursive_rows)
    write_csv(TABLE_DIR / "039_00_treatment_ic_audit.csv", treatment_rows)

    station_summary: dict[str, dict] = {}
    for station in stations:
        rows = [r for r in compare_dicts if r["station"] == station and r["row_index"] != ""]
        issues = [r for r in compare_dicts if r["station"] == station and r["status"] != "ok"]
        station_summary[station] = {
            "ic_rows_compared": len(rows),
            "issue_count": len(issues),
            "direct_file_row": next((r for r in direct_file_rows if r["station"] == station), {}),
            "mean_sh2o_ratio": mean(float(r["sh2o_ratio"]) for r in rows if r["sh2o_ratio"] != "") if rows else None,
            "mean_snh4_ratio": mean(float(r["snh4_ratio"]) for r in rows if r["snh4_ratio"] != "") if rows else None,
            "mean_sno3_ratio": mean(float(r["sno3_ratio"]) for r in rows if r["sno3_ratio"] != "") if rows else None,
        }

    result = {
        "task": "039_00_low_initial_soil_water_nitrogen_input_audit",
        "original_dir": str(ORIGINAL_DIR.relative_to(ROOT)),
        "lowIC_dir": str(LOWIC_DIR.relative_to(ROOT)),
        "direct_lowIC_mzx_count": sum(int(r["lowIC_direct_mzx_count"]) for r in direct_file_rows),
        "recursive_lowIC_mzx_count": len(recursive_low_mzx),
        "issue_count": len(issue_rows),
        "authoritative_issue_count": len(authoritative_issue_rows),
        "has_recursive_extra_mzx": len(recursive_low_mzx) != sum(int(r["lowIC_direct_mzx_count"]) for r in direct_file_rows),
        "next_step_allowed": len(authoritative_issue_rows) == 0,
        "station_summary": station_summary,
        "outputs": {
            "comparison_csv": str((TABLE_DIR / "039_00_ic_profile_comparison.csv").relative_to(ROOT)),
            "issues_csv": str((TABLE_DIR / "039_00_ic_profile_issues.csv").relative_to(ROOT)),
            "authoritative_comparison_csv": str((TABLE_DIR / "039_00_authoritative_template_comparison.csv").relative_to(ROOT)),
            "authoritative_issues_csv": str((TABLE_DIR / "039_00_authoritative_template_issues.csv").relative_to(ROOT)),
            "direct_mzx_csv": str((TABLE_DIR / "039_00_direct_mzx_file_audit.csv").relative_to(ROOT)),
            "recursive_inventory_csv": str((TABLE_DIR / "039_00_lowIC_recursive_mzx_inventory.csv").relative_to(ROOT)),
            "treatment_ic_csv": str((TABLE_DIR / "039_00_treatment_ic_audit.csv").relative_to(ROOT)),
        },
    }
    JSON_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 039_00 初始土壤水分/氮减半输入审计记录",
        "",
        "## 目的",
        "",
        "确认派生输入目录是否只改变了 `*INITIAL CONDITIONS` 中的 `SH2O/SNH4/SNO3`，并保持 `ICBL`、处理 IC 编号和主线文件结构可追溯。此任务不运行 DSSAT、不训练 RL。",
        "",
        "## 结论",
        "",
        f"- 原始目录：`{ORIGINAL_DIR.relative_to(ROOT)}`",
        f"- lowIC 目录：`{LOWIC_DIR.relative_to(ROOT)}`",
        f"- 直接位于站点目录下的 lowIC MZX 数：{result['direct_lowIC_mzx_count']}",
        f"- 递归发现的 lowIC MZX 数：{result['recursive_lowIC_mzx_count']}",
        f"- 全部顶层 MZX 比对问题行数：{len(issue_rows)}",
        f"- 主线 authoritative template 问题行数：{len(authoritative_issue_rows)}",
        f"- 是否存在递归额外 MZX：`{str(result['has_recursive_extra_mzx']).lower()}`",
        f"- 是否允许进入 DSSAT smoke：`{str(result['next_step_allowed']).lower()}`",
        "",
        "## 站点摘要",
        "",
        "| 站点 | IC 行数 | 问题数 | 平均 SH2O 比例 | 平均 SNH4 比例 | 平均 SNO3 比例 | lowIC 直接 MZX |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for station, summary in station_summary.items():
        row = summary["direct_file_row"]
        def fmt(v):
            return "" if v is None else f"{v:.3f}"
        lines.append(
            f"| {station} | {summary['ic_rows_compared']} | {summary['issue_count']} | "
            f"{fmt(summary['mean_sh2o_ratio'])} | {fmt(summary['mean_snh4_ratio'])} | {fmt(summary['mean_sno3_ratio'])} | "
            f"`{row.get('lowIC_direct_mzx', '')}` |"
        )
    lines.extend(
        [
            "",
            "## 需要先处理的问题",
            "",
        ]
    )
    if authoritative_issue_rows:
        lines.extend(
            [
                "详见 `tables/039_00_authoritative_template_issues.csv`。当前主线模板风险包括：",
                "",
                "- 某些 authoritative template 缺失同名 lowIC MZX，未来把 `MULTISITE_INPUT_ROOT` 指向 lowIC 时会找不到模板。",
                "- 某些 authoritative template 字段不是约 0.5 倍。",
            ]
        )
    else:
        lines.append("主线 authoritative template 未发现阻塞问题。可以进入 DSSAT smoke。")
    if result["has_recursive_extra_mzx"]:
        lines.extend(
            [
                "",
                "另外：lowIC 目录存在递归额外 MZX。主线代码当前按固定模板名读取，通常不受影响；但为了避免后续人工或脚本误选，建议保持派生输入目录洁净。",
            ]
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `tables/039_00_ic_profile_comparison.csv`",
            "- `tables/039_00_ic_profile_issues.csv`",
            "- `tables/039_00_authoritative_template_comparison.csv`",
            "- `tables/039_00_authoritative_template_issues.csv`",
            "- `tables/039_00_direct_mzx_file_audit.csv`",
            "- `tables/039_00_lowIC_recursive_mzx_inventory.csv`",
            "- `tables/039_00_treatment_ic_audit.csv`",
            "- `039_00_result.json`",
        ]
    )
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["next_step_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
