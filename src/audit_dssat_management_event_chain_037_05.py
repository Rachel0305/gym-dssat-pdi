from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TASK = "037_05_dssat_management_event_chain_preflight"
SNAP_ROOT = ROOT / "benchmark_results" / "034_00_multisite_input_ic1_four_baseline_rebuild" / "snapshots"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
TABLE = OUT / "tables" / "037_05_management_event_chain_audit.csv"
ISSUES = OUT / "tables" / "037_05_management_event_chain_issues.csv"

REQUIRED = ["fileX.MZX", "DSSAT48.INP", "MgmtEvent.OUT"]
TOL = 1e-6


def ensure_dirs() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        (OUT / "configs" / PROMPT.name).write_bytes(PROMPT.read_bytes())


def is_data_line(line: str) -> bool:
    return bool(re.match(r"^\s*\d+\s+", line))


def to_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return 0.0


def parse_treatment_groups(path: Path) -> dict[str, int]:
    """Return MP/MI/MF group ids for the first treatment selected in DSSAT48.INP."""
    lines = path.read_text(errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("@N") and "TNAME" in line and "MP" in line and "MI" in line and "MF" in line:
            for data in lines[idx + 1 :]:
                if data.startswith("*") or data.lstrip().startswith("@"):
                    break
                if not is_data_line(data):
                    continue
                parts = data.split()
                # Expected layout: N R O C TNAME CU FL SA IC MP MI MF ...
                if len(parts) >= 12:
                    return {"MP": int(float(parts[9])), "MI": int(float(parts[10])), "MF": int(float(parts[11]))}
    return {"MP": 1, "MI": 1, "MF": 1}


def parse_filex_events(path: Path, selected: dict[str, int]) -> dict[str, Any]:
    """Parse planned non-zero management events from rendered fileX.MZX."""
    lines = path.read_text(errors="ignore").splitlines()
    irrigation_by_group: dict[int, list[float]] = {}
    fert_by_group: dict[int, list[float]] = {}
    section = ""
    current_group: int | None = None
    reading_events = False

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("*IRRIGATION"):
            section = "irrigation"
            current_group = None
            reading_events = False
            continue
        if upper.startswith("*FERTILIZERS"):
            section = "fertilizer"
            current_group = None
            reading_events = False
            continue
        if stripped.startswith("*"):
            section = ""
            current_group = None
            reading_events = False
            continue
        if not section:
            continue
        if stripped.startswith("@"):
            if section == "irrigation" and "EFIR" in upper:
                reading_events = False
            elif section == "irrigation" and "IDATE" in upper:
                reading_events = True
            elif section == "fertilizer" and "FDATE" in upper:
                reading_events = True
                current_group = selected.get("MF", 1)
            continue
        if not is_data_line(line):
            continue
        parts = stripped.split()
        if section == "irrigation":
            if not reading_events:
                current_group = int(float(parts[0]))
                irrigation_by_group.setdefault(current_group, [])
            elif current_group is not None and len(parts) >= 4:
                amount = to_float(parts[3])
                if amount > TOL:
                    irrigation_by_group.setdefault(current_group, []).append(amount)
        elif section == "fertilizer" and len(parts) >= 6:
            # In rendered fileX fertilizer rows are event rows for selected MF.
            amount = to_float(parts[5])
            if amount > TOL:
                fert_by_group.setdefault(selected.get("MF", 1), []).append(amount)

    selected_i = irrigation_by_group.get(selected.get("MI", 1), [])
    selected_f = fert_by_group.get(selected.get("MF", 1), [])
    return {
        "planned_irrigation_event_count": len(selected_i),
        "planned_irrigation_total_mm": sum(selected_i),
        "planned_n_event_count": len(selected_f),
        "planned_n_total_kg_ha": sum(selected_f),
    }


def parse_dssat48_inp_events(path: Path) -> dict[str, Any]:
    """Parse non-zero management events from DSSAT48.INP, the actual DSSAT input file."""
    lines = path.read_text(errors="ignore").splitlines()
    section = ""
    i_amounts: list[float] = []
    n_amounts: list[float] = []
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("*IRRIGATION"):
            section = "irrigation"
            continue
        if upper.startswith("*FERTILIZERS"):
            section = "fertilizer"
            continue
        if stripped.startswith("*"):
            section = ""
            continue
        if not section or stripped.startswith("@") or not is_data_line(line):
            continue
        parts = stripped.split()
        if section == "irrigation" and len(parts) >= 3:
            amount = to_float(parts[2])
            if amount > TOL:
                i_amounts.append(amount)
        elif section == "fertilizer" and len(parts) >= 5:
            amount = to_float(parts[4])
            if amount > TOL:
                n_amounts.append(amount)
    return {
        "inp_irrigation_event_count": len(i_amounts),
        "inp_irrigation_total_mm": sum(i_amounts),
        "inp_n_event_count": len(n_amounts),
        "inp_n_total_kg_ha": sum(n_amounts),
    }


def parse_mgmt_events(path: Path) -> dict[str, Any]:
    """Parse unique executed non-zero events from MgmtEvent.OUT.

    DSSAT sometimes writes duplicate RUN blocks in MgmtEvent.OUT. We count unique
    event prefixes plus amounts rather than raw matching lines.
    """
    i_events: set[tuple[str, float]] = set()
    n_events: set[tuple[str, float]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if "Irrigation" in line:
            m = re.search(r"Irrigation\s+([-+]?\d+(?:\.\d+)?)", line)
            if m:
                amount = to_float(m.group(1))
                if amount > TOL:
                    key = line.split("Irrigation", 1)[0].strip()
                    i_events.add((key, round(amount, 6)))
        if "Fertilizer" in line:
            m = re.search(r"Fertilizer\s+([-+]?\d+(?:\.\d+)?)", line)
            if m:
                amount = to_float(m.group(1))
                if amount > TOL:
                    key = line.split("Fertilizer", 1)[0].strip()
                    n_events.add((key, round(amount, 6)))
    i_amounts = [amount for _, amount in sorted(i_events)]
    n_amounts = [amount for _, amount in sorted(n_events)]
    return {
        "mgmt_irrigation_event_count": len(i_amounts),
        "mgmt_irrigation_total_mm": sum(i_amounts),
        "mgmt_n_event_count": len(n_amounts),
        "mgmt_n_total_kg_ha": sum(n_amounts),
    }


def audit_snapshot(path: Path) -> dict[str, Any]:
    rel = path.relative_to(ROOT).as_posix()
    parts = path.relative_to(SNAP_ROOT).parts
    station = parts[0] if len(parts) > 0 else ""
    year = parts[1] if len(parts) > 1 else ""
    scenario = parts[2] if len(parts) > 2 else ""
    row: dict[str, Any] = {"station_code": station, "year": year, "scenario": scenario, "snapshot_path": rel}
    missing = [name for name in REQUIRED if not (path / name).exists()]
    row["missing_required_files"] = ";".join(missing)
    if missing:
        row["status"] = "missing_required_file"
        return row

    selected = parse_treatment_groups(path / "DSSAT48.INP")
    row.update({f"selected_{k}": v for k, v in selected.items()})
    row.update(parse_filex_events(path / "fileX.MZX", selected))
    row.update(parse_dssat48_inp_events(path / "DSSAT48.INP"))
    row.update(parse_mgmt_events(path / "MgmtEvent.OUT"))

    issues: list[str] = []
    for kind, planned_count, planned_total, inp_count, inp_total, mgmt_count, mgmt_total in [
        (
            "irrigation",
            row["planned_irrigation_event_count"],
            row["planned_irrigation_total_mm"],
            row["inp_irrigation_event_count"],
            row["inp_irrigation_total_mm"],
            row["mgmt_irrigation_event_count"],
            row["mgmt_irrigation_total_mm"],
        ),
        (
            "nitrogen",
            row["planned_n_event_count"],
            row["planned_n_total_kg_ha"],
            row["inp_n_event_count"],
            row["inp_n_total_kg_ha"],
            row["mgmt_n_event_count"],
            row["mgmt_n_total_kg_ha"],
        ),
    ]:
        if planned_count > inp_count:
            issues.append(f"{kind}_event_drop_filex_to_inp")
        if inp_count != mgmt_count:
            issues.append(f"{kind}_event_count_inp_vs_mgmt")
        if abs(planned_total - inp_total) > 0.2:
            issues.append(f"{kind}_amount_filex_vs_inp")
        if abs(inp_total - mgmt_total) > 0.2:
            issues.append(f"{kind}_amount_inp_vs_mgmt")
    row["status"] = "ok" if not issues else "event_chain_issue"
    row["issues"] = ";".join(issues)
    return row


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录"
    show = df.head(max_rows).copy()
    for col in show.select_dtypes(include=["number"]).columns:
        show[col] = pd.to_numeric(show[col], errors="coerce").round(3)
    show = show.astype(object).where(pd.notna(show), "")
    lines = [
        "| " + " | ".join(map(str, show.columns)) + " |",
        "| " + " | ".join(["---"] * len(show.columns)) + " |",
    ]
    for row in show.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    rows = []
    for path in sorted(SNAP_ROOT.glob("*/*/*")):
        if path.is_dir():
            rows.append(audit_snapshot(path))
    df = pd.DataFrame(rows)
    df.to_csv(TABLE, index=False, encoding="utf-8-sig")
    issue_df = df[df["status"].ne("ok")].copy()
    issue_df.to_csv(ISSUES, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby(["station_code", "scenario", "status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["station_code", "scenario", "status"])
    )
    issue_cols = [
        "station_code",
        "year",
        "scenario",
        "status",
        "issues",
        "planned_irrigation_event_count",
        "inp_irrigation_event_count",
        "mgmt_irrigation_event_count",
        "planned_n_event_count",
        "inp_n_event_count",
        "mgmt_n_event_count",
        "snapshot_path",
    ]
    lines = [
        "# 037_05：DSSAT 管理事件生效链路预检记录",
        "",
        "## 结论先说",
        "",
        f"- 审计 snapshot 数：{len(df)}",
        f"- 异常 snapshot 数：{len(issue_df)}",
        f"- 明细表：`{TABLE.relative_to(ROOT).as_posix()}`",
        f"- 异常表：`{ISSUES.relative_to(ROOT).as_posix()}`",
        "",
        "## 按站点/情景/状态汇总",
        "",
        markdown_table(summary),
        "",
        "## 异常明细",
        "",
        markdown_table(issue_df[issue_cols] if not issue_df.empty else issue_df),
        "",
        "## 使用规则",
        "",
        "- 后续正式训练、指标汇总、五情景图绘制前，必须先通过本类 preflight。",
        "- 只要出现 `event_chain_issue`，该 snapshot 不进入正式比较，先修正管理事件生效链路。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    result = {
        "task": TASK,
        "audited_snapshots": int(len(df)),
        "issue_snapshots": int(len(issue_df)),
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "detail_csv": TABLE.relative_to(ROOT).as_posix(),
        "issues_csv": ISSUES.relative_to(ROOT).as_posix(),
    }
    (OUT / "037_05_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    if not issue_df.empty:
        print(issue_df[["station_code", "year", "scenario", "issues"]].to_string(index=False))


if __name__ == "__main__":
    main()
