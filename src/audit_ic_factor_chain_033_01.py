from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "benchmark_results" / "033_01_ic_factor_chain_audit"
DOC_PATH = PROJECT_ROOT / "docs" / "033_01_ic_factor_chain_audit_record.md"

FACTORS = ["CU", "FL", "SA", "IC", "MP", "MI", "MF", "MR", "MC", "MT", "ME", "MH", "SM"]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_treatment_factors(text: str) -> dict:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("@N R O C TNAME"):
            for row in lines[idx + 1 :]:
                stripped = row.strip()
                if not stripped or stripped.startswith("*") or stripped.startswith("@"):
                    continue
                parts = stripped.split()
                if len(parts) < 5 + len(FACTORS):
                    return {
                        "treatment_parse_status": "too_few_columns",
                        "treatment_row": stripped,
                    }
                factor_values = parts[5 : 5 + len(FACTORS)]
                return {
                    "treatment_parse_status": "ok",
                    "treatment_row": stripped,
                    "treatment_number": parts[0],
                    "treatment_name": parts[4],
                    **dict(zip(FACTORS, factor_values)),
                }
            return {"treatment_parse_status": "header_without_row", "treatment_row": ""}
    return {"treatment_parse_status": "missing_treatment_header", "treatment_row": ""}


def parse_initial_condition_ids(text: str) -> list[str]:
    ids: set[str] = set()
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("*INITIAL CONDITIONS"):
            in_section = True
            continue
        if in_section and stripped.startswith("*"):
            break
        if not in_section or not stripped or stripped.startswith("@"):
            continue
        parts = stripped.split()
        if parts and re.fullmatch(r"\d+", parts[0]):
            ids.add(parts[0])
    return sorted(ids, key=lambda x: int(x))


def infer_group(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    rel_text = str(rel).replace("\\", "/")
    if rel_text.startswith("my_data/"):
        return "source_templates"
    if "033_00_initial_soil_water_sensitivity_wspd_audit" in rel_text:
        return "033_00_derived_inputs"
    if "032_22_five_site_half_split_stress_aware_maskableppo_batch" in rel_text:
        return "032_22_maskableppo_rendered"
    if "031_35_missing_four_baseline_completion_for_03134" in rel_text:
        return "031_35_four_baseline_snapshots"
    if "031_36_missing_dssat_auto_completion_for_03134" in rel_text:
        return "031_36_auto_baseline_snapshots"
    if "032_" in rel_text:
        return "032_other_outputs"
    if "031_" in rel_text:
        return "031_other_outputs"
    return "other"


def infer_station_year(path: Path, text: str) -> tuple[str, str]:
    rel_text = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    station = ""
    year = ""

    for token in ["HLA", "SYA", "LCA", "YCA", "FQA"]:
        if f"/{token}/" in rel_text or f"_{token}_" in rel_text or f"{token}20" in rel_text:
            station = token
            break
    if not station:
        name = path.name.upper()
        station_map = {"-HL": "HLA", "-SY": "SYA", "-LC": "LCA", "-YC": "YCA", "-FQ": "FQA"}
        for key, val in station_map.items():
            if key in name:
                station = val
                break

    m = re.search(r"\b(?:Sim)?(20\d{2})\b", text)
    if m:
        year = m.group(1)
    else:
        m = re.search(r"(20\d{2})", rel_text)
        if m:
            year = m.group(1)
    return station, year


def classify(row: dict) -> str:
    if row.get("treatment_parse_status") != "ok":
        return "parse_problem"
    ic = row.get("IC", "")
    ids = set(filter(None, row.get("initial_condition_ids", "").split(";")))
    if not ic.isdigit():
        return "ic_not_numeric"
    if int(ic) == 0:
        return "ic_zero"
    if ic in ids:
        return "ic_nonzero_valid"
    return "ic_nonzero_missing_block"


def collect_paths() -> list[Path]:
    paths: list[Path] = []
    paths.extend(sorted((PROJECT_ROOT / "my_data").glob("UFGA8201-*.jinja2")))

    targeted_roots = [
        PROJECT_ROOT / "benchmark_results" / "033_00_initial_soil_water_sensitivity_wspd_audit",
        PROJECT_ROOT / "benchmark_results" / "032_22_five_site_half_split_stress_aware_maskableppo_batch",
        PROJECT_ROOT / "benchmark_results" / "031_35_missing_four_baseline_completion_for_03134",
        PROJECT_ROOT / "benchmark_results" / "031_36_missing_dssat_auto_completion_for_03134",
        PROJECT_ROOT / "benchmark_results" / "032_23_lc_half_split_maskableppo_demo",
        PROJECT_ROOT / "benchmark_results" / "032_24_lc_half_split_maskableppo_all_year_daily_plots",
    ]
    for root in targeted_roots:
        if not root.exists():
            continue
        for pattern in ("*.jinja2", "fileX.MZX", "*.MZX"):
            paths.extend(sorted(root.rglob(pattern)))

    # 去重并保持顺序。
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in collect_paths():
        text = read_text(path)
        factors = parse_treatment_factors(text)
        ic_ids = parse_initial_condition_ids(text)
        station, year = infer_station_year(path, text)
        row = {
            "group": infer_group(path),
            "relative_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "station": station,
            "year": year,
            "initial_condition_ids": ";".join(ic_ids),
            **factors,
        }
        row["status"] = classify(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    rows_path = OUT_DIR / "033_01_ic_factor_rows.csv"
    df.to_csv(rows_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    if not df.empty:
        for group, sub in df.groupby("group", dropna=False):
            counts = Counter(sub["status"])
            summary_rows.append(
                {
                    "group": group,
                    "files": len(sub),
                    "ic_zero": counts.get("ic_zero", 0),
                    "ic_nonzero_valid": counts.get("ic_nonzero_valid", 0),
                    "ic_nonzero_missing_block": counts.get("ic_nonzero_missing_block", 0),
                    "parse_problem": sum(v for k, v in counts.items() if k.startswith("parse") or k in {"ic_not_numeric"}),
                    "unique_ic_values": ";".join(sorted(map(str, sub.get("IC", pd.Series(dtype=str)).dropna().unique()))),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "033_01_ic_factor_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    problem = df[df["status"].isin(["ic_zero", "ic_nonzero_missing_block", "parse_problem", "ic_not_numeric"])].copy()
    problem_path = OUT_DIR / "033_01_problem_files.csv"
    problem.to_csv(problem_path, index=False, encoding="utf-8-sig")

    status_counts = Counter(df["status"]) if not df.empty else Counter()
    group_lines = []
    for _, r in summary.sort_values("group").iterrows():
        group_lines.append(
            f"| {r['group']} | {int(r['files'])} | {int(r['ic_zero'])} | "
            f"{int(r['ic_nonzero_valid'])} | {int(r['ic_nonzero_missing_block'])} | {r['unique_ic_values']} |"
        )

    source_note = ""
    if not df.empty:
        source = df[df["group"] == "source_templates"]
        if not source.empty:
            source_note = (
                f"源模板共 {len(source)} 个；IC=0 的源模板 {int((source['status'] == 'ic_zero').sum())} 个，"
                f"IC 非零且可对应 INITIAL CONDITIONS 的源模板 {int((source['status'] == 'ic_nonzero_valid').sum())} 个。"
            )

    conclusion = []
    if status_counts.get("ic_zero", 0) > 0:
        conclusion.append(
            "发现大量文件的 treatment `IC` 因子为 0；这些文件即使包含 `*INITIAL CONDITIONS` 表，也不会通过 treatment 因子启用该初始条件。"
        )
    if status_counts.get("ic_nonzero_valid", 0) > 0:
        conclusion.append("存在 IC 非零且能对应 `*INITIAL CONDITIONS` treatment id 的文件，说明审计脚本能识别已启用 IC 的分支。")
    if status_counts.get("ic_nonzero_missing_block", 0) > 0:
        conclusion.append("存在 IC 非零但找不到对应初始条件块的文件，需要单独检查。")
    if not conclusion:
        conclusion.append("未发现可解释的 IC 因子记录，请检查解析范围。")

    DOC_PATH.write_text(
        "\n".join(
            [
                "# 033_01 IC 因子启用链条审计记录",
                "",
                "## 任务性质",
                "",
                "本任务只读取现有模板和结果快照，审计 DSSAT treatment 行中 `IC` 因子是否启用；不训练、不跑 DSSAT、不修改代码。",
                "",
                "## 审计输出",
                "",
                f"- 逐文件表：`{rows_path.relative_to(PROJECT_ROOT)}`",
                f"- 汇总表：`{summary_path.relative_to(PROJECT_ROOT)}`",
                f"- 问题文件表：`{problem_path.relative_to(PROJECT_ROOT)}`",
                "",
                "## 总体计数",
                "",
                f"- 审计文件数：{len(df)}",
                f"- `IC=0`：{status_counts.get('ic_zero', 0)}",
                f"- `IC>0` 且能对应初始条件块：{status_counts.get('ic_nonzero_valid', 0)}",
                f"- `IC>0` 但找不到对应初始条件块：{status_counts.get('ic_nonzero_missing_block', 0)}",
                f"- 解析问题：{status_counts.get('parse_problem', 0) + status_counts.get('ic_not_numeric', 0)}",
                "",
                "## 按文件组汇总",
                "",
                "| 文件组 | 文件数 | IC=0 | IC有效启用 | IC非零但缺块 | IC取值 |",
                "|---|---:|---:|---:|---:|---|",
                *group_lines,
                "",
                "## 源模板情况",
                "",
                source_note or "未在本次目标范围内找到源模板。",
                "",
                "## 初步结论",
                "",
                *[f"- {x}" for x in conclusion],
                "",
                "## 下一步建议",
                "",
                "若主流程渲染输入确认为 `IC=0`，应先修复安全渲染函数，使其显式启用当前模板中的 `*INITIAL CONDITIONS`；修复后先做小规模 smoke，确认渲染后的 treatment 行 `IC=1` 且 WSPD 对初始水分有响应，再决定全量重跑范围。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"wrote {rows_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {problem_path}")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
