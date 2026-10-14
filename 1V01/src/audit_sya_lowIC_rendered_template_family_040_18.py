"""040_18: classify SYA rendered templates as ORIGIN/LOWIC/UNKNOWN.

This converts an ad-hoc provenance check into a reproducible audit. It does not
run DSSAT and does not modify any input files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = "040_18_sya_lowIC_rendered_template_family_audit"
OUT = ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK}_record.md"

ORIGIN_MARKERS = [" 1    10   .27", " 1    10  0.27"]
LOWIC_MARKERS = [" 1    10   .14", " 1    10  0.14"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(text: str) -> str:
    if any(marker in text for marker in LOWIC_MARKERS):
        return "LOWIC"
    if any(marker in text for marker in ORIGIN_MARKERS):
        return "ORIGIN"
    return "UNKNOWN"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for task_dir in sorted((ROOT / "benchmark_results").glob("040_*")):
        rendered_root = task_dir / "rendered_inputs" / "SYA" / "2014"
        if not rendered_root.exists():
            continue
        for path in sorted(rendered_root.rglob("*.jinja2")):
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            rows.append(
                {
                    "task_dir": task_dir.name,
                    "year": 2014,
                    "rendered_template": str(path.relative_to(ROOT)),
                    "ic_family": classify(text),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    summary_rows = []
    by_task: dict[str, Counter] = {}
    for row in rows:
        by_task.setdefault(row["task_dir"], Counter())[row["ic_family"]] += 1
    for task, counts in sorted(by_task.items()):
        summary_rows.append(
            {
                "task_dir": task,
                "LOWIC": counts.get("LOWIC", 0),
                "ORIGIN": counts.get("ORIGIN", 0),
                "UNKNOWN": counts.get("UNKNOWN", 0),
                "total_templates": sum(counts.values()),
                "dominant_family": counts.most_common(1)[0][0] if counts else "",
            }
        )

    family_csv = TABLE_DIR / "040_18_rendered_template_family_files.csv"
    summary_csv = TABLE_DIR / "040_18_rendered_template_family_by_task.csv"
    result_json = OUT / "040_18_result.json"
    write_csv(family_csv, rows)
    write_csv(summary_csv, summary_rows)

    origin_tasks = [r["task_dir"] for r in summary_rows if r["ORIGIN"] > 0]
    result = {
        "task": TASK,
        "scanned_template_count": len(rows),
        "task_count": len(summary_rows),
        "origin_family_task_dirs": origin_tasks,
        "outputs": {
            "family_csv": str(family_csv.relative_to(ROOT)),
            "summary_csv": str(summary_csv.relative_to(ROOT)),
            "record_md": str(DOC.relative_to(ROOT)),
        },
    }
    result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 040_18 SYA rendered template IC 来源家族审计记录",
        "",
        "## 结论",
        "",
        f"- 扫描 SYA2014 rendered template：{len(rows)} 个。",
        f"- 出现 ORIGIN IC 的任务目录：{', '.join(origin_tasks) if origin_tasks else '无'}。",
        "- 若任务名含 lowIC 但 rendered template 被标记为 ORIGIN，该任务结果必须隔离，不能作为当前 lowIC 证据。",
        "",
        "## 任务级汇总",
        "",
        "| task_dir | LOWIC | ORIGIN | UNKNOWN | total | dominant |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['task_dir']} | {r['LOWIC']} | {r['ORIGIN']} | {r['UNKNOWN']} | {r['total_templates']} | {r['dominant_family']} |"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            f"- 文件级表：`{family_csv.relative_to(ROOT)}`",
            f"- 任务级表：`{summary_csv.relative_to(ROOT)}`",
            f"- JSON：`{result_json.relative_to(ROOT)}`",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
