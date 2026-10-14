from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "benchmark_results" / "033_02_enable_ic_render_smoke"
DOC_PATH = PROJECT_ROOT / "docs" / "033_02_enable_ic_render_smoke_record.md"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "ppo_observed_years" / "config_ppo_observed_years.yaml"
FACTORS = ["CU", "FL", "SA", "IC", "MP", "MI", "MF", "MR", "MC", "MT", "ME", "MH", "SM"]


def parse_treatment(text: str) -> dict:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("@N R O C TNAME"):
            for row in lines[idx + 1 :]:
                stripped = row.strip()
                if not stripped or stripped.startswith("@") or stripped.startswith("*"):
                    continue
                parts = stripped.split()
                values = parts[5 : 5 + len(FACTORS)]
                return {
                    "treatment_row": stripped,
                    **dict(zip(FACTORS, values)),
                }
    return {"treatment_row": ""}


def parse_ic_ids(text: str) -> list[str]:
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


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from ppo_safe_rendering import load_yaml, safe_render_template

    config = load_yaml(CONFIG_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for station, entries in config["observed_years"].items():
        entry = entries[0]
        year = int(entry["year"])
        planting_date = str(entry["planting_date"])
        rendered = safe_render_template(
            station=station,
            year=year,
            planting_date=planting_date,
            output_root=OUT_DIR,
            run_tag="033_02_ic_smoke",
        )
        text = rendered.read_text(encoding="utf-8", errors="replace")
        treatment = parse_treatment(text)
        ic_ids = parse_ic_ids(text)
        ic = treatment.get("IC", "")
        mi = treatment.get("MI", "")
        mf = treatment.get("MF", "")
        rows.append(
            {
                "station": station,
                "year": year,
                "planting_date": planting_date,
                "rendered_path": str(rendered.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "IC": ic,
                "MI": mi,
                "MF": mf,
                "initial_condition_ids": ";".join(ic_ids),
                "ic_valid": ic in ic_ids,
                "pass": (ic == "1" and mi == "1" and mf == "1" and ic in ic_ids),
                "treatment_row": treatment.get("treatment_row", ""),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "033_02_render_smoke_rows.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    passed = int(df["pass"].sum())
    total = len(df)

    table_lines = [
        f"| {r.station} | {r.year} | {r.IC} | {r.MI} | {r.MF} | {r.initial_condition_ids} | {'通过' if r['pass'] else '失败'} |"
        for _, r in df.iterrows()
    ]

    DOC_PATH.write_text(
        "\n".join(
            [
                "# 033_02 启用 IC 后的渲染 smoke 记录",
                "",
                "## 任务性质",
                "",
                "本任务只验证补丁后的安全渲染函数是否把 treatment 行中的 `IC/MI/MF` 显式启用；不跑 DSSAT，不训练。",
                "",
                "## 输出",
                "",
                f"- CSV：`{csv_path.relative_to(PROJECT_ROOT)}`",
                "",
                "## 结果",
                "",
                f"- 通过：{passed}/{total}",
                "",
                "| 站点 | 年份 | IC | MI | MF | INITIAL CONDITIONS id | 判定 |",
                "|---|---:|---:|---:|---:|---|---|",
                *table_lines,
                "",
                "## 结论",
                "",
                "- 若 5/5 均通过，则说明未来通过 `ppo_safe_rendering.safe_render_template` 生成的工作输入会启用当前模板中的第一组 `*INITIAL CONDITIONS`。",
                "- 这只修复未来渲染链条，不会改变既有历史结果；既有结果若要使用 IC，必须重新生成输入并重跑。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"passed {passed}/{total}")
    print(csv_path)
    print(DOC_PATH)
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
