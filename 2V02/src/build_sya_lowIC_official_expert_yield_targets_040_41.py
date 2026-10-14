from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_lowIC_four_baseline_rebuild_040_21 as base04021


TASK = "040_41_sya_lowIC_official_expert_yield_targets"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
TARGET_CSV = OUT / "evaluation" / "040_41_official_expert_yield_targets.csv"

STATION = "SYA"
SITE = "SY"
YEARS = list(range(2005, 2024))


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def selected_years(_mode: str, _year: int | None) -> pd.DataFrame:
    split = pd.read_csv(base04021.baseline034.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selected = split[
        split["station_code"].astype(str).eq(STATION)
        & split["year"].isin(YEARS)
    ].copy()
    if selected.empty:
        raise RuntimeError("未找到 SYA 2005–2023 年份划分。")
    return selected.sort_values("year").reset_index(drop=True)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def write_clean_record(result: dict) -> None:
    targets = pd.read_csv(TARGET_CSV, keep_default_na=False) if TARGET_CSV.exists() else pd.DataFrame()
    lines = [
        "# 040_41 SYA lowIC official expert 产量目标表记录",
        "",
        "## 任务目的",
        "",
        "本任务只补齐 040_40 terminal yield guardrail 所需的 official expert 年度产量目标，不训练 PPO/DQN。",
        "",
        "## 结果",
        "",
        f"- 成功年份数：`{result.get('successful_years')}/19`",
        f"- 输出：`{TARGET_CSV.relative_to(ROOT).as_posix()}`",
        "",
        "## 年度目标表",
        "",
        md_table(targets, 40),
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()

    old_task = base04021.TASK
    old_out = base04021.OUT
    old_doc = base04021.DOC
    old_prompt = base04021.PROMPT
    old_scenarios = base04021.SCENARIOS
    old_loader = base04021.load_selected_years
    try:
        base04021.TASK = TASK
        base04021.OUT = OUT
        base04021.DOC = DOC
        base04021.PROMPT = PROMPT
        base04021.SCENARIOS = ["official_extension_expert"]
        base04021.load_selected_years = selected_years  # type: ignore[assignment]
        result = base04021.run("full", None)
    finally:
        base04021.TASK = old_task
        base04021.OUT = old_out
        base04021.DOC = old_doc
        base04021.PROMPT = old_prompt
        base04021.SCENARIOS = old_scenarios
        base04021.load_selected_years = old_loader  # type: ignore[assignment]

    summary = OUT / "evaluation" / "040_21_baseline_summary.csv"
    if not summary.exists():
        raise FileNotFoundError(summary)
    df = pd.read_csv(summary, keep_default_na=False)
    df = df[df["scenario"].astype(str).eq("official_extension_expert")].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    keep = [
        "station_code",
        "site",
        "year",
        "scenario",
        "grain_yield_kg_ha",
        "biomass_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "source_input_root",
        "source_status",
        "snapshot_path",
    ]
    out = df[[c for c in keep if c in df.columns]].sort_values("year")
    out.to_csv(TARGET_CSV, index=False, encoding="utf-8-sig")
    result.update(
        {
            "task": TASK,
            "target_csv": TARGET_CSV.relative_to(ROOT).as_posix(),
            "successful_years": int(out["year"].nunique()),
            "next_step_allowed": int(out["year"].nunique()) == len(YEARS),
        }
    )
    write_clean_record(result)
    (OUT / "040_41_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

