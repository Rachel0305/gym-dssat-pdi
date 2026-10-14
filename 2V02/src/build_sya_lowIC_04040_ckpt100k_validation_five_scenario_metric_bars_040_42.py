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

import build_sya_lowIC_04036_ckpt100k_validation_five_scenario_metric_bars_040_38 as base04038


TASK = "040_42_sya_lowIC_04040_ckpt100k_validation_five_scenario_metric_bars"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"
PPO_EVAL = (
    ROOT
    / "benchmark_results"
    / "040_40_sya_lowIC_ppo_yield_guardrail_v3"
    / "evaluation"
    / "040_40_checkpoint_validation_summary.csv"
)

OLD_PREFIX = "040_38_sya_lowIC_04036"
NEW_PREFIX = "040_42_sya_lowIC_04040"


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


def patch_base_module() -> None:
    base04038.TASK = TASK
    base04038.OUT = OUT
    base04038.FIG_DIR = OUT / "figures"
    base04038.TAB_DIR = OUT / "tables"
    base04038.SNAP_DIR = OUT / "snapshots"
    base04038.DAILY_DIR = OUT / "daily_outputs"
    base04038.DOC = DOC
    base04038.PROMPT = PROMPT
    base04038.PPO_EVAL = PPO_EVAL
    base04038.SCENARIO_LABELS = dict(base04038.SCENARIO_LABELS)
    base04038.SCENARIO_LABELS["rl_candidate"] = "PPO 040_40 ckpt100k"


def rename_and_clean_outputs() -> dict[str, str]:
    mapping: dict[str, str] = {}
    replacements = {
        OLD_PREFIX: NEW_PREFIX,
        "04036_ckpt100k": "04040_ckpt100k",
        "040_38": "040_42",
        "040_36": "040_40",
    }
    for folder in [OUT / "tables", OUT / "figures"]:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            new_name = path.name
            for old, new in replacements.items():
                new_name = new_name.replace(old, new)
            new_path = path.with_name(new_name)
            if new_path != path:
                if new_path.exists():
                    new_path.unlink()
                path.rename(new_path)
                mapping[path.name] = new_path.name

    # Clean status strings inside CSV tables so future readers do not see 040_38/040_36 labels.
    for path in (OUT / "tables").glob("*.csv"):
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace("fixed_replay_of_040_36_daily_actions", "fixed_replay_of_040_40_daily_actions")
        path.write_text(text, encoding="utf-8-sig")
    return mapping


def write_clean_record() -> dict[str, object]:
    gap_path = OUT / "tables" / f"{NEW_PREFIX}_ckpt100k_metric_gaps_vs_four_max.csv"
    combined_path = OUT / "tables" / f"{NEW_PREFIX}_ckpt100k_five_scenario_metric_summary.csv"
    gaps = pd.read_csv(gap_path, keep_default_na=False) if gap_path.exists() else pd.DataFrame()
    combined = pd.read_csv(combined_path, keep_default_na=False) if combined_path.exists() else pd.DataFrame()

    for col in [
        "win_yield_vs_four_max",
        "win_wp_et_vs_four_max",
        "win_pfp_n_vs_four_max",
        "any_metric_win_four",
    ]:
        if col in gaps:
            gaps[col] = gaps[col].astype(str).str.lower().isin(["true", "1", "yes"])

    win_counts = {
        "years": int(len(gaps)),
        "any_metric_win": int(gaps["any_metric_win_four"].sum()) if "any_metric_win_four" in gaps else 0,
        "yield_win": int(gaps["win_yield_vs_four_max"].sum()) if "win_yield_vs_four_max" in gaps else 0,
        "wp_et_win": int(gaps["win_wp_et_vs_four_max"].sum()) if "win_wp_et_vs_four_max" in gaps else 0,
        "pfp_n_win": int(gaps["win_pfp_n_vs_four_max"].sum()) if "win_pfp_n_vs_four_max" in gaps else 0,
    }
    figs = sorted((OUT / "figures").glob("*.png"))
    lines = [
        "# 040_42 SYA lowIC 040_40 checkpoint100k 验证年份五情景指标柱状图记录",
        "",
        "## 结论先说",
        "",
        f"- 验证年份：{win_counts['years']} 年（2014–2023）。",
        f"- PPO 至少一项指标达到四情景最高：{win_counts['any_metric_win']} / {win_counts['years']} 年。",
        f"- 产量超过四情景最高：{win_counts['yield_win']} / {win_counts['years']} 年。",
        f"- WP_ET 超过四情景最高：{win_counts['wp_et_win']} / {win_counts['years']} 年。",
        f"- PFP_N 超过四情景最高：{win_counts['pfp_n_win']} / {win_counts['years']} 年。",
        "- 本任务不训练；PPO 指标由 040_40 checkpoint100000 动作序列固定重放 DSSAT 后从 Summary.OUT 补齐。",
        "",
        "## 指标差距表",
        "",
        md_table(gaps, 40),
        "",
        "## 五情景指标列表",
        "",
        md_table(
            combined[
                [
                    c
                    for c in [
                        "year",
                        "scenario",
                        "grain_yield_kg_ha",
                        "WP_ET_kg_m3",
                        "PFP_N_kg_kg",
                        "actual_irrigation_mm",
                        "actual_nitrogen_kg_ha",
                        "source_status",
                    ]
                    if c in combined.columns
                ]
            ],
            80,
        ),
        "",
        "## 输出图",
        "",
        *[f"- `{p.relative_to(ROOT).as_posix()}`" for p in figs],
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "win_counts": win_counts,
        "combined_summary": combined_path.relative_to(ROOT).as_posix(),
        "gaps": gap_path.relative_to(ROOT).as_posix(),
        "figures": [p.relative_to(ROOT).as_posix() for p in figs],
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }


def main() -> int:
    if not PPO_EVAL.exists():
        raise FileNotFoundError(PPO_EVAL)
    patch_base_module()
    base04038.main()
    rename_and_clean_outputs()
    result = {
        "task": TASK,
        "checkpoint": 100000,
        **write_clean_record(),
    }
    (OUT / "040_42_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

