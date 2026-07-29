"""039_02 originalIC vs lowIC three-baseline DSSAT audit.

This script runs DSSAT forward simulations only.  It compares original initial
soil conditions against manually lowered initial soil water/mineral nitrogen
conditions for null, official expert, and DSSAT auto scenarios.

No PPO/DQN training is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_static_level1_four_baseline_rebuild_037_07 as baseline037


TASK = "039_02_original_vs_lowIC_three_baseline_audit"
OUT = ROOT / "benchmark_results" / TASK
PROMPT = ROOT / "prompts" / f"{TASK}.md"
DOC = OUT / f"{TASK}_record.md"
BASE_CONFIG = baseline034.BASE_CONFIG
SPLIT_CSV = baseline034.SPLIT_CSV
ORIGINAL_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
LOWIC_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual"
CONDITIONS = {
    "original": ORIGINAL_INPUT_ROOT,
    "lowIC": LOWIC_INPUT_ROOT,
}
SCENARIOS = ["null", "official_extension_expert", "dssat_auto"]
MAX_STEPS = 260
SEED = 0


def ensure_dirs() -> None:
    for rel in ["configs", "tables", "figures", "snapshots", "logs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    shutil.copy2(BASE_CONFIG, OUT / "configs" / BASE_CONFIG.name)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def load_selection(mode: str, station: str | None, year: int | None) -> pd.DataFrame:
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(f"缺少年份清单：{SPLIT_CSV}")
    df = pd.read_csv(SPLIT_CSV, keep_default_na=False)
    df = df[df["station_code"].isin(baseline034.STATIONS)].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df[df["year"].ge(2000)].copy()
    if station:
        df = df[df["station_code"].astype(str).eq(station)].copy()
    if year is not None:
        df = df[df["year"].eq(int(year))].copy()
    df = df.sort_values(["station_code", "year"]).reset_index(drop=True)
    if mode == "smoke" and station is None and year is None:
        rows = []
        for _, group in df.groupby("station_code", sort=True):
            rows.append(group.sort_values("year").iloc[0])
        df = pd.DataFrame(rows).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("没有选中任何站点年份。")
    return df


def build_env_config(run_config: dict[str, Any], selected_years: pd.DataFrame) -> dict[str, Any]:
    env_config = baseline034.build_env_config(run_config, selected_years)
    env_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    env_config["runtime"]["max_steps"] = MAX_STEPS
    env_config["seed"] = SEED
    return env_config


def configure_baseline_globals() -> None:
    baseline034.TASK_ID = "039_02"
    baseline034.OUT = OUT
    baseline034.DOC = DOC
    baseline034.PROMPT = PROMPT
    baseline034.MAX_STEPS = MAX_STEPS
    baseline034.replace_static_application_rows = baseline037.replace_static_application_rows_level1  # type: ignore[assignment]


def run_one(
    condition: str,
    input_root: Path,
    run_config: dict[str, Any],
    env_config: dict[str, Any],
    station: str,
    year: int,
    scenario: str,
    recorded_templates: dict[str, dict[int, dict[str, float]]],
) -> tuple[pd.DataFrame | None, dict[str, Any], list[dict[str, Any]], str]:
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root
    try:
        daily, summary, checks = baseline034.evaluate_scenario(
            run_config,
            env_config,
            station,
            int(year),
            scenario,
            recorded_templates,
        )
        daily = daily.copy()
        daily.insert(0, "condition", condition)
        summary = dict(summary)
        summary["condition"] = condition
        summary["input_root"] = str(input_root.relative_to(ROOT)).replace("\\", "/")
        summary["render_checks_pass"] = all(x.get("status") == "pass" for x in checks)

        old_snapshot = OUT / "snapshots" / station / str(year) / scenario
        new_snapshot = OUT / "snapshots" / condition / station / str(year) / scenario
        if old_snapshot.exists():
            if new_snapshot.exists():
                shutil.rmtree(new_snapshot)
            new_snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_snapshot), str(new_snapshot))
            summary["snapshot_path"] = str(new_snapshot.relative_to(ROOT)).replace("\\", "/")
        summary["run_status"] = "ok"
        return daily, summary, checks, ""
    except Exception:
        summary = {
            "condition": condition,
            "station_code": station,
            "site": baseline034.STATION_TO_SITE.get(station, ""),
            "year": int(year),
            "scenario": scenario,
            "input_root": str(input_root.relative_to(ROOT)).replace("\\", "/"),
            "run_status": "failed",
        }
        return None, summary, [], traceback.format_exc()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root


def build_effect_table(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary[summary["run_status"].eq("ok")].copy()
    metrics = [
        "grain_yield_kg_ha",
        "biomass_kg_ha",
        "actual_irrigation_mm",
        "actual_nitrogen_kg_ha",
        "etcp_mm",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "max_water_stress",
        "max_nitrogen_stress",
    ]
    for col in metrics:
        if col in ok.columns:
            ok[col] = pd.to_numeric(ok[col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (station, year, scenario), group in ok.groupby(["station_code", "year", "scenario"]):
        if {"original", "lowIC"} <= set(group["condition"]):
            orig = group[group["condition"].eq("original")].iloc[0]
            low = group[group["condition"].eq("lowIC")].iloc[0]
            row: dict[str, Any] = {
                "station_code": station,
                "site": orig.get("site", ""),
                "year": int(year),
                "scenario": scenario,
            }
            for col in metrics:
                row[f"original_{col}"] = orig.get(col, math.nan)
                row[f"lowIC_{col}"] = low.get(col, math.nan)
                row[f"delta_{col}"] = low.get(col, math.nan) - orig.get(col, math.nan)
            row["lowIC_increased_stress"] = (
                row["delta_max_water_stress"] > 1e-9 or row["delta_max_nitrogen_stress"] > 1e-9
            )
            row["lowIC_too_severe_candidate"] = (
                scenario != "null"
                and (
                    row["delta_grain_yield_kg_ha"] < -1000
                    or row["lowIC_max_water_stress"] >= 0.5
                    or row["lowIC_max_nitrogen_stress"] >= 0.2
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_station_overview(effect: pd.DataFrame) -> list[Path]:
    if effect.empty:
        return []
    paths: list[Path] = []
    for station, group in effect.groupby("station_code", sort=True):
        group = group.sort_values(["year", "scenario"])
        years = sorted(group["year"].unique())
        scenarios = SCENARIOS
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        fig.suptitle(f"039_02 {station}: lowIC - originalIC response by scenario", fontsize=13, fontweight="bold")
        width = 0.25
        x = list(range(len(years)))
        offsets = {"null": -width, "official_extension_expert": 0.0, "dssat_auto": width}
        colors = {"null": "#555555", "official_extension_expert": "#6A51A3", "dssat_auto": "#C99700"}
        labels = {"null": "Null", "official_extension_expert": "Official expert", "dssat_auto": "DSSAT auto"}
        for scenario in scenarios:
            sub = group[group["scenario"].eq(scenario)].set_index("year")
            vals_y = [sub.loc[y, "delta_grain_yield_kg_ha"] if y in sub.index else math.nan for y in years]
            vals_w = [sub.loc[y, "delta_max_water_stress"] if y in sub.index else math.nan for y in years]
            vals_n = [sub.loc[y, "delta_max_nitrogen_stress"] if y in sub.index else math.nan for y in years]
            xx = [i + offsets[scenario] for i in x]
            axes[0].bar(xx, vals_y, width=width, color=colors[scenario], alpha=0.85, label=labels[scenario])
            axes[1].bar(xx, vals_w, width=width, color=colors[scenario], alpha=0.85)
            axes[2].bar(xx, vals_n, width=width, color=colors[scenario], alpha=0.85)
        axes[0].axhline(0, color="#222222", lw=0.8)
        axes[1].axhline(0, color="#222222", lw=0.8)
        axes[2].axhline(0, color="#222222", lw=0.8)
        axes[0].set_ylabel("Δ yield kg/ha")
        axes[1].set_ylabel("Δ max WSPD/SWFAC")
        axes[2].set_ylabel("Δ max NSTD/NSTRES")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([str(y) for y in years], rotation=45)
        for ax in axes:
            ax.grid(True, axis="y", alpha=0.25)
        axes[0].legend(ncol=3, fontsize=8, loc="best")
        out = OUT / "figures" / f"039_02_{station}_lowIC_minus_original_overview.png"
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out, dpi=180)
        plt.close(fig)
        paths.append(out)
    return paths


def write_record(
    mode: str,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    effect: pd.DataFrame,
    failures: pd.DataFrame,
    figure_paths: list[Path],
    elapsed: float,
) -> None:
    ok = summary[summary["run_status"].eq("ok")].copy() if not summary.empty else pd.DataFrame()
    by_station = pd.DataFrame()
    if not effect.empty:
        by_station = effect.groupby(["station_code", "scenario"]).agg(
            n=("year", "count"),
            mean_delta_yield=("delta_grain_yield_kg_ha", "mean"),
            mean_delta_WSPD=("delta_max_water_stress", "mean"),
            mean_delta_NSTD=("delta_max_nitrogen_stress", "mean"),
            lowIC_increased_stress_n=("lowIC_increased_stress", "sum"),
            too_severe_candidate_n=("lowIC_too_severe_candidate", "sum"),
        ).reset_index()
    lines = [
        "# 039_02 originalIC vs lowIC 三情景 DSSAT 基线响应审计记录",
        "",
        "## 任务边界",
        "",
        "- 本轮只运行 DSSAT 前向模拟，不训练 PPO/DQN。",
        "- 对比 originalIC 与 lowIC_manual 两套输入。",
        "- 情景包括：null、official_extension_expert、dssat_auto。",
        "- DSSAT 原始输出保存到 `snapshots/<condition>/<station>/<year>/<scenario>/`。",
        "",
        "## 执行信息",
        "",
        f"- 模式：`{mode}`",
        f"- 选中站点年份数：{len(selected)}",
        f"- 预期运行数：{len(selected) * len(CONDITIONS) * len(SCENARIOS)}",
        f"- 成功运行数：{len(ok)}",
        f"- 失败运行数：{len(failures)}",
        f"- 耗时：{elapsed:.1f} 秒",
        "",
        "## 站点年份覆盖",
        "",
        md_table(selected.groupby(["station_code", "site", "split"]).size().reset_index(name="n"), 120),
        "",
        "## 按站点与情景的 lowIC 响应摘要",
        "",
        md_table(by_station, 120),
        "",
        "## 失败记录",
        "",
        md_table(failures[["condition", "station_code", "year", "scenario", "details"]] if not failures.empty else failures, 120),
        "",
        "## 图件",
        "",
    ]
    lines.extend([f"- `{p.relative_to(ROOT).as_posix()}`" for p in figure_paths])
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- summary: `benchmark_results/{TASK}/tables/039_02_{mode}_summary.csv`",
            f"- effect: `benchmark_results/{TASK}/tables/039_02_{mode}_lowIC_minus_original_effect.csv`",
            f"- daily: `benchmark_results/{TASK}/tables/039_02_{mode}_daily.csv`",
            f"- failures: `benchmark_results/{TASK}/tables/039_02_{mode}_failures.csv`",
            f"- render check: `benchmark_results/{TASK}/tables/039_02_{mode}_render_check.csv`",
        ]
    )
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--station", default="", help="可选：只跑一个站点，例如 FQA/HLA/LCA/SYA/YCA")
    parser.add_argument("--year", type=int, default=None, help="可选：只跑某一年")
    args = parser.parse_args()

    start = time.time()
    ensure_dirs()
    configure_baseline_globals()

    run_config = direct_ppo.load_yaml(BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = SEED
    run_config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    run_config["runtime"]["max_steps"] = MAX_STEPS

    selected = load_selection(args.mode, args.station or None, args.year)
    selected.to_csv(OUT / "configs" / f"039_02_{args.mode}_selected_station_years.csv", index=False, encoding="utf-8-sig")
    env_config = build_env_config(run_config, selected)
    direct_ppo.write_yaml(env_config, OUT / "configs" / f"039_02_{args.mode}_resolved_env_config.yaml")
    recorded_templates = baseline034.recorded_template_schedules()

    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []

    for row in selected.itertuples(index=False):
        station = str(row.station_code)
        year = int(row.year)
        for condition, input_root in CONDITIONS.items():
            for scenario in SCENARIOS:
                print(f"[039_02] {args.mode} {condition} {station}{year} {scenario}", flush=True)
                daily, summary, checks, tb = run_one(
                    condition,
                    input_root,
                    run_config,
                    env_config,
                    station,
                    year,
                    scenario,
                    recorded_templates,
                )
                summary_rows.append(summary)
                if daily is not None:
                    daily_frames.append(daily)
                for check in checks:
                    render_rows.append({"condition": condition, "station_code": station, "year": year, "scenario": scenario, **check})
                if tb:
                    failure_rows.append(
                        {
                            "condition": condition,
                            "station_code": station,
                            "site": baseline034.STATION_TO_SITE.get(station, ""),
                            "year": year,
                            "scenario": scenario,
                            "details": summary.get("run_status", "failed"),
                            "traceback": tb[-4000:],
                        }
                    )

                partial = pd.DataFrame(summary_rows)
                partial.to_csv(OUT / "tables" / f"039_02_{args.mode}_summary_partial.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(summary_rows)
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    failures = pd.DataFrame(failure_rows)
    render_check = pd.DataFrame(render_rows)
    effect = build_effect_table(summary)
    figure_paths = plot_station_overview(effect)

    summary.to_csv(OUT / "tables" / f"039_02_{args.mode}_summary.csv", index=False, encoding="utf-8-sig")
    daily_all.to_csv(OUT / "tables" / f"039_02_{args.mode}_daily.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(OUT / "tables" / f"039_02_{args.mode}_failures.csv", index=False, encoding="utf-8-sig")
    render_check.to_csv(OUT / "tables" / f"039_02_{args.mode}_render_check.csv", index=False, encoding="utf-8-sig")
    effect.to_csv(OUT / "tables" / f"039_02_{args.mode}_lowIC_minus_original_effect.csv", index=False, encoding="utf-8-sig")

    elapsed = time.time() - start
    write_record(args.mode, selected, summary, effect, failures, figure_paths, elapsed)
    result = {
        "task": TASK,
        "mode": args.mode,
        "selected_station_years": int(len(selected)),
        "expected_runs": int(len(selected) * len(CONDITIONS) * len(SCENARIOS)),
        "successful_runs": int(summary["run_status"].eq("ok").sum()) if "run_status" in summary else 0,
        "failed_runs": int(len(failures)),
        "output_root": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "elapsed_seconds": elapsed,
    }
    (OUT / f"039_02_{args.mode}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0 if len(failures) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
