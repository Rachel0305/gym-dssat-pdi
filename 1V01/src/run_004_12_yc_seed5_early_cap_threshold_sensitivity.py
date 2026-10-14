from __future__ import annotations

import gc
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/yc_random_weather_ppo/004_12_seed5_early_cap_threshold_sensitivity"
OUT = OUT_BASE / "attempt_01"
REPORT = ROOT / "docs/yc_random_weather_004_12_seed5_early_cap_threshold_sensitivity.md"
PROMPT = ROOT / "prompt_02/004_12_yc_seed5_early_cap_threshold_sensitivity.md"
PRIOR = ROOT / "results/yc_random_weather_ppo/004_11_seed5_mask_sensitivity/attempt_02"
MODEL = ROOT / "results/yc_random_weather_ppo/004_05/training/random_weather/ppo_seed_5/models/checkpoint_100000.zip"
CONFIG = ROOT / "results/yc_random_weather_ppo/004_05/config/canonical_yc_ppo_config.json"
YEARS = list(range(2014, 2024))
CAPS: dict[str, float] = {"cap_090": 90.0, "cap_105": 105.0, "cap_120": 120.0}
CAP_LABELS = {"cap_075": "75 mm", "cap_090": "90 mm", "cap_105": "105 mm", "cap_120": "120 mm", "cap_inf": "∞ mm"}
COLORS = {"cap_075": "#2A9D55", "cap_090": "#2878B5", "cap_105": "#D9822B", "cap_120": "#8A5A9E", "cap_inf": "#5B6573"}
CHECKPOINT_SHA256 = "814A5C73404D2B046927BA0ECA42DCE7F8F1D804B550A226C96B61CFDA97E2BF"
CONFIG_SHA256 = "ABF6B9C11A000DE7F3EBCA680573C0AAAE24CF3F7E4BEAAFEC18943037AFC850"
RSS_LIMIT_MB = 2048.0


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def write_json(path: Path, obj: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def import_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics_existing() -> tuple[pd.DataFrame, pd.DataFrame]:
    old = pd.read_csv(PRIOR / "annual_metrics_A_B_C_D.csv")
    a = old[old.variant.eq("A_canonical")].copy()
    b = old[old.variant.eq("B_no_early")].copy()
    if len(a) != 10 or len(b) != 10 or set(a.year.astype(int)) != set(YEARS) or set(b.year.astype(int)) != set(YEARS):
        raise RuntimeError("004_11 reused endpoints do not cover all ten years")
    a["threshold"] = "cap_075"
    a["source"] = "004_07_02 canonical formal result, reconciled in 004_11"
    b["threshold"] = "cap_inf"
    b["source"] = "004_11 B_no_early full evaluation-only rollout"
    return a, b


def make_pareto(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = frame.groupby(["threshold", "early_cap_mm"], as_index=False).agg(
        yield_mean_kg_ha=("yield_kg_ha", "mean"), yield_sd_kg_ha=("yield_kg_ha", "std"),
        irrigation_mean_mm=("irrigation_mm", "mean"), nitrogen_mean_kg_ha=("nitrogen_kg_ha", "mean"),
        wp_et_mean=("wp_et_kg_m3", "mean"), pfp_n_mean=("pfp_n_kg_kg", "mean"),
        common_reward_mean=("common_reward", "mean"), canonical_reward_mean=("canonical_reward", "mean"),
    )
    means["pareto_3obj"] = True
    for idx, x in means.iterrows():
        dominates = means.apply(lambda y: (
            y.yield_mean_kg_ha >= x.yield_mean_kg_ha
            and y.irrigation_mean_mm <= x.irrigation_mean_mm
            and y.nitrogen_mean_kg_ha <= x.nitrogen_mean_kg_ha
            and (y.yield_mean_kg_ha > x.yield_mean_kg_ha
                 or y.irrigation_mean_mm < x.irrigation_mean_mm
                 or y.nitrogen_mean_kg_ha < x.nitrogen_mean_kg_ha)
        ), axis=1)
        means.loc[idx, "pareto_3obj"] = not bool(dominates.any())
    annual = frame.copy()
    annual["pareto_3obj"] = True
    for year, group in annual.groupby("year"):
        for idx, x in group.iterrows():
            dom = group.apply(lambda y: (
                y.yield_kg_ha >= x.yield_kg_ha and y.irrigation_mm <= x.irrigation_mm
                and y.nitrogen_kg_ha <= x.nitrogen_kg_ha
                and (y.yield_kg_ha > x.yield_kg_ha or y.irrigation_mm < x.irrigation_mm or y.nitrogen_kg_ha < x.nitrogen_kg_ha)
            ), axis=1)
            annual.loc[idx, "pareto_3obj"] = not bool(dom.any())
    return means, annual


def plots(frame: pd.DataFrame, means: pd.DataFrame) -> list[str]:
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8})
    result = []
    for col, ylabel, name in [
        ("yield_kg_ha", "Yield (kg/ha)", "figure_yield_by_year.png"),
        ("irrigation_mm", "Total irrigation (mm)", "figure_irrigation_by_year.png"),
        ("nitrogen_kg_ha", "Total N (kg/ha)", "figure_nitrogen_by_year.png"),
        ("wp_et_kg_m3", "WP_ET (kg/m3)", "figure_wp_et_by_year.png"),
        ("common_reward", "Common reward", "figure_common_reward_by_year.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9.2, 4.5), dpi=170)
        for key in CAP_LABELS:
            part = frame[frame.threshold.eq(key)].sort_values("year")
            ax.plot(part.year, part[col], marker="o", linewidth=1.8, markersize=4, color=COLORS[key], label=CAP_LABELS[key])
        ax.set_title(ylabel); ax.set_xlabel("Year"); ax.set_ylabel(ylabel); ax.set_xticks(YEARS)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.7); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
        fig.tight_layout()
        path = figdir / name; fig.savefig(path, bbox_inches="tight"); plt.close(fig); result.append(path.relative_to(ROOT).as_posix())
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=170)
    for row in means.itertuples(index=False):
        if row.threshold == "cap_120" and means.loc[means.threshold.eq("cap_105"), "yield_mean_kg_ha"].iloc[0] == row.yield_mean_kg_ha:
            continue
        point_label = "105 / 120 mm (same result point)" if row.threshold == "cap_105" else f"{CAP_LABELS[row.threshold]} (N={row.nitrogen_mean_kg_ha:.0f})"
        ax.scatter(row.irrigation_mean_mm, row.yield_mean_kg_ha, s=45 + 0.15 * row.nitrogen_mean_kg_ha,
                   c=COLORS[row.threshold], marker="o" if row.pareto_3obj else "X", edgecolors="black", linewidths=.6,
                   label=point_label)
        point_text = "105 / 120 mm" if row.threshold == "cap_105" else CAP_LABELS[row.threshold]
        ax.annotate(point_text, (row.irrigation_mean_mm, row.yield_mean_kg_ha), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean annual irrigation (mm; lower is better)"); ax.set_ylabel("Mean yield (kg/ha; higher is better)")
    ax.set_title("Seed 5 fixed-checkpoint trade-off\nmarker size ∝ mean N; circle = mean 3-objective Pareto")
    ax.grid(color="#D9DEE3", linewidth=.7); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); path = figdir / "figure_pareto_yield_irrigation_nitrogen.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); result.append(path.relative_to(ROOT).as_posix())
    return result


def make_report(frame: pd.DataFrame, means: pd.DataFrame, annual: pd.DataFrame, plot_paths: list[str], manifest: dict[str, Any], overwrite: bool = False) -> None:
    base = means.set_index("threshold").loc["cap_075"]
    lines = [
        "# YC 004_12：seed 5 early-cap threshold sensitivity", "", "## 执行边界", "",
        "- 固定 seed 5 checkpoint 100K；75 mm 及 ∞ mm 端点复用既有已核验结果，只对 90/105/120 mm 执行 2014–2023 完整闭环 evaluation-only rollout。",
        "- early cap 是唯一变化项。mid cap 150 mm、DAP 窗口、I/N 间隔、季节与日预算、late reserve、动作空间、observation、reward、天气、DSSAT/runtime 均冻结。未训练、未修改正式配置/源码/模型，未提交或推送。",
        f"- 新增 rollout：{manifest['completed_new_variant_years']}/30 variant-years；峰值 process-tree RSS {manifest['peak_rss_mb']:.1f} MiB（硬限 {RSS_LIMIT_MB:.0f} MiB）。Checkpoint SHA256 `{manifest['checkpoint_sha256']}`；config SHA256 `{manifest['config_sha256']}`。",
        "", "## 十年均值 trade-off", "",
        "核心目标是 yield 最大化、灌溉与 N 最小化。表中 Pareto 指基于三个十年均值目标的 nondominated 状态；圆点图用 X 标识被支配点。WP_ET、PFP_N 和 reward 是辅助指标。", "",
        "| Early cap | Yield 均值±SD (kg/ha) | Irrigation (mm) | N (kg/ha) | WP_ET | PFP_N | Common reward | Canonical reward | 3目标 Pareto | ΔYield vs 75 | ΔI vs 75 | ΔN vs 75 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|",
    ]
    for r in means.itertuples(index=False):
        lines.append(f"| {CAP_LABELS[r.threshold]} | {r.yield_mean_kg_ha:.1f} ± {r.yield_sd_kg_ha:.1f} | {r.irrigation_mean_mm:.1f} | {r.nitrogen_mean_kg_ha:.1f} | {r.wp_et_mean:.3f} | {r.pfp_n_mean:.2f} | {r.common_reward_mean:.2f} | {r.canonical_reward_mean:.4f} | {'是' if r.pareto_3obj else '否'} | {r.yield_mean_kg_ha-base.yield_mean_kg_ha:+.1f} | {r.irrigation_mean_mm-base.irrigation_mean_mm:+.1f} | {r.nitrogen_mean_kg_ha-base.nitrogen_mean_kg_ha:+.1f} |")
    lines += ["", "## 逐年 Pareto 与相对 75 mm", "", "| 年份 | 阈值 | Yield Δ | I Δ | N Δ | WP_ET Δ | PFP_N Δ | 3目标 Pareto |", "|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    byyear = annual.pivot(index="year", columns="threshold")
    for year in YEARS:
        base_row = frame[(frame.year.eq(year)) & frame.threshold.eq("cap_075")].iloc[0]
        for key in CAP_LABELS:
            row = frame[(frame.year.eq(year)) & frame.threshold.eq(key)].iloc[0]
            flag = bool(annual[(annual.year.eq(year)) & annual.threshold.eq(key)].pareto_3obj.iloc[0])
            lines.append(f"| {year} | {CAP_LABELS[key]} | {row.yield_kg_ha-base_row.yield_kg_ha:+.1f} | {row.irrigation_mm-base_row.irrigation_mm:+.1f} | {row.nitrogen_kg_ha-base_row.nitrogen_kg_ha:+.1f} | {row.wp_et_kg_m3-base_row.wp_et_kg_m3:+.3f} | {row.pfp_n_kg_kg-base_row.pfp_n_kg_kg:+.2f} | {'是' if flag else '否'} |")
    best_trade = means[(means.threshold.ne("cap_075")) & means.pareto_3obj]
    if best_trade.empty:
        verdict = "没有新增阈值进入十年均值 Pareto 前沿；75 mm 以外未发现三目标折中优势。"
    else:
        labels = "、".join(CAP_LABELS[x] for x in best_trade.threshold)
        verdict = f"除 75 mm 外，{labels} 位于十年均值 Pareto 前沿，与 75 mm 构成不同资源—产量折中；这不等于存在单一最优阈值。"
    means_by_threshold = means.set_index("threshold")
    same_105_120 = frame[frame.threshold.eq("cap_105")].sort_values("year")[
        ["yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "action_signature"]
    ].reset_index(drop=True).equals(frame[frame.threshold.eq("cap_120")].sort_values("year")
        [["yield_kg_ha", "irrigation_mm", "nitrogen_kg_ha", "action_signature"]].reset_index(drop=True))
    wins_105 = int(((frame[frame.threshold.eq("cap_105")].set_index("year").yield_kg_ha
                     - frame[frame.threshold.eq("cap_075")].set_index("year").yield_kg_ha) > 0).sum())
    lines += ["", "## 结论", "", f"- {verdict}",
              f"- 105 mm 相对 75 mm：十年平均产量 {means_by_threshold.loc['cap_105','yield_mean_kg_ha']-base.yield_mean_kg_ha:+.1f} kg/ha，灌溉 {means_by_threshold.loc['cap_105','irrigation_mean_mm']-base.irrigation_mean_mm:+.1f} mm，N {means_by_threshold.loc['cap_105','nitrogen_mean_kg_ha']-base.nitrogen_mean_kg_ha:+.1f} kg/ha；产量更高年份 {wins_105}/10。105/120 的年度 yield、I、N 和 action signature 完全相同：{'是' if same_105_120 else '否'}。因此 120 mm 在这份固定 checkpoint 结果中未显示超过 105 mm 的收益；105 mm 是较紧且观测等效的候选点。",
              "- 是否值得：需按前沿上的产量增益是否足以补偿新增水/N判断，并查看逐年一致性；报告不以 reward 单指标选阈值，也不把更多资源消耗自动视作改善。",
              "- 本分析只刻画已训练 seed 5 actor 在改变 early mask 后的闭环反事实。不能外推为新 mask 下重新训练的效果，也不构成正式修改 canonical mask 的批准。",
              "- 75 与 ∞ 端点来自历史正式结果，90/105/120 来自本轮；所有端点 checkpoint/config 指纹一致，weather/template 输入经逐季哈希核验。", "", "## 执行记录", "",
              "- 初次启动时，checkpoint SHA256 常量存在一位录入错误，provenance gate 在创建输出目录或运行 DSSAT 前停止；核对宿主/容器哈希后修正常量再启动。",
              "- 30 季 rollout 全部完成后，首次聚合因复用基线 early_cap_mm 为空而在报告阶段失败；按 threshold 标签补齐数值映射后仅重做后处理，没有重跑模拟。原始逐季 trace 和 manifest 保留了过程证据。", "", "## 图表", ""]
    lines.extend(f"- `{p}`" for p in plot_paths)
    lines += ["", "## 数据与复现", "", "- 新增逐年指标：`annual_metrics_5_thresholds.csv`；相对 75 mm：`annual_differences_vs_75.csv`。",
              "- action/mask traces：每阈值 episode CSV 位于 `episodes/`，汇总 `decision_trace_new_thresholds.csv`；管理事件见 `action_events_new_thresholds.csv`。",
              "- 完整 provenance、输入哈希与闭合检查见 `formal_manifest.json` 和 `provenance.json`。", "", "未提交、未推送。", ""]
    if REPORT.exists() and not overwrite:
        raise FileExistsError(REPORT)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run() -> None:
    if OUT.exists() or REPORT.exists():
        raise FileExistsError(f"Output exists; refusing overwrite: {OUT} / {REPORT}")
    if sha256_file(MODEL) != CHECKPOINT_SHA256 or sha256_file(CONFIG) != CONFIG_SHA256:
        raise RuntimeError("Frozen checkpoint/config hash differs from audited 004_11 provenance")
    OUT.mkdir(parents=True)
    for sub in ("logs", "episodes", "figures"):
        (OUT / sub).mkdir()
    manifest: dict[str, Any] = {
        "task": "004_12 YC seed5 early-cap threshold sensitivity", "status": "running",
        "started_utc": pd.Timestamp.utcnow().isoformat(), "seed": 5, "checkpoint": 100000,
        "checkpoint_sha256": sha256_file(MODEL), "config_sha256": sha256_file(CONFIG),
        "new_thresholds_mm": list(CAPS.values()), "reused_thresholds": {"75": "004_07_02/004_11 A_canonical", "infinity": "004_11 B_no_early"},
        "frozen_mid_cap_mm": 150, "rss_limit_mb": RSS_LIMIT_MB, "completed_new_variant_years": 0,
        "additional_training": False, "config_modified": False, "wrapper_source_modified": False,
    }
    (OUT / "logs/formal.log").touch(exist_ok=False)
    oldout, stdout, stderr = None, sys.stdout, sys.stderr
    log = (OUT / "logs/formal.log").open("w", encoding="utf-8", newline="\n")
    class Tee:
        def write(self, s):
            stdout.write(s); log.write(s); stdout.flush(); log.flush(); return len(s)
        def flush(self): stdout.flush(); log.flush()
    sys.stdout = sys.stderr = Tee()
    try:
        parent = import_module(ROOT / "src/run_004_11_yc_seed5_mask_sensitivity.py", "yc_00411_runtime_for_00412")
        parent.OUT = OUT
        runner, pilot, staged, _ = parent.load_runtime()
        pilot.MAX_PROCESS_TREE_RSS_MB = RSS_LIMIT_MB
        helper = parent.import_module(ROOT / "src/build_relaxed_success_five_scenario_daily_evidence_027_05.py", "yc_02705_metrics_00412")
        model_path = MODEL
        steps, formal_eps, trace_by_year = parent.read_formal_refs(runner)
        base_a, base_inf = metrics_existing()
        all_rows = [base_a, base_inf]
        all_traces: list[pd.DataFrame] = []
        run_order = [(key, 2019) for key in CAPS]
        run_order += [(key, year) for key in CAPS for year in YEARS if year != 2019]
        rss_peak = 0.0
        for key, year in run_order:
            cap = CAPS[key]
            variant_root = OUT / "runtime" / key
            variant_root.mkdir(parents=True, exist_ok=True)
            parent.OUT = variant_root
            runner.TASK_ROOT = variant_root
            parent.VARIANTS["B_no_early"] = (cap, 150.0)
            row, trace_rows = parent.run_episode(runner, pilot, staged, helper, model_path, "B_no_early", year, steps[year], smoke=(year == 2019))
            row["threshold"] = key; row["early_cap_mm"] = cap; row["source"] = "new complete evaluation-only rollout"
            all_rows.append(pd.DataFrame([row]))
            trace = pd.DataFrame(trace_rows)
            trace["threshold"] = key
            trace_path = OUT / "episodes" / f"trace_{key}_{year}.csv"
            trace.to_csv(trace_path, index=False, encoding="utf-8-sig", lineterminator="\n")
            all_traces.append(trace)
            rss_peak = max(rss_peak, float(row["peak_rss_mb"]))
            manifest["completed_new_variant_years"] += 1
            manifest["peak_rss_mb"] = rss_peak
            manifest["last_completed"] = {"threshold": key, "year": year, "yield_kg_ha": row["yield_kg_ha"], "irrigation_mm": row["irrigation_mm"], "nitrogen_kg_ha": row["nitrogen_kg_ha"]}
            if year == 2019:
                manifest.setdefault("smoke_passed_thresholds", []).append(key)
            # Incremental progress evidence prevents losing finished seasons on interruption.
            progress_path = OUT / f"progress_{key}_{year}.json"
            write_json(progress_path, {"status": "passed", "row": row, "completed_utc": pd.Timestamp.utcnow().isoformat()})
            (OUT / "formal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metrics = pd.concat(all_rows, ignore_index=True, sort=False)
        metrics["early_cap_mm"] = metrics.threshold.map({"cap_075": 75.0, "cap_090": 90.0, "cap_105": 105.0, "cap_120": 120.0, "cap_inf": float("inf")})
        metrics = metrics.sort_values(["year", "early_cap_mm"]).reset_index(drop=True)
        if len(metrics) != 50 or metrics.duplicated(["threshold", "year"]).any():
            raise RuntimeError(f"Expected 50 unique threshold-years, got {len(metrics)}")
        metrics.to_csv(OUT / "annual_metrics_5_thresholds.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        base = metrics[metrics.threshold.eq("cap_075")].set_index("year")
        deltas = []
        for row in metrics.itertuples(index=False):
            if row.threshold == "cap_075": continue
            b = base.loc[int(row.year)]
            deltas.append({"threshold": row.threshold, "early_cap_mm": row.early_cap_mm, "year": int(row.year),
                           "yield_delta_kg_ha": row.yield_kg_ha-b.yield_kg_ha, "irrigation_delta_mm": row.irrigation_mm-b.irrigation_mm,
                           "nitrogen_delta_kg_ha": row.nitrogen_kg_ha-b.nitrogen_kg_ha, "wp_et_delta": row.wp_et_kg_m3-b.wp_et_kg_m3,
                           "pfp_n_delta": row.pfp_n_kg_kg-b.pfp_n_kg_kg, "common_reward_delta": row.common_reward-b.common_reward,
                           "canonical_reward_delta": row.canonical_reward-b.canonical_reward})
        delta_df = pd.DataFrame(deltas)
        delta_df.to_csv(OUT / "annual_differences_vs_75.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        trace_df = pd.concat(all_traces, ignore_index=True)
        trace_df.to_csv(OUT / "decision_trace_new_thresholds.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        events = trace_df[trace_df.irrigation_mm.gt(0) | trace_df.nitrogen_kg_ha.gt(0)].copy()
        events.to_csv(OUT / "action_events_new_thresholds.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        means, annual = make_pareto(metrics)
        means.to_csv(OUT / "mean_tradeoff_pareto.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        annual.to_csv(OUT / "annual_pareto_membership.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        plot_paths = plots(metrics, means)
        manifest.update({"status": "passed", "passed": True, "completed_utc": pd.Timestamp.utcnow().isoformat(),
                         "completed_new_variant_years": 30, "peak_rss_mb": rss_peak, "plot_paths": plot_paths,
                         "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"), "config_and_wrapper_source_modified": False})
        (OUT / "formal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_json(OUT / "provenance.json", {
            "checkpoint": {"path": str(MODEL.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(MODEL)},
            "config": {"path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(CONFIG), "modified": False},
            "prompt_sha256": sha256_file(PROMPT), "new_thresholds": CAPS,
            "frozen_rules": {"mid_cap_mm": 150, "late_pre_dap90_cap_mm": 195, "season_irrigation_mm": 240, "season_nitrogen_kg_ha": 250,
                             "interval_days": 7, "observation_reward_action_grid_weather_dssat": "unchanged"},
            "reused_endpoints": {"75_mm": str((PRIOR / "annual_metrics_A_B_C_D.csv").relative_to(ROOT)).replace("\\", "/"),
                                 "infinity_mm": str((PRIOR / "annual_metrics_A_B_C_D.csv").relative_to(ROOT)).replace("\\", "/")},
            "runtime": "serial CPU; one model/environment; max_cached_years=1", "training": False,
        })
        make_report(metrics, means, annual, plot_paths, manifest)
        print(f"[004_12 pass] rollout_years=30 peak_rss={rss_peak:.1f}MiB report={REPORT}", flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "passed": False, "completed_utc": pd.Timestamp.utcnow().isoformat(), "error": repr(exc), "traceback": traceback.format_exc()})
        (OUT / "formal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        log.close(); gc.collect()


def analyze_existing() -> None:
    metrics = pd.read_csv(OUT / "annual_metrics_5_thresholds.csv")
    metrics["early_cap_mm"] = metrics.threshold.map({"cap_075": 75.0, "cap_090": 90.0, "cap_105": 105.0, "cap_120": 120.0, "cap_inf": float("inf")})
    if len(metrics) != 50 or metrics.early_cap_mm.isna().any():
        raise RuntimeError("Saved annual metrics are incomplete or contain an unknown threshold")
    means, annual = make_pareto(metrics)
    metrics.to_csv(OUT / "annual_metrics_5_thresholds.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    means.to_csv(OUT / "mean_tradeoff_pareto.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    annual.to_csv(OUT / "annual_pareto_membership.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    plot_paths = plots(metrics, means)
    manifest = json.loads((OUT / "formal_manifest.json").read_text(encoding="utf-8"))
    manifest.pop("error", None)
    manifest.pop("traceback", None)
    manifest.update({"status": "passed", "passed": True, "analysis_repair": "explicit threshold-to-mm mapping; no simulation rerun",
                     "completed_utc": pd.Timestamp.utcnow().isoformat(), "plot_paths": plot_paths,
                     "config_and_wrapper_source_modified": False,
                     "recovery_history": [
                         {"stage": "preflight", "issue": "checkpoint SHA256 constant typo", "effect": "stopped before output creation or DSSAT episode", "resolution": "verified host/container hashes and corrected constant"},
                         {"stage": "postprocess", "issue": "reused baseline early_cap_mm missing before groupby", "effect": "all 30 rollouts had passed; report aggregation failed", "resolution": "explicit threshold mapping; postprocessing only, no rerun"},
                     ]})
    (OUT / "formal_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_report(metrics, means, annual, plot_paths, manifest, overwrite=True)
    print(f"[004_12 analysis pass] report={REPORT}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["formal", "analyze"], default="formal")
    args = parser.parse_args()
    run() if args.phase == "formal" else analyze_existing()
