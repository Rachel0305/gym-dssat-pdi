"""Read-only site-year audit for the YC feedback-controller demo.

This is intentionally an audit helper, not the controller implementation.  It
uses completed baseline daily logs and writes the required audit record before
the new experiment code is created.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DOC = ROOT / "docs" / "2026-09-09_yc_feedback_controller_demo_audit.md"

SOURCES = {
    "YC": ROOT / "benchmark_results/055_02_yca_lowIC_four_baselines_static_level1/evaluation/055_02_baseline_daily.csv",
    "HLA": ROOT / "benchmark_results/054_02_hla_lowIC_four_baselines_static_level1/evaluation/054_02_baseline_daily.csv",
    "FQ": ROOT / "benchmark_results/051_02_fqa_originIC_four_baselines_static_level1/evaluation/051_02_baseline_daily.csv",
}


def audit_rows(site: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["scenario"] = df["scenario"].fillna("null")
    for col in ["year", "dap", "swfac", "nstres"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["scenario"] == "null") & (df["dap"] > 0)].copy()
    df = df.sort_values(["year", "dap"]).drop_duplicates(["year", "dap"], keep="last")
    rows: list[dict[str, object]] = []
    for year, group in df.groupby("year", sort=True):
        sw = group["swfac"].dropna().to_numpy(dtype=float)
        ns = group["nstres"].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "site": site,
                "year": int(year),
                "days": int(len(group)),
                "swfac_min": float(np.min(sw)),
                "swfac_p05": float(np.percentile(sw, 5)),
                "swfac_p25": float(np.percentile(sw, 25)),
                "swfac_median": float(np.percentile(sw, 50)),
                "swfac_p75": float(np.percentile(sw, 75)),
                "swfac_p95": float(np.percentile(sw, 95)),
                "swfac_max": float(np.max(sw)),
                "swfac_active_days_pct": float(100 * np.mean(sw > 0.05)),
                "nstres_min": float(np.min(ns)),
                "nstres_p05": float(np.percentile(ns, 5)),
                "nstres_p25": float(np.percentile(ns, 25)),
                "nstres_median": float(np.percentile(ns, 50)),
                "nstres_p75": float(np.percentile(ns, 75)),
                "nstres_p95": float(np.percentile(ns, 95)),
                "nstres_max": float(np.max(ns)),
                "nstres_active_days_pct": float(100 * np.mean(ns > 0.05)),
            }
        )
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    work = frame[columns or list(frame.columns)].copy()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].map(lambda value: f"{value:.3f}" if isinstance(value, float) else str(value))
    return work.to_markdown(index=False)


def main() -> None:
    tables = {site: audit_rows(site, path) for site, path in SOURCES.items()}
    all_rows = pd.concat(tables.values(), ignore_index=True)
    selected = tables["YC"].loc[tables["YC"]["year"].eq(2019)].iloc[0]
    reference = pd.read_csv(
        ROOT / "benchmark_results/055_02_yca_lowIC_four_baselines_static_level1/evaluation/055_02_baseline_summary.csv"
    )
    reference["year"] = pd.to_numeric(reference["year"], errors="coerce")
    reference_2019 = reference[(reference["year"] == 2019) & (reference["scenario"] == "official_extension_expert")].iloc[0]

    active_cols = [
        "site",
        "year",
        "days",
        "swfac_p95",
        "swfac_max",
        "swfac_active_days_pct",
        "nstres_p95",
        "nstres_max",
        "nstres_active_days_pct",
    ]
    full_cols = [
        "site",
        "year",
        "days",
        "swfac_min",
        "swfac_p05",
        "swfac_p25",
        "swfac_median",
        "swfac_p75",
        "swfac_p95",
        "swfac_max",
        "swfac_active_days_pct",
        "nstres_min",
        "nstres_p05",
        "nstres_p25",
        "nstres_median",
        "nstres_p75",
        "nstres_p95",
        "nstres_max",
        "nstres_active_days_pct",
    ]

    lines = [
        "# YC feedback-controller demo：开工前审计",
        "",
        "## 审计结论",
        "",
        "- 选择 `YC / YCA, 2019` 作为单一 demo site-year。它来自现有 YC lowIC 完成日志，102 个去重后的 DAP>0 日记录中，`SWFAC>0.05` 的明显水胁迫日占 46.875%，`NSTRES>0.05` 的明显氮胁迫日占 52.083%；两者都出现连续动态范围，但没有全季贴近极端值。",
        "- 选择 YC 而不是 HLA/FQ 遵循任务给定优先级 `YC → HLA → FQ`。YC 2019 的水、氮信号同时存在；HLA 的候选年通常也有信号，但其水胁迫与低IC/固定管理组合更容易受输入管理模式影响；FQ 多数年份至少一个信号接近无变化。",
        "- `reference_yield` 使用同一 DSSAT/gym-DSSAT lowIC 环境下、YC2019、`official_extension_expert` 方案的模拟产量 `7453.510 kg/ha`，产量约束为 `yield >= 7229.905 kg/ha`。这不是田间实测值。",
        "",
        "## 1. 环境、目录与运行方式",
        "",
        "- 项目包含五站点标签：SY/HLA/YC/FQ/LC；当前站点模板位于 `configs/sites/*.yaml`，YC 模板为 `configs/sites/yca.yaml`，输入目录为 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013[_lowIC_manual]/YC`。",
        "- 已确认 YC lowIC 既有工作流：`src/055_yca_lowIC_site_transfer/`，其 055_02 基线重建结果覆盖 2014–2023 并保存逐日 `swfac/nstres`；2019 在 `033_04` split registry 中为 validation 年。",
        "- 当前可用运行容器为 `nifty_taussig`；项目内 Python/DSSAT 检查采用：`docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python ...`。不要在 Windows 主机直接调用容器专用解释器路径。",
        "- 现有 `pymoo` 在主机和容器环境均未安装；本 demo 不安装或升级系统/项目环境依赖，后续在新实验目录中实现显式 mixed-variable NSGA-II 算子并记录这一点。",
        "",
        "## 2. SWFAC、NSTRES、DAP 的实际语义",
        "",
        "- `references/utils/utils.py:106-110` 对 maize 执行 `state['nstres'] = 1 - state['nstres']` 与 `state['swfac'] = 1 - state['swfac']`；项目诊断文档也明确说明当前表中的两个值是 gym 后处理后的胁迫指数。",
        "- 因此本任务使用：`0` 近似无胁迫，数值越大表示胁迫越强；DSSAT 原始 factor 的方向不能直接套到 controller。",
        "- 现有项目的可复用明显胁迫诊断线为 `>0.05`（对应 `run_all_year_direct_action_safe_ppo.py` 的 `*_stress_days_gt_0p05` 字段）。本次阈值搜索会以分布为锚点并两侧适度扩展，不把 P05–P95 当成硬边界。",
        "- 逐日日志的 `DAP` 来自环境 observation；审计统计只保留 `DAP>0` 并按 `(year,DAP)` 去重，避免初始 observation 重复计数。",
        "",
        "## 3. 候选 site-year 的完整逐年统计（null/no in-season action）",
        "",
        "`SWFAC/NSTRES` 统计均来自现有已完成的 055_02/054_02/051_02 baseline daily CSV，按站点分别读取；`stress-active days` 定义为对应后处理指标 `>0.05` 的日占比。",
        "",
        md_table(all_rows, full_cols),
        "",
        "## 4. 选择理由与参照管理",
        "",
        md_table(all_rows, active_cols),
        "",
        f"YC2019 的具体统计：SWFAC min/P05/P25/median/P75/P95/max = {selected.swfac_min:.3f}/{selected.swfac_p05:.3f}/{selected.swfac_p25:.3f}/{selected.swfac_median:.3f}/{selected.swfac_p75:.3f}/{selected.swfac_p95:.3f}/{selected.swfac_max:.3f}；NSTRES min/P05/P25/median/P75/P95/max = {selected.nstres_min:.3f}/{selected.nstres_p05:.3f}/{selected.nstres_p25:.3f}/{selected.nstres_median:.3f}/{selected.nstres_p75:.3f}/{selected.nstres_p95:.3f}/{selected.nstres_max:.3f}。",
        f"同一环境中的 YC2019 `official_extension_expert` 模拟产量为 {float(reference_2019['grain_yield_kg_ha']):.3f} kg/ha，模拟投入为 N={float(reference_2019['actual_nitrogen_kg_ha']):.3f} kg/ha、I={float(reference_2019['actual_irrigation_mm']):.3f} mm；这些只用于产量约束参照，不把现场观测产量混入约束。",
        "",
        "## 5. 施氮、灌溉与 basal/pre-season 管理审计",
        "",
        "- 当前 DSSAT 输入并非没有自动施氮：`CNYC0801.MZX` 的 automatic-management block 含 `NMTHR/NAMNT/NCODE/NAOFF`，因此 DSSAT 历史上提供氮胁迫触发的自动施肥例程。但该例程在当前 DSSAT-gym-DSSAT 外部逐日反馈链中没有作为本 demo 的 controller 机制使用，且既有项目已记录其未被可靠验证；本任务实现并显式验证外部氮反馈 controller。",
        "- 既有 055_01 工作流已验证可把 YC lowIC treatment 1 改成 `IRRIG=A, FERTI=L`，以 DSSAT native AUTOIRR 做灌溉、gym-DSSAT action channel 做外部 N；因此 AUTOIRR 在该站点可用。但本 demo 的两个 controller 都必须由外部 daily feedback 独立决定，不能把 AUTOIRR 混入优化目标。",
        "- 原始 YC lowIC `CNYC0801.MZX` treatment 2 含 planting-time `96 kg N/ha`、DAP43 `278 kg N/ha` 以及 DAP43 `120 mm` 固定管理行。为避免把既有专家/农民日历偷偷带入 candidate，demo 将在实验专用输入副本中保留 `96 kg N/ha` 作为所有 candidate 共享的固定 basal N，移除 DAP43 固定 N 和固定灌溉；controller 只控制 DAP 窗口内 in-season N/I。原始 `.MZX` 不修改。",
        "- 因此 demo 统计闭合式为：`total_nitrogen = 96 + sum(controller_n_actions)`；`total_irrigation = 0 + sum(controller_irrigation_actions)`。所有 scenario/candidate 使用完全相同的 basal/pre-season 起点；固定 basal N 计入最终 total nitrogen。",
        "- 现有 055_02 YC2019 记录的 `recorded_farmer_template` 是 `template_reused_not_year_specific`，模拟投入 N=374 kg/ha、I=120 mm，不能当作 YC2019 的已验证现场观测产量。若最终结果表中没有独立田间实测产量，将明确标记为 unavailable，而不伪造 farmer/expert 实测值。",
        "",
        "## 6. 审计后允许进入实现阶段的固定决定",
        "",
        "- site-year: `YC / YCA 2019`, lowIC input profile, single-year proof-of-concept。",
        "- fixed safety windows: `start_dap=7`, `stop_dap=90`（宽泛农艺窗口，不贴合专家具体节点；具体 stop 以该年作物完成期与运行日志复核）。",
        "- optimized parameters: `n_threshold, n_dose, n_min_interval_days, n_season_budget, water_threshold, irrigation_dose, irrigation_min_interval_days, irrigation_season_budget`；固定参数仅为两侧 start/stop 窗口，共 12 参数、优化 8 参数。",
        "- optimization: NSGA-II, first round population=32/generations=20/optimizer seed=1; only if front is non-degenerate may run optional 40/30 confirmation. Mixed variables must use explicit type-aware sampling/crossover/mutation; discrete dose/budget are never rounded from continuous intervals.",
        "- This audit is complete; only after this document is written may the new controller and optimization code be added.",
        "",
    ]
    AUDIT_DOC.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(AUDIT_DOC)


if __name__ == "__main__":
    main()
