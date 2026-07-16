from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calculate_five_site_wue_nue_from_summary_019_10 import num, parse_summary_out


OUT = ROOT / "benchmark_results" / "021_18"
DOC = ROOT / "docs" / "2026-07-15_021_18_sy2014_deterministic_upper_bound_oracle_search.md"
BASELINES = {
    "null": ROOT / "DSSAT_auto_validation/sy2014_ic2_forward_validation_021_02/runs/null/pdi_tmp_snapshot/Summary.OUT",
    "recorded": ROOT / "DSSAT_auto_validation/sy2014_ic2_forward_validation_021_02/runs/recorded/pdi_tmp_snapshot/Summary.OUT",
    "dssat_auto": ROOT / "DSSAT_auto_validation/sy2014_ic2_forward_validation_021_02/runs/dssat_auto/pdi_tmp_snapshot/Summary.OUT",
    "official_extension_expert": ROOT / "DSSAT_auto_validation/extension_expert_baseline_018_03/SY2014/extension_expert_fixed_dap/pdi_tmp_snapshot_eval/Summary.OUT",
}
WP_THRESHOLD = 2.26
YIELD_THRESHOLD = 11077.0
PFP_THRESHOLD = 36.9


def metric_row(name: str, path: Path) -> dict[str, object]:
    row = parse_summary_out(path)[-1]
    hwam, ircm, nicm, nucm, etcp = (num(row, k) for k in ("HWAM", "IRCM", "NICM", "NUCM", "ETCP"))
    ypem, ypim, ypnam, ypnum = (num(row, k) for k in ("YPEM", "YPIM", "YPNAM", "YPNUM"))
    return {
        "scenario": name,
        "yield_kg_ha": hwam,
        "irrigation_mm_summary": ircm,
        "nitrogen_kg_ha_summary": nicm,
        "nitrogen_uptake_kg_ha": nucm,
        "ETCP_mm": etcp,
        "WP_ET_kg_m3": ypem * 0.1 if ypem is not None else np.nan,
        "WP_ET_recalculated": hwam / etcp / 10 if hwam and etcp else np.nan,
        "IWP_gross_kg_m3": ypim * 0.1 if ircm and ircm > 0 and ypim is not None else np.nan,
        "PFP_N_kg_kg": ypnam if nicm and nicm > 0 and ypnam is not None else np.nan,
        "NUtE_kg_kg": ypnum if nucm and nucm > 0 and ypnum is not None else np.nan,
        "PNB_N": nucm / nicm if nicm and nicm > 0 and nucm is not None else np.nan,
        "source_summary_out": str(path.relative_to(ROOT)),
    }


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    subset = frame[columns].copy()
    def fmt(value: object) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(fmt(value) for value in row) + " |" for row in subset.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def main() -> None:
    candidates = pd.read_csv(OUT / "021_18_candidate_summary.csv")
    baselines = pd.DataFrame([metric_row(name, path) for name, path in BASELINES.items()])
    baselines.to_csv(OUT / "021_18_baseline_metrics.csv", index=False)

    candidates["wp_pass"] = candidates["WP_ET_kg_m3"].ge(WP_THRESHOLD)
    candidates["strict_success"] = (
        candidates["final_gwad"].ge(YIELD_THRESHOLD)
        & candidates["wp_pass"]
        & candidates["PFP_N_kg_kg"].ge(PFP_THRESHOLD)
        & candidates["summary_irrigation_total"].le(120.0)
        & candidates["summary_nitrogen_total"].le(300.0)
    )
    candidates["yield_minus_expert"] = candidates["final_gwad"] - YIELD_THRESHOLD
    candidates["water_saved_vs_expert"] = 266.0 - candidates["summary_irrigation_total"]
    candidates["nitrogen_saved_vs_expert"] = 300.0 - candidates["summary_nitrogen_total"]
    candidates.to_csv(OUT / "021_18_candidate_summary_with_success.csv", index=False)
    success = candidates[candidates["strict_success"]].sort_values(
        ["summary_nitrogen_total", "summary_irrigation_total", "final_gwad"],
        ascending=[True, True, False],
    )
    success.to_csv(OUT / "021_18_success_candidates.csv", index=False)
    if success.empty:
        raise RuntimeError("No candidate satisfied preregistered success criteria")
    selected = success.iloc[0]
    summary = {
        "status": "completed",
        "candidate_count": int(len(candidates)),
        "strict_success_count": int(len(success)),
        "selected_demonstration": selected["scenario"],
        "selected_yield_kg_ha": float(selected["final_gwad"]),
        "selected_irrigation_mm": float(selected["summary_irrigation_total"]),
        "selected_nitrogen_kg_ha": float(selected["summary_nitrogen_total"]),
        "selected_wp_et_kg_m3": float(selected["WP_ET_kg_m3"]),
        "selected_iwp_kg_m3": float(selected["IWP_gross_kg_m3"]),
        "selected_pfp_n_kg_kg": float(selected["PFP_N_kg_kg"]),
        "selected_nute_kg_kg": float(selected["NUtE_kg_kg"]),
        "interpretation": "A feasible schedule exists; this is deterministic oracle evidence, not a learned DQN result.",
    }
    (OUT / "021_18_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    baseline_cols = ["scenario", "yield_kg_ha", "irrigation_mm_summary", "nitrogen_kg_ha_summary", "WP_ET_kg_m3", "IWP_gross_kg_m3", "PFP_N_kg_kg", "NUtE_kg_kg"]
    success_cols = ["scenario", "phase", "final_gwad", "summary_irrigation_total", "summary_nitrogen_total", "WP_ET_kg_m3", "IWP_gross_kg_m3", "PFP_N_kg_kg", "NUtE_kg_kg"]
    text = f"""# 021_18 SY2014 确定性上界 / oracle 调度搜索记录

## 背景与问题

导师的目标是让 DQN 尽量同时超过 expert 与 DSSAT auto 的产量和水氮利用效率。021_18 不训练 DQN，而是先验证当前 SY2014 IC=2、离散动作、共享 7 d 间隔、I≤120/N≤300 约束内是否存在客观可达的高质量策略。

## 冻结条件与输入

- 输入：复用 `021_14` 的 SY2014 IC=2 PDI/DSSAT 4.8.0 输入。
- 动作：I=0/15/30 mm，N=0/50/100 kg ha-1；DAP 1–120；共享间隔≥7 d。
- 无 DQN 训练；无 reward、IC、DSSAT 输入修改。
- 所有候选保存原始 `Summary.OUT`、`PlantGro.OUT`、日值、动作和输入快照。

## 同输入基准

{md_table(baselines, baseline_cols)}

注意：recorded 的管理事件请求总氮是 293 kg ha-1，但 DSSAT `Summary.OUT` 的实际 `NICM` 为 247 kg ha-1；本表和利用效率统一采用 Summary.OUT 实际值。官方 expert 的事件汇总为266.1 mm，Summary.OUT 为266 mm；利用效率采用 Summary.OUT。

## 预注册成功条件

- 产量≥11077 kg ha-1；
- WP_ET≥2.26 kg m-3；
- 对 N>0 候选，PFP_N≥36.9 kg kg-1；
- I≤120 mm、N≤300 kg ha-1。

IWP、NUtE、PNB 同时报告，但 DSSAT auto 的 N=0，PFP_N 不可定义，不能解释为无穷大。

## Smoke

scaled 25K 的 I120/N300 高产时序被确定性回放为11199 kg ha-1，与历史结果一致；IRCM/NICM=120/300，未漏动作。因此搜索链路通过。

## 搜索过程

1. 固定 I120，扫描 early/mid/late × N150/200/250/300，共12组。
2. 选取前三个候选，测试 I90/I60/I30，共9组。
3. 粗扫描发现 I60–I90 跨越预注册阈值，因此追加登记两种 I75 × N200/250/300，共6组 adaptive refinement；没有修改成功阈值。

汇总脚本第一次执行时因容器缺少 pandas 可选依赖 `tabulate` 而停止；没有重新模拟，也没有修改结果。随后改为脚本内置 Markdown 表格生成，未安装或升级环境包。

## 严格通过候选

{md_table(success, success_cols)}

## 最省氮的严格通过候选

- `{selected['scenario']}`：产量 {selected['final_gwad']:.0f} kg ha-1，I={selected['summary_irrigation_total']:.0f} mm，N={selected['summary_nitrogen_total']:.0f} kg ha-1。
- 相对官方 expert：增产 {selected['yield_minus_expert']:.0f} kg ha-1，少灌 {selected['water_saved_vs_expert']:.0f} mm，少施氮 {selected['nitrogen_saved_vs_expert']:.0f} kg ha-1。
- WP_ET={selected['WP_ET_kg_m3']:.2f} kg m-3，IWP={selected['IWP_gross_kg_m3']:.2f} kg m-3，PFP_N={selected['PFP_N_kg_kg']:.1f} kg kg-1，NUtE={selected['NUtE_kg_kg']:.1f} kg kg-1。
- 实际时序：灌溉 DAP22/29/42/56/79，各15 mm；施氮 DAP29=50、42=100、56=50 kg ha-1。

## 结论

当前约束内确实存在同时超过官方 expert 和 DSSAT auto 产量、并提高主要水肥投入效率的确定性策略。因此 SY2014 的困难不是“动作空间里没有好策略”，而是 DQN 尚未稳定学到这种时序。

该结论不能写成“DQN 已经成功”。021_18 的策略是人工剪枝搜索得到的 oracle，只能作为：

1. 可达上界证据；
2. demonstration-guided DQN 的示范轨迹；
3. 后续 DQN 是否真正学到合理时序的独立评估标准。

## 限制

- 只验证 SY2014；不能直接外推到其他站点或年份。
- 搜索不是全局穷举；“oracle”指当前剪枝候选集中的确定性上界，不是数学全局最优。
- NUtE=40.0 低于 N=0 的 DSSAT auto（53.4），但 auto 的低产且无外施氮情景不能用 PFP_N 比较；因此“全面超过每一个效率指标”仍不成立。

## 下一步

以 `{selected['scenario']}` 及其他严格通过候选构建小型示范集，先做纯离线数据校验和行为克隆预训练 smoke，再在保持 DQN、reward、IC、动作约束不变的条件下进行短步数微调。必须与无示范 DQN 做单变量、多 seed 对照；不直接启动大规模训练。
"""
    DOC.write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
