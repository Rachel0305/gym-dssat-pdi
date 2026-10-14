"""Fit and audit LCA train-only CLI parameters without running WGEN."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/lca_cli_parameter_audit_016"
REPORT = ROOT / "docs/lca_cli_parameter_audit_016.md"
SRC = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC"
FLAGS = ROOT / "results/sy_lc_random_weather_015/nonrain_provenance_resolution_484.csv"
sys.path.insert(0, str(ROOT / "scripts"))
import build_dssat_cli as cli


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_wth(path: Path, year: int) -> list[cli.WeatherDay]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = next((line.split()[1:] for line in lines if line.startswith("@") and "DATE" in line), None)
    if header is None or not set(cli.WEATHER_VARIABLES).issubset(header):
        raise ValueError(f"Missing WTH columns: {path}")
    result = []
    for line in lines:
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        token = parts[0]
        y, doy = int(token[:-3]), int(token[-3:])
        if len(token) == 5:
            y += 2000 if y < 50 else 1900
        day = date(y, 1, 1) + timedelta(days=doy - 1)
        if y != year or day.year != y:
            raise ValueError(f"Invalid WTH date: {path}: {token}")
        values = {name: float(parts[header.index(name)]) for name in cli.WEATHER_VARIABLES}
        if not all(math.isfinite(x) and x not in (-99, -999, 9999, 99999) for x in values.values()):
            raise ValueError(f"Missing/nonfinite weather: {day}")
        if values["SRAD"] < 0 or values["RAIN"] < 0 or values["TMAX"] < values["TMIN"]:
            raise ValueError(f"Physical weather error: {day}")
        result.append(cli.WeatherDay(day, values["SRAD"], values["TMAX"], values["TMIN"], values["RAIN"]))
    expected = (date(year + 1, 1, 1) - date(year, 1, 1)).days
    if len(result) != expected or [x.day for x in result] != [date(year, 1, 1) + timedelta(days=i) for i in range(expected)]:
        raise ValueError(f"Incomplete year: {path}")
    return result


def main() -> None:
    if REPORT.exists() or OUT.exists():
        raise SystemExit("Refusing to overwrite existing 016 audit artifacts")
    paths = [SRC / f"CNLC{year % 100:02d}01.WTH" for year in range(2005, 2014)]
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    records = [row for year, path in zip(range(2005, 2014), paths) for row in read_wth(path, year)]
    assert len(records) == 3287
    fit = cli.fit_monthly_statistics(records)
    monthly = cli.monthly_weather_summary(records)
    # YC's serialization helpers are reused, but its station/year-specific wrapper is not.
    original_start, original_end = cli.EXPECTED_START, cli.EXPECTED_END
    cli.EXPECTED_START, cli.EXPECTED_END = date(2005, 1, 1), date(2013, 12, 31)
    try:
        content = cli._cli_text(records, "CNLC", 37.890, 114.694, 50, fit, monthly)
    finally:
        cli.EXPECTED_START, cli.EXPECTED_END = original_start, original_end
    lines = content.splitlines()
    first = lines.index("*WGEN PARAMETERS")
    rows = [line for line in lines[first + 2:] if line.strip()][:12]
    parsed = [cli.parse_wgen_parameter_row(line) for line in rows]
    serialization_errors = []
    for row, fixed in zip(fit, parsed):
        if row["month"] != fixed["MTH"]:
            serialization_errors.append(f"month {row['month']}: index mismatch")
        for name in cli.WGEN_PARAMETER_FIELDS:
            precision = cli.WGEN_PARAMETER_PRECISION[name]
            if abs(float(row[name]) - float(fixed[name])) > 0.5 * 10 ** (-precision) + 1e-9:
                serialization_errors.append(f"month {row['month']}: {name} rounding mismatch")
    with FLAGS.open(encoding="utf-8-sig", newline="") as stream:
        flagged = [r for r in csv.DictReader(stream) if r["site"] == "LCA"]
    assert len(flagged) == 22
    counts = Counter((r["date"][:7], r["variable"]) for r in flagged)
    by_day = {r.day.isoformat(): r for r in records}
    replacement = {}
    for r in flagged:
        day, name = r["date"], r["variable"]
        month = int(day[5:7])
        pool = [getattr(x, name.lower()) for x in records if x.day.month == month and
                (x.day.isoformat()[:7], name) not in counts and x.day.isoformat() != day]
        # Exclude every flagged cell for this variable, not whole month-years.
        flag_dates = {f["date"] for f in flagged if f["variable"] == name}
        pool = [getattr(x, name.lower()) for x in records if x.day.month == month and x.day.isoformat() not in flag_dates]
        replacement[(day, name)] = statistics.mean(pool)
    changed = []
    for row in records:
        values = {name.lower(): replacement[(row.day.isoformat(), name)] for name in ("SRAD", "TMAX", "TMIN")
                  if (row.day.isoformat(), name) in replacement}
        changed.append(replace(row, **values) if values else row)
    alternative = cli.fit_monthly_statistics(changed)
    impact = []
    for base, other in zip(fit, alternative):
        for name in cli.WGEN_PARAMETER_FIELDS:
            delta = float(other[name]) - float(base[name])
            impact.append({"month": base["month"], "parameter": name, "original": base[name],
                           "replacement": other[name], "absolute_delta": abs(delta),
                           "relative_delta": abs(delta) / abs(float(base[name])) if float(base[name]) else None})
    january = [r.rain for r in records if r.day.month == 1 and r.rain > 0]
    loo = [cli.gamma_shape_greenwood_durand(january[:i] + january[i+1:]) for i in range(len(january))]
    # A month with only four positive amounts cannot provide a robust gamma fit.
    jan_alpha = float(fit[0]["ALPHA"])
    jan_loo_spread = max(loo) - min(loo)
    alpha_clamped = [int(r["month"]) for r in fit if float(r["ALPHA"]) >= 0.998 - 1e-12]
    physical_errors = []
    for row in fit:
        month = row["month"]
        for name in ("SDSD", "SWSD", "XDSD", "XWSD", "NASD", "RTOT", "RNUM"):
            if float(row[name]) < 0:
                physical_errors.append(f"{month}:{name}<0")
        if not (0 < float(row["ALPHA"]) <= 0.998 and 0 <= float(row["PDW"]) <= 1):
            physical_errors.append(f"{month}:ALPHA/PDW range")
        if not all(math.isfinite(float(row[name])) for name in cli.WGEN_PARAMETER_FIELDS):
            physical_errors.append(f"{month}:nonfinite")
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    if before != after:
        raise ValueError("WTH changed during audit")
    if physical_errors or serialization_errors:
        gate = "BLOCKED_BY_PARAMETER_QC"
    elif len(january) < 10 or jan_loo_spread > 0.2 or 1 in alpha_clamped:
        gate = "BLOCKED_BY_JANUARY_GAMMA_SAMPLE"
    else:
        gate = "PARAMETER_QC_PASS"
    qc = {"site": "LCA", "period": "2005-2013", "days": len(records), "wet_rule": "RAIN > 0.0",
          "fitting_code": "scripts/build_dssat_cli.py:fit_monthly_statistics",
          "wth_sha256_before_after_equal": before == after, "wth_sha256": before,
          "parameter_count": 14 * 12, "all_months_complete": len(fit) == 12 and all(len([r[n] for n in cli.WGEN_PARAMETER_FIELDS]) == 14 for r in fit),
          "physical_errors": physical_errors, "serialization_errors": serialization_errors,
          "january": {"wet_days": len(january), "rain_mm": january, "alpha": jan_alpha,
                      "alpha_leave_one_out": loo, "alpha_leave_one_out_spread": jan_loo_spread},
          "alpha_clamped_months": alpha_clamped,
          "flagged_cells": len(flagged), "max_flag_impact": max(impact, key=lambda x: x["absolute_delta"]),
          "gate": gate, "wgen_executed": False}
    OUT.mkdir(parents=True)
    cli_path = OUT / "CNLC_2005_2013_candidate.CLI"
    cli_path.write_text(content, encoding="ascii", newline="\n")
    with (OUT / "monthly_wgen_statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fit[0]))
        writer.writeheader(); writer.writerows(fit)
    with (OUT / "flagged_cell_parameter_sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(impact[0]))
        writer.writeheader(); writer.writerows(impact)
    (OUT / "parameter_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flagged_months = sorted({int(r["date"][5:7]) for r in flagged})
    table = "\n".join(f"| {r['month']} | {r['wet_day_count']} | {r['ALPHA']:.3f} | {r['PDW']:.3f} | {r['RTOT']:.1f} |"
                      for r in fit)
    report = f"""# LCA 2005–2013 CLI 参数拟合与审计（016）

只读输入为冻结的 CNLC 2005–2013 九个 WTH，共 {len(records)} 天。采用 `RAIN > 0.0` 判湿日，使用 `scripts/build_dssat_cli.py` 的 14 项逐月拟合公式和 `I6,14(1X,F5.0)` 写入格式。输入 SHA256 前后完全相同，见 `parameter_qc.json`。未运行 WGEN、DSSAT、PPO，也未生成随机天气。

## 结果及 gate

**{gate}。** 12 个月 × 14 项共 168 个参数均已计算；物理/有限值错误 {len(physical_errors)} 项，固定宽度和舍入错误 {len(serialization_errors)} 项。候选 CLI 可供检查，但目前**不放行下一阶段 WGEN 输入**：1 月只有 {len(january)} 个湿日，Gamma `ALPHA={jan_alpha:.3f}`，逐一删去一个湿日后范围 {min(loo):.3f}–{max(loo):.3f}（跨度 {jan_loo_spread:.3f}）。四个样本不足以证明月降水分布形状稳定；此门槛是本轮保守方法学判断，并非 CLI 语法错误。

| 月份 | 湿日数（9 年合计） | ALPHA | PDW | RTOT（mm/月） |
|---:|---:|---:|---:|---:|
{table}

需人工重点关注 **1 月** 的湿日量和 Gamma 拟合；ALPHA 达 0.998 上限的月份：{alpha_clamped or '无'}。22 个来源标记值分布月份：{flagged_months}。在内存中以训练期同月未标记均值替换这些位置后，参数最大绝对变化为 {qc['max_flag_impact']['absolute_delta']:.4f}（{qc['max_flag_impact']['month']} 月 {qc['max_flag_impact']['parameter']}）。降水参数本身不受这 22 个非降水值影响；完整逐项影响见 `flagged_cell_parameter_sensitivity.csv`。这项敏感性只检验 22 项，不证明其他历史补充值来源。

候选文件：`results/lca_cli_parameter_audit_016/CNLC_2005_2013_candidate.CLI`；未舍入数值：`monthly_wgen_statistics.csv`；机器检查：`parameter_qc.json`。原 015 审计和 gate 未更改。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"gate": gate, "january": qc["january"], "alpha_clamped_months": alpha_clamped,
                      "max_flag_impact": qc["max_flag_impact"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
