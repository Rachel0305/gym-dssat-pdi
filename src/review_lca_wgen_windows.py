"""Read-only LCA historical-window sensitivity review; never runs WGEN."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC"
OUT = ROOT / "results/lca_wgen_window_review"
REPORT = ROOT / "docs/lca_wgen_window_review.md"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_lca_cli_parameters_016 import read_wth
import build_dssat_cli as cli


def raw_alpha(amounts):
    if len(amounts) < 3 or any(v <= 0 for v in amounts):
        return None
    mean = math.fsum(amounts) / len(amounts)
    y = math.log(mean) - math.fsum(math.log(v) for v in amounts) / len(amounts)
    if y <= 0:
        return None
    return (8.898919 + 9.05995*y + 0.9775373*y*y) / (y*(17.79728 + 11.968477*y + y*y))


def get_year(y):
    p = SRC / f"CNLC{y%100:02d}01.WTH"
    if not p.exists():
        return None, "FILE_ABSENT", None
    try:
        rows = read_wth(p, y)
        return rows, "COMPLETE_QC_PASS", hashlib.sha256(p.read_bytes()).hexdigest()
    except (ValueError, IndexError) as e:
        return None, f"INVALID:{e}", hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (OUT.exists() or REPORT.exists()) and not (ROOT/'backups/lca_wgen_window_review_before_detail_revision/window_comparison.csv').exists():
        raise SystemExit("Refusing to overwrite window review artifacts without backup")
    years = {}
    status = {}
    hashes = {}
    for y in range(1995, 2014):
        years[y], status[y], hashes[y] = get_year(y)
    windows = [(2005, 2013), (2001, 2013), (2000, 2013), (1995, 2013)]
    rows = []
    summary = []
    for start, end in windows:
        label = f"{start}-{end}"
        missing = [y for y in range(start, end+1) if years[y] is None]
        if missing:
            summary.append(dict(window=label, available=False, missing_or_invalid_years=missing, gate="UNAVAILABLE"))
            continue
        daily = [d for y in range(start,end+1) for d in years[y]]
        fit = cli.fit_monthly_statistics(daily)
        jan = fit[0]
        clamped = [int(r["month"]) for r in fit if r["ALPHA"] >= 0.998-1e-12]
        for r in fit:
            month = int(r["month"])
            amounts = [d.rain for d in daily if d.day.month==month and d.rain>0]
            loo_alpha = []
            loo_raw = []
            loo_wet = []
            for omitted in range(start,end+1):
                subset = [d.rain for d in daily if d.day.year != omitted and d.day.month==month and d.rain>0]
                loo_wet.append(len(subset))
                a = raw_alpha(subset)
                loo_raw.append(a)
                loo_alpha.append(min(a,0.998) if a is not None else None)
            raw = raw_alpha(amounts)
            valid = [v for v in loo_alpha if v is not None]
            valid_raw = [v for v in loo_raw if v is not None]
            annual_wet = [sum(d.day.year==y and d.day.month==month and d.rain>0 for d in daily) for y in range(start,end+1)]
            rows.append(dict(window=label, start_year=start, end_year=end, month=month,
                days=int(r["day_count"]), wet_days=len(amounts), wet_years=len({d.day.year for d in daily if d.day.month==month and d.rain>0}),
                max_single_year_wet=max(annual_wet), max_single_year_wet_fraction=max(annual_wet)/len(amounts),
                rain_total_mm=round(sum(amounts),3), alpha_raw=raw, alpha_clamped=float(r["ALPHA"]),
                is_clamped=month in clamped, pdw=float(r["PDW"]), rtot=float(r["RTOT"]), rnum=float(r["RNUM"]),
                leave_one_year_out_min_wet=min(loo_wet),
                leave_one_year_out_alpha_min=min(valid) if valid else None,
                leave_one_year_out_alpha_max=max(valid) if valid else None,
                leave_one_year_out_alpha_range=max(valid)-min(valid) if valid else None,
                leave_one_year_out_raw_alpha_min=min(valid_raw) if valid_raw else None,
                leave_one_year_out_raw_alpha_max=max(valid_raw) if valid_raw else None,
                leave_one_year_out_raw_alpha_range=max(valid_raw)-min(valid_raw) if valid_raw else None,
                leave_one_year_out_clamped_count=sum(v is not None and v>=0.998-1e-12 for v in loo_alpha),
                leave_one_year_out_valid_count=len(valid)))
        jan_row=next(r for r in rows if r['window']==label and r['month']==1)
        # Review criterion: at least ten January wet days, represented in >=5 years,
        # and an identifiable (unclamped) January gamma shape in full and LOO-year fits.
        jan_ok = (jan_row['wet_days']>=10 and jan_row['wet_years']>=5 and
                  not jan_row['is_clamped'] and jan_row['leave_one_year_out_clamped_count']==0 and
                  jan_row['leave_one_year_out_valid_count']==end-start+1)
        gate = 'JANUARY_SAMPLE_DIAGNOSTIC_PASS' if jan_ok else 'BLOCKED_BY_JANUARY_GAMMA_SAMPLE'
        summary.append(dict(window=label, available=True, days=len(daily), january_wet_days=jan_row['wet_days'],
            january_wet_years=jan_row['wet_years'], january_alpha=jan_row['alpha_clamped'],
            january_raw_alpha=jan_row['alpha_raw'], january_loo_alpha_range=jan_row['leave_one_year_out_alpha_range'],
            january_loo_raw_alpha_range=jan_row['leave_one_year_out_raw_alpha_range'],
            january_max_single_year_wet=jan_row['max_single_year_wet'],
            january_max_single_year_wet_fraction=jan_row['max_single_year_wet_fraction'],
            clamped_months=clamped, min_monthly_wet_days=min(int(r['wet_day_count']) for r in fit),
            gate=gate))
    # Preserve the previously audited 2005-2013 result and verify numerical continuity.
    prior=json.loads((ROOT/'results/lca_cli_parameter_audit_016/parameter_qc.json').read_text(encoding='utf-8'))
    baseline=next(s for s in summary if s['window']=='2005-2013')
    assert baseline['january_wet_days']==prior['january']['wet_days']
    assert baseline['january_alpha']==prior['january']['alpha']
    for y,h in hashes.items():
        if h is not None:
            assert hashlib.sha256((SRC/f'CNLC{y%100:02d}01.WTH').read_bytes()).hexdigest()==h
    OUT.mkdir(parents=True, exist_ok=True)
    fields=list(rows[0])
    with (OUT/'window_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (OUT/'year_availability_and_hashes.json').write_text(json.dumps({str(y):{'status':status[y], 'sha256':hashes[y]} for y in years},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'window_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table='\n'.join(f"| {s['window']} | {s.get('days','—')} | {s.get('january_wet_days','—')} | {s.get('january_wet_years','—')} | {s.get('january_max_single_year_wet','—')} | {s['january_raw_alpha']:.3f} | {s.get('clamped_months','—')} | {s['gate']} |" if s['available'] else f"| {s['window']} | — | — | — | — | — | — | UNAVAILABLE |" for s in summary)
    report=f"""# LCA WGEN 长历史窗口可行性审查

本次只读检查 `DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC` 内现有 CNLC WTH。湿日定义为 `RAIN > 0.0 mm`；Gamma 使用 `scripts/build_dssat_cli.py` 的 Greenwood–Durand 公式，原始估计若大于等于 1，CLI 算法截为 0.998。逐年文件完整性、连续日期、有限值、RAIN/SRAD 非负和 TMAX≥TMIN 均检查，所有已读取文件的 SHA256 前后相同。未运行 WGEN、DSSAT、PPO，未修改 WTH 或 016 审计。

| 窗口 | 日数 | 1 月湿日 | 有湿日的1月年份数 | 最多单年湿日 | 1 月原始 ALPHA | 被 clamp 月份 | 窗口判断 |
|---|---:|---:|---:|---:|---:|---|---|
{table}

1995–2013 不可用：缺少或不合格年份 {next(s for s in summary if s['window']=='1995-2013')['missing_or_invalid_years']}。项目经清洗的 LCA 连续天气记录起于 2000 年；1998 年另有历史 WTH，但含缺测天气值；1999 年 `CNLC9901.WTH` 只有文件头，不能连接到 2000 年，也不能补成 1995–2013。详见 `year_availability_and_hashes.json`。

判定使用保守的预先明示诊断门槛：1 月至少 10 个湿日、分布于至少 5 个年份；全窗口及逐一剔除任一年后的 Gamma 估计都不触发 0.998 截断。10 日/5 年不是 DSSAT 官方合格阈值，而是本次检验极稀疏月份可识别性的审查准则。逐月湿日数、原始与截断后 ALPHA、被截断状态、逐一剔年范围及转移/降水参数全部保存在 `window_comparison.csv`。被截断的 ALPHA 即使剔年后不变，也不能当作稳定证据。

2000–2013 虽有 26 个 1 月湿日，但 2000 年独占 14 个（53.8%）；剔除该年只余 12 个。该窗口 1 月原始 ALPHA 为 {next(s for s in summary if s['window']=='2000-2013')['january_raw_alpha']:.3f}，逐一剔年原始估计范围见 CSV（跨度 {next(s for s in summary if s['window']=='2000-2013')['january_loo_raw_alpha_range']:.3f}），而写入 CLI 的值始终被截为 0.998。2001–2013 的 12 个湿日仅分布于 4 个年份。增加年份改善了数量，但没有证明 1 月降水形状估计的稳健性；年份集中问题仍在。

**结论：** {('存在满足该 1 月稳定性诊断的历史窗口。' if any(s['gate']=='JANUARY_SAMPLE_DIAGNOSTIC_PASS' for s in summary) else '现有候选窗口均不能解除 1 月 Gamma 样本阻塞。')} 此处只评估窗口与参数稳定性；即使某窗口通过，也还需单独审查扩展年份的天气来源与所有 14 项 CLI 参数，不能据此运行 WGEN。2005–2013 原候选 CLI 及其阻塞结论保持原样。
"""
    REPORT.write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
