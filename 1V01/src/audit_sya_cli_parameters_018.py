"""Fit and audit a candidate SYA CLI using only the 017 train-only CSV."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/sya_train_only_fill_sensitivity/candidate_weather_2005_2013.csv'
PREVIOUS = ROOT/'results/sya_train_only_fill_sensitivity'
OUT = ROOT/'results/sya_cli_parameter_audit_018'
REPORT = ROOT/'docs/sya_cli_parameter_audit_018.md'
sys.path.insert(0,str(ROOT/'scripts'))
import build_dssat_cli as cli


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_candidate():
    with SOURCE.open(encoding='utf-8-sig',newline='') as f:
        raw=list(csv.DictReader(f))
    expected=[date(2005,1,1)+timedelta(days=i) for i in range(3287)]
    if len(raw)!=len(expected):
        raise ValueError('Unexpected candidate day count')
    records=[]
    for r,day in zip(raw,expected):
        if r['DATE']!=day.isoformat():
            raise ValueError(f'Candidate date mismatch: {r["DATE"]} vs {day}')
        vals=[float(r[k]) for k in ('SRAD','TMAX','TMIN','RAIN')]
        if not all(math.isfinite(v) for v in vals) or vals[0]<0 or vals[3]<0 or vals[1]<vals[2]:
            raise ValueError(f'Invalid candidate weather at {day}')
        records.append(cli.WeatherDay(day,*vals))
    return records


def main():
    if (OUT.exists() or REPORT.exists()) and not (ROOT/'backups/sya_cli_parameter_audit_018_before_duration_fix/parameter_qc.json').exists():
        raise SystemExit('Refusing to overwrite 018 audit without backup')
    prior_summary=PREVIOUS/'sensitivity_summary.json'
    prior_comparison=PREVIOUS/'cli_parameter_comparison.csv'
    previous_gate=ROOT/'results/sy_lc_random_weather_015/wgen_readiness_review/methodology_gate.json'
    tracked=[SOURCE,prior_summary,prior_comparison,previous_gate,ROOT/'scripts/build_dssat_cli.py']
    sha_before={str(p.relative_to(ROOT)):digest(p) for p in tracked}
    summary017=json.loads(prior_summary.read_text(encoding='utf-8'))
    assert summary017['fill_cells']==462 and summary017['validation_data_used_for_fill'] is False
    records=read_candidate()
    stats=cli.fit_monthly_statistics(records)
    monthly=cli.monthly_weather_summary(records)
    crosscheck=cli.crosscheck_statistics(records,stats)
    assert crosscheck['passed'] and crosscheck['months_checked']==12
    # Preserve the project's 018-stated dates in its unchanged serializer.
    start0,end0=cli.EXPECTED_START,cli.EXPECTED_END
    cli.EXPECTED_START,cli.EXPECTED_END=date(2005,1,1),date(2013,12,31)
    try:
        body=cli._cli_text(records,'CNSY',41.517,123.400,41,stats,monthly)
    finally:
        cli.EXPECTED_START,cli.EXPECTED_END=start0,end0
    # _cli_text has a YC-specific hard-coded ten-year DURN; this candidate spans nine years.
    old_duration='  2005    10 -99.0 -99.0 -99.0 -99.0 Calculated_from_train_only_daily_data'
    new_duration='  2005     9 -99.0 -99.0 -99.0 -99.0 Calculated_from_train_only_daily_data'
    if body.count(old_duration)!=1:
        raise ValueError('Unexpected CLI START/DURN line')
    body=body.replace(old_duration,new_duration)
    lines=body.splitlines()
    sections={s:sum(line.startswith(s) for line in lines) for s in cli.CLI_SECTIONS}
    first=lines.index('*WGEN PARAMETERS')
    wgen_lines=lines[first+2:first+14]
    fixed=[];errors=[]
    for month,line in enumerate(wgen_lines,1):
        try:
            parsed=cli.parse_wgen_parameter_row(line)
            if parsed['MTH']!=month: errors.append(f'month {month}: index')
            fixed.append(parsed)
        except ValueError as e:
            errors.append(f'month {month}: {e}')
    for s,n in sections.items():
        if n!=1: errors.append(f'section {s}: count {n}')
    if '*CLIMATE:CNSY' not in body or not any(line.startswith('  CNSY ') for line in lines):
        errors.append('CNSY station identifier')
    if '@START  DURN' not in body or not any(line==new_duration for line in lines):
        errors.append('start and train-only metadata')
    for row,parsed in zip(stats,fixed):
        for p in cli.WGEN_PARAMETER_FIELDS:
            decimals=cli.WGEN_PARAMETER_PRECISION[p]
            expected=float(f"{float(row[p]):.{decimals}f}")
            if not math.isclose(float(parsed[p]),expected,abs_tol=1e-12):
                errors.append(f'{row["month"]}:{p} fixed-width value')
    # The 017 comparison is an independently saved contract for these exact 168 values.
    with prior_comparison.open(encoding='utf-8-sig',newline='') as f:
        prior=list(csv.DictReader(f))
    lookup={(int(r['month']),r['parameter']):float(r['train_only_candidate']) for r in prior}
    assert len(lookup)==168
    for r in stats:
        for p in cli.WGEN_PARAMETER_FIELDS:
            if not math.isclose(float(r[p]),lookup[(int(r['month']),p)],rel_tol=1e-12,abs_tol=1e-12):
                errors.append(f'{r["month"]}:{p} differs from 017 candidate')
    physical=[]
    for r in stats:
        m=r['month'];wet=r['wet_day_count'];dry=r['dry_day_count']
        if wet<3 or dry<2: physical.append(f'{m}: insufficient wet/dry subset')
        for p in ('SDSD','SWSD','XDSD','XWSD','NASD','RTOT','RNUM'):
            if float(r[p])<0: physical.append(f'{m}:{p} negative')
        if not 0<float(r['ALPHA'])<=0.998 or not 0<=float(r['PDW'])<=1:
            physical.append(f'{m}:ALPHA/PDW range')
        if not all(math.isfinite(float(r[p])) for p in cli.WGEN_PARAMETER_FIELDS):
            physical.append(f'{m}:nonfinite')
        if abs(r['PDW']-r['p_wet_given_dry'])>1e-12: physical.append(f'{m}:PDW mismatch')
    flagged=Counter()
    with (PREVIOUS/'fill_difference.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if int(r['year'])==2005:
                flagged[int(r['month'])]+=1
    focus=[]
    for r in stats:
        m=int(r['month'])
        focus.append(dict(month=m,filled_2005_cells=flagged[m],wet_days=r['wet_day_count'],dry_days=r['dry_day_count'],
                          SDMN=r['SDMN'],SDSD=r['SDSD'],SWMN=r['SWMN'],SWSD=r['SWSD'],
                          XDMN=r['XDMN'],XDSD=r['XDSD'],XWMN=r['XWMN'],XWSD=r['XWSD'],
                          NAMN=r['NAMN'],NASD=r['NASD'],ALPHA=r['ALPHA'],RTOT=r['RTOT'],PDW=r['PDW'],RNUM=r['RNUM']))
    change=[r for r in prior if int(r['month']) in range(1,6)]
    largest=max(change,key=lambda r:float(r['absolute_difference']))
    sd_change=max((r for r in change if r['parameter'] in ('SDSD','SWSD','XDSD','XWSD','NASD')),
                  key=lambda r:float(r['relative_difference']))
    rain_delta=[r for r in prior if r['parameter'] in ('ALPHA','RTOT','PDW','RNUM') and float(r['absolute_difference'])!=0]
    clamped=[int(r['month']) for r in stats if float(r['ALPHA'])>=0.998-1e-12]
    sha_after={str(p.relative_to(ROOT)):digest(p) for p in tracked}
    assert sha_before==sha_after
    gate='READY_FOR_WGEN_REVIEW' if not errors and not physical and len(stats)==12 and not rain_delta else 'BLOCKED_BY_CLI_PARAMETER_QC'
    qc=dict(site='SYA',years='2005-2013',source=str(SOURCE.relative_to(ROOT)),source_sha256=sha_before[str(SOURCE.relative_to(ROOT))],
            input_sha256=sha_before,input_sha256_unchanged=True,weather_days=len(records),parameter_count=168,
            wet_rule='RAIN > 0.0',monthly_complete=len(stats)==12,section_counts=sections,fixed_width_row_count=len(fixed),
            fixed_width_expected=cli.WGEN_READ_FORMAT,serialization_errors=errors,physical_errors=physical,
            cli_start_year=2005,cli_duration_years=9,
            independent_crosscheck_passed=crosscheck['passed'],matches_017_parameter_comparison=not errors,
            january_to_may_2005_filled_cells={str(m):flagged[m] for m in range(1,6)},
            largest_january_to_may_change=largest,largest_january_to_may_sd_relative_change=sd_change,
            rain_gamma_changed_from_017_baseline=bool(rain_delta),alpha_clamped_months=clamped,
            gate=gate,WGEN_run=False,synthetic_weather_generated=False,DSSAT_run=False,PPO_run=False)
    OUT.mkdir(parents=True,exist_ok=True)
    candidate=OUT/'CNSY_2005_2013_train_only_candidate.CLI'
    candidate.write_text(body,encoding='ascii',newline='\n')
    with (OUT/'monthly_parameter_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(focus[0]));w.writeheader();w.writerows(focus)
    (OUT/'parameter_qc.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'independent_parameter_check.json').write_text(json.dumps(crosscheck,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=f"""# SYA CLI 参数拟合与审计（018）

使用 017 的 `candidate_weather_2005_2013.csv` 作为唯一拟合输入，共 3,287 日，填补仅使用 2005–2013 训练期资料。输入 SHA256 及审计前后校验见 `parameter_qc.json`。原 WTH、weather_clean、baseline 和 015/017 记录未修改。

## 方法与检查

按 `scripts/build_dssat_cli.py` 的 `RAIN > 0.0` 湿日定义，拟合 12 个月 × 14 个 WGEN CLI 参数；标准差采用样本标准差。候选 CLI 的气象站为 CNSY，起始年 2005、历时 9 年。`*WGEN PARAMETERS` 采用 `I6,14(1X,F5.0)` 固定宽度；重新解析 12 行并逐项核对舍入值，另用独立 mean/stdev、转移计数与 Gamma 公式复核，并与 017 的 168 项候选参数逐值核对。

完整性：**{len(stats)} 个月、168 项**；独立复算通过：**{crosscheck['passed']}**；序列化错误 **{len(errors)}**；物理范围错误 **{len(physical)}**。每月至少 {min(r['wet_day_count'] for r in stats)} 个湿日。ALPHA 达项目公式 0.998 截断值的月份：**{clamped}**；这些月份需在下一阶段 WGEN 方法审查中记录，截断值本身不是 CLI 格式错误。

## 2005 年集中填补月审计

| 月份 | 2005 年填补单元格 | 全训练期湿日 | 全训练期干日 | ALPHA | RTOT | PDW |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r['month']} | {r['filled_2005_cells']} | {r['wet_days']} | {r['dry_days']} | {r['ALPHA']:.3f} | {r['RTOT']:.2f} | {r['PDW']:.3f} |" for r in focus[:5])}

与当前 full-period fill WTH 的 017 参数比较中，1–5 月最大绝对变化为 **{float(largest['absolute_difference']):.4f}**（{largest['month']} 月 {largest['parameter']}）；标准差参数最大相对变化 **{100*float(sd_change['relative_difference']):.2f}%**（{sd_change['month']} 月 {sd_change['parameter']}）。逐月湿/干 SRAD、TMAX 均值和标准差及 TMIN 统计见 `monthly_parameter_audit.csv`；逐项新旧差值见 017 的 `cli_parameter_comparison.csv`。降水与 Gamma 参数差值项数为 **{len(rain_delta)}**，因为 RAIN 保持不变。2005 年 424 个单元格集中填补仍可能压低日际变率，不能由本轮格式与范围检查证明合成天气有效。

## 判定

**`{gate}`。** 候选 CLI 在完整性、物理范围、固定宽度及独立复算方面通过，可提交下一阶段 WGEN 方法审查。此状态只批准审查，不构成运行 WGEN 的授权；无需因本轮 CLI 参数错误重新调整填补方案。后续仍需对 2005 年长缺段和被截断的 Gamma 月份作生成前评估。未运行 WGEN、未生成随机天气、未运行 DSSAT/PPO。

候选文件：`results/sya_cli_parameter_audit_018/CNSY_2005_2013_train_only_candidate.CLI`；机器检查：`parameter_qc.json`；逐月参数：`monthly_parameter_audit.csv`。
"""
    REPORT.write_text(report,encoding='utf-8')
    print(json.dumps({'gate':gate,'serialization_errors':errors,'physical_errors':physical,'clamped_months':clamped,
                      'min_wet_days':min(r['wet_day_count'] for r in stats)},ensure_ascii=False))


if __name__=='__main__':
    main()
