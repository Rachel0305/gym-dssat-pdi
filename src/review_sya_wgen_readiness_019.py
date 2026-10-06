"""Read-only SYA candidate weather statistics and WGEN readiness review."""
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

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/sya_wgen_readiness_review_019'
REPORT=ROOT/'docs/sya_wgen_readiness_review_019.md'
CAND=ROOT/'results/sya_train_only_fill_sensitivity/candidate_weather_2005_2013.csv'
FILL=ROOT/'results/sya_train_only_fill_sensitivity/fill_difference.csv'
CLI=ROOT/'results/sya_cli_parameter_audit_018/CNSY_2005_2013_train_only_candidate.CLI'
QC=ROOT/'results/sya_cli_parameter_audit_018/parameter_qc.json'
WTH=ROOT/'DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY'
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
import build_dssat_cli as cli
from audit_lca_cli_parameters_016 import read_wth


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sd(values): return statistics.stdev(values) if len(values)>1 else None


def main():
    if (OUT.exists() or REPORT.exists()) and not (ROOT/'backups/sya_wgen_readiness_019_before_diagnostic_clarification/wgen_readiness_gate.json').exists():
        raise SystemExit('Refusing to overwrite 019 review without backup')
    paths=[WTH/f'CNSY{y%100:02d}01.WTH' for y in range(2005,2014)]
    tracked=[CAND,FILL,CLI,QC]+paths
    before={str(p.relative_to(ROOT)):sha(p) for p in tracked}
    q=json.loads(QC.read_text(encoding='utf-8'))
    assert q['gate']=='READY_FOR_WGEN_REVIEW' and q['source_sha256']==before[str(CAND.relative_to(ROOT))]
    source=read_csv(CAND)
    assert len(source)==3287
    days=[]
    for i,r in enumerate(source):
        d=date(2005,1,1)+timedelta(days=i)
        assert r['DATE']==d.isoformat()
        vals=[float(r[v]) for v in ('SRAD','TMAX','TMIN','RAIN')]
        assert all(math.isfinite(v) for v in vals) and vals[0]>=0 and vals[3]>=0 and vals[1]>=vals[2]
        days.append(cli.WeatherDay(d,*vals))
    original=[d for y,p in zip(range(2005,2014),paths) for d in read_wth(p,y)]
    flags=read_csv(FILL);assert len(flags)==462
    keyed={(r['date'],r['variable']) for r in flags};assert len(keyed)==462
    by_day={d.day.isoformat():d for d in days}
    for a,b in zip(days,original):
        assert a.day==b.day and a.rain==b.rain
        for v in ('SRAD','TMAX','TMIN'):
            if (a.day.isoformat(),v) not in keyed:
                assert getattr(a,v.lower())==getattr(b,v.lower())
    variability=[];impact=[]
    for month in range(1,13):
        for v in ('SRAD','TMAX','TMIN'):
            for period,selector in (('2005',lambda d:d.day.year==2005),('2006-2013',lambda d:d.day.year>=2006),('2005-2013',lambda d:True)):
                selected=[d for d in days if d.day.month==month and selector(d)]
                values=[getattr(d,v.lower()) for d in selected]
                filled=sum((d.day.isoformat(),v) in keyed for d in selected)
                variability.append(dict(month=month,variable=v,period=period,days=len(selected),filled_cells=filled,
                    filled_fraction=filled/len(selected),mean=statistics.mean(values),sample_sd=sd(values),
                    minimum=min(values),maximum=max(values),distinct_values=len(set(values))))
            current=[getattr(d,v.lower()) for d in days if d.day.month==month]
            old=[getattr(d,v.lower()) for d in original if d.day.month==month]
            rest=[getattr(d,v.lower()) for d in days if d.day.month==month and d.day.year>=2006]
            subset=[d for d in days if d.day.month==month and d.day.year==2005]
            runs=[];run=0
            for d in subset:
                run=run+1 if (d.day.isoformat(),v) in keyed else 0
                runs.append(run)
            impact.append(dict(month=month,variable=v,training_days=len(current),filled_cells=sum((d.day.isoformat(),v) in keyed for d in days if d.day.month==month),
                filled_2005_cells=sum((d.day.isoformat(),v) in keyed for d in subset),
                longest_consecutive_2005_fill_days=max(runs),
                current_wth_sample_sd=sd(old),train_only_sample_sd=sd(current),
                sd_change_train_minus_current=sd(current)-sd(old),
                sd_relative_change_vs_current=(sd(current)-sd(old))/sd(old) if sd(old) else None,
                year_2005_sample_sd=sd([getattr(d,v.lower()) for d in subset]),
                rest_2006_2013_sample_sd=sd(rest),
                rest_minus_full_sample_sd=sd(rest)-sd(current)))
    # Longest consecutive calendar days with at least one filled cell (not sum of variable runs).
    filled_dates={r['date'] for r in flags};longest=0;run=0
    for d in days:
        run=run+1 if d.day.isoformat() in filled_dates else 0
        longest=max(longest,run)
    fit=cli.fit_monthly_statistics(days)
    fit8=cli.fit_monthly_statistics([d for d in days if d.day.year>=2006])
    assert cli.crosscheck_statistics(days,fit)['passed']
    assert len(fit)==12 and all(len([r[p] for p in cli.WGEN_PARAMETER_FIELDS])==14 for r in fit)
    check=[]
    for a,b in zip(fit,fit8):
        for p in cli.WGEN_PARAMETER_FIELDS:
            check.append(dict(month=a['month'],parameter=p,all_2005_2013=a[p],exclude_2005_2006_2013=b[p],
                absolute_difference=abs(float(b[p])-float(a[p])),
                relative_difference=abs(float(b[p])-float(a[p]))/abs(float(a[p])) if float(a[p]) else None))
    clamped=[int(r['month']) for r in fit if float(r['ALPHA'])>=0.998-1e-12]
    sections={s:sum(line.startswith(s) for line in CLI.read_text(encoding='ascii').splitlines()) for s in cli.CLI_SECTIONS}
    assert all(n==1 for n in sections.values())
    text=CLI.read_text(encoding='ascii')
    head=text.splitlines().index('*WGEN PARAMETERS')
    parsed=[cli.parse_wgen_parameter_row(line) for line in text.splitlines()[head+2:head+14]]
    assert [r['MTH'] for r in parsed]==list(range(1,13))
    for row,p in zip(fit,parsed):
        for name in cli.WGEN_PARAMETER_FIELDS:
            assert p[name]==float(f"{float(row[name]):.{cli.WGEN_PARAMETER_PRECISION[name]}f}")
    # Identical monthly fill on every date makes 2005 Jan-Apr zero-variance for all three variables.
    zero=[r for r in variability if r['period']=='2005' and r['sample_sd']==0]
    first4=[r for r in zero if r['month'] in (1,2,3,4)]
    assert len(first4)==12
    risk='BLOCKED_BY_WEATHER_STATISTICS' if len(first4)==12 and longest>=120 else 'READY_WITH_DOCUMENTED_LIMITATION' if zero or flags else 'READY_FOR_WGEN'
    after={str(p.relative_to(ROOT)):sha(p) for p in tracked};assert before==after
    gate=dict(site='SYA',training_years='2005-2013',decision=risk,
        input_sha256=before,input_sha256_unchanged=True,weather_days=len(days),filled_cells=len(flags),
        filled_2005_cells=sum(int(r['year'])==2005 for r in flags),longest_consecutive_fill_days=longest,
        zero_sd_2005_month_variable_count=len(zero),zero_sd_2005_january_april_count=len(first4),
        alpha_clamped_months=clamped,cli_sections=sections,cli_wgen_parameter_rows=len(parsed),
        cli_parameter_count=168,min_monthly_wet_days=min(r['wet_day_count'] for r in fit),
        blocker='2005 January-April SRAD/TMAX/TMIN are constant for complete months due to mean fill; 139-day affected run suppresses within-year daily variability' if risk=='BLOCKED_BY_WEATHER_STATISTICS' else None,
        exclusion_of_2005_is_diagnostic_only=True,WGEN_run=False,synthetic_weather_generated=False,DSSAT_run=False,PPO_run=False)
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/'weather_variability_summary.csv',variability)
    write_csv(OUT/'fill_impact_summary.csv',impact)
    write_csv(OUT/'exclude_2005_parameter_diagnostic.csv',check)
    (OUT/'wgen_readiness_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    focus=[r for r in impact if r['month']<=5]
    table='\n'.join(f"| {m} | {sum(r['filled_2005_cells'] for r in focus if r['month']==m)} | "+' | '.join(f"{next(r for r in variability if r['period']=='2005' and r['month']==m and r['variable']==v)['sample_sd']:.2f} / {next(r for r in variability if r['period']=='2006-2013' and r['month']==m and r['variable']==v)['sample_sd']:.2f}" for v in ('SRAD','TMAX','TMIN'))+' |' for m in range(1,6))
    max_full=max(impact,key=lambda r:abs(r['sd_relative_change_vs_current']))
    max_exclusion=max((r for r in check if r['parameter'] in ('SDSD','SWSD','XDSD','XWSD','NASD')),key=lambda r:r['relative_difference'])
    max_exclusion_focus=max((r for r in check if int(r['month'])<=5 and r['parameter'] in ('SDSD','SWSD','XDSD','XWSD','NASD')),key=lambda r:r['relative_difference'])
    report=f"""# SYA WGEN readiness review（019）

**结论：`{risk}`。** 018 CLI 在格式、完整性和基本物理范围上通过，但 2005 年 1–4 月的 SRAD、TMAX、TMIN 每月均为常数，12 个月×变量组合的样本标准差均为 0。连续受填补影响的最长时段为 {longest} 天（至少一个变量）；2005 年共有 424/462 个填补单元格。完整四个月的日际变化被均值填补消除，已构成 WGEN 气候变率拟合的实质风险，不能仅以九年合并参数有值而放行。此判定使用任务给定的 `BLOCKED_BY_WEATHER_STATISTICS` 类别，不另设 DSSAT 官方数值阈值。

## 输入和统计检查

唯一候选输入是 017 的 `candidate_weather_2005_2013.csv`，2005–2013 共 3,287 个连续日；验证期 2014–2023 未纳入。四变量均为有限值，无负 SRAD/RAIN 或 TMAX<TMIN。训练期整体、2005 年、2006–2013 的逐月均值、样本标准差和极值见 `weather_variability_summary.csv`。输入及 018 CLI 的 SHA256 前后保持一致，原 WTH 与既有 015/017/018 审计未修改。

表内为 2005 年样本 SD / 2006–2013 同月样本 SD，单位 SRAD 为 MJ/m²/day，温度为 ℃：

| 月份 | 2005 填补单元格 | SRAD | TMAX | TMIN |
|---:|---:|---:|---:|---:|
{table}

1–4 月的三个变量各只有一个不同值。这是直接可见的异常平滑；5 月仍有部分观测，变率没有完全归零。填补前后是“当前全时期均值填补 WTH”与“训练期均值填补候选”的对比，不是原始缺测位置的真实观测前后对比。九年合并月度 SD 的两方案最大相对差异为 {100*abs(max_full['sd_relative_change_vs_current']):.2f}%（{max_full['month']} 月 {max_full['variable']}）；两种均值填补都无法恢复 2005 缺段的日际变率。每月填补数量、变量连续长度和 SD 差异见 `fill_impact_summary.csv`。

## CLI 与降水参数

重新核对 018 CLI 的 12×14 参数与固定宽度序列化；每月至少 {gate['min_monthly_wet_days']} 个湿日。ALPHA 达 0.998 截断的月份为 {clamped}。降水未被 017 候选填补改变，Gamma、RTOT、PDW、RNUM 的新旧差异为 0；这些参数不是本轮阻塞主因。湿/干条件 SRAD/TMAX 和 TMIN 的拟合参数可计算，但 2005 年连续常数段对条件变率有影响。仅作诊断地剔除 2005 年后，**1–5 月** CLI 标准差参数最大相对变化为 {100*max_exclusion_focus['relative_difference']:.2f}%（{max_exclusion_focus['month']} 月 {max_exclusion_focus['parameter']}）。全 12 个月最大变化为 {100*max_exclusion['relative_difference']:.2f}%（{max_exclusion['month']} 月 {max_exclusion['parameter']}），该月不属于集中填补月份，不能归因于填补。逐项见 `exclude_2005_parameter_diagnostic.csv`。此诊断不选择或替换拟合窗口。

## 后续要求

本候选不得直接进入 WGEN。应先处理 2005 年长缺段对月内及湿/干条件变率的影响，并对新的候选重新做完整 CLI 和 readiness 审查；具体方案需有独立记录。未运行 WGEN、未生成随机天气、未运行 DSSAT/PPO，正式天气目录与原 WTH 未改。
"""
    REPORT.write_text(report,encoding='utf-8')
    print(json.dumps({'gate':risk,'longest':longest,'zero_month_variables':len(first4),'max_focus_8yr_sd_parameter_relative_change':max_exclusion_focus['relative_difference']},ensure_ascii=False))


if __name__=='__main__':main()
