"""SYA train-only monthly fill sensitivity; no simulations or original edits."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sya_train_only_fill_sensitivity'
REPORT = ROOT / 'docs/sya_train_only_fill_sensitivity_review.md'
AUDIT = ROOT / 'results/sy_lc_random_weather_015'
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from audit_lca_cli_parameters_016 import read_wth
import build_dssat_cli as cli


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists() or REPORT.exists():
        raise SystemExit('Refusing to overwrite previous 017 review')
    source = AUDIT/'nonrain_provenance_resolution_484.csv'
    gaps_path = AUDIT/'weather_gap_details.csv'
    cleaned_path = ROOT/'weather_clean/SYA_weather_cleaned.csv'
    paths=[ROOT/'DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY'/f'CNSY{y%100:02d}01.WTH' for y in range(2005,2014)]
    protected=paths+[source,gaps_path,cleaned_path,ROOT/'scripts/build_dssat_cli.py',AUDIT/'wgen_readiness_review/methodology_gate.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    baseline=[d for y,p in zip(range(2005,2014),paths) for d in read_wth(p,y)]
    assert len(baseline)==3287
    flags=[r for r in read_csv(source) if r['site']=='SYA']
    selected={(r['date'],r['variable']) for r in flags}
    assert len(flags)==len(selected)==462
    gaps=read_csv(gaps_path)
    missing={v:{r['date'] for r in gaps if r['site']=='SYA' and r['variable']==v and r['kind']=='RAW_MISSING_VALUE'} for v in ('SRAD','TMAX','TMIN')}
    assert selected=={(d,v) for v in missing for d in missing[v] if '2005-01-01'<=d<='2013-12-31'}
    # Only training rows are eligible donors; original missing positions are excluded per variable.
    train=[r for r in read_csv(cleaned_path) if '2005-01-01'<=r['date']<='2013-12-31']
    assert len(train)==3287
    clean_by_day={r['date']:r for r in train}
    pools={}
    means={}
    observed_mismatches=[]
    for v in missing:
        for month in range(1,13):
            donors=[r for r in train if int(r['month'])==month and r['date'] not in missing[v]]
            values=[float(r[v]) for r in donors]
            assert values and all(math.isfinite(x) for x in values)
            pools[(v,month)]=len(values)
            means[(v,month)]=statistics.mean(values)
    for d in baseline:
        for v in missing:
            if d.day.isoformat() not in missing[v] and abs(getattr(d,v.lower())-round(float(clean_by_day[d.day.isoformat()][v]),1))>1e-8:
                observed_mismatches.append({'date':d.day.isoformat(),'variable':v})
    # Observed WTH deviations are retained; means come from pre-WTH precision cleaned observed values.
    candidate=[]; unrounded=[]; differences=[]
    flag_lookup={ (r['date'],r['variable']):r for r in flags }
    for d in baseline:
        vals={}; full={}
        for v in missing:
            key=(d.day.isoformat(),v)
            if key not in selected: continue
            value=means[(v,d.day.month)]
            rounded=round(value,1)
            vals[v.lower()]=rounded;full[v.lower()]=value
            old=getattr(d,v.lower()); delta=rounded-old
            historical=float(flag_lookup[key]['historical_month_mean_unrounded'])
            differences.append(dict(date=key[0],year=d.day.year,month=d.day.month,variable=v,
                wet=d.rain>0,donor_count=pools[(v,d.day.month)],historical_mean_unrounded=historical,
                current_wth_value=old,train_only_mean_unrounded=value,candidate_value_0p1=rounded,
                signed_difference=delta,absolute_difference=abs(delta),
                relative_difference=abs(delta)/abs(old) if old else None))
        candidate.append(replace(d,**vals) if vals else d)
        unrounded.append(replace(d,**full) if full else d)
    violations=[dict(date=d.day.isoformat(),SRAD=d.srad,TMAX=d.tmax,TMIN=d.tmin,RAIN=d.rain)
                for d in candidate if d.srad<0 or d.rain<0 or d.tmax<d.tmin or not all(math.isfinite(x) for x in (d.srad,d.tmax,d.tmin,d.rain))]
    assert all(a.rain==b.rain for a,b in zip(baseline,candidate))
    assert all(getattr(a,v.lower())==getattr(b,v.lower()) for a,b in zip(baseline,candidate)
               for v in missing if (a.day.isoformat(),v) not in selected)
    fitted=[cli.fit_monthly_statistics(x) for x in (baseline,candidate,unrounded)]
    checks=[cli.crosscheck_statistics(x,f) for x,f in zip((baseline,candidate,unrounded),fitted)]
    assert all(c['passed'] for c in checks)
    comparison=[]
    for a,b,u in zip(*fitted):
        for p in cli.WGEN_PARAMETER_FIELDS:
            old=float(a[p]);new=float(b[p]);delta=new-old
            comparison.append(dict(month=a['month'],parameter=p,current_wth=old,train_only_candidate=new,
                candidate_unrounded=float(u[p]),absolute_difference=abs(delta),signed_difference=delta,
                relative_difference=abs(delta)/abs(old) if old else None,
                cli_rounded_current=f"{old:.{cli.WGEN_PARAMETER_PRECISION[p]}f}",
                cli_rounded_candidate=f"{new:.{cli.WGEN_PARAMETER_PRECISION[p]}f}"))
    assert len(comparison)==168
    conditional=[]
    for month in range(1,13):
        for v in missing:
            for group in ('all','wet','dry'):
                indices=[i for i,d in enumerate(baseline) if d.day.month==month and
                         (group=='all' or (d.rain>0)==(group=='wet'))]
                a=[getattr(baseline[i],v.lower()) for i in indices]
                b=[getattr(candidate[i],v.lower()) for i in indices]
                m0,m1=statistics.mean(a),statistics.mean(b)
                s0,s1=statistics.stdev(a),statistics.stdev(b)
                conditional.append(dict(month=month,variable=v,group=group,days=len(indices),
                    replaced_cells=sum((baseline[i].day.isoformat(),v) in selected for i in indices),
                    current_mean=m0,candidate_mean=m1,mean_absolute_change=abs(m1-m0),
                    current_sample_sd=s0,candidate_sample_sd=s1,sd_absolute_change=abs(s1-s0),
                    sd_relative_change=abs(s1-s0)/s0 if s0 else None))
    by_var={v:{'cells':sum(r['variable']==v for r in differences),
               'max_absolute':max((r for r in differences if r['variable']==v),key=lambda r:r['absolute_difference']),
               'max_relative':max((r for r in differences if r['variable']==v and r['relative_difference'] is not None),key=lambda r:r['relative_difference'])}
            for v in missing}
    weather_params=[r for r in comparison if r['parameter'] not in ('ALPHA','RTOT','PDW','RNUM')]
    max_cli=max(weather_params,key=lambda r:r['absolute_difference'])
    max_sd=max((r for r in comparison if r['parameter'] in ('SDSD','SWSD','XDSD','XWSD','NASD')),key=lambda r:r['relative_difference'])
    rain_unchanged=all(r['absolute_difference']==0 for r in comparison if r['parameter'] in ('ALPHA','RTOT','PDW','RNUM'))
    annual=Counter(r['year'] for r in differences)
    summary={'site':'SYA','train_years':'2005-2013','validation_years':'2014-2023',
        'validation_data_used_for_fill':False,'donor_method':'train-only cleaned observed rows; original missing positions excluded separately per variable',
        'candidate_precision':'0.1; unrounded sensitivity also computed','days':3287,'fill_cells':462,
        'annual_fill_cells':dict(annual),'fill_difference_by_variable':by_var,
        'candidate_qc_violations':violations,'observed_wth_cleaned_rounding_mismatches':observed_mismatches,
        'cli_parameters_compared':168,'max_cli_absolute_difference':max_cli,'max_cli_sd_relative_difference':max_sd,
        'rain_parameters_identical':rain_unchanged,'independent_parameter_checks_passed':True,
        'decision':'C' if violations else 'A',
        'gate':'BLOCKED_BY_CANDIDATE_QC' if violations else 'CONDITIONAL_READY_FOR_SYA_CLI_PARAMETER_AUDIT',
        'decision_scope':'A/B sensitivity only; not proof that monthly-mean imputation preserves observed variability',
        'wth_modified':False,'WGEN_run':False,'synthetic_weather_generated':False,'DSSAT_run':False,'PPO_run':False}
    after={str(p.relative_to(ROOT)):sha(p) for p in protected}
    assert before==after
    summary['input_sha256']=before;summary['input_sha256_unchanged']=True
    OUT.mkdir(parents=True)
    write_csv('fill_difference.csv',differences)
    write_csv('cli_parameter_comparison.csv',comparison)
    write_csv('conditional_statistics_comparison.csv',conditional)
    write_csv('candidate_weather_2005_2013.csv',[dict(DATE=d.day.isoformat(),SRAD=d.srad,TMAX=d.tmax,TMIN=d.tmin,RAIN=d.rain) for d in candidate])
    write_csv('monthly_fill_donor_summary.csv',[dict(variable=v,month=m,donor_count=pools[(v,m)],train_only_mean=means[(v,m)]) for v,m in sorted(means)])
    (OUT/'sensitivity_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'independent_parameter_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    table='\n'.join(f"| {v} | {by_var[v]['cells']} | {by_var[v]['max_absolute']['absolute_difference']:.3f} | {100*by_var[v]['max_relative']['relative_difference']:.2f}% |" for v in missing)
    focus=[]
    for m in range(1,6):
        cells=sum(r['year']==2005 and r['month']==m for r in differences)
        hits=[r for r in weather_params if r['month']==m]
        largest=max(hits,key=lambda r:r['absolute_difference'])
        focus.append(f"| {m} | {cells} | {largest['parameter']} | {largest['absolute_difference']:.4f} |")
    report=f"""# SYA train-only fill 敏感性审查（017）

日期：2026-10-06。固定 2005–2013 训练期，2014–2023 验证数据未参与候选填补。只生成独立候选 CSV 和参数对比，不生成 CLI 或随机天气、不运行 WGEN/DSSAT/PPO。原 WTH、weather_clean、配置及已有审计均未覆盖，输入 SHA256 前后一致。

## 方法与原问题

现有正式 WTH 的 462 项非降水缺口曾采用包含验证期的站点×月份均值填补。按 015 provenance 表固定逐日变量位置，SRAD/TMAX/TMIN 分别为 160/151/151 项；原缺口集合与原始缺测审计逐项一致。2005 年有 {annual[2005]} 项，占 91.8%，历史最长连续受影响段 139 天。

只读取 cleaned CSV 中 2005–2013 的行作为均值供体，逐变量排除全部原始缺测位置，计算训练期同月非缺测均值。cleaned 数据用于恢复观测值在 WTH 舍入前的精度，不使用其中任何缺测填补值或验证期行。候选仅替换这 462 项，其他观测 WTH 值、RAIN 和日期保持原样；替换值按 WTH 一位小数精度冻结，并另算未舍入版本的参数以显示精度影响。观测 WTH 与 cleaned 一位小数对照差异 {len(observed_mismatches)} 项，原值均保留（详见 JSON）。

湿日固定 RAIN>0，使用 `scripts/build_dssat_cli.py:fit_monthly_statistics` 计算两套 14×12 参数；标准差为样本标准差，RTOT/RNUM 为同月跨训练年的平均总量/湿日数。独立 mean/stdev、转移计数、降水及 Gamma 复算均通过。

## 填补值差异

| 变量 | 替换项 | 最大绝对变化（SRAD: MJ/m²/day；温度: ℃） | 最大相对变化 |
|---|---:|---:|---:|
{table}

相对变化用 abs(B−A)/abs(A)；温度近 0℃时百分比会放大，仅作为补充，不单独判定风险。逐项未舍入均值、供体数、候选值和差异见 `fill_difference.csv`。

## CLI 影响与集中填补月份

168 项比较完整，最大参数绝对变化为 **{max_cli['absolute_difference']:.4f}**（{max_cli['month']} 月 {max_cli['parameter']}）；CLI 中标准差参数最大相对变化 **{100*max_sd['relative_difference']:.2f}%**（{max_sd['month']} 月 {max_sd['parameter']}）。湿/干及全月条件均值/标准差完整保存于 `conditional_statistics_comparison.csv`。ALPHA、RTOT、PDW、RNUM 在全部月份完全相同，因为 RAIN 未改变。

| 2005 年月份 | 当月替换项 | 当月最大绝对变化的 CLI 参数 | 参数变化 |
|---|---:|---|---:|
{chr(10).join(focus)}

候选四变量 QC 违规 **{len(violations)} 项**。是否出现温度倒置、非有限值或负辐射/降水均已检查，没有对异常值自动修正。

## 最终建议

**判定 {summary['decision']}：{('候选 QC 未通过，不适合进入 WGEN。' if violations else '与现有结果基本一致，可作为 WGEN 拟合输入候选；进入独立 SYA CLI 参数审计阶段。')}** 本次没有套用未经说明的 DSSAT 官方阈值：该判断基于完整 QC、绝对均值变化、相对标准差变化、降水参数不变和参数独立复算的共同证据。机器 gate 为 `{summary['gate']}`，仅针对本候选；015 历史 gate 保留不改。

train-only 方案消除了已知 462 项填补对验证期均值的依赖，但均值填补本身仍压低缺段的日际变率，尤其 2005 年长连续缺段。因此 A 只说明两种填补窗口的参数结果接近，不代表实测天气已恢复或合成天气已验证。正式生成前应对候选开展 CLI 参数/序列化及集中填补条件方差审计；本任务完成后停止。

输出目录：`results/sya_train_only_fill_sensitivity/`。`candidate_weather_2005_2013.csv` 是独立候选数据，不是随机天气，也未安装到任何 DSSAT 输入目录。
"""
    REPORT.write_text(report,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('decision','gate','fill_cells','max_cli_absolute_difference','max_cli_sd_relative_difference','rain_parameters_identical','candidate_qc_violations')},ensure_ascii=False))


if __name__=='__main__':
    main()
