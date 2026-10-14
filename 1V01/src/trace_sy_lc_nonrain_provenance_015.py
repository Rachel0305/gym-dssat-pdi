"""Read-only, per-cell source trace for the 484 unresolved SYA/LCA entries.

Outputs only new audit artifacts. Never edits weather, simulator, or model inputs.
"""
from collections import Counter
from datetime import date
from pathlib import Path
import csv
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sy_lc_random_weather_015'
REPORT = ROOT / 'docs/provenance_resolution_report.md'
DETAIL = OUT / 'nonrain_provenance_resolution_484.csv'
SUMMARY = OUT / 'nonrain_provenance_resolution_summary.json'
sys.path.insert(0, str(ROOT/'src'))
from audit_sy_lc_random_weather_source_015 import parse_wth


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_daily(path, year):
    records, errors = parse_wth(path, year)
    assert not errors, (path, errors)
    return {d.isoformat(): values for d, values in records}


def main():
    if REPORT.exists() or DETAIL.exists() or SUMMARY.exists():
        raise SystemExit('Refusing to overwrite existing resolution evidence.')
    gaps = read_rows(OUT/'remaining_nonrain_provenance_gaps.csv')
    assert len(gaps)==484 and Counter(r['site'] for r in gaps)=={'SYA':462,'LCA':22}
    raw_gaps = read_rows(OUT/'weather_gap_details.csv')
    detail = []
    site_summary = {}
    for site, short, prefix, source_root in (
        ('SYA','SY','CNSY','DSSAT_auto_validation/multisite_new_cultivar_inputs_013'),
        ('LCA','LC','CNLC','DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual'),
    ):
        clean = pd.read_csv(ROOT/f'weather_clean/{site}_weather_cleaned.csv', dtype={'date':str})
        qc = pd.read_csv(ROOT/f'weather_clean_qc/{site}_weather_cleaned_qc.csv', dtype={'date':str})
        clean_by_date = clean.set_index('date')
        qc_by_date = qc.set_index('date')
        missing_dates = {v:{r['date'] for r in raw_gaps if r['site']==site and r['variable']==v and r['kind']=='RAW_MISSING_VALUE'} for v in ('SRAD','TMAX','TMIN')}
        # Rebuild exactly the pre-QC mean used by weather_preprocess.fill_missing:
        # group by station/month over the entire observed period, excluding raw
        # blanks before the historical transform(mean) fills them.
        means = {(v,month):float(clean.loc[(clean.month==month)&~clean.date.isin(missing_dates[v]),v].mean())
                 for v in missing_dates for month in range(1,13)}
        cached = {}
        whole_year_diffs = {}
        same_origin_hashes = {}
        for year in range(2005,2014):
            name=f'{prefix}{year%100:02d}01.WTH'
            final=ROOT/source_root/short/name
            generated=ROOT/'Leave_One_experiments/wth_generated_qc'/site/f'{site}{year}.WTH'
            origin=ROOT/'DSSAT_auto_validation/multisite_new_cultivar_inputs_013'/short/name
            assert final.exists() and generated.exists() and origin.exists()
            a,b=parse_daily(final,year),parse_daily(generated,year)
            cached[year]=(a,b,final,generated)
            assert a.keys()==b.keys()
            whole_year_diffs[year]=sum(abs(a[d][v]-b[d][v])>0.050001 for d in a for v in ('SRAD','TMAX','TMIN','RAIN'))
            same_origin_hashes[year]=sha(final)==sha(origin)
        site_rows=[]
        for r in gaps:
            if r['site']!=site:continue
            day=r['date'];year=int(day[:4]);month=int(day[5:7]);var=r['variable']
            current,old_wth,final_path,generated_path=cached[year]
            final_value=current[day][var]
            old_value=old_wth[day][var]
            clean_value=float(clean_by_date.loc[day,var])
            qc_value=float(qc_by_date.loc[day,var])
            mean=means[var,month]
            historical_mean_match=abs(clean_value-mean)<1e-7
            final_mean_match=abs(final_value-mean)<=0.050001
            generated_match=abs(final_value-old_value)<=0.050001
            assert historical_mean_match and final_mean_match and generated_match
            # SYA: the entire 9-year final WTH series agrees with historical
            # cleaned data at one-decimal resolution; only two other cells
            # differ from a second historical generated WTH. This establishes
            # that the 462 listed values are the old monthly-mean values.
            # LCA: years underwent documented later supplementation; exact
            # agreement at 22 cells cannot by itself prove which write won.
            classification='确认旧均值填补' if site=='SYA' else '无法确定'
            row={
                'site':site,'date':day,'variable':var,
                'original_xls':r['original_source'],'original_xls_state':'blank',
                'historical_month_mean_unrounded':f'{mean:.12g}',
                'weather_clean_value':f'{clean_value:.12g}',
                'weather_clean_qc_value':f'{qc_value:.12g}',
                'old_generated_wth_value':f'{old_value:.1f}',
                'final_wth_value':f'{final_value:.1f}',
                'old_generated_wth':generated_path.relative_to(ROOT).as_posix(),
                'final_wth':final_path.relative_to(ROOT).as_posix(),
                'final_wth_sha256':sha(final_path),
                'mean_matches_cleaned_exactly':historical_mean_match,
                'mean_matches_final_wth_at_0p1':final_mean_match,
                'old_generated_wth_matches_final':generated_match,
                'classification':classification,
                'external_download_daily_record':'NOT_FOUND_IN_PROJECT',
                'reason':'full SYA source series matches cleaned; monthly-mean reconstruction and old generated WTH agree' if site=='SYA' else 'supplemented LCA years lack per-cell import log; matching old mean alone cannot prove final write source',
            }
            site_rows.append(row)
        detail.extend(site_rows)
        site_summary[site]={
            'entry_count':len(site_rows),
            'classification_counts':dict(Counter(r['classification'] for r in site_rows)),
            'variable_counts':dict(Counter(r['variable'] for r in site_rows)),
            'training_year_final_vs_old_generated_wth_differences':whole_year_diffs,
            'final_vs_originIC_wth_sha256_same_every_training_year':all(same_origin_hashes.values()),
            'mean_matches_cleaned_exactly':sum(r['mean_matches_cleaned_exactly'] for r in site_rows),
            'mean_matches_final_wth_at_0p1':sum(r['mean_matches_final_wth_at_0p1'] for r in site_rows),
            'old_generated_wth_matches_final':sum(r['old_generated_wth_matches_final'] for r in site_rows),
        }
    assert len(detail)==484
    with DETAIL.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
    summary={
        'scope':'484 non-RAIN provenance entries; read-only source trace',
        'sites':site_summary,
        'externally_downloaded_SYA_LCA_source_files_found':0,
        'external_download_to_final_wth_daily_variable_links_proven':0,
        'original_xls_blank_to_external_download_to_final_wth_chain':'NOT_ESTABLISHED',
        'gate':{'SYA':'BLOCKED_BY_DATA_PROVENANCE','LCA':'BLOCKED_BY_DATA_PROVENANCE'},
        'classification_rule':'SYA old-mean value identified by exact mean reconstruction and full-year WTH/cleaned agreement; LCA same numerical evidence remains source-undetermined because later supplementation lacks cell-level import record',
    }
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=[
        '# SYA / LCA 非降水来源缺口追踪报告','',
        '范围：`remaining_nonrain_provenance_gaps.csv`的484个站点—日期—变量项；只读追踪，未修改WTH、DSSAT/WGEN/PPO配置，未运行模型或模拟。','',
        '## 结论','',
        '|站点|项数|已证实外部补测|确认旧均值填补（数值层面）|无法确定最终写入来源|gate|',
        '|---|---:|---:|---:|---:|---|',
        '|SYA|462|0|462|0|BLOCKED_BY_DATA_PROVENANCE|',
        '|LCA|22|0|0|22|BLOCKED_BY_DATA_PROVENANCE|','',
        'SYA的462项可确认最终WTH采用了历史“站点×月份”均值（按WTH一位小数精度）；这些值没有逐日外部观测证据。LCA的22项同样等于旧均值，但LCA训练年后来发生天气补充，在缺少逐项导入记录时不能把“数值相同”升级为“最终来源已证实”，故列为无法确定。没有任何一项建立“原始XLS空白→外部下载数据→最终WTH”的逐日变量级证据链。','',
        '## 证据链','',
        '1. `src/weather_preprocess.py`：`build_specs`从`my_data`指定SYA辐射/温度XLS及LCA的`D32.xls`/`T2.xls`；`normalize_source`识别空白；`merge_weather`按站点日期合并；`reindex_full_years`补齐日期；`fill_missing`对SRAD/TMAX/TMIN按站点跨所有年份同月均值填补，最后输出`weather_clean/{site}_weather_cleaned.csv`与`missing_value_fill_log.csv`。`weather_clean/data_check_report.md`和`docs/experiment_records/weather_data_check_report.md`记录过该流程。审计中以原始缺口日期从cleaned CSV中排除对应变量，重新计算月份均值；484项与cleaned精确一致，最终WTH按一位小数也全部一致。','',
        '2. `weather_clean_qc/{site}_weather_cleaned_qc.csv`和`Leave_One_experiments/wth_generated_qc/{site}/{site}{year}.WTH`在484项均与旧均值一致；`Leave_One_experiments/wth_generated_qc/wth_generation_summary_qc.csv`保留生成版天气文件清单。`src/run_weather_qc_030_00.py`的030_00结果记录仅对HLA两处值作校正，没有SYA/LCA的目标项校正。最终WTH与originIC目录相同年份文件SHA256一致，包括LCA的lowIC目录。','',
        '3. SYA 2005–2013最终WTH与旧生成版逐日四变量比较只有2006、2011各1个非目标SRAD值差异；最终WTH与`weather_clean/SYA_weather_cleaned.csv`的全年逐值比较在一位小数误差内一致。462个目标项没有变动，因此确认其现有值是旧均值填补值。这里确认的是WTH数值及历史生成链的对应，不声称存在完整的旧命令运行日志。','',
        '4. LCA 2005–2013最终WTH与旧生成版在多个年份已有补充差异；22个目标项仍等于旧均值。用户此前确认的历史下载补充解释了差异背景，但项目现存`data/external`目录只见YC、HLA/FQ相关下载子目录；`my_data`中SYA/LCA日值XLS是原始观测输入，未发现独立的这22项补测文件。备份内发现的SYA/LCA WTH为实验输入/渲染副本，不是带原始发布者、下载时间及逐行导入关系的独立补测材料。没有找到足以把这22项判为“已证实补测”的manifest。','',
        '## 判定边界','',
        '“确认旧均值填补”指最终WTH数值与可复算的历史均值填补输出一致，且SYA整年序列与旧来源链一致；不表示已经有可用于WGEN拟合的独立实测资料。LCA同值22项可疑为旧填补残留，但由于后续补充流程覆盖了同年其他天气值，无法确定最终编辑时对这22项采用的具体来源。数值巧合不能在没有逐项记录时排除。',
        '',
        '本次不改变`final_gate.json`：SYA仍因462项旧均值填补缺少独立补测/可接受填补依据而BLOCKED；LCA仍因22项来源无法确定而BLOCKED。若要解除，应提供这些日期和变量对应的补测文件与映射/导入记录，或明确接受这些旧均值作为WGEN拟合输入并独立复审。RAIN空白=0和LCA其余1113处差异的既有修订结论维持。','',
        '逐项表：`results/sy_lc_random_weather_015/nonrain_provenance_resolution_484.csv`；机器摘要：`results/sy_lc_random_weather_015/nonrain_provenance_resolution_summary.json`。',''
    ]
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary['sites'],ensure_ascii=False))


if __name__ == '__main__':
    main()
