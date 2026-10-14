"""Revise only the 015 provenance interpretation using frozen audit evidence.

Does not read or change WTH, simulation inputs, or training configuration.
The original 015 report and edited outputs must be backed up before running.
"""
from collections import Counter
from datetime import date
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sy_lc_random_weather_015'
REPORT = ROOT / 'docs/sy_lc_random_weather_source_audit_015.md'
BACKUP = ROOT / 'backups/sy_lc_weather_015_before_provenance_revision'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    for name in ('final_gate.json','raw_source_audit.csv','weather_gap_details.csv', REPORT.name):
        assert (BACKUP/name).exists(), f'Missing protected baseline: {name}'
    audit, _ = read_csv(OUT/'weather_source_audit.csv')
    assert len(audit) == 38 and all(r['qc_status'] == 'PASS' for r in audit)
    mismatches = json.loads((OUT/'derived_weather_comparison_mismatches.json').read_text(encoding='utf-8'))
    assert len(mismatches) == 1113 and all(r['site']=='LCA' and int(r['date'][:4])<=2013 for r in mismatches)
    mismatch_keys = {(r['site'], r['date'], r['variable']) for r in mismatches}
    original, fields = read_csv(BACKUP/'weather_gap_details.csv')
    rain_blanks = [r for r in original if r['kind']=='RAW_MISSING_VALUE' and r['variable']=='RAIN']
    actual_gaps = [r for r in original if not (r['kind']=='RAW_MISSING_VALUE' and r['variable']=='RAIN')]
    assert len(rain_blanks) > 0
    raw, raw_fields = read_csv(BACKUP/'raw_source_audit.csv')
    for row in raw:
        if row['variable']=='RAIN':
            row['rain_blank_as_zero_count'] = row['missing_values']
            row['missing_values'] = '0'
        else:
            row['rain_blank_as_zero_count'] = '0'
    write_csv(OUT/'raw_source_audit.csv', raw, raw_fields+['rain_blank_as_zero_count'])
    write_csv(OUT/'weather_gap_details.csv', actual_gaps, fields)
    rain_rows = []
    count = Counter((r['site'], r['date'][:4]) for r in rain_blanks)
    for (site, year), n in sorted(count.items()):
        rain_rows.append(dict(site=site, year=year, rain_blank_as_zero_count=n, interpretation='zero_rainfall_mm', missing_weather_count=0,
                              evidence='weather_clean/data_check_report.md:57; current user provenance correction'))
    write_csv(OUT/'rain_blank_zero_summary.csv', rain_rows, list(rain_rows[0]))
    remaining = []
    for r in actual_gaps:
        if r['kind']!='RAW_MISSING_VALUE' or r['variable']=='RAIN' or not r['date'][:4].isdigit() or int(r['date'][:4])>2013:
            continue
        key = (r['site'], r['date'], r['variable'])
        # A changed final WTH value is consistent with the historical
        # supplementation context. An unchanged cleaned value is not proof
        # that a new daily observation replaced the former imputation.
        if key not in mismatch_keys:
            remaining.append(dict(site=r['site'],date=r['date'],variable=r['variable'],
                                  final_wth_equals_cleaned='yes',
                                  disposition='unresolved_original_gap_or_historical_imputation',
                                  original_source=r['source']))
    write_csv(OUT/'remaining_nonrain_provenance_gaps.csv', remaining, list(remaining[0]))
    unresolved = Counter(r['site'] for r in remaining)
    assert unresolved == {'SYA':462,'LCA':22}, unresolved
    revised_gate = {
        'SYA_weather_ready':'BLOCKED_BY_DATA_PROVENANCE',
        'LCA_weather_ready':'BLOCKED_BY_DATA_PROVENANCE',
        'need_gap_fill':'UNDETERMINED_PENDING_NONRAIN_SOURCE_REVIEW',
        'next_step':'仅核实剩余训练期SRAD/TMAX/TMIN来源或既有填补依据；RAIN空白=0，LCA 1113处cleaned/WTH差异不再作为阻塞。核实后复审来源门槛。',
        'resolved_blockers':['RAIN blank fields are zero rainfall, not missing data',
                             'LCA 1113 cleaned CSV versus final WTH differences reflect historical supplementary weather data'],
        'remaining_blockers':{
            'SYA':{'nonrain_original_gap_cells_matching_cleaned_and_final_wth':462,'variables':{'SRAD':160,'TMAX':151,'TMIN':151}},
            'LCA':{'nonrain_original_gap_cells_matching_cleaned_and_final_wth':22,'variables':{'SRAD':13,'TMAX':8,'TMIN':1}}
        },
        'scope':'provenance audit revision only; WTH QC unchanged'
    }
    (OUT/'final_gate.json').write_text(json.dumps(revised_gate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    provenance = {
        'revision':'015 provenance context rev1',
        'historical_context_lca':'Historical supplementary weather data were incorporated into the final WTH after downloading source files; cleaned CSV comparison differences reflect this historical supplementation process.',
        'historical_context_authority':'current user clarification',
        'rain_blank_semantics':'Blank RAIN in the original XLS denotes zero rainfall (0 mm), not missing weather data.',
        'rain_evidence':'weather_clean/data_check_report.md:57 records earlier user confirmation and 67681 blanks interpreted as zero across sites',
        'lca_cleaned_wth_difference_count':1113,
        'lca_difference_disposition':'historical supplementation context; not a provenance blocker',
        'remaining_nonrain_source_gap_count':dict(unresolved),
        'caution':'Matched cleaned/WTH values at original nonrain blanks do not independently prove a downloaded observation replaced historical imputation.'
    }
    (OUT/'provenance_context_revision.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    verification = json.loads((BACKUP/'verification_log.json').read_text(encoding='utf-8'))
    verification.pop('unresolved_provenance', None)
    verification['provenance_revision'] = {
        'rain_blank_semantics':'confirmed zero rainfall; excluded from missing weather',
        'lca_1113_differences':'historical supplementary weather context; excluded from blocker',
        'remaining_nonrain_source_gaps':dict(unresolved),
        'gate':'both sites remain BLOCKED_BY_DATA_PROVENANCE for remaining nonrain source gaps'
    }
    (OUT/'verification_log.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    old_report = (BACKUP/REPORT.name).read_text(encoding='utf-8')
    config_start = old_report.index('## Baseline 与配置继承')
    weather_start = old_report.index('## 天气来源与逐日检查')
    baseline = old_report[config_start:weather_start]
    lines = [
        '# SYA / LCA 随机天气增强前置来源审计 015（provenance 修订版）','',
        '修订日期：2026-10-06。原版报告与被修订的CSV/JSON保存在 `backups/sy_lc_weather_015_before_provenance_revision/`。本次只修订历史来源解释与门槛；WTH逐日QC、配置、模拟和训练均未改动。','',
        '## 修订结论','',
        '|站点|WTH结构QC|已解除的阻塞依据|剩余来源阻塞|gate|',
        '|---|---|---|---|---|',
        '|SYA|19/19年通过|RAIN空白=0 mm|训练期SRAD 160、TMAX 151、TMIN 151个原始空白，与cleaned及最终WTH值相同，仍需确认填补或补测来源|BLOCKED_BY_DATA_PROVENANCE|',
        '|LCA|19/19年通过|RAIN空白=0 mm；1113处cleaned/WTH差异有历史下载补充背景，不再作为异常|训练期SRAD 13、TMAX 8、TMIN 1个原始空白，与cleaned及最终WTH值相同，仍需确认是否为原有填补值|BLOCKED_BY_DATA_PROVENANCE|',
        '',
        '两站仍有真正阻止进入WGEN的非降水来源缺口。本结论只针对拟合来源可信度；WTH四变量、日期连续性和数值QC已经通过。不存在需要对RAIN插值或另寻降水来源的事项。','',
        baseline.rstrip(),'',
        '## 天气文件和核查范围','',
        '- SYA训练/验证WTH：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013/SY/CNSY{YY}01.WTH`。',
        '- LCA训练/验证WTH：`DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/LC/CNLC{YY}01.WTH`。',
        '- 训练2005–2013，验证2014–2023；两站合计38个年份的WTH逐日检查均PASS。`weather_source_audit.csv`保留原审计数值，不因来源解释而重写。','',
        '## 两项历史背景修订','',
        '### LCA 补充天气','',
        '> Historical supplementary weather data were incorporated into the final WTH after downloading source files; cleaned CSV comparison differences reflect this historical supplementation process.','',
        '这条历史说明来自本次用户补充。既有实验配置显示LCA 053_00使用当前lowIC目录中的最终WTH；此前报告的1113处差异全部在训练期，涉及RAIN 647、SRAD 126、TMAX 169、TMIN 171个变量单元格。现将差异解释为历史补充流程造成，不再把差异本身作为BLOCKED依据；逐日差异清单保留供追溯。当前项目内未定位到这批下载源文件的逐日导入manifest，因此不把“有补充流程”扩大表述为“全部原始非降水空白都已由下载数据逐日闭合”。','',
        '### RAIN空白','',
        '过往`weather_clean/data_check_report.md`明确写明“已根据用户确认，将RAIN原表空白解释为无降雨，并转换为0”。本次再次获得同一语义确认：原始XLS的RAIN空白=0 mm。修订后的`raw_source_audit.csv`中RAIN `missing_values=0`，原空白计入`rain_blank_as_zero_count`；`weather_gap_details.csv`已移除这些“RAW_MISSING_VALUE/RAIN”记录；`rain_blank_zero_summary.csv`逐站点逐年保存0雨日空白数量。这不需要插值或重新找降水来源。','',
        '## 仍需核实的非降水来源','',
        '原始XLS在训练期的SRAD/TMAX/TMIN空白，SYA分别160/151/151个；LCA分别139/172/172个。将这些日期与最终WTH、旧cleaned CSV比较，SYA的462个空白位置全部与cleaned值一致；LCA有461个位置值已不同，符合历史补充背景，剩余22个仍与cleaned值一致（SRAD13、TMAX8、TMIN1）。`remaining_nonrain_provenance_gaps.csv`列出462+22个具体日期和变量。',
        '',
        '“与cleaned相同”本身不能证明最终WTH仍是均值填补，也不能证明已有独立补测。旧`weather_preprocess.py`确实按站点月份均值填补非降水空白，且计算范围覆盖训练和验证年份。SYA的WTH与cleaned逐值一致，故这462处尤需核对；LCA的22处可能是旧填补残留，也可能有值恰好一致的补充记录。未见足够证据解除这484处来源缺口，故两站维持BLOCKED，但不再以RAIN空白或1113处差异为理由。',
        '',
        '下一步只需针对`remaining_nonrain_provenance_gaps.csv`核查历史补充文件、导入记录或明确的插补接受依据，再独立复核WGEN拟合天气来源。当前不生成CLI，不计算月参数，不运行WGEN、DSSAT或PPO。','',
        '## 产物与验证','',
        '`results/sy_lc_random_weather_015/`包含原有`baseline_config_summary.csv`、`weather_source_audit.csv`、`config_evidence.json`、`input_sha256.json`、差异清单，以及修订后的`raw_source_audit.csv`、`weather_gap_details.csv`、`final_gate.json`和新增的`rain_blank_zero_summary.csv`、`remaining_nonrain_provenance_gaps.csv`、`provenance_context_revision.json`。','',
        '原WTH与配置文件的哈希见`input_sha256.json`；本修订脚本不写这些文件。`src/revise_sy_lc_weather_provenance_015.py`基于冻结原审计证据生成修订版。',''
    ]
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'rain_blanks_reclassified':len(rain_blanks),'lca_mismatch_context_count':len(mismatches),'remaining_source_gaps':dict(unresolved),'gate':{'SYA':revised_gate['SYA_weather_ready'],'LCA':revised_gate['LCA_weather_ready']}},ensure_ascii=False))


if __name__ == '__main__':
    main()
