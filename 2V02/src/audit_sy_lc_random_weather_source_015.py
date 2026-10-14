"""Read-only SYA/LCA weather audit. Never imports simulator/training modules.

Run: python src/audit_sy_lc_random_weather_source_015.py
Outputs are refused if already present, preserving earlier evidence.
"""
from pathlib import Path
from datetime import date, timedelta
from collections import Counter
import csv
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sy_lc_random_weather_015'
REPORT = ROOT / 'docs/sy_lc_random_weather_source_audit_015.md'
VARS = ['SRAD', 'TMAX', 'TMIN', 'RAIN']
SITES = {'SYA': ('SY', '046_10_sya_originIC_expanded_action_maskableppo'),
         'LCA': ('LC', '053_00_lca_lowIC_expanded_action_maskableppo')}


def rel(p):
    return p.relative_to(ROOT).as_posix()


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv_write(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_wth(p, year):
    records, errors, header = [], [], None
    if not p.exists():
        return [], ['FILE_MISSING']
    for n, line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == '@' and 'DATE' in parts:
            header = parts[1:]
            if not all(v in header for v in VARS):
                errors.append('MISSING_COLUMN')
            continue
        if header is None or line.startswith(('$', '@', '*', '!')):
            continue
        if not parts[0].isdigit():
            errors.append(f'INVALID_ROW:{n}')
            continue
        try:
            token = parts[0]
            y, doy = int(token[:-3]), int(token[-3:])
            if len(token) == 5:
                y += 2000 if y < 50 else 1900
            d = date(y, 1, 1) + timedelta(days=doy - 1)
            if doy < 1 or d.year != y or y != year:
                raise ValueError('unexpected year or day')
        except ValueError:
            errors.append(f'INVALID_DATE:{n}:{parts[0]}')
            continue
        vals = {}
        for v in VARS:
            try:
                val = float(parts[header.index(v)])
                vals[v] = val if math.isfinite(val) and val not in (-99, -999, 9999, 99999, 999999, 32766) else None
            except (ValueError, IndexError):
                vals[v] = None
        records.append((d, vals))
    if header is None:
        errors.append('NO_DATE_HEADER')
    return records, errors


def main():
    if REPORT.exists() or (OUT.exists() and any(OUT.iterdir())):
        raise SystemExit('Refusing to overwrite previous audit outputs; archive them under backups first.')
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots, configs, audit, issues, hashes = {}, [], [], [], {}
    def track(p):
        if p.exists():
            hashes[rel(p)] = digest(p)
    for site, (short, run) in SITES.items():
        cp = ROOT / 'configs' / (run + '.json')
        mp = ROOT / 'benchmark_results' / run / (run.split('_')[0] + '_' + run.split('_')[1] + '_run_manifest.json')
        cfg = json.loads(cp.read_text(encoding='utf-8'))
        manifest = json.loads(mp.read_text(encoding='utf-8'))
        assert manifest['config'] == cfg, f'Current/frozen config mismatch: {site}'
        pf = manifest['preflight']
        assert pf['engine_train_years'] == cfg['scope']['train_years']
        assert pf['engine_validation_years'] == cfg['scope']['validation_years']
        source = ROOT / pf['resolved_input_root'] / short
        snapshots[site] = {'config': cfg, 'manifest': rel(mp), 'input_dir': rel(source),
                           'historical_status': manifest['status']}
        track(cp); track(mp)
        configs.append(dict(site=site, config_file=rel(cp), IC=cfg['input_profile'],
            reward='032_00 stress relief yield-cost reward + 040_28 SWFAC guardrail (50,0.05,scale=0.001)',
            action_space=json.dumps(cfg['actions'], separators=(',', ':')),
            mask_version='040_26 staged I240 + 040_36 DAP90 reserve, inherited via 042_10/046_02',
            train_weather_source=rel(source), train_year_range='2005-2013',
            validation_weather_source=rel(source), validation_year_range='2014-2023',
            irrigation_limit_mm=240, nitrogen_limit_kg_ha=250,
            ppo='MaskablePPO;100K;seed0;lr0.0003;gamma1;gae1;n_steps144;batch144;epochs5;ent0.01;net[64,64]'))
        for year in cfg['scope']['train_years'] + cfg['scope']['validation_years']:
            p = source / f'CN{short}{year % 100:02}01.WTH'
            track(p)
            records, errors = parse_wth(p, year)
            expected = {date(year, 1, 1) + timedelta(days=i) for i in range((date(year+1, 1, 1)-date(year,1,1)).days)}
            counts = Counter(d for d, _ in records)
            missing = expected - set(counts)
            nmissing = {v: sum(vals[v] is None for _, vals in records) for v in VARS}
            neg_rain = sum(vals['RAIN'] is not None and vals['RAIN'] < 0 for _, vals in records)
            neg_srad = sum(vals['SRAD'] is not None and vals['SRAD'] < 0 for _, vals in records)
            temp_logic = sum(vals['TMAX'] is not None and vals['TMIN'] is not None and vals['TMAX'] < vals['TMIN'] for _, vals in records)
            ranges = sum(any(vals[v] is not None and not (lo <= vals[v] <= hi) for v,lo,hi in [('SRAD',0,45),('TMAX',-60,60),('TMIN',-60,60),('RAIN',0,500)]) for _,vals in records)
            duplicates = sum(c-1 for c in counts.values())
            ordered = [d for d,_ in records] == sorted(d for d,_ in records)
            status = 'PASS' if not (errors or missing or duplicates or any(nmissing.values()) or neg_rain or neg_srad or temp_logic or ranges or not ordered) else 'FAIL'
            audit.append(dict(site=site, year=year, file=rel(p), days_expected=len(expected), days_found=len(records), missing_days=len(missing), duplicate_days=duplicates,
                rain_missing=nmissing['RAIN'], srad_missing=nmissing['SRAD'], tmax_missing=nmissing['TMAX'], tmin_missing=nmissing['TMIN'], qc_status=status,
                split='train' if year <= 2013 else 'validation', negative_rain=neg_rain, negative_srad=neg_srad, temperature_logic_errors=temp_logic,
                physical_range_warning_rows=ranges, date_order_ok=ordered, parse_errors=';'.join(errors), sha256=digest(p) if p.exists() else ''))
            for d in sorted(missing):
                issues.append(dict(site=site,date=d.isoformat(),variable='ALL',kind='WTH_MISSING_DATE',source=rel(p),value='',possible_reason='源文件缺行；未自动修复'))
            for d, vals in records:
                for v,val in vals.items():
                    if val is None or (v in ('RAIN','SRAD') and val < 0):
                        issues.append(dict(site=site,date=d.isoformat(),variable=v,kind='WTH_INVALID_VALUE',source=rel(p),value=val,possible_reason='缺测标记或非法数值'))
                if vals['TMAX'] is not None and vals['TMIN'] is not None and vals['TMAX'] < vals['TMIN']:
                    issues.append(dict(site=site,date=d.isoformat(),variable='TMAX/TMIN',kind='WTH_TEMPERATURE_LOGIC',source=rel(p),value=str(vals),possible_reason='温度大小关系错误'))
    print('WTH audit completed; reading original XLS one file at a time.', flush=True)
    # Use only pure parsing helpers. In particular, never call normalize_source:
    # it converts missing rainfall to zero and deduplicates observations.
    import pandas as pd
    sys.path.insert(0, str(ROOT / 'src'))
    import weather_preprocess as wp
    source_map = list(csv.DictReader((ROOT / 'weather_clean/recognized_weather_fields.csv').open(encoding='utf-8-sig')))
    raw_summary, provenance = [], []
    grouped = {}
    for m in source_map:
        sites = [s for s in SITES if s in m['stations_found'].split(',')]
        if sites:
            grouped.setdefault(m['source_file'], []).append((m, sites))
    for name, specs in grouped.items():
        p = ROOT / 'my_data' / name
        track(p)
        frame = wp.read_excel_all_sheets(p)
        frame.columns = [wp.normalize_text(c) for c in frame.columns]
        sc = wp.pick_column(frame.columns, ['生态站代码'])
        yc = wp.pick_column(frame.columns, ['年'])
        mc = wp.pick_column(frame.columns, ['月'])
        dc = wp.pick_column(frame.columns, ['日'])
        dt = pd.to_datetime(dict(year=wp.coerce_numeric(frame[yc]), month=wp.coerce_numeric(frame[mc]), day=wp.coerce_numeric(frame[dc])), errors='coerce')
        for m, sites in specs:
            v = m['variable']; values = wp.coerce_numeric(frame[m['source_column']])
            for site in sites:
                mask = frame[sc].astype(str).str.strip().eq(site)
                data = pd.DataFrame({'date': dt[mask], 'value': values[mask]})
                for year in range(2005,2024):
                    sub = data[data.date.dt.year.eq(year)]
                    exp = pd.date_range(f'{year}-01-01', f'{year}-12-31')
                    absent = exp.difference(sub.date.dropna())
                    nmiss = int(sub.value.isna().sum())
                    dup = int(sub.date.duplicated().sum())
                    raw_summary.append(dict(site=site,year=year,variable=v,source=rel(p),source_column=m['source_column'],rows=len(sub),missing_values=nmiss,missing_dates=len(absent),duplicate_dates=dup))
                    for d in absent:
                        provenance.append(dict(site=site,date=d.date().isoformat(),variable=v,kind='RAW_DATE_ABSENT',source=rel(p),value='',possible_reason='原始观测文件未记录该日'))
                    for row in sub[sub.value.isna()].itertuples():
                        provenance.append(dict(site=site,date=row.date.date().isoformat(),variable=v,kind='RAW_MISSING_VALUE',source=rel(p),value='',possible_reason='原始空白或缺测标记；旧流程RAIN转0，其他变量站点月均值填补，未证明该日真实观测'))
                    for row in sub[sub.date.duplicated(keep=False)].itertuples():
                        provenance.append(dict(site=site,date=row.date.date().isoformat(),variable=v,kind='RAW_DUPLICATE_DATE',source=rel(p),value=row.value,possible_reason='同日多条来源记录，旧流程keep=last'))
                    for row in sub[sub.value.notna() & ((sub.value < 0) if v in ('RAIN','SRAD') else ((sub.value < -60)|(sub.value > 60)))].itertuples():
                        provenance.append(dict(site=site,date=row.date.date().isoformat(),variable=v,kind='RAW_INVALID_VALUE',source=rel(p),value=row.value,possible_reason='原始物理范围异常，需要来源核查'))
                invalid_dates = int(data.date.isna().sum())
                if invalid_dates:
                    provenance.append(dict(site=site,date='UNKNOWN',variable=v,kind='RAW_INVALID_DATE',source=rel(p),value=invalid_dates,possible_reason='日期不可解析，不能确定所属年份'))
        del frame
        print(f'Parsed {name}', flush=True)
    track(ROOT / 'weather_clean/recognized_weather_fields.csv')
    track(ROOT / 'weather_clean/missing_value_fill_log.csv')
    gate = {}
    totals = {}
    for site in SITES:
        rows = [r for r in raw_summary if r['site']==site and r['year']<=2013]
        totals[site] = {v: sum(r['missing_values']+r['missing_dates'] for r in rows if r['variable']==v) for v in VARS}
        blockers = [r for r in provenance if r['site']==site and (r['date']=='UNKNOWN' or int(r['date'][:4])<=2013)]
        failed = [r for r in audit if r['site']==site and r['split']=='train' and r['qc_status']!='PASS']
        gate[f'{site}_weather_ready'] = 'BLOCKED_BY_DATA_PROVENANCE' if blockers or failed else 'COMPLETE_READY_FOR_WGEN'
    gate['need_gap_fill'] = 'SOURCE_VERIFICATION_REQUIRED_NO_AUTOMATIC_FILL' if any(v=='BLOCKED_BY_DATA_PROVENANCE' for v in gate.values()) else 'NO'
    gate['next_step'] = '先核实原始缺测、既有填补和降水空白语义；逐日闭合训练期来源并复审。CLI/WGEN/DSSAT/PPO需另行任务授权。'
    csv_write('baseline_config_summary.csv', configs, list(configs[0]))
    csv_write('weather_source_audit.csv', audit, list(audit[0]))
    csv_write('raw_source_audit.csv', raw_summary, list(raw_summary[0]))
    csv_write('weather_gap_details.csv', issues + provenance, ['site','date','variable','kind','source','value','possible_reason'])
    (OUT/'final_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'config_evidence.json').write_text(json.dumps(snapshots,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    changed = [name for name, h in hashes.items() if digest(ROOT/name)!=h]
    assert not changed, f'Audit input changed during run: {changed}'
    (OUT/'input_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    lines = ['# SYA / LCA 随机天气增强前置来源审计 015', '', '日期：2026-10-06。执行范围：只读文件审计；未生成 CLI，未运行 WGEN、DSSAT 或 PPO；未填补或修改天气、配置和既有结果。', '',
        '## 结论', '', '|站点|训练期|验证期|WTH QC通过年数|来源门槛|', '|---|---|---|---|---|']
    for site in SITES:
        rows=[r for r in audit if r['site']==site]
        lines.append(f"|{site}|2005–2013|2014–2023|{sum(r['qc_status']=='PASS' for r in rows)}/{len(rows)}|{gate[site+'_weather_ready']}|")
    lines += ['', 'WTH 结构完整性和真实观测来源是两个独立门槛。既有 WTH 可以连续、有值，但历史填补不等于原始观测。训练期来源未闭合时，不能直接用这些值拟合 WGEN 月统计参数或生成站点 CLI。', '',
        '## Baseline 与配置继承', '',
        '本审计选择与 YC 055_00 同一继承链的冻结历史 baseline：SYA 046_10 originIC、LCA 053_00 lowIC；不采用其他 forecast、reward 或天气增强分支。当前 JSON 与 completed_formal manifest 一致，训练/验证年份与 manifest 中 engine split 一致。证据见 config_evidence.json。', '',
        '两站动作均为 I=[0,15,30,45] mm × N=[0,40,80,120] kg/ha，共16个离散动作。有效灌溉季节上限240 mm，氮250 kg/ha；I/N事件间隔7天，灌溉DAP1–120、施氮DAP1–90。040_26 灌溉阶段上限DAP≤30:75、DAP≤60:150；040_36 DAP≤90:195。', '',
        'reward 继承032_00 yield-minus-water/nitrogen-cost-plus-stress-relief（yield_coef0.158、N cost1.58、water cost1.1、water relief10、N relief5、scale0.001），并继承040_28的50×max(0,SWFAC−0.05)×scale过程惩罚。仅照录既有代码，不在本任务修订其物理解释。', '',
        '重要：运行目录复制的基础 YAML 中 season_irrigation_soft_limit=160 是基础值；runner 的 load_i240_config 在内存中覆盖为240，且 wrapper 增加阶段掩码，不能仅据复制 YAML 推断有效配置。PPO采用现有100K/seed0、[64,64]网络及原始无预报观测；具体参数见 baseline_config_summary.csv。', '',
        '配置与代码证据：configs/046_10_sya_originIC_expanded_action_maskableppo.json；configs/053_00_lca_lowIC_expanded_action_maskableppo.json；src/run_sya_ppo_configured_046_02.py；src/053_lca_lowIC_site_transfer/run_053_00_lca_lowIC_expanded_action_maskableppo.py；src/run_sya_lowIC_binary_timing_maskableppo_042_10.py；src/run_sya_lowIC_ppo_i240_staged_reserve_040_26.py；src/run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py；src/run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py。', '',
        '## 天气来源与逐日检查', '',
        '来源目录来自既有正式 manifest 的 resolved_input_root；runner 与 ppo_safe_rendering.source_weather_path 选择对应逐年 CN{site}{YY}01.WTH，build_env_args 显式 random_weather=False。按这条配置解析路径审计；本次不重新渲染，不声称已验证历史 DSSAT 实际读取字节。', '']
    for site in SITES:
        lines.append(f"- {site}：`{snapshots[site]['input_dir']}`。")
    lines += ['', '逐年检查日期范围、闰年天数、日期顺序、缺失日、重复日、四变量列、缺测哨兵/非有限值、负降水、负辐射、TMAX<TMIN。保守范围警报：SRAD0–45 MJ/m²/d、温度−60–60℃、RAIN0–500 mm/d；超出为核查警报，不自动修改。days_found为该年可解析记录数，duplicate_days为多余记录数，变量missing为现有WTH行上的缺测单元格，缺整日另计missing_days。', '',
        '## 原始观测缺口', '', '|站点|训练期SRAD缺口|TMAX缺口|TMIN缺口|RAIN未核实缺口|', '|---|---|---|---|---|']
    for site,t in totals.items():
        lines.append(f"|{site}|{t['SRAD']}|{t['TMAX']}|{t['TMIN']}|{t['RAIN']}|")
    lines += ['', '上表是原始 XLS 缺测单元格加缺日期数量；RAIN 缺测包含旧流程自动转0的空白，不能直接认定全部需要插值或全部是降雨缺失，必须先核实编码语义。所有日期、变量、来源文件与可能原因保存在 weather_gap_details.csv；raw_source_audit.csv 分年分变量列出数量。验证期也审计并单独标记，禁止用于训练期拟合。', '',
        '原始数据按 recognized_weather_fields.csv 的映射读取 my_data XLS，一次一个文件，只使用weather_preprocess的读取/数值识别函数，避开normalize_source的降水转0与去重逻辑。历史fill_missing按站点月份均值填补且涵盖全来源年份（SYA2001–2023、LCA2000–2023），存在训练派生值混入验证期统计信息的风险；本任务不据此重训或重定义baseline。', '',
        '既有clean/qc CSV与WTH的逐值一致性、每条历史填补的传递链及其原始补测授权尚未全部闭合。本次保持保守来源门槛；不能将现有填补值自动认作WGEN拟合的真实观测。可能原因只依据缺测编码/旧代码处理方式，不能确定仪器故障等原因。', '',
        '## 后续门槛与产物', '', gate['next_step'], '',
        '月参数/CLI在格式上需要连续四变量天气；在来源上须先解决训练期原始缺测、降水空白及既有全期均值填补问题，再以训练期独立来源审核。当前未计算月参数、未生成CLI，也未验证运行时WGEN兼容性。', '',
        '输出目录：results/sy_lc_random_weather_015/。必需baseline_config_summary.csv、weather_source_audit.csv、final_gate.json均已生成；附加raw_source_audit.csv、weather_gap_details.csv、config_evidence.json、input_sha256.json用于复核。', '',
        '复现：`python src/audit_sy_lc_random_weather_source_015.py`。脚本拒绝覆盖已有产物；复跑前需将旧审计产物备份到backups。输入SHA256在运行前后核验一致。仅串行读取气象文件，没有加载模型、GPU或多进程。', '',
        '探索记录：初始全仓文件搜索输出过大，后续改为限定配置/代码/站点目录；PowerShell中直接向rg传通配路径报错，改用-g过滤；误猜weather_preprocess_report.md不存在，实际报告是weather_clean/data_check_report.md。这些探索错误不改变输入或门槛。', '',
        'Git：未修改已有用户工作区变更，未提交原始天气，未push。任务文件commit文字为建议，本次交付保留可审阅的工作区产物。']
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'gate':gate,'training_raw_missing':totals,'wth_failures':sum(r['qc_status']!='PASS' for r in audit),'source_files_verified_unchanged':len(hashes)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
