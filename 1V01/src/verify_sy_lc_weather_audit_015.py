"""Lightweight post-audit checks and derived CSV/WTH comparison (read-only inputs)."""
from pathlib import Path
from datetime import date, timedelta
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/sy_lc_random_weather_015'


def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    audit = rows(OUT / 'weather_source_audit.csv')
    raw = rows(OUT / 'raw_source_audit.csv')
    assert len(audit) == 38 and len(raw) == 152
    assert len({(r['site'],r['year']) for r in audit}) == 38
    assert all(r['qc_status'] == 'PASS' and r['days_expected'] == r['days_found'] for r in audit)
    historical = rows(ROOT / 'weather_clean/data_check_by_year_before_fill.csv')
    for r in raw:
        if r['variable'] != 'RAIN':
            old = next(x for x in historical if x['station'] == r['site'] and x['year'] == r['year'])
            assert int(old[r['variable']]) == int(r['missing_values'])
    for name, h in json.loads((OUT/'input_sha256.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == h, name
    cleaned = {s: {r['date']:r for r in rows(ROOT/f'weather_clean/{s}_weather_cleaned.csv')} for s in ('SYA','LCA')}
    mismatches, compared = [], 0
    for r in audit:
        for line in (ROOT/r['file']).read_text(encoding='utf-8-sig').splitlines():
            bits = line.split()
            if len(bits) != 5 or not bits[0].isdigit():
                continue
            d = (date(int(bits[0][:-3]),1,1) + timedelta(days=int(bits[0][-3:])-1)).isoformat()
            for i,v in enumerate(('SRAD','TMAX','TMIN','RAIN'),1):
                compared += 1
                val = cleaned[r['site']][d][v]
                if abs(float(bits[i])-float(val)) > 0.050001:
                    mismatches.append(dict(site=r['site'],date=d,variable=v,WTH=bits[i],cleaned=val))
    evidence = OUT/'derived_weather_comparison_mismatches.json'
    if evidence.exists():
        assert json.loads(evidence.read_text(encoding='utf-8')) == mismatches
    else:
        evidence.write_text(json.dumps(mismatches,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(validation='PASS',weather_years=len(audit),raw_variable_years=len(raw),compared_cells=compared,mismatches=len(mismatches))))


if __name__ == '__main__':
    main()
