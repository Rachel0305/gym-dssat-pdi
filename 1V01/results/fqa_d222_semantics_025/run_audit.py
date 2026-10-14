"""Read-only FQA D222 field and event audit for 2011–2013."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import xlrd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {
    "current": ROOT / "data/external/人工大气观测降雨蒸发能见度日值.xls",
    "old_raw": ROOT / "data/external/cern_d222/raw/excel.xls",
    "legacy": ROOT / "my_data/HLLCYCFQ降雨数据.xls",
}
FIELDS = ["20-8时降水量(mm)", "8-20时降水量(mm)", "20-20合计(mm)", "8-8时降水量(mm)"]
RAIN = "20-20合计(mm)"
START, END = dt.date(2011, 1, 1), dt.date(2013, 12, 31)
FIELD_DESCRIPTIONS = {
    "生态站代码": "生态站代码；FQA 由项目站点映射为封丘站",
    "年": "导出记录的年份",
    "月": "导出记录的月份",
    "日": "导出记录的日期；时段归属待原始字典确认",
    "小型蒸发皿20时蒸发量(mm)": "20 时小型蒸发皿读数",
    "E601蒸发皿20点蒸发量(mm)": "20 时 E601 蒸发皿读数",
    "20-8时降水量(mm)": "20 时至 8 时的人工观测降水分段；日期锚定待字典确认",
    "8-20时降水量(mm)": "8 时至 20 时的人工观测降水分段；日期锚定待字典确认",
    "20-20合计(mm)": "20 时至 20 时的降水合计；本项目采用的 D222 雨量列",
    "8-8时降水量(mm)": "8 时至 8 时的降水合计；不同于本项目采用列",
    "8时能见度(km)": "8 时定时能见度",
    "14时能见度(km)": "14 时定时能见度",
    "20时能见度(km)": "20 时定时能见度",
}


def blank(v):
    return v is None or str(v).strip() == ""


def number(v):
    if blank(v):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def write(name, rows, fields):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def percentile(values, p):
    a = sorted(values)
    if not a:
        return None
    i = (len(a) - 1) * p
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    return a[lo] + (a[hi] - a[lo]) * (i - lo)


def load_workbook(role, path):
    book = xlrd.open_workbook(str(path), on_demand=True)
    records = {}
    inventory = []
    for sheet in book.sheets():
        if sheet.nrows < 2:
            continue
        headers = [str(v).strip() for v in sheet.row_values(1)]
        if not {"生态站代码", "年", "月", "日", RAIN}.issubset(headers):
            continue
        cols = {name: headers.index(name) for name in headers}
        vals = defaultdict(list)
        types = defaultdict(Counter)
        for ri in range(2, sheet.nrows):
            row = sheet.row_values(ri)
            if str(row[cols["生态站代码"]]).strip() != "FQA":
                continue
            try:
                date = dt.date(*(int(float(row[cols[k]])) for k in ("年", "月", "日")))
            except (TypeError, ValueError, OverflowError):
                continue
            if not START <= date <= END:
                continue
            if date in records:
                raise ValueError(f"Duplicate FQA date in {path}: {date}")
            item = {"date": date, "source_row": ri + 1, "sheet": sheet.name}
            for ci, name in enumerate(headers):
                val = row[ci]
                item[name] = val
                vals[name].append(val)
                types[name][xlrd.sheet.ctype_text.get(sheet.cell_type(ri, ci), "unknown")] += 1
            records[date] = item
        for name in headers:
            series = vals[name]
            nums = [number(v) for v in series if not blank(v)]
            nums = [x for x in nums if x is not None]
            decimals = set()
            for v in series:
                if blank(v) or number(v) is None:
                    continue
                s = str(v).strip()
                decimals.add(len(s.split(".", 1)[1].rstrip("0")) if "." in s else 0)
            inventory.append({
                "source_role": role, "file": str(path.relative_to(ROOT)), "sheet": sheet.name,
                "field_name": name, "field_description": FIELD_DESCRIPTIONS.get(name, "原始导出列名；观测细则未随文件提供"),
                "unit_from_header": "mm" if "(mm)" in name else "km" if "(km)" in name else "none_stated",
                "excel_cell_types": json.dumps(types[name], ensure_ascii=False, sort_keys=True),
                "missing_encoding": "empty text/blank cell" if any(blank(v) for v in series) else "none observed",
                "decimal_places_observed": ";".join(map(str, sorted(decimals))) if decimals else "",
                "min_numeric": min(nums) if nums else "", "max_numeric": max(nums) if nums else "",
                "nonzero_numeric_count": sum(x != 0 for x in nums),
                "zero_numeric_count": sum(x == 0 for x in nums),
                "blank_count": sum(blank(v) for v in series),
                "nonblank_nonnumeric_count": sum(not blank(v) and number(v) is None for v in series),
                "scope": "FQA 2011-01-01..2013-12-31", "date_rows": len(records),
            })
    book.release_resources()
    return records, inventory


def status(item, field):
    if item is None:
        return "ABSENT_DATE"
    x = item.get(field)
    if blank(x):
        return "BLANK"
    return "NUMERIC" if number(x) is not None else "NONNUMERIC"


def main():
    datasets, inventory = {}, []
    for role, path in SOURCES.items():
        data, inv = load_workbook(role, path)
        datasets[role] = data
        inventory.extend(inv)
    write("field_inventory.csv", inventory, ["source_role", "file", "sheet", "field_name", "field_description", "unit_from_header", "excel_cell_types", "missing_encoding", "decimal_places_observed", "min_numeric", "max_numeric", "nonzero_numeric_count", "zero_numeric_count", "blank_count", "nonblank_nonnumeric_count", "scope", "date_rows"])

    dates = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
    current = datasets["current"]
    distributions = []
    for year in (2011, 2012, 2013):
        ys = [d for d in dates if d.year == year]
        vals = [number(current[d][RAIN]) for d in ys if d in current and number(current[d][RAIN]) is not None]
        distributions.append({"year": year, "expected_days": len(ys), "record_days": sum(d in current for d in ys), "numeric_count": len(vals), "blank_or_absent_count": len(ys)-len(vals), "mean_mm": statistics.mean(vals), "median_mm": statistics.median(vals), "std_sample_mm": statistics.stdev(vals) if len(vals)>1 else "", "max_mm": max(vals), "p90_mm": percentile(vals,.9), "p95_mm": percentile(vals,.95), "p99_mm": percentile(vals,.99), "numeric_subtotal_mm": round(sum(vals),6), "zero_count": sum(v==0 for v in vals), "note": "非空数值分布；小计不是年总量；样本标准差；线性插值分位数"})
    vals = [number(current[d][RAIN]) for d in dates if d in current and number(current[d][RAIN]) is not None]
    distributions.append({"year": "2011-2013_POOL", "expected_days": len(dates), "record_days": sum(d in current for d in dates), "numeric_count": len(vals), "blank_or_absent_count": len(dates)-len(vals), "mean_mm": statistics.mean(vals), "median_mm": statistics.median(vals), "std_sample_mm": statistics.stdev(vals), "max_mm": max(vals), "p90_mm": percentile(vals,.9), "p95_mm": percentile(vals,.95), "p99_mm": percentile(vals,.99), "numeric_subtotal_mm": round(sum(vals),6), "zero_count": sum(v==0 for v in vals), "note": "三年合并非空数值分布；不是完整逐日降水分布"})
    write("fqa_precip_distribution_2011_2013.csv", distributions, list(distributions[0]))

    event_summaries = []
    for day in (dt.date(2012,9,2), dt.date(2013,5,26)):
        rows = []
        windows = {"seven": [day+dt.timedelta(days=i) for i in range(-3,4)], "month": [d for d in dates if d.year==day.year and d.month==day.month]}
        coverage = {}
        for key, ds in windows.items():
            vv = [number(current[d][RAIN]) for d in ds if d in current and number(current[d][RAIN]) is not None]
            coverage[key] = {"numeric_days": len(vv), "expected_days": len(ds), "numeric_subtotal_mm": round(sum(vv),6), "complete_total_mm": round(sum(vv),6) if len(vv)==len(ds) else None}
        for role, data in datasets.items():
            for d in [day+dt.timedelta(days=i) for i in range(-3,4)]:
                item = data.get(d)
                row = {"source_role":role,"file":str(SOURCES[role].relative_to(ROOT)),"date":d.isoformat(),"offset_days":(d-day).days,"sheet":item["sheet"] if item else "","source_row":item["source_row"] if item else "","rain_field":RAIN,"rain_raw":item[RAIN] if item else "","rain_mm":number(item[RAIN]) if item else "","rain_status":status(item,RAIN)}
                for field in FIELDS:
                    row[field] = item[field] if item and field in item else ""
                a,b = (number(item.get(FIELDS[i])) if item else None for i in (0,1))
                total = number(item.get(RAIN)) if item else None
                row["halfday_closure_error_mm"] = round(total-a-b,6) if None not in (a,b,total) else ""
                row["current_7d_numeric_days"] = coverage["seven"]["numeric_days"]
                row["current_7d_numeric_subtotal_mm"] = coverage["seven"]["numeric_subtotal_mm"]
                row["current_7d_complete_total_mm"] = coverage["seven"]["complete_total_mm"] if coverage["seven"]["complete_total_mm"] is not None else ""
                row["current_month_numeric_days"] = coverage["month"]["numeric_days"]
                row["current_month_expected_days"] = coverage["month"]["expected_days"]
                row["current_month_numeric_subtotal_mm"] = coverage["month"]["numeric_subtotal_mm"]
                row["current_month_complete_total_mm"] = coverage["month"]["complete_total_mm"] if coverage["month"]["complete_total_mm"] is not None else ""
                rows.append(row)
        write(f"event_{day:%Y%m%d}_audit.csv", rows, list(rows[0]))
        event_summaries.append({"date":day.isoformat(),"fields":{field:current[day][field] for field in FIELDS},"previous":current.get(day-dt.timedelta(days=1),{}).get(RAIN),"next":current.get(day+dt.timedelta(days=1),{}).get(RAIN),"window":coverage})

    # Missingness is compared within the *same raw D222 row*, not against filled weather.
    missing_days = [d for d in dates if status(current.get(d),RAIN) != "NUMERIC"]
    runs = []
    for d in missing_days:
        if not runs or d != runs[-1][-1]+dt.timedelta(days=1):
            runs.append([d])
        else:
            runs[-1].append(d)
    run_lookup = {d:(r[0].isoformat(),r[-1].isoformat(),len(r)) for r in runs for d in r}
    missing_rows = []
    met_fields = ["小型蒸发皿20时蒸发量(mm)","E601蒸发皿20点蒸发量(mm)","8时能见度(km)","14时能见度(km)","20时能见度(km)"]
    for d in dates:
        item = current.get(d)
        if status(item,RAIN) == "NUMERIC":
            continue
        co = {f"{field}_status":status(item,field) for field in FIELDS+met_fields}
        missing_rows.append({"row_type":"MISSING_DATE","date":d.isoformat(),"year":d.year,"month":d.month,"rain_status":status(item,RAIN),"run_start":run_lookup[d][0],"run_end":run_lookup[d][1],"run_length_days":run_lookup[d][2],"other_meteorological_numeric_count":sum(status(item,f)=="NUMERIC" for f in met_fields),"other_meteorological_field_count":len(met_fields),**co})
    for year in (2011,2012,2013):
        for month in range(1,13):
            ds = [d for d in dates if d.year==year and d.month==month]
            miss = [d for d in ds if status(current.get(d),RAIN)!="NUMERIC"]
            missing_rows.append({"row_type":"MONTH_SUMMARY","year":year,"month":month,"rain_status":"SUMMARY","expected_days":len(ds),"rain_missing_days":len(miss),"rain_numeric_days":len(ds)-len(miss),"rain_missing_with_any_other_meteorological_numeric_days":sum(any(status(current.get(d),f)=="NUMERIC" for f in met_fields) for d in miss),"rain_missing_with_all_other_meteorological_blank_days":sum(all(status(current.get(d),f)!="NUMERIC" for f in met_fields) for d in miss),"rain_missing_runs_starting_in_month":sum(r[0].year==year and r[0].month==month for r in runs),"longest_missing_run_intersecting_month":max((len(r) for r in runs if any(d in ds for d in r)),default=0)})
    write("missing_pattern_analysis.csv", missing_rows, list(dict.fromkeys(k for r in missing_rows for k in r)))

    with (ROOT/"results/hl_fq_global_precip_023/fqa_2011_2013_sanity.csv").open(encoding="utf-8-sig",newline="") as f:
        ref={r["date"]:r for r in csv.DictReader(f) if r["row_type"]=="DAILY_2011_2013" and r["date"] in {"2012-09-02","2013-05-26"}}
    external=[]
    for e in event_summaries:
        r=ref[e["date"]]
        external.append({"date":e["date"],"d222_current_mm":number(current[dt.date.fromisoformat(e["date"])][RAIN]),"d222_field":RAIN,"gpcc_nearest_mm":r["gpcc_nearest_grid_precip_mm"],"gpcc_same_day_3x3_max_mm":r["gpcc_neighbor_max_mm"],"cpc_nearest_mm":r["cpc_nearest_grid_precip_mm"],"cpc_same_day_3x3_max_mm":r["cpc_neighbor_max_mm"],"reference_role":"REGIONAL_REFERENCE_ONLY; no station validation or replacement","source":"results/hl_fq_global_precip_023/fqa_2011_2013_sanity.csv"})
    write("external_reference_summary.csv",external,list(external[0]))

    # This machine summary makes exact report numbers inspectable without rerunning extraction.
    closure=Counter()
    for d in dates:
        item=current.get(d)
        if not item: continue
        a,b,t=(number(item.get(f)) for f in (FIELDS[0],FIELDS[1],RAIN))
        if None not in (a,b,t):
            closure["both_halves_and_total"]+=1
            if abs(a+b-t)<0.051: closure["both_halves_match_total"]+=1
        elif t is not None and (a is not None or b is not None):
            closure["total_with_one_half"]+=1
            if abs((a or 0)+(b or 0)-t)<0.051: closure["one_half_matches_total"]+=1
        elif t is None and a is None and b is None:
            closure["all_three_blank"]+=1
        else:
            closure["other"]+=1
    cross_day=[]
    cross_counts=Counter()
    for d in dates[:-1]:
        now,nxt=current.get(d),current.get(d+dt.timedelta(days=1))
        eight_eight=number(now.get(FIELDS[3])) if now else None
        daytime=number(now.get(FIELDS[1])) if now else None
        next_overnight=number(nxt.get(FIELDS[0])) if nxt else None
        if eight_eight is None:
            continue
        if daytime is not None and next_overnight is not None:
            case="BOTH_COMPONENTS"
        elif daytime is not None or next_overnight is not None:
            case="ONE_COMPONENT_ONLY"
        else:
            case="NO_COMPONENTS"
        residual=round(eight_eight-(daytime or 0)-(next_overnight or 0),6)
        cross_counts[case]+=1
        if abs(residual)<.051: cross_counts[case+"_MATCH"]+=1
        cross_day.append({"date":d.isoformat(),"8-8_current_date_mm":eight_eight,"8-20_current_date_mm":daytime if daytime is not None else "","20-8_next_date_mm":next_overnight if next_overnight is not None else "","component_case":case,"residual_mm_if_blank_component_zero_for_closure_only":residual,"interpretation":"internal field relation only; blank component is not imputed in source"})
    write("cross_day_field_closure.csv",cross_day,list(cross_day[0]))
    source_agreement={}
    for role in ("old_raw","legacy"):
        other=datasets[role]
        source_agreement[role]={"same_numeric_or_blank_days":sum(status(current.get(d),RAIN)==status(other.get(d),RAIN) and number(current[d][RAIN])==number(other[d][RAIN]) for d in dates),"expected_days":len(dates)}
    summary={"source_rows":{k:len(v) for k,v in datasets.items()},"source_agreement":source_agreement,"distribution":distributions,"events":event_summaries,"missing_run_count":len(runs),"longest_missing_run":{"start":max(runs,key=len)[0].isoformat(),"end":max(runs,key=len)[-1].isoformat(),"days":len(max(runs,key=len))},"rain_missing_days":len(missing_days),"rain_missing_with_any_other_met_numeric_days":sum(any(status(current.get(d),f)=="NUMERIC" for f in met_fields) for d in missing_days),"rain_missing_with_all_other_met_missing_days":sum(all(status(current.get(d),f)!="NUMERIC" for f in met_fields) for d in missing_days),"halfday_closure":dict(closure),"cross_day_8_8_closure":dict(cross_counts)}
    (OUT/"computed_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__":
    main()
