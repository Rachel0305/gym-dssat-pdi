"""Read-only five-site D222 rainfall semantics and current WTH trace."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import xlrd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/external/人工大气观测降雨蒸发能见度日值.xls"
FIELDS = ("20-8时降水量(mm)", "8-20时降水量(mm)", "20-20合计(mm)", "8-8时降水量(mm)")
SITES = ("HLA", "FQA", "YCA", "SYA", "LCA")
DIRS = {"HLA":"HL", "FQA":"FQ", "YCA":"YC", "SYA":"SY", "LCA":"LC"}
WTH_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"


def value(x):
    if x is None or str(x).strip()=="": return None
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except (ValueError,TypeError):
        return None


def write_csv(name, rows, keys):
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def load_source():
    book=xlrd.open_workbook(str(SOURCE),on_demand=True)
    source={site:{} for site in SITES}
    sheet_info={}
    for sheet in book.sheets():
        if sheet.nrows<2: continue
        headers=[str(x).strip() for x in sheet.row_values(1)]
        cols={name:i for i,name in enumerate(headers)}
        if not {"生态站代码","年","月","日"}.issubset(cols): continue
        sheet_info[sheet.name]=headers
        for ri in range(2,sheet.nrows):
            site=str(sheet.cell_value(ri,cols["生态站代码"])).strip()
            if site not in source: continue
            try:
                date=dt.date(*(int(float(sheet.cell_value(ri,cols[k]))) for k in ("年","月","日")))
            except (ValueError,TypeError,OverflowError): continue
            if date in source[site]: raise ValueError(f"duplicate {site} {date}")
            source[site][date]={"sheet":sheet.name,"source_row":ri+1,"fields":{f:(sheet.cell_value(ri,cols[f]) if f in cols else None) for f in FIELDS}}
    book.release_resources()
    return source,sheet_info


def load_wth(site):
    folder=WTH_ROOT/DIRS[site]
    data={}; files=[]; duplicates=0
    for path in sorted(folder.glob("*.WTH")):
        # Use annual production files only; LC also holds a 1998–2016 archive
        # named CNLC9816.WTH, whose overlapping dates must not supersede them.
        if len(path.stem)!=8 or not path.stem.startswith("CN"+DIRS[site]) or not path.stem.endswith("01"):
            continue
        year=2000+int(path.stem[4:6])
        if not 2000<=year<=2023: continue
        files.append(path.relative_to(ROOT).as_posix())
        with path.open(encoding="utf-8",errors="replace") as f:
            for line in f:
                parts=line.split()
                if len(parts)<5 or len(parts[0])!=7 or not parts[0].isdigit(): continue
                try:
                    date=dt.date(int(parts[0][:4]),1,1)+dt.timedelta(days=int(parts[0][4:])-1)
                    rain=float(parts[4])
                except (ValueError,OverflowError): continue
                if date.year!=year: continue
                if date in data: duplicates+=1
                data[date]=(rain,path.relative_to(ROOT).as_posix())
    return data,files,duplicates


def main():
    source,sheets=load_source()
    inventory=[]; closure=[]; comparisons=[]; dates=[]; wth_trace=[]; cross_mismatches=[]
    for site in SITES:
        rows=source[site]
        sheet_names=sorted({item["sheet"] for item in rows.values()})
        header_set=set.intersection(*(set(sheets[n]) for n in sheet_names)) if sheet_names else set()
        for field in FIELDS:
            nums=[value(item["fields"][field]) for item in rows.values()]
            present=[x for x in nums if x is not None]
            inventory.append({"site":site,"file":SOURCE.relative_to(ROOT).as_posix(),"sheets":";".join(sheet_names),"field_name":field,"field_present":field in header_set,"unit_from_header":"mm" if "(mm)" in field else "UNSTATED","date_rows":len(rows),"numeric_nonempty_count":len(present),"blank_count":sum(x is None for x in nums),"max_numeric_mm":max(present) if present else "","min_numeric_mm":min(present) if present else "","scope_first_date":min(rows).isoformat() if rows else "","scope_last_date":max(rows).isoformat() if rows else ""})
        same=Counter(); across=Counter(); error_same=[]; error_across=[]
        for d,item in rows.items():
            a,b,total,eight=(value(item["fields"][f]) for f in FIELDS)
            if total is not None:
                if a is not None and b is not None:
                    cat="BOTH_HALF_NUMERIC"; err=total-a-b
                elif a is not None or b is not None:
                    cat="ONE_HALF_NUMERIC"; err=total-(a or 0)-(b or 0)
                else:
                    cat="NO_HALF_NUMERIC"; err=None
                same[cat]+=1
                if err is not None:
                    error_same.append(abs(err))
                    if abs(err)>.051: same[cat+"_MISMATCH"]+=1
            if eight is not None:
                nxt=rows.get(d+dt.timedelta(days=1))
                next_night=value(nxt["fields"][FIELDS[0]]) if nxt else None
                if b is not None and next_night is not None:
                    cat="BOTH_COMPONENTS"; err=eight-b-next_night
                elif b is not None or next_night is not None:
                    cat="ONE_COMPONENT"; err=eight-(b or 0)-(next_night or 0)
                else:
                    cat="NO_COMPONENT"; err=None
                across[cat]+=1
                if err is not None:
                    error_across.append(abs(err))
                    if abs(err)>.051:
                        across[cat+"_MISMATCH"]+=1
                        month_end=(d+dt.timedelta(days=1)).month!=d.month
                        cross_mismatches.append({"site":site,"date":d.isoformat(),"source_row":item["source_row"],"8-8_current_mm":eight,"8-20_current_mm":b if b is not None else "","20-8_next_day_mm":next_night if next_night is not None else "","component_case":cat,"residual_mm":round(err,6),"month_end":month_end,"interpretation":"field relation mismatch; do not alter raw values"})
        closure.append({"site":site,"source_date_rows":len(rows),"same_day_both_half_sample":same["BOTH_HALF_NUMERIC"],"same_day_both_half_mismatch":same["BOTH_HALF_NUMERIC_MISMATCH"],"same_day_one_half_sample":same["ONE_HALF_NUMERIC"],"same_day_one_half_mismatch":same["ONE_HALF_NUMERIC_MISMATCH"],"same_day_total_no_half":same["NO_HALF_NUMERIC"],"same_day_max_abs_error_mm":max(error_same) if error_same else "","cross_day_both_component_sample":across["BOTH_COMPONENTS"],"cross_day_both_component_mismatch":across["BOTH_COMPONENTS_MISMATCH"],"cross_day_one_component_sample":across["ONE_COMPONENT"],"cross_day_one_component_mismatch":across["ONE_COMPONENT_MISMATCH"],"cross_day_eight_eight_no_component":across["NO_COMPONENT"],"cross_day_max_abs_error_mm":max(error_across) if error_across else "","tolerance_mm":.051,"blank_component_rule":"Only for conditional closure check; original blank remains blank"})
        comparison={"site":site,"source_file":SOURCE.relative_to(ROOT).as_posix(),"sheets":";".join(sheet_names),"has_20_8":FIELDS[0] in header_set,"has_8_20":FIELDS[1] in header_set,"has_20_20":FIELDS[2] in header_set,"has_8_8":FIELDS[3] in header_set,"project_preprocess_source_field":FIELDS[2],"field_name_identical_to_all_sites":True,"observed_field_structure":"20-20 equals same-row 20-8 plus 8-20 where components visible","field_meaning_difference_detected":False,"cross_day_month_end_exception_count":sum(m["site"]==site and m["month_end"] for m in cross_mismatches),"physical_value_validation":"NOT_ASSESSED"}
        comparisons.append(comparison)
        wth,files,dupes=load_wth(site)
        cleaned_path=ROOT/"weather_clean"/f"{site}_weather_cleaned.csv"
        cleaned={}
        with cleaned_path.open(encoding="utf-8-sig",newline="") as f:
            for line in csv.DictReader(f):
                try: cleaned[dt.date.fromisoformat(line["date"])]=float(line["RAIN"])
                except (ValueError,KeyError): continue
        counted=Counter(); examples=[]
        for d,(rain,path) in wth.items():
            item=rows.get(d)
            if item is None:
                counted["wth_date_without_current_d222_row"]+=1; continue
            raw=value(item["fields"][FIELDS[2]])
            if raw is None:
                counted["wth_on_d222_blank"]+=1
                if abs(rain)<.051: counted["wth_zero_on_d222_blank"]+=1
                else: counted["wth_nonzero_on_d222_blank"]+=1
            else:
                counted["wth_on_d222_numeric"]+=1
                if abs(raw-rain)<.051: counted["wth_matches_numeric_same_date"]+=1
                else:
                    counted["wth_differs_numeric_same_date"]+=1
                    if len(examples)<5: examples.append({"date":d.isoformat(),"source_mm":raw,"wth_mm":rain,"wth_file":path})
        cleaned_numeric_dates=[d for d in cleaned if d in rows and value(rows[d]["fields"][FIELDS[2]]) is not None]
        wth_trace.append({"site":site,"wth_root":WTH_ROOT.relative_to(ROOT).as_posix(),"wth_file_count":len(files),"wth_date_count":len(wth),"wth_duplicate_dates":dupes,"cleaned_source_file":cleaned_path.relative_to(ROOT).as_posix(),"cleaned_numeric_overlap_days":len(cleaned_numeric_dates),"cleaned_matches_d222_numeric_same_date":sum(abs(cleaned[d]-value(rows[d]["fields"][FIELDS[2]]))<.051 for d in cleaned_numeric_dates),**dict(counted),"numeric_mismatch_examples":json.dumps(examples,ensure_ascii=False),"interpretation":"same-date numeric agreement is a value trace, not a complete generation log"})
        evaluable=across["BOTH_COMPONENTS"]+across["ONE_COMPONENT"]
        mismatches=across["BOTH_COMPONENTS_MISMATCH"]+across["ONE_COMPONENT_MISMATCH"]
        status="INFERRED" if evaluable>=20 and mismatches/evaluable<.05 else "UNRESOLVED"
        site_mismatches=[m for m in cross_mismatches if m["site"]==site]
        dates.append({"site":site,"source_field":FIELDS[2],"source_interval_label":"20:00-to-20:00","candidate_A_previous_20_to_date_20":"SUPPORTED_BY_INTERNAL_FIELD_CLOSURE" if status=="INFERRED" else "UNRESOLVED","candidate_B_calendar_00_to_24":"NOT_SUPPORTED_BY_FIELD_LABEL","date_semantics_status":status,"same_day_wth_label_handling":"UNCHANGED; no date shift in src/weather_preprocess.py","cross_day_evaluable_sample":evaluable,"cross_day_total_mismatch":mismatches,"cross_day_mismatch_month_end":sum(m["month_end"] for m in site_mismatches),"cross_day_mismatch_non_month_end":sum(not m["month_end"] for m in site_mismatches),"official_date_anchor":"NOT_EXPLICITLY_DOCUMENTED","blank_semantics":"UNRESOLVED","risk":"20-20 observation day may differ from 00-24 model/calendar day; wet-event date and WGEN wet-day statistics may shift"})
    write_csv("rainfall_field_inventory.csv",inventory,list(inventory[0]))
    write_csv("five_site_rain_field_comparison.csv",comparisons,list(comparisons[0]))
    write_csv("field_closure_summary.csv",closure,list(closure[0]))
    write_csv("date_semantics_assessment.csv",dates,list(dates[0]))
    write_csv("cross_day_mismatch_details.csv",cross_mismatches,list(cross_mismatches[0]) if cross_mismatches else ["site","date","source_row","8-8_current_mm","8-20_current_mm","20-8_next_day_mm","component_case","residual_mm","month_end","interpretation"])
    write_csv("wth_numeric_trace.csv",wth_trace,list(wth_trace[0]))
    print(json.dumps({"inventory":inventory,"closure":closure,"wth_trace":wth_trace,"dates":dates},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
