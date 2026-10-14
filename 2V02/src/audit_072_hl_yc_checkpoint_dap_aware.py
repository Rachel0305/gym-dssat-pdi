#!/usr/bin/env python3
"""Read-only DAP-aware audit for HL/YC existing daily outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


REQUIRED = {"year", "dap", "safe_action_amir", "safe_action_anfer"}


def entropy(values):
    c = Counter(values); n = sum(c.values())
    if not n: return 0.0, 0.0
    h = -sum((v/n) * math.log(v/n) for v in c.values())
    return h, h / math.log(len(c)) if len(c) > 1 else 0.0


def hash_events(events):
    raw = "|".join(f"{d}:{i:g}:{n:g}" for d, i, n in events).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def inspect(path: Path, year: int):
    if not path.exists(): return {"path": str(path), "exists": False, "error": "missing"}
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f); fields = set(r.fieldnames or []); rows = list(r)
    missing = sorted(REQUIRED - fields)
    if missing: return {"path": str(path), "exists": True, "error": "missing_columns", "missing_columns": ";".join(missing)}
    events=[]; pairs=[]; post=0; mismatches={k:0 for k in ["off_grid_rows","requested_to_safe_mismatch_rows","raw_to_safe_mismatch_rows","safe_to_dssat_mismatch_rows"]}
    for row in rows:
        d=int(float(row["dap"])); i=float(row["safe_action_amir"]); n=float(row["safe_action_anfer"])
        if abs(i)>1e-9 or abs(n)>1e-9:
            events.append((d,i,n)); pairs.append(f"I{i:g}/N{n:g}"); post += int(d>1)
        if "requested_discrete_action_index" in row and row.get("requested_discrete_action_index") not in ("", row.get("discrete_action_index")):
            mismatches["requested_to_safe_mismatch_rows"] += 1
        if "raw_action_amir" in row and (abs(float(row.get("raw_action_amir") or 0)-i)>1e-6 or abs(float(row.get("raw_action_anfer") or 0)-n)>1e-6):
            mismatches["raw_to_safe_mismatch_rows"] += 1
        if "season_cumulative_irrigation" in row and abs(float(row.get("season_cumulative_irrigation") or 0)-float(row.get("previous_cumulative_irrigation") or 0)-i)>1e-6:
            mismatches["safe_to_dssat_mismatch_rows"] += 1
        if "season_cumulative_n" in row and abs(float(row.get("season_cumulative_n") or 0)-float(row.get("previous_cumulative_n") or 0)-n)>1e-6:
            mismatches["safe_to_dssat_mismatch_rows"] += 1
        if i not in (0.0,15.0,30.0,45.0) or n not in (0.0,40.0,80.0,120.0): mismatches["off_grid_rows"] += 1
    h, hn = entropy(pairs)
    return {"path":str(path),"exists":True,"error":None,"year":year,"rows":len(rows),"positive_action_rows":len(events),"positive_actions_after_dap1":post,"unique_nonzero_action_pairs":len(set(pairs)),"nonzero_action_pairs":";".join(sorted(set(pairs))),"pair_entropy_nats":h,"pair_entropy_normalized":hn,"trajectory_signature":hash_events(events),**mismatches}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/072_hl_yc_checkpoint_audit.json"); p.add_argument("--output-root",required=True); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8")); out=Path(a.output_root)
    if a.dry_run: print(json.dumps({"status":"dry_run","config":a.config,"output_root":a.output_root},indent=2)); return
    if out.exists() and any(out.iterdir()): raise SystemExit(f"refusing non-empty output root: {out}")
    out.mkdir(parents=True,exist_ok=True); details=[]; summaries=[]
    for site, spec in cfg["sites"].items():
        station=spec["station_code"]
        for root_spec in spec["roots"]:
            root=Path(root_spec["root"]); daily=root/"daily_outputs"/station
            files=sorted(daily.glob(f"{station}_*_seed*_ckpt*_daily.csv")) if daily.exists() else []
            for path in files:
                m=re.match(rf"{station}_(\d+)_seed(\d+)_ckpt(\d+)_daily\.csv$",path.name)
                if not m: continue
                year,seed,ckpt=map(int,m.groups()); row=inspect(path,year); row.update({"site":site,"label":root_spec["label"],"seed":seed,"checkpoint":ckpt}); details.append(row)
    detail_fields=sorted({k for r in details for k in r})
    with (out/"hl_yc_checkpoint_detail.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=detail_fields); w.writeheader(); w.writerows(details)
    groups={}
    for r in details:
        if r.get("error") is not None: continue
        key=(r["site"],r["label"],r["seed"],r["checkpoint"]); groups.setdefault(key,[]).append(r)
    for key, rows in sorted(groups.items()):
        summaries.append({"site":key[0],"label":key[1],"seed":key[2],"checkpoint":key[3],"years_with_daily":len(rows),"pair_set_signature_count":len({r["nonzero_action_pairs"] for r in rows}),"dap_aware_trajectory_signature_count":len({r["trajectory_signature"] for r in rows}),"mean_post_dap1":sum(r["positive_actions_after_dap1"] for r in rows)/len(rows),"mean_pair_entropy_nats":sum(r["pair_entropy_nats"] for r in rows)/len(rows),"max_mismatch":max(sum(r[k] for k in ["off_grid_rows","requested_to_safe_mismatch_rows","raw_to_safe_mismatch_rows","safe_to_dssat_mismatch_rows"]) for r in rows)})
    with (out/"hl_yc_checkpoint_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]) if summaries else ["site","label"]); w.writeheader(); w.writerows(summaries)
    manifest={"task":cfg["task"],"status":"completed_read_only","detail_csv":str(out/"hl_yc_checkpoint_detail.csv"),"summary_csv":str(out/"hl_yc_checkpoint_summary.csv"),"daily_files":len(details),"error_rows":sum(r.get("error") is not None for r in details),"no_training":True,"no_dssat":True,"no_replay":True}
    (out/"072_audit_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
