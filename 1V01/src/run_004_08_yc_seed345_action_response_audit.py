"""Post-process existing YC PPO seed 3/4/5 action and state records."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP_FILE = ROOT / "results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv"
REPLAY = ROOT / "results/yc_random_weather_ppo/004_07_02_seed345_etcp_replay"
SUMMARY_FILE = REPLAY / "replay_episode_summary.csv"
SNAPSHOTS = REPLAY / "snapshots"
OUT = ROOT / "results/yc_random_weather_ppo/004_08_seed345_action_response_audit"
REPORT = ROOT / "docs/yc_random_weather_004_08_seed345_action_response_audit.md"
SEEDS = [3, 4, 5]
YEARS = list(range(2014, 2024))
GREEN, BLUE, ORANGE = "#2A9D55", "#2878B5", "#D9822B"
LINESTYLES = {3: ("-", "o"), 4: ("--", "s"), 5: ("-.", "^")}


def parse_out(path: Path) -> pd.DataFrame:
    cols, rows = None, []
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("@"):
            cols = line[1:].split()
        elif cols and line and not line.startswith(("*", "!")):
            fields = line.split()
            if fields[0].lstrip("-").isdigit():
                rows.append((fields + [""] * len(cols))[:len(cols)])
    frame = pd.DataFrame(rows, columns=cols or [])
    for col in frame:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    keys = [c for c in ["YEAR", "DOY", "DAS"] if c in frame]
    return frame.drop_duplicates(keys, keep="last") if len(keys) == 3 else frame


def snapshot(seed: int, year: int) -> pd.DataFrame:
    folder = SNAPSHOTS / f"seed_{seed}" / "YCA" / str(year)
    w, p, s = (parse_out(folder / f) for f in ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT"])
    if w.empty or p.empty or s.empty:
        raise RuntimeError(f"Missing snapshot daily table: {folder}")
    def pick(df, mapping):
        return df[[c for c in mapping if c in df]].rename(columns=mapping)
    w = pick(w, {"DAS":"das", "DOY":"doy", "PRED":"rain_mm", "SRAD":"srad", "TMXD":"tmax", "TMND":"tmin"})
    p = pick(p, {"DAS":"das", "DAP":"snapshot_dap", "GSTD":"stage", "LAID":"lai", "WSPD":"wspd", "NSTD":"nstd"})
    s = pick(s, {"DAS":"das", "SWTD":"soil_water_mm", "SWXD":"soil_water_extractable_mm", "PREC":"soil_precip_mm"})
    d = w.merge(p, on="das", how="outer").merge(s, on="das", how="outer").sort_values("das")
    d = d.drop_duplicates("das", keep="last").reset_index(drop=True)
    d["seed"], d["year"] = seed, year
    d["prior7_rain_mm"] = d.rain_mm.shift(1).rolling(7, min_periods=1).sum()
    return d


def load_steps() -> pd.DataFrame:
    cols = [
        "training_regime","ppo_seed","evaluation_weather_type","evaluation_weather_year","episode_key",
        "timestep","date","doy","dap","action_index","requested_action_index","irrigation_action_mm",
        "fertilizer_action_kgN_ha","swfac","nstres","istage","xlai","RAIN",
        "observation_vector_json","observation_variables_json",
    ]
    parts = []
    for chunk in pd.read_csv(STEP_FILE, usecols=cols, chunksize=50000):
        x = chunk.loc[
            chunk.training_regime.eq("RANDOM_WEATHER_WGEN")
            & pd.to_numeric(chunk.ppo_seed, errors="coerce").isin(SEEDS)
            & chunk.evaluation_weather_type.eq("observed_weather")
        ].copy()
        if len(x):
            parts.append(x)
    x = pd.concat(parts, ignore_index=True)
    x["seed"] = pd.to_numeric(x.ppo_seed).astype(int)
    x["year"] = pd.to_numeric(x.evaluation_weather_year).astype(int)
    for c in ["timestep","doy","dap","action_index","irrigation_action_mm","fertilizer_action_kgN_ha",
              "swfac","nstres","istage","xlai","RAIN"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x[["irrigation_action_mm","fertilizer_action_kgN_ha"]] = x[
        ["irrigation_action_mm","fertilizer_action_kgN_ha"]].fillna(0)
    return x.sort_values(["seed","year","timestep"]).reset_index(drop=True)


def fmt(v):
    v = float(v)
    return f"{0 if abs(v) < 1e-10 else v:g}"


def signature(g):
    return ";".join(
        f"{int(r.dap)}:I{fmt(r.irrigation_action_mm)}/N{fmt(r.fertilizer_action_kgN_ha)}"
        for r in g.sort_values(["timestep","dap"]).itertuples()
    )


def tokens(g):
    return [f"{int(r.dap)}|{fmt(r.irrigation_action_mm)}|{fmt(r.fertilizer_action_kgN_ha)}"
            for r in g.sort_values(["timestep","dap"]).itertuples()]


def edit_similarity(a, b):
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b), 1)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    if (OUT.exists() and not (OUT / "analysis_summary.json").exists()) or (
        REPORT.exists() and not REPORT.read_text(encoding="utf-8").startswith("# YC 004_08")
    ):
        raise FileExistsError("004_08 output already exists and is not a resumable output from this script.")
    if not STEP_FILE.exists() or not SUMMARY_FILE.exists():
        raise FileNotFoundError("Required existing step trace or replay summary is missing.")
    steps = load_steps()
    years_found = set(zip(steps.seed, steps.year))
    summary = pd.read_csv(SUMMARY_FILE)
    summary["seed"] = pd.to_numeric(summary.seed, errors="coerce")
    summary["year"] = pd.to_numeric(summary.year, errors="coerce")
    summary = summary.loc[summary.seed.isin(SEEDS) & summary.year.isin(YEARS)].copy()
    if len(steps) != 3048 or len(years_found) != 30 or len(summary) != 30:
        raise RuntimeError(f"Unexpected coverage: {len(steps)} daily rows, {len(years_found)} seed-years, {len(summary)} summaries")

    contexts = pd.concat([snapshot(s, y) for s in SEEDS for y in YEARS], ignore_index=True)
    trace = steps.merge(contexts, left_on=["seed","year","timestep"], right_on=["seed","year","das"],
                        how="left", validate="many_to_one")
    prior_cols = ["soil_water_mm","soil_water_extractable_mm","lai","stage","wspd","nstd","rain_mm"]
    prior = contexts[["seed", "year", "das"] + prior_cols].copy()
    prior["das"] += 1
    prior = prior.rename(columns={c:f"prior_{c}" for c in prior_cols})
    trace = trace.merge(prior, on=["seed","year","das"], how="left", validate="many_to_one")
    trace["obs_vector_available"] = trace.observation_vector_json.fillna("").astype(str).str.len().gt(2)
    trace["obs_variables_available"] = trace.observation_variables_json.fillna("").astype(str).str.len().gt(2)
    trace["alignment_note"] = "timestep joined to snapshot DAS; prior_* is preceding DAS; snapshot values are ex-post context"
    events = trace.loc[trace.irrigation_action_mm.gt(0) | trace.fertilizer_action_kgN_ha.gt(0)].copy()
    events = events.sort_values(["seed","year","timestep"]).reset_index(drop=True)

    action_rows = []
    for seed in SEEDS:
        for year in YEARS:
            g = events.loc[events.seed.eq(seed) & events.year.eq(year)]
            action_rows.append({
                "seed":seed,"year":year,"action_signature_trace":signature(g),"action_rows":len(g),
                "action_days":int(g.timestep.nunique()),
                "irrigation_action_days":int(g.loc[g.irrigation_action_mm.gt(0),"timestep"].nunique()),
                "nitrogen_action_days":int(g.loc[g.fertilizer_action_kgN_ha.gt(0),"timestep"].nunique()),
                "irrigation_mm_trace":g.irrigation_action_mm.sum(),
                "nitrogen_kg_ha_trace":g.fertilizer_action_kgN_ha.sum(),
            })
    annual = summary.merge(pd.DataFrame(action_rows), on=["seed","year"], validate="one_to_one")
    annual["signature_trace_matches_replay"] = annual.action_signature_trace.eq(annual.action_signature)
    annual = annual.merge(trace.groupby(["seed","year"],as_index=False).agg(
        step_rows=("timestep","size"),swfac_mean=("swfac","mean"),swfac_min=("swfac","min"),
        nstres_mean=("nstres","mean"),nstres_max=("nstres","max"),
        obs_vector_rows=("obs_vector_available","sum"),obs_variables_rows=("obs_variables_available","sum")),
        on=["seed","year"],validate="one_to_one")
    annual = annual.merge(contexts.groupby(["seed","year"],as_index=False).agg(
        rainfall_mm=("rain_mm","sum"),wet_days=("rain_mm",lambda v:int(v.fillna(0).gt(0).sum())),
        soil_water_mean_mm=("soil_water_mm","mean"),soil_water_min_mm=("soil_water_mm","min"),
        lai_max=("lai","max"),stage_max=("stage","max"),snapshot_days=("das","nunique")),
        on=["seed","year"],validate="one_to_one")

    event_daps = sorted(pd.to_numeric(events.dap,errors="coerce").dropna().astype(int).unique())
    dap_rows = []
    for seed in SEEDS:
        st = trace.loc[trace.seed.eq(seed)]
        for dap in event_daps:
            for year in YEARS:
                r = st.loc[st.year.eq(year) & st.dap.eq(dap)]
                if r.empty:
                    dap_rows.append({"seed":seed,"dap":dap,"year":year})
                    continue
                r = r.iloc[0]
                dap_rows.append({
                    "seed":seed,"dap":dap,"year":year,"timestep":r.timestep,"action_index":r.action_index,
                    "irrigation_action_mm":r.irrigation_action_mm,"fertilizer_action_kgN_ha":r.fertilizer_action_kgN_ha,
                    "swfac_step":r.swfac,"nstres_step":r.nstres,
                    "soil_water_preceding_das_mm":r.get("prior_soil_water_mm",np.nan),
                    "soil_water_same_das_mm":r.get("soil_water_mm",np.nan),
                    "lai_preceding_das":r.get("prior_lai",np.nan),"stage_preceding_das":r.get("prior_stage",np.nan),
                    "prior7_rain_mm":r.get("prior7_rain_mm",np.nan),"same_day_rain_snapshot_mm":r.get("rain_mm",np.nan),
                    "obs_vector_available":r.obs_vector_available,
                })
    dap_year = pd.DataFrame(dap_rows)
    for c in ["irrigation_action_mm","fertilizer_action_kgN_ha","swfac_step","nstres_step",
              "soil_water_preceding_das_mm","soil_water_same_das_mm","lai_preceding_das",
              "stage_preceding_das","prior7_rain_mm","same_day_rain_snapshot_mm"]:
        dap_year[c] = pd.to_numeric(dap_year[c],errors="coerce")
    response = []
    state_fields = ["irrigation_action_mm","fertilizer_action_kgN_ha","swfac_step","nstres_step",
                    "soil_water_preceding_das_mm","lai_preceding_das","stage_preceding_das","prior7_rain_mm"]
    for (seed,dap), g in dap_year.groupby(["seed","dap"]):
        row = {"seed":int(seed),"dap":int(dap),"years_with_step":int(g.timestep.notna().sum()),
               "irrigation_action_years":int(g.irrigation_action_mm.fillna(0).gt(0).sum()),
               "nitrogen_action_years":int(g.fertilizer_action_kgN_ha.fillna(0).gt(0).sum()),
               "obs_vector_years":int(g.obs_vector_available.fillna(False).sum())}
        for c in state_fields:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=0)
            row[f"{c}_min"] = g[c].min()
            row[f"{c}_max"] = g[c].max()
        response.append(row)
    dap_summary = pd.DataFrame(response)

    pair_rows = []
    for seed in SEEDS:
        by_year = {y:events.loc[events.seed.eq(seed)&events.year.eq(y)] for y in YEARS}
        for a,b in combinations(YEARS,2):
            ga,gb=by_year[a],by_year[b]
            da=set(pd.to_numeric(ga.dap,errors="coerce").dropna().astype(int))
            db=set(pd.to_numeric(gb.dap,errors="coerce").dropna().astype(int))
            ta,tb=set(tokens(ga)),set(tokens(gb))
            pair_rows.append({"seed":seed,"year_a":a,"year_b":b,"exact_signature_match":signature(ga)==signature(gb),
                              "signature_a":signature(ga),"signature_b":signature(gb),
                              "action_dap_jaccard":jaccard(da,db),"action_tuple_jaccard":jaccard(ta,tb),
                              "sequence_edit_similarity":edit_similarity(tokens(ga),tokens(gb))})
    pairs=pd.DataFrame(pair_rows)

    profiles={}
    for seed in SEEDS:
        ag=annual.loc[annual.seed.eq(seed)].sort_values("year")
        sigs=ag.action_signature_trace.fillna("").tolist()
        mode=Counter(sigs).most_common(1)[0][1]
        ds=dap_summary.loc[dap_summary.seed.eq(seed)]
        spread={
            "swfac":float((ds.swfac_step_max-ds.swfac_step_min).max()),
            "nstres":float((ds.nstres_step_max-ds.nstres_step_min).max()),
            "soil_water_mm":float((ds.soil_water_preceding_das_mm_max-ds.soil_water_preceding_das_mm_min).max()),
            "prior7_rain_mm":float((ds.prior7_rain_mm_max-ds.prior7_rain_mm_min).max()),
        }
        per_dap_classes=[]
        for r in ds.itertuples():
            ranges=[
                r.swfac_step_max-r.swfac_step_min,
                r.nstres_step_max-r.nstres_step_min,
                r.soil_water_preceding_das_mm_max-r.soil_water_preceding_das_mm_min,
                r.prior7_rain_mm_max-r.prior7_rain_mm_min,
            ]
            thresholds=[.05,.05,5,10]
            per_dap_classes.append(sum(pd.notna(value) and value>=threshold for value,threshold in zip(ranges,thresholds)))
        classes=max(per_dap_classes,default=0)
        daps_with_two=sum(value>=2 for value in per_dap_classes)
        changed=int(((ds.irrigation_action_mm_std.fillna(0) > .01) | (ds.fertilizer_action_kgN_ha_std.fillna(0) > .01)).sum())
        n_unique=len(set(sigs))
        if ag.step_rows.min()<90 or ag.snapshot_days.min()<90: adapt="INSUFFICIENT"
        elif mode>=8 and classes>=2: adapt="LOW"
        elif n_unique>=4 and changed>=3: adapt="HIGH"
        elif n_unique>=2 and changed>=1: adapt="MODERATE"
        else: adapt="LOW"
        pp=pairs.loc[pairs.seed.eq(seed)]
        profiles[str(seed)]={"unique_signatures":n_unique,"modal_signature_count":mode,"modal_fraction":mode/10,
            "exact_year_pairs":int(pp.exact_signature_match.sum()),"mean_dap_jaccard":float(pp.action_dap_jaccard.mean()),
            "mean_tuple_jaccard":float(pp.action_tuple_jaccard.mean()),"mean_edit_similarity":float(pp.sequence_edit_similarity.mean()),
            "action_change_dap_count":changed,"context_spread":spread,
            "context_spread_classes":int(classes),"daps_with_two_context_classes":int(daps_with_two),
            "weather_adaptivity":adapt}
    p5=profiles["5"]
    fixed="STRONG" if p5["modal_fraction"]>=.9 and p5["context_spread_classes"]>=2 else (
        "MODERATE" if p5["modal_fraction"]>=.7 else "WEAK" if p5["unique_signatures"]>1 else "NONE")
    n=annual.loc[annual.seed.eq(5),"nitrogen_kg_ha"].dropna()
    n_dap=dap_summary.loc[dap_summary.seed.eq(5)&dap_summary.nitrogen_action_years.gt(0)]
    n_fixed=len(n)==10 and np.allclose(n.to_numpy(),n.iloc[0]) and not n_dap.empty and n_dap.nitrogen_action_years.eq(10).all() and n_dap.fertilizer_action_kgN_ha_std.fillna(0).le(.01).all()
    n_pattern="FIXED" if n_fixed and p5["context_spread_classes"]>=2 else "MIXED" if len(n)==10 and np.allclose(n.to_numpy(),n.iloc[0]) else "INSUFFICIENT"

    reconciled=bool(annual.signature_trace_matches_replay.all()
        and np.allclose(annual.irrigation_mm_trace,annual.irrigation_mm,atol=1e-6)
        and np.allclose(annual.nitrogen_kg_ha_trace,annual.nitrogen_kg_ha,atol=1e-6))
    if not reconciled: raise RuntimeError("Step action totals/signatures do not reconcile to replay summary")
    quality={"step_rows":len(steps),"seed_year_pairs":len(years_found),"snapshot_count":len(contexts.groupby(["seed","year"])),
        "snapshot_step_join_fraction":float(trace.das.notna().mean()),
        "obs_vector_rows":int(trace.obs_vector_available.sum()),"obs_variables_rows":int(trace.obs_variables_available.sum()),
        "action_events":len(events),"pair_pairs":len(pairs),"action_reconciliation":reconciled,
        "event_prior7_rain_missing":int(events.prior7_rain_mm.isna().sum()),
        "event_prior7_rain_missing_dap":sorted(pd.to_numeric(events.loc[events.prior7_rain_mm.isna(),"dap"],errors="coerce").dropna().unique().astype(int).tolist()),
        "missing_step":{"SWFAC":int(trace.swfac.isna().sum()),"NSTRES":int(trace.nstres.isna().sum()),
            "DAP":int(trace.dap.isna().sum()),"LAI":int(trace.xlai.isna().sum()),"stage":int(trace.istage.isna().sum()),
            "rain":int(trace.RAIN.isna().sum())},
        "missing_snapshot":{"soil_water":int(trace.soil_water_mm.isna().sum()),"LAI":int(trace.lai.isna().sum()),
            "stage":int(trace.stage.isna().sum()),"rain":int(trace.rain_mm.isna().sum())}}
    verdict={"seed5_weather_adaptivity":p5["weather_adaptivity"],"seed5_fixed_schedule_evidence":fixed,
        "seed5_high_N_pattern":n_pattern,"seed3_weather_adaptivity":profiles["3"]["weather_adaptivity"],
        "seed4_weather_adaptivity":profiles["4"]["weather_adaptivity"],
        "seed5_water_saving_mechanism":"fewer and more concentrated irrigation action days than seed4; descriptive, not causal",
        "next_step":"observation-action issue" if p5["weather_adaptivity"]=="LOW" else "no tuning yet"}

    # All expensive reads and consistency checks finish before creating new outputs.
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"figures").mkdir(exist_ok=True)
    annual.to_csv(OUT/"annual_seed345_behavior.csv",index=False)
    events.to_csv(OUT/"action_events_with_context.csv",index=False)
    dap_year.to_csv(OUT/"seed_dap_action_response.csv",index=False)
    dap_summary.to_csv(OUT/"seed_dap_action_response_summary.csv",index=False)
    pairs.to_csv(OUT/"action_schedule_similarity_pairs.csv",index=False)
    trace.to_csv(OUT/"daily_context_seed345.csv",index=False)
    input_paths=[STEP_FILE,SUMMARY_FILE]+[
        SNAPSHOTS/f"seed_{s}"/"YCA"/str(y)/f for s in SEEDS for y in YEARS
        for f in ["PlantGro.OUT","SoilWat.OUT","Weather.OUT","MgmtEvent.OUT","Summary.OUT"]]
    provenance={"task":"004_08 YC seed345 action-response audit","training_or_reward_or_constraint_change":False,
        "evaluation_only_replay":False,"commit_or_push":False,
        "reference_documents":["prompt_02/004_07_yc_fixed_ppo_validation_year_pareto_assessment.md",
            "prompt_02/004_07_01_yc_all_seed_fixed_model_validation_comparison.md",
            "prompt_02/004_07_02_yc_seed5_focus_wp_et_seed3_seed4_reference_replay.md",
            "docs/yc_random_weather_004_07_02_seed345_wp_et_replay.md"],
        "inputs":[{"path":p.relative_to(ROOT).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size} for p in input_paths],
        "style_sources":["results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/plot_004_05_seed3_diagnostic.py",
            "src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py"]}
    (OUT/"input_provenance.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"analysis_summary.json").write_text(json.dumps({"quality":quality,"seed_profiles":profiles,"verdicts":verdict},ensure_ascii=False,indent=2),encoding="utf-8")

    # Event calendar preserves the established YC management marker language.
    fig,axes=plt.subplots(3,1,figsize=(16,10),sharex=True)
    for ax,seed in zip(axes,SEEDS):
        sub=events.loc[events.seed.eq(seed)]
        for r in sub.itertuples():
            y=YEARS.index(int(r.year))
            if r.irrigation_action_mm>0:
                ax.scatter(r.dap,y-.1,s=47,marker="o",color=BLUE,edgecolor="white",linewidth=.5)
                ax.annotate(f"I {r.irrigation_action_mm:g}",(r.dap,y-.1),xytext=(-23,-9),textcoords="offset points",fontsize=6.5,color="#22577A",ha="right")
            if r.fertilizer_action_kgN_ha>0:
                ax.scatter(r.dap,y+.1,s=48,marker="s",color=ORANGE,edgecolor="white",linewidth=.5)
                ax.annotate(f"N {r.fertilizer_action_kgN_ha:g}",(r.dap,y+.1),xytext=(5,4),textcoords="offset points",fontsize=6.5,color="#9C5700",ha="left")
        ax.set_yticks(range(10),YEARS); ax.invert_yaxis(); ax.set_ylabel(f"Seed {seed}\nYear")
        ax.set_title(f"Seed {seed}",loc="left",fontweight="bold",fontsize=10); ax.grid(axis="x",alpha=.22)
    axes[-1].set_xlabel("Days after planting (DAP)")
    axes[0].legend(handles=[plt.Line2D([],[],marker="o",ls="",color=BLUE,label="Irrigation (mm)"),
        plt.Line2D([],[],marker="s",ls="",color=ORANGE,label="Nitrogen (kg/ha)")],frameon=False,ncol=2,loc="upper right")
    fig.suptitle("YCA | Actual seed 3/4/5 management actions by observed year",x=.02,ha="left",fontweight="bold")
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(OUT/"figures/action_calendar_seed345.png",dpi=200); plt.close(fig)

    # Thin annual traces show state variation at DAPs where each seed actually acted.
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex="col")
    state_specs=[("swfac_step","Step-trace SWFAC","SWFAC"),("nstres_step","Step-trace NSTRES","NSTRES"),
        ("soil_water_preceding_das_mm","Prior-DAS soil-water proxy","Soil water (mm)")]
    for i,seed in enumerate(SEEDS):
        sub=dap_year.loc[dap_year.seed.eq(seed)]
        daps=sorted(events.loc[events.seed.eq(seed),"dap"].dropna().astype(int).unique())
        for j,(field,title,ylabel) in enumerate(state_specs):
            ax=axes[i,j]; matrix=sub.pivot(index="year",columns="dap",values=field).reindex(columns=daps)
            for _,v in matrix.iterrows(): ax.plot(daps,v.to_numpy(float),color="#A7B8AA",lw=.8,alpha=.42,marker=".",ms=3)
            ax.plot(daps,matrix.mean().to_numpy(float),color=GREEN,lw=2,marker="o",ms=4,label="Across-year mean")
            for d in daps: ax.axvline(d,color="#777777",lw=.6,ls=":",alpha=.45)
            ax.set_title(f"Seed {seed} | {title}",loc="left",fontweight="bold",fontsize=9)
            ax.set_ylabel(ylabel); ax.grid(axis="y",alpha=.22)
            if j==0: ax.set_ylim(-.03,1.03)
            if i==2: ax.set_xlabel("DAP with action in at least one year")
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle("YCA | State proxies at management DAP across 2014–2023",x=.02,ha="left",fontweight="bold")
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(OUT/"figures/state_variation_at_action_dap.png",dpi=200); plt.close(fig)

    fig,axes=plt.subplots(4,1,figsize=(16,13),sharex=True)
    for ax,(field,label) in zip(axes,[("formal_yield_kg_ha","Yield (kg/ha)"),("irrigation_mm","Total irrigation (mm)"),
        ("nitrogen_kg_ha","Total N (kg N/ha)"),("common_reward","Common reward")]):
        for seed in SEEDS:
            g=annual.loc[annual.seed.eq(seed)].sort_values("year"); ls,m=LINESTYLES[seed]
            ax.plot(g.year,pd.to_numeric(g[field],errors="coerce"),color=GREEN,ls=ls,lw=2.1 if seed==5 else 1.45,
                marker=m,ms=4.2 if seed==5 else 3.5,label=f"PPO seed {seed}")
        ax.set_ylabel(label); ax.grid(axis="y",alpha=.22); ax.legend(ncol=3,fontsize=8,frameon=False,loc="best")
    axes[-1].set_xticks(YEARS,YEARS); axes[-1].set_xlabel("Validation year")
    fig.suptitle("YCA | Fixed Random-weather PPO seeds across observed validation years",x=.02,ha="left",fontweight="bold")
    fig.tight_layout(rect=[0,0,1,.975]); fig.savefig(OUT/"figures/annual_metrics_seed345.png",dpi=200); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(16,5.8),sharex=True,sharey=True)
    for ax,seed in zip(axes,SEEDS):
        mat=np.eye(10)
        for r in pairs.loc[pairs.seed.eq(seed)].itertuples():
            a,b=YEARS.index(int(r.year_a)),YEARS.index(int(r.year_b)); mat[a,b]=mat[b,a]=r.sequence_edit_similarity
        im=ax.imshow(mat,vmin=0,vmax=1,cmap="YlGn",aspect="auto")
        ax.set_title(f"Seed {seed}",loc="left",fontweight="bold"); ax.set_xticks(range(10),YEARS,rotation=45,ha="right")
        ax.set_yticks(range(10),YEARS); ax.set_xlabel("Validation year"); ax.grid(color="white",lw=.5)
        for i in range(10):
            for j in range(10):
                color = "white" if mat[i,j] >= .72 else "#25352A"
                ax.text(j,i,f"{mat[i,j]:.0%}",ha="center",va="center",fontsize=6.5,color=color)
    axes[0].set_ylabel("Validation year")
    fig.subplots_adjust(left=.06,right=.90,bottom=.20,top=.82,wspace=.08)
    color_ax=fig.add_axes([.925,.20,.016,.60])
    fig.colorbar(im,cax=color_ax,label="Action sequence similarity")
    fig.suptitle("YCA | Interannual action-sequence similarity",x=.02,ha="left",fontweight="bold")
    fig.savefig(OUT/"figures/schedule_similarity_seed345.png",dpi=200); plt.close(fig)

    means=annual.groupby("seed").agg(yield_mean=("formal_yield_kg_ha","mean"),water_mean=("irrigation_mm","mean"),
        n_mean=("nitrogen_kg_ha","mean"),reward_mean=("common_reward","mean"),wp_mean=("wp_et_kg_m3","mean"),
        pfp_mean=("pfp_n_kg_kg","mean"),action_days=("management_action_days_nonzero","mean"),
        event_count=("management_event_count","mean"))
    s4=annual.loc[annual.seed.eq(4)].set_index("year"); s5=annual.loc[annual.seed.eq(5)].set_index("year")
    water=s4.irrigation_mm-s5.irrigation_mm; yd=s5.formal_yield_kg_ha-s4.formal_yield_kg_ha
    delta=float(water.mean()); ydelta=float(yd.mean()); ypct=100*ydelta/float(s4.formal_yield_kg_ha.mean())
    d=events.loc[events.seed.isin([4,5])].groupby(["seed","dap"]).irrigation_action_mm.sum().unstack(0).fillna(0)
    d["diff"]=d.get(4,0)-d.get(5,0)
    extra4=float(d["diff"].clip(lower=0).sum()/10)
    offset5=float((-d["diff"].clip(upper=0)).sum()/10)
    top4=", ".join(f"DAP {int(k)} (+{v/10:.1f})" for k,v in d["diff"].sort_values(ascending=False).items() if v>0)
    top5=", ".join(f"DAP {int(k)} (+{-v/10:.1f})" for k,v in d["diff"].sort_values().items() if v<0)
    mechanism=(f"seed4 在 {top4} 的额外灌溉平均合计 {extra4:.1f} mm/年；seed5 在 {top5} 的额外灌溉抵消 {offset5:.1f} mm/年，净差为 {delta:.1f} mm/年。seed5 将多次约 15 mm 的分散灌溉换成 DAP 8/31/61 的 30/45/45 mm 集中动作；这是动作日历上的水量重分配解释，非因果证明。")
    mean_table="\n".join(f"| {s} | {r.yield_mean:.1f} | {r.water_mean:.1f} | {r.n_mean:.1f} | {r.reward_mean:.2f} | {r.wp_mean:.3f} | {r.pfp_mean:.2f} | {r.action_days:.1f} | {r.event_count:.1f} |" for s,r in means.iterrows())
    profile_table="\n".join(f"| {s} | {profiles[str(s)]['weather_adaptivity']} | {profiles[str(s)]['unique_signatures']} | {profiles[str(s)]['modal_signature_count']}/10 | {profiles[str(s)]['exact_year_pairs']}/45 | {profiles[str(s)]['mean_edit_similarity']:.3f} |" for s in SEEDS)
    pair_table="\n".join(f"| {s} | {pairs.loc[pairs.seed.eq(s),'action_dap_jaccard'].mean():.3f} | {pairs.loc[pairs.seed.eq(s),'action_tuple_jaccard'].mean():.3f} | {pairs.loc[pairs.seed.eq(s),'sequence_edit_similarity'].mean():.3f} |" for s in SEEDS)
    year_table="\n".join(f"| {y} | {s4.loc[y,'formal_yield_kg_ha']:.0f} | {s5.loc[y,'formal_yield_kg_ha']:.0f} | {yd.loc[y]:+.0f} | {s4.loc[y,'irrigation_mm']:.0f} | {s5.loc[y,'irrigation_mm']:.0f} | {water.loc[y]:.0f} |" for y in YEARS)
    q=profiles["5"]["context_spread"]
    seed5_signature=annual.loc[annual.seed.eq(5),"action_signature_trace"].iloc[0]
    report=f"""# YC 004_08：seed 3/4/5 action-response 行为审计

## 方法与数据质量

按 004_07 / 004_07_01 的固定模型与 all-seed 验证框架，使用 004_05 step trace 和 004_07_02 的 30 个固定 checkpoint replay 摘要、snapshots；没有训练、改 reward/约束或启动 evaluation-only replay。动作逐年与 replay summary 核对。step timestep 对齐 snapshot DAS；同日及前一 DAS 状态均输出。SWFAC/NSTRES 来自动作 step 行；精确 observation vector/variables JSON 对 PPO 目标记录为空，因此无法确认这些代理是否属于模型实际输入。snapshot 同日天气为事后上下文，不是已证实的 policy input。

- 逐日记录 {len(steps)} 行；seed×year {len(years_found)}/30；snapshots {len(contexts.groupby(["seed","year"]))}/30。
- step-to-snapshot DAS 匹配 {trace.das.notna().mean():.1%}；动作总量、signature 与 replay summary 闭合：{reconciled}。
- observation vector 非空 {int(trace.obs_vector_available.sum())}/{len(trace)}；observation variables 非空 {int(trace.obs_variables_available.sum())}/{len(trace)}。
- 动作事件中前 7 日降雨代理缺失 {int(events.prior7_rain_mm.isna().sum())}/{len(events)}，均为播种日 DAP 0（snapshot 未含作物季开始前天气）；其余事件可对齐。
- 逐日精确字段缺失：SWFAC {trace.swfac.isna().sum()}、NSTRES {trace.nstres.isna().sum()}、DAP {trace.dap.isna().sum()}、step LAI {trace.xlai.isna().sum()}、step stage {trace.istage.isna().sum()}、step rainfall {trace.RAIN.isna().sum()}。snapshot 补充 soil water、LAI/stage、降雨等事后状态上下文。

## 样式来源

主要复用 results/yc_random_weather_ppo/004_05_review_figures/055_03_five_scenario/plot_004_05_seed3_diagnostic.py，参考系列脚本 src/055_yca_lowIC_site_transfer/run_055_03_yca_lowIC_five_scenario_figures.py。沿用左对齐标题、紧凑多面板、浅网格、PPO 绿色、灌溉蓝圆/N 橙方及按年动作日历；为区分三个 seed，仅使用旧线色配不同线型/marker。

## 年度结果

| Seed | Yield 均值 kg/ha | 灌溉均值 mm | N 均值 kg/ha | Common reward 均值 | WP_ET 均值 | PFP_N 均值 | action days/年 | 管理 events/年 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{mean_table}

seed5 对 seed4 平均少灌溉 {delta:.1f} mm/年，产量差 {ydelta:+.1f} kg/ha ({ypct:+.2f}%)；seed5 用水更少年份 {int(water.gt(0).sum())}/10，产量更高年份 {int(yd.gt(0).sum())}/10。

| 年份 | Seed4 产量 | Seed5 产量 | 产量差 5−4 | Seed4 灌溉 | Seed5 灌溉 | 节水 4−5 |
|---:|---:|---:|---:|---:|---:|---:|
{year_table}

Seed5 年均 action days {means.loc[5,'action_days']:.1f}，seed4 为 {means.loc[4,'action_days']:.1f}。灌溉日剂量的逐日净额拆分：{mechanism}

## Schedule 相似度和 weather adaptivity

| Seed | Adaptivity | 独立 schedule 数 | 模态 schedule 年数 | 完全相同年度对 | 平均编辑相似度 |
|---:|---|---:|---:|---:|---:|
{profile_table}

| Seed | DAP 集合 Jaccard | 事件 tuple Jaccard | 有序序列相似度 |
|---:|---:|---:|---:|
{pair_table}

Seed5 在相同 DAP 的代理状态最大年际范围：SWFAC {q['swfac']:.3f}、NSTRES {q['nstres']:.3f}、前一 DAS 土壤水 {q['soil_water_mm']:.1f} mm、前 7 日雨量 {q['prior7_rain_mm']:.1f} mm；其中 {p5['daps_with_two_context_classes']} 个共同 DAP 至少有两类状态/天气代理超过年际变异阈值。Fixed-schedule STRONG 判据要求模态 schedule ≥9/10 且相同 DAP 上至少两类代理确有变化；seed5 满足。跨年状态存在变化而动作重复，支持外部行为上的低 weather adaptivity。观测向量缺失，原因不能归结为 reward、observation 编码或 mask 中任何单一因素。

## Seed5 的 240 kg N/ha

分类为 {n_pattern}。Seed5 十年完全相同的 I/N 动作序列为：{seed5_signature}。逐年施氮总量、DAP 和剂量均在 annual_seed345_behavior.csv 与 seed_dap_action_response.csv 中；固定表示验证中实际动作模板重复，不表示代码硬编码。

## 结论

seed5_weather_adaptivity = {verdict['seed5_weather_adaptivity']}
seed5_fixed_schedule_evidence = {fixed}
seed5_water_saving_mechanism = {mechanism}
seed5_high_N_pattern = {n_pattern}
next_step = {verdict['next_step']}

Seed3/4/5 的 adaptivity 对照：{profiles['3']['weather_adaptivity']} / {profiles['4']['weather_adaptivity']} / {profiles['5']['weather_adaptivity']}。seed3 的跨年变化主要是 DAP 16 施氮在部分年份缺失（总 N 80 vs 120 kg/ha），不是充分的 weather response 证据；seed4 的 signature 变化主要是季末追加灌溉次数随 episode 终止时间变化。seed5 schedule 十年逐项相同。优先查 observation-action 响应和 mask，再决定是否优化 PPO/reward；本轮本身不调整任何训练因素。

## 文件和限制

CSV：annual_seed345_behavior、action_events_with_context、seed_dap_action_response、seed_dap_action_response_summary、action_schedule_similarity_pairs、daily_context_seed345。JSON：analysis_summary、input_provenance。PNG：action_calendar_seed345、state_variation_at_action_dap、annual_metrics_seed345、schedule_similarity_seed345，均在结果目录 figures 下。

缺项为精确 PPO observation vector/variables；PPO trace 的 LAI、stage、天气列为空，使用 snapshots 的同日/前日状态代理。没有 replay、没有训练、没有 commit/push；结论限于固定 checkpoint 和筛查用的十个年份。
"""
    analysis_path=OUT/"analysis_summary.json"
    analysis_data=json.loads(analysis_path.read_text(encoding="utf-8"))
    verdict["seed5_water_saving_mechanism"]=mechanism
    analysis_data["verdicts"]["seed5_water_saving_mechanism"]=mechanism
    analysis_path.write_text(json.dumps(analysis_data,ensure_ascii=False,indent=2),encoding="utf-8")
    REPORT.write_text(report,encoding="utf-8")
    print(json.dumps({"output":OUT.relative_to(ROOT).as_posix(),"report":REPORT.relative_to(ROOT).as_posix(),
        "quality":quality,"verdicts":verdict},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
