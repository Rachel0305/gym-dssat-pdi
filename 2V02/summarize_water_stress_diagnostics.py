from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_water_stress_sites import (
    ETCP_FROM_DSSAT_485,
    SITE_NAMES,
    build_env_args,
    crop_season_prcp,
    parse_pdate,
    parse_template_irrigation,
)


SITES = ["HL", "SY", "LC", "FQ", "YC"]
AGENTS = ["null", "expert"]
NEW_TRACE_DIR = Path("output_hl/water_stress_diagnostics")
OLD_TRACE_DIRS = {
    "LC": Path("output_lc/output_lc/irrigation"),
}


def numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def load_trace(site: str, agent: str) -> tuple[pd.DataFrame | None, str]:
    new_path = NEW_TRACE_DIR / f"{site}_{agent}_water_stress_trace.csv"
    if new_path.exists():
        return pd.read_csv(new_path), "new_irrigation_env_original_inputs"

    old_dir = OLD_TRACE_DIRS.get(site)
    if old_dir:
        old_path = old_dir / f"{agent}_decision_trace_ep1.csv"
        if old_path.exists():
            return pd.read_csv(old_path), "fallback_existing_old_irrigation_trace"

    return None, "missing"


def summarize(site: str, agent: str) -> dict:
    env_args = build_env_args(site=site, mode="irrigation", data_dir="./my_data", prefer_suffix=None)
    pdate = parse_pdate(env_args["fileX_template_path"])
    df, source = load_trace(site, agent)
    n_days = len(df) if df is not None else 0
    prcp = crop_season_prcp(env_args["auxiliary_file_paths"][1], pdate, n_days or 1)
    etcp = ETCP_FROM_DSSAT_485.get(site, np.nan)
    template_irrig = parse_template_irrigation(env_args["fileX_template_path"])

    if df is None:
        return {
            "site": site,
            "site_name": SITE_NAMES.get(site, site),
            "agent": agent,
            "trace_source": source,
            "PRCP": prcp,
            "ETCP": etcp,
            "PRCP_minus_ETCP": prcp - etcp,
            "swfac_stress_days_gt_0.05": np.nan,
            "turfac_stress_days_gt_0.05": np.nan,
            "min_swfac": np.nan,
            "mean_nstres": np.nan,
            "max_grnwt": np.nan,
            "irrigation_mm": np.nan,
            "final_totir": np.nan,
            "template_irrigation_mm": template_irrig,
            "note": "trace missing",
        }

    swfac = numeric(df.get("swfac"))
    turfac = numeric(df.get("turfac"))
    nstres = numeric(df.get("nstres"))
    grnwt = numeric(df.get("grnwt"))
    final_totir_series = numeric(df.get("totir")).dropna()
    if "history_action_amir" in df:
        irrigation = numeric(df.get("history_action_amir")).sum(skipna=True)
    elif "real_action_amir" in df:
        irrigation = numeric(df.get("real_action_amir")).sum(skipna=True)
    else:
        irrigation = final_totir_series.iloc[-1] if not final_totir_series.empty else np.nan

    swfac_available = not swfac.dropna().empty
    turfac_available = not turfac.dropna().empty
    nstres_available = not nstres.dropna().empty
    note_parts = []
    if not swfac_available:
        note_parts.append("swfac not exposed by maize irrigation state")
    if not turfac_available:
        note_parts.append("turfac not exposed by maize irrigation state")
    if not nstres_available:
        note_parts.append("nstres not exposed by maize irrigation state")
    if source.startswith("fallback"):
        note_parts.append("LC used existing trace because original LC env call stalled")

    return {
        "site": site,
        "site_name": SITE_NAMES.get(site, site),
        "agent": agent,
        "trace_source": source,
        "PDATE": pdate,
        "n_days": n_days,
        "PRCP": prcp,
        "ETCP": etcp,
        "PRCP_minus_ETCP": prcp - etcp,
        "swfac_stress_days_gt_0.05": int((swfac.dropna() > 0.05).sum()) if swfac_available else np.nan,
        "turfac_stress_days_gt_0.05": int((turfac.dropna() > 0.05).sum()) if turfac_available else np.nan,
        "min_swfac": swfac.min(skipna=True) if swfac_available else np.nan,
        "mean_nstres": nstres.mean(skipna=True) if nstres_available else np.nan,
        "max_grnwt": grnwt.max(skipna=True),
        "irrigation_mm": irrigation,
        "final_totir": final_totir_series.iloc[-1] if not final_totir_series.empty else np.nan,
        "template_irrigation_mm": template_irrig,
        "note": "; ".join(note_parts),
    }


def main() -> None:
    rows = [summarize(site, agent) for site in SITES for agent in AGENTS]
    df = pd.DataFrame(rows)
    out_dir = Path("output_hl/water_stress_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "water_stress_summary_final.csv", index=False)

    md = ["# Water Stress Diagnostic Summary", ""]
    md.append("Important: maize irrigation mode does not expose `swfac`, `nstres`, or `turfac` in the current installed env_config.yml. Blank stress values are therefore missing observations, not zero stress.")
    md.append("")
    md.append(df.round(3).to_markdown(index=False))
    (out_dir / "water_stress_summary_final.md").write_text("\n".join(md), encoding="utf-8")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
