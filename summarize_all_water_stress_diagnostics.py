from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_all_water_stress_sites import summarize_trace
from dssat_site_config import build_env_args


SITES = ["HL", "SY", "LC", "FQ", "YC"]
AGENTS = ["null", "expert"]
TRACE_DIRS = [
    Path("output_hl/all_water_stress_diagnostics"),
    Path("output_hl/all_water_stress_diagnostics_smoke"),
]
OUTPUT_DIR = Path("output_hl/all_water_stress_diagnostics")


def find_trace(site: str, agent: str) -> tuple[Path | None, str]:
    filename = f"{site}_{agent}_all_water_stress_trace.csv"
    for trace_dir in TRACE_DIRS:
        path = trace_dir / filename
        if path.exists():
            return path, trace_dir.name
    return None, "missing_or_stalled"


def missing_row(site: str, agent: str, source: str) -> dict:
    env_args = build_env_args(site=site, mode="all", data_dir="./my_data", prefer_suffix=None)
    return {
        "site": site,
        "site_name": site,
        "agent": agent,
        "mode": "all",
        "template": env_args["fileX_template_path"],
        "weather": env_args["auxiliary_file_paths"][1],
        "soil": env_args["auxiliary_file_paths"][2],
        "trace_source": source,
        "n_days": 0,
        "swfac_stress_days_gt_0.05": np.nan,
        "swfac_stress_days_gt_0.10": np.nan,
        "max_swfac": np.nan,
        "mean_swfac": np.nan,
        "nstres_days_gt_0.05": np.nan,
        "max_nstres": np.nan,
        "mean_nstres": np.nan,
        "mean_trnu": np.nan,
        "final_trnu": np.nan,
        "max_grnwt": np.nan,
        "total_anfer": np.nan,
        "total_amir": np.nan,
        "final_totir": np.nan,
        "total_reward": np.nan,
        "note": "all-mode diagnostic stalled or trace missing for this site/agent",
    }


def main() -> None:
    rows = []
    for site in SITES:
        env_args = build_env_args(site=site, mode="all", data_dir="./my_data", prefer_suffix=None)
        for agent in AGENTS:
            path, source = find_trace(site, agent)
            if path is None:
                rows.append(missing_row(site, agent, source))
                continue
            df = pd.read_csv(path)
            summary = summarize_trace(site, agent, df, env_args)
            summary["trace_source"] = source
            rows.append(summary)

    df = pd.DataFrame(rows)
    first_cols = ["site", "site_name", "agent", "trace_source", "PRCP", "ETCP", "PRCP_minus_ETCP"]
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "all_water_stress_summary_final.csv", index=False)
    md = [
        "# Final All-mode Water/Nitrogen Stress Diagnostic Summary",
        "",
        "`swfac` and `nstres` are exposed in maize `all` mode. In this installed environment they are post-processed as `1 - original`, so larger values indicate stronger stress.",
        "",
        "Rows marked `missing_or_stalled` did not finish safely and should be rerun after isolating the site configuration problem.",
        "",
        "```text",
        df.round(3).to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "all_water_stress_summary_final.md").write_text("\n".join(md), encoding="utf-8")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
