from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from diagnose_all_water_stress_sites import evaluate_site_agent
from dssat_site_config import build_env_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="my_data_rain_scaling/rain_scaling_manifest.json")
    parser.add_argument("--output-dir", default="output_hl/rain_scaling_diagnostics")
    parser.add_argument("--agents", default="null,expert")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--run-dssat-location", default="/opt/dssat_pdi/run_dssat")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    agents = [value.strip().lower() for value in args.agents.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for entry in manifest:
        site = entry["site"]
        scale = float(entry["scale"])
        data_dir = entry["data_dir"]
        env_args = build_env_args(
            site=site,
            mode="all",
            seed=123,
            data_dir=data_dir,
            prefer_suffix=None,
            run_dssat_location=args.run_dssat_location,
        )
        for agent in agents:
            label = f"{site}_{entry['scale']:g}_{agent}".replace(".", "p")
            scenario_output_dir = output_dir / f"{site}_{entry['scale']:g}".replace(".", "p")
            print(f"Running {label} data_dir={data_dir}", flush=True)
            summary = evaluate_site_agent(site, agent, env_args, scenario_output_dir, args.max_steps)
            summary["rain_scale"] = scale
            summary["trace_dir"] = scenario_output_dir.as_posix()
            summary["annual_rain_before"] = entry["annual_rain_before"]
            summary["annual_rain_after"] = entry["annual_rain_after"]
            summary["scenario_data_dir"] = data_dir
            rows.append(summary)

    df = pd.DataFrame(rows)
    first = ["site", "rain_scale", "agent", "PRCP", "ETCP", "PRCP_minus_ETCP"]
    cols = first + [col for col in df.columns if col not in first]
    df = df[cols]
    df.to_csv(output_dir / "rain_scaling_all_stress_summary.csv", index=False)
    (output_dir / "rain_scaling_all_stress_summary.md").write_text(
        "\n".join([
            "# Rainfall Scaling All-mode Diagnostic Summary",
            "",
            "Rainfall is multiplied in WTH files while templates, soil and cultivar files are copied unchanged.",
            "",
            "```text",
            df.round(3).to_string(index=False),
            "```",
        ]),
        encoding="utf-8",
    )
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
