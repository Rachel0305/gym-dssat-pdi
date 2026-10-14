from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_ROOTS = {
    "HL": Path("output_hl/output_hl/irrigation"),
    "SY": Path("output_sy/output_sy/irrigation"),
    "YC": Path("output_yc/output_yc/irrigation"),
    "LC": Path("output_lc/output_lc/irrigation"),
    "FQ": Path("output_fq/output_fq/irrigation"),
}

WEATHER_FILES = {
    "HL": Path("my_data/CNHL0701(1).WTH"),
    "SY": Path("my_data/CNSY1201(1).WTH"),
    "YC": Path("my_data/CNYC0801(1).WTH"),
    "LC": Path("my_data/CNLC0801(1).WTH"),
    "FQ": Path("my_data/CNFQ0701(1).WTH"),
}


def first_existing_column(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(dtype=float)


def summarize_trace(path: Path) -> dict:
    df = pd.read_csv(path)
    agent = path.name.split("_decision_trace_ep")[0]
    final_totir = first_existing_column(df, ["totir"]).dropna()
    real_amir = first_existing_column(df, ["real_action_amir"])
    raw_amir = first_existing_column(df, ["action_amir"])
    if not real_amir.empty:
        water_estimate = real_amir.sum()
        water_source = "real_action_amir"
    elif not final_totir.empty:
        water_estimate = final_totir.iloc[-1]
        water_source = "final_totir"
    else:
        water_estimate = pd.NA
        water_source = "missing"
    return {
        "agent": agent,
        "file": path.name,
        "reward_sum": pd.to_numeric(df.get("reward"), errors="coerce").sum(),
        "max_grnwt": pd.to_numeric(df.get("grnwt"), errors="coerce").max(),
        "mean_swfac": pd.to_numeric(df.get("swfac"), errors="coerce").mean(),
        "mean_nstres": pd.to_numeric(df.get("nstres"), errors="coerce").mean(),
        "water_estimate": water_estimate,
        "water_source": water_source,
        "raw_action_amir_sum": raw_amir.sum() if not raw_amir.empty else pd.NA,
        "final_totir": final_totir.iloc[-1] if not final_totir.empty else pd.NA,
        "days": len(df),
    }


def summarize_outputs() -> pd.DataFrame:
    records = []
    for site, root in OUTPUT_ROOTS.items():
        for path in sorted(root.glob("*_decision_trace_ep*.csv")):
            record = summarize_trace(path)
            record["site"] = site
            records.append(record)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    grouped = (
        df.groupby(["site", "agent"], as_index=False)
        .agg(
            episodes=("file", "count"),
            reward_sum=("reward_sum", "mean"),
            max_grnwt=("max_grnwt", "mean"),
            mean_swfac=("mean_swfac", "mean"),
            mean_nstres=("mean_nstres", "mean"),
            water_estimate=("water_estimate", "mean"),
            water_source=("water_source", lambda values: ",".join(sorted(set(map(str, values))))),
            raw_action_amir_sum=("raw_action_amir_sum", "mean"),
            final_totir=("final_totir", "mean"),
            days=("days", "mean"),
        )
        .sort_values(["site", "agent"])
    )
    return grouped


def summarize_weather() -> pd.DataFrame:
    rows = []
    for site, path in WEATHER_FILES.items():
        rain_values = []
        grow_values = []
        if path.exists():
            for line in path.read_text(errors="ignore").splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0].isdigit():
                    rain = float(parts[4])
                    rain_values.append(rain)
                    doy = int(parts[0][-3:])
                    if 120 <= doy <= 300:
                        grow_values.append(rain)
        rows.append(
            {
                "site": site,
                "weather_file": str(path),
                "annual_rain": sum(rain_values),
                "rain_doy_120_300": sum(grow_values),
                "days": len(rain_values),
            }
        )
    return pd.DataFrame(rows).sort_values("site")


def main() -> None:
    out_dir = Path("output_hl/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = summarize_outputs()
    weather = summarize_weather()
    outputs.to_csv(out_dir / "irrigation_policy_summary.csv", index=False)
    weather.to_csv(out_dir / "weather_rain_summary_suffix1.csv", index=False)
    print(outputs.round(3).to_string(index=False))
    print()
    print(weather.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
