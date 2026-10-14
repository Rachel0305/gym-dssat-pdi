from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_extension_expert_baseline_018_03 as extension
import run_hla_unified_dqn_long_train_015_09 as hla_base
import run_hla_new_cultivar_candidate_year_screening as hla_screen


OUT = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_five_scenario_nstep_020_11"
FIG = OUT / "figures"
DOC = ROOT / "docs" / "2026-07-10_020_11_hla_five_scenario_nstep_freeze_record.md"
FROZEN_CONFIG = OUT / "020_11_hla_nstep5_frozen_config.json"
YEARS = [2007, 2010, 2015, 2016, 2022]

SCENARIOS = [
    "null",
    "recorded_farmer",
    "dssat_auto",
    "extension_expert",
    "nstep_seed0",
    "nstep_seed1",
    "nstep_seed2",
]

LABELS = {
    "null": "Null",
    "recorded_farmer": "Recorded farmer practice",
    "dssat_auto": "DSSAT auto",
    "extension_expert": "Official extension expert",
    "nstep_seed0": "n-step DQN seed0",
    "nstep_seed1": "n-step DQN seed1",
    "nstep_seed2": "n-step DQN seed2",
}

COLORS = {
    "null": "#111111",
    "recorded_farmer": "#C9252D",
    "dssat_auto": "#B8860B",
    "extension_expert": "#5B4B9A",
    "nstep_seed0": "#126B3A",
    "nstep_seed1": "#2F8F5B",
    "nstep_seed2": "#65A96B",
}

LINESTYLES = {
    "null": "-",
    "recorded_farmer": "--",
    "dssat_auto": "-.",
    "extension_expert": (0, (1, 1)),
    "nstep_seed0": "-",
    "nstep_seed1": "--",
    "nstep_seed2": ":",
}

SELECTED_MODELS = {
    0: {
        "checkpoint": 30000,
        "run_dir": ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed0_020_08" / "nstep5_seed0_50000steps",
    },
    1: {
        "checkpoint": 10000,
        "run_dir": ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed1_020_06" / "nstep5_seed1_50000steps",
    },
    2: {
        "checkpoint": 20000,
        "run_dir": ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_dqn_seed2_020_07" / "nstep5_seed2_50000steps",
    },
}

YEAR_INPUTS = {
    2007: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_transfer_2007_2016_2022_020_10" / "test_cases" / "2007" / "input",
    2010: SELECTED_MODELS[0]["run_dir"] / "input",
    2015: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_to_hla2015_transfer_020_09" / "test_case_hla2015" / "input",
    2016: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_transfer_2007_2016_2022_020_10" / "test_cases" / "2016" / "input",
    2022: ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_transfer_2007_2016_2022_020_10" / "test_cases" / "2022" / "input",
}

EXPECTED_RECORDED_YIELD = {2010: 7679.0, 2015: 7296.0}
WATER_COST = 1.0
NITROGEN_COST = 5.0


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(frame: pd.DataFrame, candidates: list[str], default: float = np.nan) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def text_series(frame: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype=str)


def standardize_daily(frame: pd.DataFrame, year: int, source: str, forced_scenario: str | None = None) -> pd.DataFrame:
    raw_scenario = text_series(frame, ["scenario"], "null").replace({"": "null", "nan": "null"})
    scenario_map = {
        "expert_2007_shifted": "recorded_farmer",
        "recorded": "recorded_farmer",
        "transfer_2010_seed0": "nstep_seed0",
        "transfer_2010_seed1": "nstep_seed1",
        "transfer_2010_seed2": "nstep_seed2",
        "extension_expert_fixed_dap": "extension_expert",
    }
    scenario = raw_scenario.replace(scenario_map)
    if forced_scenario is not None:
        scenario = pd.Series(forced_scenario, index=frame.index, dtype=str)
    out = pd.DataFrame(
        {
            "year": int(year),
            "scenario": scenario,
            "dap": numeric(frame, ["dap"]),
            "doy": numeric(frame, ["doy"]),
            "rain_mm": numeric(frame, ["rain", "rain_obs"], 0.0).fillna(0.0),
            "water_stress": numeric(frame, ["water_stress", "swfac", "wspd"]),
            "nitrogen_stress": numeric(frame, ["nitrogen_stress", "nstres", "nstd"]),
            "grain_kg_ha": numeric(frame, ["grain_kg_ha", "grnwt", "gwad"]),
            "biomass_kg_ha": numeric(frame, ["biomass_kg_ha", "topwt", "cwad"]),
            "irrigation_mm": numeric(frame, ["irrigation_mm", "irrigation_mm_action"], 0.0).fillna(0.0),
            "fertilizer_kg_ha": numeric(frame, ["fertilizer_kg_ha", "fertilizer_kg_ha_action"], 0.0).fillna(0.0),
            "action_dap": numeric(frame, ["dap_action", "dap"]),
            "source_file": source,
        }
    )
    out = out[np.isfinite(out["dap"])].copy()
    out = out.sort_values(["scenario", "dap"]).drop_duplicates(["scenario", "dap"], keep="last")
    out["dqn_seed"] = out["scenario"].map({"nstep_seed0": 0, "nstep_seed1": 1, "nstep_seed2": 2})
    return out.reset_index(drop=True)


def validate_input(input_dir: Path, year: int, require_placeholders: bool = True) -> Path:
    required_suffixes = {".MZX", ".WTH", ".CUL", ".SOL"}
    found = {p.suffix.upper() for p in input_dir.iterdir() if p.is_file()}
    missing = required_suffixes - found
    if missing:
        raise FileNotFoundError(f"HLA{year} input missing {sorted(missing)} in {input_dir}")
    filex = next(input_dir.glob("*.MZX"))
    content = filex.read_text(encoding="latin-1", errors="ignore")
    if require_placeholders and ("{{ irrig }}" not in content or "{{ ferti }}" not in content):
        raise RuntimeError(f"HLA{year} template lacks PDI irrigation/fertilization placeholders")
    treatment_line = next((line for line in content.splitlines() if line.lstrip().startswith("1 1 1 0")), "")
    if not treatment_line:
        raise RuntimeError(f"HLA{year} treatment line was not found")
    if "*INITIAL CONDITIONS" not in content:
        raise RuntimeError(f"HLA{year} initial-condition section was not found")
    return filex


def prepare_fixed_case(year: int, scenario: str) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    source_input = YEAR_INPUTS[year]
    validate_input(source_input, year)
    run_dir = OUT / "runs" / str(year) / scenario
    input_dir = run_dir / "input"
    if run_dir.exists() and not (run_dir / "replay_daily.csv").exists():
        raise RuntimeError(f"Incomplete existing run directory; inspect before retry: {run_dir}")
    if (run_dir / "replay_daily.csv").exists():
        return {}, run_dir, []
    input_dir.mkdir(parents=True, exist_ok=False)
    audit_rows: list[dict[str, Any]] = []
    files_to_copy = [src for src in sorted(source_input.iterdir()) if src.is_file()]
    if scenario == "dssat_auto":
        auto_source = hla_screen.SOURCE_ROOT / "auto_irrig" / str(year) / "input"
        auto_mzx = next(auto_source.glob("*.MZX"))
        files_to_copy = [src for src in files_to_copy if src.suffix.upper() != ".MZX"] + [auto_mzx]
    for src in files_to_copy:
        if not src.is_file():
            continue
        dst = input_dir / src.name
        shutil.copyfile(src, dst)
        audit_rows.append(
            {
                "year": year,
                "scenario": scenario,
                "file": src.name,
                "source_path": src.relative_to(ROOT).as_posix(),
                "copied_path": dst.relative_to(ROOT).as_posix(),
                "source_sha256": sha256(src),
                "copied_sha256": sha256(dst),
                "hash_match": sha256(src) == sha256(dst),
            }
        )
    filex = validate_input(input_dir, year, require_placeholders=scenario != "dssat_auto")
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": [str(p) for p in sorted(input_dir.iterdir()) if p.is_file() and p != filex],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return env_args, run_dir, audit_rows


def official_schedule() -> pd.DataFrame:
    schedule = extension.build_region_schedule()
    return schedule[schedule["region"].eq("northeast_greatwall_spring_maize")].copy()


def recorded_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dap": 0, "n_kg_ha_mid": 165.0, "irrigation_mm_mid": 0.0},
            {"dap": 49, "n_kg_ha_mid": 0.0, "irrigation_mm_mid": 10.0},
            {"dap": 70, "n_kg_ha_mid": 0.0, "irrigation_mm_mid": 10.0},
            {"dap": 95, "n_kg_ha_mid": 0.0, "irrigation_mm_mid": 10.0},
        ]
    )


def run_or_load_fixed(year: int, scenario: str, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    env_args, run_dir, audit = prepare_fixed_case(year, scenario)
    daily_path = run_dir / "replay_daily.csv"
    events_path = run_dir / "replay_events.csv"
    deduplicated_events_path = run_dir / "replay_events_deduplicated.csv"
    summary_path = run_dir / "replay_summary.json"
    corrected_summary_path = run_dir / "replay_summary_deduplicated.json"
    if daily_path.exists() and summary_path.exists():
        daily = pd.read_csv(daily_path, keep_default_na=False)
        if deduplicated_events_path.exists():
            events = pd.read_csv(deduplicated_events_path, keep_default_na=False)
        else:
            events = pd.read_csv(events_path, keep_default_na=False) if events_path.exists() else pd.DataFrame()
            if "operation" not in events.columns:
                events = hla_screen.parse_mgmt_events(run_dir / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT")
                if not events.empty:
                    events["scenario"] = scenario
                    events["requested_year"] = year
                    events.to_csv(deduplicated_events_path, index=False, encoding="utf-8-sig")
        source_summary = corrected_summary_path if corrected_summary_path.exists() else summary_path
        summary = json.loads(source_summary.read_text(encoding="utf-8"))
        if not events.empty:
            operation = events["operation"].fillna("").astype(str).str.lower()
            quantity = pd.to_numeric(events["quantity"], errors="coerce").fillna(0.0)
            summary["event_irrigation_total_raw_parser"] = summary.get("event_irrigation_total")
            summary["event_fertilizer_total_raw_parser"] = summary.get("event_fertilizer_total")
            summary["event_irrigation_total"] = float(quantity[operation.str.contains("irrigation")].sum())
            parsed_fertilizer = float(quantity[operation.str.contains("fertil")].sum())
            commanded_fertilizer = float(summary.get("action_fertilizer_total", 0.0) or 0.0)
            summary["event_fertilizer_total"] = commanded_fertilizer if commanded_fertilizer > 0 and parsed_fertilizer == 0 else parsed_fertilizer
            corrected_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return daily, events, summary, audit
    case = {
        "site": "HLA",
        "station": "Hailun",
        "year": int(year),
        "region": "northeast_greatwall_spring_maize",
    }
    raw_daily, _external_events, summary = extension.run_fixed_schedule(case, env_args, schedule, run_dir)
    mgmt_path = run_dir / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT"
    events = hla_screen.parse_mgmt_events(mgmt_path)
    if not events.empty:
        events["scenario"] = scenario
        events["requested_year"] = year
    if scenario == "dssat_auto" and not events.empty:
        raw_daily["irrigation_mm_action"] = 0.0
        raw_daily["fertilizer_kg_ha_action"] = 0.0
        for event in events.itertuples():
            operation = str(event.operation)
            dap = float(event.dap)
            quantity = float(event.quantity)
            mask = pd.to_numeric(raw_daily["dap"], errors="coerce").eq(dap)
            if "Irrigation" in operation:
                raw_daily.loc[mask, "irrigation_mm_action"] += quantity
            if "Fertil" in operation:
                raw_daily.loc[mask, "fertilizer_kg_ha_action"] += quantity
    raw_daily["scenario"] = scenario
    summary["scenario"] = scenario
    if not events.empty:
        operation = events["operation"].fillna("").astype(str).str.lower()
        quantity = pd.to_numeric(events["quantity"], errors="coerce").fillna(0.0)
        summary["event_irrigation_total_raw_parser"] = summary.get("event_irrigation_total")
        summary["event_fertilizer_total_raw_parser"] = summary.get("event_fertilizer_total")
        summary["event_irrigation_total"] = float(quantity[operation.str.contains("irrigation")].sum())
        parsed_fertilizer = float(quantity[operation.str.contains("fertil")].sum())
        commanded_fertilizer = float(summary.get("action_fertilizer_total", 0.0) or 0.0)
        summary["event_fertilizer_total"] = commanded_fertilizer if commanded_fertilizer > 0 and parsed_fertilizer == 0 else parsed_fertilizer
    raw_daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    events.to_csv(deduplicated_events_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    corrected_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return raw_daily, events, summary, audit


def load_existing() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    # Existing files are used only for the three selected DQN policies.
    for seed, spec in SELECTED_MODELS.items():
        path = spec["run_dir"] / "nstep5_eval_daily.csv"
        raw = pd.read_csv(path, keep_default_na=False)
        raw = raw[pd.to_numeric(raw["checkpoint_step"], errors="coerce").eq(spec["checkpoint"])].copy()
        raw["scenario"] = f"nstep_seed{seed}"
        frames.append(standardize_daily(raw, 2010, path.relative_to(ROOT).as_posix(), f"nstep_seed{seed}"))

    # HLA2015 three transferred n-step policies.
    p2015 = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_to_hla2015_transfer_020_09" / "020_09_hla2010_nstep_to_hla2015_daily.csv"
    raw = pd.read_csv(p2015, keep_default_na=False)
    raw["scenario"] = raw["scenario"].replace({"": "null"})
    raw = raw[raw["scenario"].isin(["transfer_2010_seed0", "transfer_2010_seed1", "transfer_2010_seed2"])].copy()
    frames.append(standardize_daily(raw, 2015, p2015.relative_to(ROOT).as_posix()))

    # HLA2007/2016/2022 three transferred n-step policies.
    pother = ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla2010_nstep_transfer_2007_2016_2022_020_10" / "020_10_hla2010_nstep_transfer_daily.csv"
    raw = pd.read_csv(pother, keep_default_na=False)
    raw["scenario"] = raw["scenario"].replace({"": "null"})
    keep = ["transfer_2010_seed0", "transfer_2010_seed1", "transfer_2010_seed2"]
    for year in [2007, 2016, 2022]:
        part = raw[pd.to_numeric(raw["requested_year"], errors="coerce").eq(year) & raw["scenario"].isin(keep)].copy()
        frames.append(standardize_daily(part, year, pother.relative_to(ROOT).as_posix()))
    existing = pd.concat(frames, ignore_index=True, sort=False)
    return existing[existing["scenario"].isin(["nstep_seed0", "nstep_seed1", "nstep_seed2"])].copy()


def attach_year_rain(daily: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for year, group in daily.groupby("year", sort=True):
        weather_path = OUT / "runs" / str(int(year)) / "null" / "pdi_tmp_snapshot_eval" / "Weather.OUT"
        if not weather_path.exists():
            raise FileNotFoundError(weather_path)
        weather = extension.parse_dssat_table(weather_path)
        if "DAS" not in weather.columns or "PRED" not in weather.columns:
            raise RuntimeError(f"HLA{year} Weather.OUT lacks DAS/PRED columns")
        weather_dap = pd.to_numeric(weather["DAS"], errors="coerce") - 1.0
        weather_rain = pd.to_numeric(weather["PRED"], errors="coerce").fillna(0.0)
        rain_map = {float(dap): float(rain) for dap, rain in zip(weather_dap, weather_rain) if np.isfinite(dap)}
        if sum(rain_map.values()) <= 0:
            raise RuntimeError(f"HLA{year} Weather.OUT rainfall is all zero")
        current = group.copy()
        current["rain_mm"] = current["dap"].map(rain_map).fillna(0.0)
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True, sort=False)


def add_unified_reward(daily: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for year, year_df in daily.groupby("year", sort=True):
        null = year_df[year_df["scenario"].eq("null")].sort_values("dap")
        null_yield = float(null["grain_kg_ha"].dropna().iloc[-1])
        for scenario, group in year_df.groupby("scenario", sort=False):
            current = group.sort_values("dap").copy()
            current["reward_step_unified"] = -WATER_COST * current["irrigation_mm"] - NITROGEN_COST * current["fertilizer_kg_ha"]
            final_yield = float(current["grain_kg_ha"].dropna().iloc[-1])
            terminal_gain = max(0.0, final_yield - null_yield)
            last_index = current.index[-1]
            current.loc[last_index, "reward_step_unified"] += terminal_gain
            current["cumulative_reward_unified"] = current["reward_step_unified"].cumsum()
            current["null_baseline_yield"] = null_yield
            outputs.append(current)
    return pd.concat(outputs, ignore_index=True, sort=False)


def make_summary(daily: pd.DataFrame, fixed_summary: dict[tuple[int, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    rain_totals = {
        int(year): float(group[group["scenario"].eq("null")][["dap", "rain_mm"]].drop_duplicates("dap")["rain_mm"].sum())
        for year, group in daily.groupby("year")
    }
    for (year, scenario), group in daily.groupby(["year", "scenario"], sort=True):
        current = group.sort_values("dap")
        final_grain = float(current["grain_kg_ha"].dropna().iloc[-1])
        final_biomass = float(current["biomass_kg_ha"].dropna().iloc[-1])
        action_i = float(current["irrigation_mm"].sum())
        action_n = float(current["fertilizer_kg_ha"].sum())
        meta = fixed_summary.get((int(year), str(scenario)), {})
        executed_i = float(meta.get("event_irrigation_total", action_i))
        executed_n = float(meta.get("event_fertilizer_total", action_n))
        if np.isfinite(meta.get("final_gwad", np.nan)):
            final_grain = float(meta["final_gwad"])
        if np.isfinite(meta.get("final_cwad", np.nan)):
            final_biomass = float(meta["final_cwad"])
        rows.append(
            {
                "year": int(year),
                "scenario_family": "nstep_dqn" if scenario.startswith("nstep_seed") else scenario,
                "scenario": scenario,
                "label": LABELS[scenario],
                "dqn_seed": int(scenario[-1]) if scenario.startswith("nstep_seed") else np.nan,
                "final_grain_kg_ha": final_grain,
                "final_biomass_kg_ha": final_biomass,
                "rain_total_mm": rain_totals[int(year)],
                "irrigation_action_total_mm": action_i,
                "irrigation_executed_total_mm": executed_i,
                "nitrogen_action_total_kg_ha": action_n,
                "nitrogen_executed_total_kg_ha": executed_n,
                "irrigation_event_count": int((current["irrigation_mm"] > 1e-8).sum()),
                "fertilizer_event_count": int((current["fertilizer_kg_ha"] > 1e-8).sum()),
                "max_water_stress": float(current["water_stress"].max()),
                "mean_water_stress": float(current["water_stress"].mean()),
                "max_nitrogen_stress": float(current["nitrogen_stress"].max()),
                "mean_nitrogen_stress": float(current["nitrogen_stress"].mean()),
                "unified_reward_total": float(current["reward_step_unified"].sum()),
                "gross_irrigation_productivity_kg_m3": final_grain / executed_i / 10.0 if executed_i > 0 else np.nan,
                "partial_factor_productivity_n_kg_kg": final_grain / executed_n if executed_n > 0 else np.nan,
                "final_dap": float(current["dap"].max()),
                "source_file": str(current["source_file"].iloc[0]),
            }
        )
    summary = pd.DataFrame(rows)
    additions = []
    for year, group in summary.groupby("year", sort=True):
        null_yield = float(group[group["scenario"].eq("null")]["final_grain_kg_ha"].iloc[0])
        auto = group[group["scenario"].eq("dssat_auto")].iloc[0]
        expert = group[group["scenario"].eq("extension_expert")].iloc[0]
        for index, row in group.iterrows():
            additions.append(
                {
                    "index": index,
                    "yield_gain_vs_null_kg_ha": row["final_grain_kg_ha"] - null_yield,
                    "yield_diff_vs_auto_kg_ha": row["final_grain_kg_ha"] - auto["final_grain_kg_ha"],
                    "irrigation_saving_vs_auto_mm": auto["irrigation_executed_total_mm"] - row["irrigation_executed_total_mm"],
                    "nitrogen_saving_vs_auto_kg_ha": auto["nitrogen_executed_total_kg_ha"] - row["nitrogen_executed_total_kg_ha"],
                    "yield_diff_vs_extension_expert_kg_ha": row["final_grain_kg_ha"] - expert["final_grain_kg_ha"],
                    "irrigation_saving_vs_extension_expert_mm": expert["irrigation_executed_total_mm"] - row["irrigation_executed_total_mm"],
                    "nitrogen_saving_vs_extension_expert_kg_ha": expert["nitrogen_executed_total_kg_ha"] - row["nitrogen_executed_total_kg_ha"],
                }
            )
    extra = pd.DataFrame(additions).set_index("index")
    summary = summary.join(extra)
    order = {scenario: i for i, scenario in enumerate(SCENARIOS)}
    summary["scenario_order"] = summary["scenario"].map(order)
    return summary.sort_values(["year", "scenario_order"]).drop(columns="scenario_order").reset_index(drop=True)


def plot_year(daily: pd.DataFrame, year: int, base: Path) -> None:
    data = daily[daily["year"].eq(year)].copy()
    rain = data[data["scenario"].eq("null")][["dap", "rain_mm"]].drop_duplicates("dap").sort_values("dap")
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(7.2, 10.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.65, 1.0, 1.0, 0.85, 0.85, 1.25, 1.0]},
    )
    fig.suptitle(
        f"HLA {year}: five scenario families with three n-step DQN seeds",
        x=0.075,
        y=0.988,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    axes[0].bar(rain["dap"], rain["rain_mm"], width=0.9, color="#BFC5CF", edgecolor="#68717D", linewidth=0.3)
    axes[0].set_ylabel("Rain\n(mm)")
    offsets = dict(zip(SCENARIOS, np.linspace(-0.36, 0.36, len(SCENARIOS))))
    for zorder, scenario in enumerate(SCENARIOS, start=2):
        current = data[data["scenario"].eq(scenario)].sort_values("dap")
        color = COLORS[scenario]
        linestyle = LINESTYLES[scenario]
        axes[1].plot(current["dap"], current["water_stress"], color=color, ls=linestyle, lw=1.45, label=LABELS[scenario], zorder=zorder)
        axes[2].plot(current["dap"], current["nitrogen_stress"], color=color, ls=linestyle, lw=1.45, zorder=zorder)
        irrigation = current[current["irrigation_mm"] > 1e-8]
        fertilizer = current[current["fertilizer_kg_ha"] > 1e-8]
        if not irrigation.empty:
            axes[3].vlines(
                irrigation["action_dap"] + offsets[scenario],
                0,
                irrigation["irrigation_mm"],
                color=color,
                linestyles=linestyle,
                linewidth=1.7,
                alpha=0.92,
                zorder=zorder,
            )
        if not fertilizer.empty:
            axes[4].scatter(
                fertilizer["action_dap"] + offsets[scenario],
                fertilizer["fertilizer_kg_ha"],
                marker="^",
                s=26,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=zorder,
            )
        axes[5].plot(current["dap"], current["grain_kg_ha"], color=color, ls=linestyle, lw=1.55, zorder=zorder)
        axes[5].plot(current["dap"], current["biomass_kg_ha"], color=color, ls=":", lw=1.05, alpha=0.9, zorder=zorder)
        axes[6].plot(current["dap"], current["cumulative_reward_unified"], color=color, ls=linestyle, lw=1.5, zorder=zorder)
    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[3].set_ylabel("Irrigation\n(mm)")
    axes[4].set_ylabel("Fertilizer\n(kg N ha$^{-1}$)")
    axes[5].set_ylabel("Crop mass\n(kg ha$^{-1}$)")
    axes[6].set_ylabel("Cumulative\nreward")
    axes[6].set_xlabel("Days after planting (DAP)")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.52, 0.965), ncol=4, fontsize=6.5, handlelength=2.6, columnspacing=1.0)
    axes[3].text(0.995, 0.94, "vertical lines = irrigation", transform=axes[3].transAxes, ha="right", va="top", fontsize=7, color="#555555")
    axes[4].text(0.995, 0.94, "triangles = fertilization", transform=axes[4].transAxes, ha="right", va="top", fontsize=7, color="#555555")
    axes[5].text(0.995, 0.94, "scenario style = grain; dotted = biomass", transform=axes[5].transAxes, ha="right", va="top", fontsize=7, color="#555555")
    for panel, axis in zip("abcdefg", axes):
        axis.grid(True, axis="x", color="#E0E4EA", linewidth=0.55)
        axis.grid(True, axis="y", color="#ECEFF3", linewidth=0.45, linestyle="--")
        axis.margins(x=0.01)
        axis.text(-0.075, 1.02, panel, transform=axis.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.055, top=0.92, hspace=0.20)
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    use = frame[columns].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in use.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.3f}".rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_frozen_config(summary: pd.DataFrame) -> None:
    import stable_baselines3

    model_entries = []
    for seed, spec in SELECTED_MODELS.items():
        path = spec["run_dir"] / "models" / f"nstep5_checkpoint_{spec['checkpoint']}.zip"
        if not path.exists():
            raise FileNotFoundError(path)
        model_entries.append(
            {
                "seed": seed,
                "selected_checkpoint": spec["checkpoint"],
                "selection_rule": "maximum deterministic total_reward; earliest checkpoint on ties",
                "model_path": path.relative_to(ROOT).as_posix(),
                "model_sha256": sha256(path),
            }
        )
    source_scripts = [
        ROOT / "src" / "run_hla_baseline_relative_dqn_checkpoint_015_12.py",
        ROOT / "src" / "run_hla_unified_dqn_long_train_015_09.py",
        ROOT / "src" / "run_hla2010_nstep_dqn_seed0_020_08.py",
        ROOT / "src" / "run_hla2010_nstep_dqn_seed1_020_06.py",
        ROOT / "src" / "run_hla2010_nstep_dqn_seed2_020_07.py",
    ]
    config = {
        "schema": "hla_nstep5_frozen_v1",
        "status": "frozen_for_cross_site_validation",
        "frozen_date": "2026-07-10",
        "algorithm": "stable_baselines3.DQN",
        "stable_baselines3_version": stable_baselines3.__version__,
        "training_site_year": "HLA2010",
        "reward": {
            "formula": "terminal max(0, GWAD_final - local_null_GWAD) - 1.0*irrigation - 5.0*nitrogen",
            "water_cost": WATER_COST,
            "nitrogen_cost": NITROGEN_COST,
            "negative_terminal_yield_gain_clipped_to_zero": True,
            "local_baseline_rule": "each site-year uses its own null yield; this is normalization, not station-specific coefficient tuning",
            "leaching_term": None,
        },
        "action_table": hla_base.ACTION_TABLE_9,
        "constraints": {
            "irrigation_budget_mm": 120.0,
            "nitrogen_budget_kg_ha": 300.0,
            "single_irrigation_cap_mm": 30.0,
            "single_nitrogen_cap_kg_ha": 100.0,
            "minimum_interval_days": 7,
            "irrigation_window_dap": [[1, 120]],
            "nitrogen_window_dap": [[1, 120]],
            "template_management": {"IRRIG": "L", "FERTI": "L"},
            "initial_condition_pointer": 1,
        },
        "dqn_hyperparameters": {
            "policy": "MlpPolicy",
            "policy_network": {
                "hidden_layers": [64, 64],
                "activation": "ReLU",
                "optimizer": "Adam",
                "source": "Stable-Baselines3 DQN MlpPolicy defaults (policy_kwargs=None)",
            },
            "learning_rate": 0.0001,
            "buffer_size": 10000,
            "learning_starts": 50,
            "batch_size": 32,
            "train_freq": 1,
            "gradient_steps": 1,
            "gamma": 0.99,
            "tau": 1.0,
            "target_update_interval": 10000,
            "max_grad_norm": 10.0,
            "n_steps": 5,
            "exploration_fraction": 0.35,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "total_timesteps": 50000,
            "checkpoint_interval": 5000,
        },
        "evaluation": {
            "deterministic_action": True,
            "checkpoint_selection": "maximum deterministic total_reward; earliest checkpoint on ties",
        },
        "environment": {
            "seed": 0,
            "random_weather": False,
            "evaluation_mode": True,
        },
        "selected_models": model_entries,
        "source_code": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in source_scripts
        ],
        "hla_summary_source": (OUT / "020_11_hla_five_scenario_summary.csv").relative_to(ROOT).as_posix(),
        "cross_site_change_policy": [
            "replace only station input package and local null baseline",
            "keep reward coefficients, action table, budgets, windows, DQN hyperparameters, n_steps and checkpoint rule unchanged",
            "train new models independently at the target site; do not label direct HLA checkpoint replay as cross-site training validation",
        ],
        "verified_hla_years": sorted(summary["year"].unique().astype(int).tolist()),
    }
    FROZEN_CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    hla_base.configure_shared_settings()
    extension.install_official_reward_module()

    existing = load_existing()
    fixed_frames: list[pd.DataFrame] = []
    fixed_events: list[pd.DataFrame] = []
    fixed_summary: dict[tuple[int, str], dict[str, Any]] = {}
    hash_audit_rows: list[dict[str, Any]] = []
    replay_summary_rows: list[dict[str, Any]] = []

    for year in YEARS:
        for scenario, schedule in [
            ("null", pd.DataFrame(columns=["dap", "n_kg_ha_mid", "irrigation_mm_mid"])),
            ("recorded_farmer", recorded_schedule()),
            ("dssat_auto", pd.DataFrame(columns=["dap", "n_kg_ha_mid", "irrigation_mm_mid"])),
            ("extension_expert", official_schedule()),
        ]:
            raw_daily, events, summary, audit = run_or_load_fixed(year, scenario, schedule)
            fixed_frames.append(
                standardize_daily(
                    raw_daily,
                    year,
                    (OUT / "runs" / str(year) / scenario / "replay_daily.csv").relative_to(ROOT).as_posix(),
                    scenario,
                )
            )
            if not events.empty:
                fixed_events.append(events)
            fixed_summary[(year, scenario)] = summary
            hash_audit_rows.extend(audit)
            replay_summary_rows.append(summary)

    daily = pd.concat([existing, *fixed_frames], ignore_index=True, sort=False)
    daily = attach_year_rain(daily)
    daily = add_unified_reward(daily)
    daily = daily.sort_values(["year", "scenario", "dap"]).reset_index(drop=True)
    summary = make_summary(daily, fixed_summary)

    expected = {(year, scenario) for year in YEARS for scenario in SCENARIOS}
    observed = set(zip(summary["year"].astype(int), summary["scenario"]))
    if observed != expected:
        raise RuntimeError(f"Scenario coverage mismatch; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    if summary.groupby("year").size().ne(7).any():
        raise RuntimeError("Every year must contain four baselines plus three DQN seeds")

    recorded_audit = []
    for year, expected_yield in EXPECTED_RECORDED_YIELD.items():
        actual = float(summary[(summary["year"].eq(year)) & (summary["scenario"].eq("recorded_farmer"))]["final_grain_kg_ha"].iloc[0])
        recorded_audit.append(
            {
                "year": year,
                "existing_recorded_yield": expected_yield,
                "replay_recorded_yield": actual,
                "absolute_difference": abs(actual - expected_yield),
                "within_5kg_tolerance": abs(actual - expected_yield) <= 5.0,
            }
        )
    recorded_audit_df = pd.DataFrame(recorded_audit)
    recorded_audit_df.to_csv(OUT / "020_11_recorded_replay_audit.csv", index=False, encoding="utf-8-sig")
    if not recorded_audit_df["within_5kg_tolerance"].all():
        raise RuntimeError("Recorded-farmer replay differs materially from existing results; inspect 020_11_recorded_replay_audit.csv")

    daily.to_csv(OUT / "020_11_hla_five_scenario_daily.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "020_11_hla_five_scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(replay_summary_rows).to_csv(OUT / "020_11_fixed_replay_summary.csv", index=False, encoding="utf-8-sig")
    if fixed_events:
        pd.concat(fixed_events, ignore_index=True, sort=False).to_csv(OUT / "020_11_fixed_replay_events.csv", index=False, encoding="utf-8-sig")
    hash_audit = pd.DataFrame(hash_audit_rows)
    if not hash_audit.empty:
        hash_audit.to_csv(OUT / "020_11_input_hash_audit.csv", index=False, encoding="utf-8-sig")
        if not hash_audit["hash_match"].all():
            raise RuntimeError("Input copy hash mismatch")

    official_schedule().to_csv(OUT / "020_11_extension_expert_schedule.csv", index=False, encoding="utf-8-sig")
    recorded_schedule().to_csv(OUT / "020_11_recorded_farmer_schedule.csv", index=False, encoding="utf-8-sig")
    for year in YEARS:
        plot_year(daily, year, FIG / f"020_11_hla{year}_five_scenario_three_seed_process")
    write_frozen_config(summary)

    view_cols = [
        "year",
        "scenario",
        "dqn_seed",
        "final_grain_kg_ha",
        "final_biomass_kg_ha",
        "irrigation_executed_total_mm",
        "nitrogen_executed_total_kg_ha",
        "irrigation_event_count",
        "fertilizer_event_count",
        "max_water_stress",
        "max_nitrogen_stress",
        "unified_reward_total",
        "yield_diff_vs_auto_kg_ha",
        "yield_diff_vs_extension_expert_kg_ha",
    ]
    lines = [
        "# 020_11 HLA 五情景补齐与 n-step 框架冻结记录",
        "",
        "## 口径纠正",
        "",
        "- 旧 `expert_2007_shifted` 实际为 2007 年农民/试验田管理记录平移，本轮统一命名为 `recorded_farmer`。",
        "- 推文方案单独命名为 `extension_expert`，采用东北春玉米区表 1 中值的固定 DAP 实现。",
        "- 五类情景为 null、recorded farmer、DSSAT auto、official extension expert、n-step DQN；DQN 展开为 seed0/1/2，所以每年汇总有 7 行。",
        "",
        "## Figure contract",
        "",
        "- 核心结论：在同一年度输入下，分别显示农民记录、官方推广 expert、DSSAT auto 与三个 n-step seed 的资源投入、胁迫、产量和统一奖励，避免将 record 误当 expert。",
        "- 证据链：降雨是外部驱动；水/氮胁迫是状态响应；灌溉/施氮是管理动作；籽粒/生物量是作物结果；累积奖励是统一评价。",
        "- 图型：quantitative grid；Python/matplotlib；白底；PNG/SVG/PDF 同源导出。",
        "- 审阅风险：七条策略曲线可能重合，故使用颜色与线型双编码，并将水、氮管理拆为两个面板。",
        "",
        "## 汇总",
        "",
        markdown_table(summary, view_cols),
        "",
        "## Recorded replay 一致性",
        "",
        markdown_table(recorded_audit_df, list(recorded_audit_df.columns)),
        "",
        "## 冻结的 HLA n-step 框架",
        "",
        "- DQN `n_steps=5`；其余超参数、9动作、I120/N300预算、单次上限、7天间隔和 baseline-relative reward 均保持不变。",
        "- 三个选中模型：seed0/30K、seed1/10K、seed2/20K。",
        "- 跨站点验证只允许替换站点输入和该站点年份自己的 null baseline；不能改奖励系数、动作、约束、n-step 或 checkpoint 规则。",
        "",
        "## 输出",
        "",
        f"- 汇总：`{(OUT / '020_11_hla_five_scenario_summary.csv').relative_to(ROOT)}`",
        f"- 日值：`{(OUT / '020_11_hla_five_scenario_daily.csv').relative_to(ROOT)}`",
        f"- 冻结配置：`{FROZEN_CONFIG.relative_to(ROOT)}`",
        f"- 图：`{FIG.relative_to(ROOT)}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary[view_cols].to_string(index=False))


if __name__ == "__main__":
    main()
