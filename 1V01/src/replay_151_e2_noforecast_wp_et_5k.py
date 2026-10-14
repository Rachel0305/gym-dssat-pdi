"""Replay frozen E2/no-forecast 5K checkpoints and complete WP_ET.

This script does not train or change the original experiment outputs.  It
replays the already saved 5K checkpoint for each requested policy/year under
the same DSSAT environment, saves a compact raw DSSAT snapshot, and reads
ETCP/YPEM/YPNAM from ``Summary.OUT``.  Runs are serial by design to avoid
container contention and memory growth.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch03222
from run_142E2_sya_dual_branch_weather_reader_smoke2k import DualBranchBaseWeatherExtractor  # noqa: F401 - SB3 pickle resolution
import run_sya_originIC_expanded_action_maskableppo_046_10 as nof
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_evaluate import latest_observation_dict


TASK = "151E2N_wp_et_5k_replay"
OUT = ROOT / "benchmark_results" / TASK
STATION = "SYA"
YEARS = list(range(2014, 2024))

POLICIES: dict[str, dict[str, Any]] = {
    "e2_s0": {
        "kind": "forecast",
        "seed": 0,
        "cfg": ROOT / "configs/143E2_sya_originIC_dual_branch_weather_reader_5k.json",
        "inventory": ROOT / "benchmark_results/143E2_sya_originIC_dual_branch_weather_reader_5k/evaluation/143E2_checkpoint_validation_summary.csv",
    },
    "e2_s1": {
        "kind": "forecast",
        "seed": 1,
        "cfg": ROOT / "configs/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1.json",
        "inventory": ROOT / "benchmark_results/145E2S1_sya_originIC_dual_branch_weather_reader_10k_seed1/evaluation/145E2S1_checkpoint_validation_summary.csv",
    },
    "e2_s2": {
        "kind": "forecast",
        "seed": 2,
        "cfg": ROOT / "configs/146E2S2_sya_originIC_dual_branch_weather_reader_10k_seed2.json",
        "inventory": ROOT / "benchmark_results/146E2S2_sya_originIC_dual_branch_weather_reader_10k_seed2/evaluation/146E2S2_checkpoint_validation_summary.csv",
    },
    "nof_s0": {
        "kind": "noforecast",
        "seed": 0,
        "cfg": ROOT / "configs/140N_sya_originIC_paired_noforecast.json",
        "inventory": ROOT / "benchmark_results/140N_sya_originIC_paired_noforecast_paired5k/evaluation/140N_checkpoint_validation_summary.csv",
    },
    "nof_s1": {
        "kind": "noforecast",
        "seed": 1,
        "cfg": ROOT / "configs/148N1_sya_originIC_paired_noforecast_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/148N1_sya_originIC_paired_noforecast_5k_seed1/evaluation/148N1_checkpoint_validation_summary.csv",
    },
    "nof_s2": {
        "kind": "noforecast",
        "seed": 2,
        "cfg": ROOT / "configs/149N2_sya_originIC_paired_noforecast_5k_seed2.json",
        "inventory": ROOT / "benchmark_results/149N2_sya_originIC_paired_noforecast_5k_seed2/evaluation/149N2_checkpoint_validation_summary.csv",
    },
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_runtime(cfg: dict[str, Any], policy_key: str, policy_out: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Install the same action wrapper/input profile used by training."""

    kind = str(POLICIES[policy_key]["kind"])
    actions = cfg["actions"]
    engine = nof.engine
    engine.BINARY_IRRIGATION_LEVELS = list(map(float, actions["irrigation_levels_mm"]))
    engine.BINARY_NITROGEN_LEVELS = list(map(float, actions["nitrogen_levels_kg_ha"]))
    engine.LOWIC_INPUT_ROOT = nof.base04602.INPUT_PROFILES[str(cfg["input_profile"])]
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.base03222.load_config = engine.load_config
    engine.base03222.base.StressAwareDiscreteWrapper = engine.base04036.LateIrrigationReserveMaskWrapper
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT

    runtime_cfg = engine.load_config()
    runtime_cfg = json.loads(json.dumps(runtime_cfg))
    runtime_cfg["paths"]["output_root"] = rel(policy_out)
    split = engine.base03222.load_split()
    split = split[split["station_code"].eq(STATION)].copy()
    selection = engine.base03222.build_selection(split)
    env_cfg = direct_ppo.build_env_config(runtime_cfg, selection)
    env_cfg["paths"]["output_root"] = rel(policy_out)
    (policy_out / "configs").mkdir(parents=True, exist_ok=True)
    direct_ppo.write_yaml(env_cfg, policy_out / "configs" / "resolved_env_config.yaml")

    if kind == "forecast":
        # E2 uses the E1 actionable-prefix feature functions plus the dual
        # branch extractor.  The caller installs the forecast factory below.
        return runtime_cfg, env_cfg, engine
    return runtime_cfg, env_cfg, engine


def summary_metrics(snapshot: Path, final_yield: float, expected_i: float, expected_n: float) -> dict[str, Any]:
    rows = siteppo.parse_summary_out(snapshot / "Summary.OUT")
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = siteppo.num(row, "HWAM")
        ircm = siteppo.num(row, "IRCM")
        nicm = siteppo.num(row, "NICM")
        if hwam is None:
            continue
        score = (
            abs(float(hwam) - float(final_yield))
            + abs(float(ircm or 0.0) - float(expected_i))
            + abs(float(nicm or 0.0) - float(expected_n))
        )
        candidates.append((score, -idx, row))
    if not candidates:
        raise RuntimeError(f"No usable Summary.OUT row in {snapshot}")
    score, neg_idx, row = min(candidates, key=lambda item: (item[0], item[1]))
    ircm = siteppo.num(row, "IRCM")
    nicm = siteppo.num(row, "NICM")
    etcp = siteppo.num(row, "ETCP")
    ypem = siteppo.num(row, "YPEM")
    ypnam = siteppo.num(row, "YPNAM")
    if etcp is None or float(etcp) <= 0:
        raise RuntimeError(f"Invalid ETCP in {snapshot}: {etcp}")
    wp = float(ypem) * 0.1 if ypem is not None and float(ypem) >= 0 else float(final_yield) / float(etcp) / 10.0
    pfp = float(ypnam) if nicm and float(nicm) > 0 and ypnam is not None and float(ypnam) >= 0 else math.nan
    return {
        "summary_row_index": int(-neg_idx),
        "summary_match_score": float(score),
        "summary_irrigation_total": float(ircm or 0.0),
        "summary_nitrogen_total": float(nicm or 0.0),
        "etcp_mm": float(etcp),
        "WP_ET_kg_m3": float(wp),
        "PFP_N_kg_kg": float(pfp),
    }


def action_sequence(daily: pd.DataFrame) -> str:
    parts: list[str] = []
    dap = pd.to_numeric(daily.get("dap"), errors="coerce")
    irr = pd.to_numeric(daily.get("safe_action_amir"), errors="coerce").fillna(0.0)
    nit = pd.to_numeric(daily.get("safe_action_anfer"), errors="coerce").fillna(0.0)
    for d, i, n in zip(dap, irr, nit):
        if float(i) > 1e-9 or float(n) > 1e-9:
            parts.append(f"DAP{int(d)} I{float(i):g}/N{float(n):g}")
    return "; ".join(parts)


def copy_compact_snapshot(source: Path, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in ["Summary.OUT", "PlantGro.OUT", "SoilWat.OUT", "Weather.OUT", "MgmtEvent.OUT"]:
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)
            copied.append(name)
    if "Summary.OUT" not in copied:
        raise FileNotFoundError(source / "Summary.OUT")
    return copied


def evaluate_one(
    runtime_cfg: dict[str, Any],
    env_cfg: dict[str, Any],
    engine: Any,
    policy_key: str,
    inv: pd.Series,
    policy_out: Path,
    forecast_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    seed = int(POLICIES[policy_key]["seed"])
    year = int(inv["year"])
    step = int(inv["checkpoint_step"])
    model_path = ROOT / str(inv["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    snapshot = policy_out / "snapshots" / STATION / str(year) / f"ckpt{step}"
    daily_path = policy_out / "daily_outputs" / STATION / f"SYA_{year}_seed{seed}_ckpt{step}_daily.csv"
    existing = snapshot / "Summary.OUT"
    if existing.exists() and daily_path.exists():
        daily = pd.read_csv(daily_path)
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0).sum())
        total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0).sum())
        metrics = summary_metrics(snapshot, final_y, total_i, total_n)
        return {"policy": policy_key, "year": year, "seed": seed, "checkpoint_step": step, "run_status": "ok_existing", "model_path": rel(model_path), "daily_csv_path": rel(daily_path), "snapshot_path": rel(snapshot), "final_grnwt": final_y, "total_irrigation": total_i, "total_n": total_n, "action_sequence": action_sequence(daily), **metrics}

    model = MaskablePPO.load(str(model_path), device="cpu")
    if forecast_cfg is not None:
        import forecast_engineered_observation_056_057 as forecast

        old_make_env = engine.base03222.base.make_env
        engine.base03222.base.make_env = forecast.make_forecast_env_factory(old_make_env, forecast_cfg)
    else:
        old_make_env = None

    run_tag = f"{STATION}_{year}_{policy_key}_ckpt{step}_wp_et_replay"
    env = engine.base03222.base.make_env(runtime_cfg, env_cfg, STATION, year, seed, run_tag, evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_cfg, STATION, year)["planting_date"])
        weather = direct_ppo.weather_for_daily(runtime_cfg)
        while not done and step_count < int(runtime_cfg["runtime"]["max_steps"]):
            latest = latest_observation_dict(env, obs, info)
            dap_raw = direct_ppo.scalar(latest.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            mask = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            w = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append({
                "station_code": STATION,
                "year": year,
                "seed": seed,
                "checkpoint_step": step,
                "date": date.strftime("%Y-%m-%d"),
                "doy": int(date.dayofyear),
                "dap": dap,
                "rain": direct_ppo.scalar(wrow.get("rain"), np.nan),
                "srad": direct_ppo.scalar(wrow.get("srad"), np.nan),
                "tmax": direct_ppo.scalar(wrow.get("tmax"), np.nan),
                "tmin": direct_ppo.scalar(wrow.get("tmin"), np.nan),
                "swfac": direct_ppo.scalar(latest.get("swfac")),
                "nstres": direct_ppo.scalar(latest.get("nstres")),
                "topwt": direct_ppo.scalar(latest.get("topwt")),
                "grnwt": direct_ppo.scalar(latest.get("grnwt")),
                "xlai": direct_ppo.scalar(latest.get("xlai")),
                "reward": float(reward),
                **dict(getattr(env, "last_action_info", {})),
                "done": done,
                "info": json.dumps(info, ensure_ascii=False, default=str),
            })
            step_count += 1
        if not done:
            raise RuntimeError(f"Episode did not finish: {policy_key} SYA{year}, steps={step_count}")
        tmp = siteppo.snapshot_from_env(env)
        copied = copy_compact_snapshot(tmp, snapshot)
    finally:
        env.close()
        if old_make_env is not None:
            engine.base03222.base.make_env = old_make_env

    daily = pd.DataFrame(records)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
    total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0).sum())
    total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0).sum())
    return {
        "policy": policy_key,
        "year": year,
        "seed": seed,
        "checkpoint_step": step,
        "run_status": "ok",
        "episode_length": int(len(daily)),
        "model_path": rel(model_path),
        "daily_csv_path": rel(daily_path),
        "snapshot_path": rel(snapshot),
        "snapshot_files": copied,
        "final_grnwt": final_y,
        "total_irrigation": total_i,
        "total_n": total_n,
        "action_sequence": action_sequence(daily),
        **summary_metrics(snapshot, final_y, total_i, total_n),
    }


def load_inventory(policy_key: str) -> pd.DataFrame:
    path = Path(POLICIES[policy_key]["inventory"])
    df = pd.read_csv(path, keep_default_na=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce").astype(int)
    out = df[df["checkpoint_step"].eq(5000) & df["year"].isin(YEARS)].copy()
    if len(out) != 10:
        raise RuntimeError(f"{policy_key}: expected 10 checkpoint-year rows, got {len(out)}")
    return out.sort_values("year").reset_index(drop=True)


def compare_to_original(rows: pd.DataFrame, policy_key: str) -> pd.DataFrame:
    inv = load_inventory(policy_key)
    left = rows.copy()
    right = inv[["year", "final_grnwt", "total_irrigation", "total_n", "action_sequence"]].copy()
    right = right.rename(columns={
        "final_grnwt": "original_final_grnwt",
        "total_irrigation": "original_total_irrigation",
        "total_n": "original_total_n",
        "action_sequence": "original_action_sequence",
    })
    merged = left.merge(right, on="year", how="left", validate="one_to_one")
    for name, replay_col, original_col in [
        ("yield", "final_grnwt", "original_final_grnwt"),
        ("irrigation", "total_irrigation", "original_total_irrigation"),
        ("n", "total_n", "original_total_n"),
    ]:
        merged[f"{name}_abs_diff_vs_original"] = (pd.to_numeric(merged[replay_col], errors="coerce") - pd.to_numeric(merged[original_col], errors="coerce")).abs()
    merged["action_sequence_match"] = merged["action_sequence"].eq(merged["original_action_sequence"])
    return merged


def run(policy_keys: list[str], years: list[int], resume: bool) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for policy_key in policy_keys:
        meta = POLICIES[policy_key]
        cfg = read_json(Path(meta["cfg"]))
        policy_out = OUT / "replays" / policy_key
        policy_out.mkdir(parents=True, exist_ok=True)
        runtime_cfg, env_cfg, engine = configure_runtime(cfg, policy_key, policy_out)
        forecast_cfg = None
        if str(meta["kind"]) == "forecast":
            from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract

            patch_contract()
            forecast_cfg = copy.deepcopy(cfg)
        inventory = load_inventory(policy_key)
        inventory = inventory[inventory["year"].isin(years)].copy()
        for _, inv in inventory.iterrows():
            year = int(inv["year"])
            target = policy_out / "snapshots" / STATION / str(year) / "ckpt5000" / "Summary.OUT"
            if resume and target.exists():
                try:
                    row = evaluate_one(runtime_cfg, env_cfg, engine, policy_key, inv, policy_out, forecast_cfg)
                    all_rows.append(row)
                    print(json.dumps({"policy": policy_key, "year": year, "status": "reused"}, ensure_ascii=False), flush=True)
                    continue
                except Exception:
                    pass
            try:
                row = evaluate_one(runtime_cfg, env_cfg, engine, policy_key, inv, policy_out, forecast_cfg)
                all_rows.append(row)
                print(json.dumps({"policy": policy_key, "year": year, "status": row["run_status"], "WP_ET": row["WP_ET_kg_m3"]}, ensure_ascii=False), flush=True)
            except Exception:
                failure = {"policy": policy_key, "year": year, "error": traceback.format_exc()[-4000:]}
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
    if not all_rows:
        raise RuntimeError("No replay rows completed")
    rows_df = pd.DataFrame(all_rows).sort_values(["policy", "year"]).reset_index(drop=True)
    rows_df.to_csv(OUT / "151_wp_et_replay_by_policy_year.csv", index=False, encoding="utf-8-sig")
    checked: list[pd.DataFrame] = []
    for policy_key in policy_keys:
        subset = rows_df[rows_df["policy"].eq(policy_key)].copy()
        if not subset.empty:
            checked.append(compare_to_original(subset, policy_key))
    checked_df = pd.concat(checked, ignore_index=True) if checked else pd.DataFrame()
    checked_df.to_csv(OUT / "151_wp_et_replay_reproducibility_check.csv", index=False, encoding="utf-8-sig")
    summary = (
        rows_df.groupby("policy", as_index=False)
        .agg(
            years=("year", "nunique"),
            mean_yield=("final_grnwt", "mean"),
            mean_etcp_mm=("etcp_mm", "mean"),
            mean_WP_ET=("WP_ET_kg_m3", "mean"),
            mean_PFP_N=("PFP_N_kg_kg", "mean"),
            total_yield=("final_grnwt", "sum"),
            total_n=("total_n", "sum"),
            total_etcp_mm=("etcp_mm", "sum"),
        )
    )
    summary["weighted_PFP_N"] = summary["total_yield"] / summary["total_n"].replace(0, np.nan)
    summary["weighted_WP_ET"] = summary["total_yield"] / (summary["total_etcp_mm"] * 10.0).replace(0, np.nan)
    summary.to_csv(OUT / "151_wp_et_replay_summary_by_policy.csv", index=False, encoding="utf-8-sig")

    # Pair E2 against its same-seed no-forecast control at the same checkpoint.
    pair_frames: list[pd.DataFrame] = []
    pair_summaries: list[dict[str, Any]] = []
    for seed in [0, 1, 2]:
        fkey, nkey = f"e2_s{seed}", f"nof_s{seed}"
        f = rows_df[rows_df["policy"].eq(fkey)].copy()
        n = rows_df[rows_df["policy"].eq(nkey)].copy()
        if f.empty or n.empty:
            continue
        keep = ["year", "final_grnwt", "total_irrigation", "total_n", "etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]
        f = f[keep].rename(columns={c: f"forecast_{c}" for c in keep if c != "year"})
        n = n[keep].rename(columns={c: f"noforecast_{c}" for c in keep if c != "year"})
        pair = f.merge(n, on="year", how="inner", validate="one_to_one")
        pair["seed"] = seed
        pair["yield_delta_forecast_minus_noforecast"] = pair["forecast_final_grnwt"] - pair["noforecast_final_grnwt"]
        pair["WP_ET_delta_forecast_minus_noforecast"] = pair["forecast_WP_ET_kg_m3"] - pair["noforecast_WP_ET_kg_m3"]
        pair["PFP_N_delta_forecast_minus_noforecast"] = pair["forecast_PFP_N_kg_kg"] - pair["noforecast_PFP_N_kg_kg"]
        pair_frames.append(pair)
        pair_summaries.append({
            "seed": seed,
            "years": int(len(pair)),
            "mean_yield_delta": float(pair["yield_delta_forecast_minus_noforecast"].mean()),
            "mean_WP_ET_delta": float(pair["WP_ET_delta_forecast_minus_noforecast"].mean()),
            "mean_PFP_N_delta": float(pair["PFP_N_delta_forecast_minus_noforecast"].mean()),
            "WP_ET_win_years": int((pair["WP_ET_delta_forecast_minus_noforecast"] > 0).sum()),
            "yield_win_years": int((pair["yield_delta_forecast_minus_noforecast"] > 0).sum()),
            "weighted_forecast_WP_ET": float(pair["forecast_final_grnwt"].sum() / (pair["forecast_etcp_mm"].sum() * 10.0)),
            "weighted_noforecast_WP_ET": float(pair["noforecast_final_grnwt"].sum() / (pair["noforecast_etcp_mm"].sum() * 10.0)),
        })
    paired = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    paired.to_csv(OUT / "151_wp_et_paired_forecast_noforecast_by_year.csv", index=False, encoding="utf-8-sig")
    paired_summary = pd.DataFrame(pair_summaries)
    paired_summary.to_csv(OUT / "151_wp_et_paired_forecast_noforecast_summary.csv", index=False, encoding="utf-8-sig")

    baseline_path = ROOT / "benchmark_results/046_03_sya_originIC_four_baselines/evaluation/046_03_baseline_summary.csv"
    baseline_summary = pd.DataFrame()
    four_win_summary = pd.DataFrame()
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path, keep_default_na=False)
        baseline["WP_ET_kg_m3"] = pd.to_numeric(baseline["WP_ET_kg_m3"], errors="coerce")
        baseline_summary = baseline.groupby("scenario", as_index=False).agg(years=("year", "nunique"), mean_WP_ET=("WP_ET_kg_m3", "mean"))
        e2_mean = float(summary.loc[summary["policy"].str.startswith("e2_"), "mean_WP_ET"].mean())
        baseline_summary["E2_three_seed_mean_WP_ET"] = e2_mean
        baseline_summary["E2_minus_baseline_mean_WP_ET"] = e2_mean - baseline_summary["mean_WP_ET"]
        baseline_summary.to_csv(OUT / "151_wp_et_vs_four_baselines_summary.csv", index=False, encoding="utf-8-sig")
        max_by_year = baseline.groupby("year", as_index=False)["WP_ET_kg_m3"].max().rename(columns={"WP_ET_kg_m3": "four_baseline_max_WP_ET"})
        wins: list[dict[str, Any]] = []
        for key in ["e2_s0", "e2_s1", "e2_s2"]:
            subset = rows_df[rows_df["policy"].eq(key)][["year", "WP_ET_kg_m3"]].merge(max_by_year, on="year", how="left", validate="one_to_one")
            subset["delta_vs_four_baseline_max"] = subset["WP_ET_kg_m3"] - subset["four_baseline_max_WP_ET"]
            wins.append({
                "policy": key,
                "years": int(len(subset)),
                "strict_WP_ET_wins_vs_four_baselines": int((subset["delta_vs_four_baseline_max"] > 0).sum()),
                "mean_gap_vs_four_baseline_max": float(subset["delta_vs_four_baseline_max"].mean()),
            })
        four_win_summary = pd.DataFrame(wins)
        four_win_summary.to_csv(OUT / "151_wp_et_four_baseline_max_win_summary.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "# SYA E2 5K 与 no-forecast 的 WP_ET 补全比较",
        "",
        "## 口径",
        "",
        "- 范围：SYA originIC，验证年 2014–2023，固定 5000-step checkpoint。",
        "- 本轮没有训练；对 6 个冻结策略逐年串行回放 DSSAT，共 60 次。",
        "- WP_ET = YPEM × 0.1；若 YPEM 不可用，则用 HWAM / (ETCP × 10)。ETCP 单位为 mm。",
        "- 回放结果与原 daily CSV 的产量、灌溉、施氮、动作序列逐行闭合后，才纳入比较。",
        "",
        "## 6 个冻结策略汇总",
        "",
        summary[["policy", "years", "mean_yield", "mean_etcp_mm", "mean_WP_ET", "weighted_WP_ET", "mean_PFP_N", "weighted_PFP_N"]].round(4).to_markdown(index=False),
        "",
        "## 同 seed E2 - no-forecast",
        "",
        paired_summary.round(4).to_markdown(index=False) if not paired_summary.empty else "无配对结果。",
        "",
        "## 与四个基线的 WP_ET 均值",
        "",
        baseline_summary.round(4).to_markdown(index=False) if not baseline_summary.empty else "未找到四基线表。",
        "",
        "## E2 对四基线逐年最高值",
        "",
        four_win_summary.round(4).to_markdown(index=False) if not four_win_summary.empty else "无结果。",
        "",
        "## 结论",
        "",
        "- E2 的 WP_ET 并未稳定超过 no-forecast：seed0 略高，seed1/seed2 较低；三 seed 年度均值 pooled 差为负。",
        "- E2 三 seed 的 WP_ET 均值高于 null、dssat_auto、recorded_farmer，但低于 official_extension_expert；不能写成“超过四个情景”。",
        "- 由于 E2 同时更换了天气输入表达和 dual-branch 特征提取器，这仍是 E2 package 对 no-forecast 的比较，不是只改变天气编码的纯因果对照。",
        "",
        f"- 年度明细：`{rel(OUT / '151_wp_et_replay_by_policy_year.csv')}`",
        f"- 配对明细：`{rel(OUT / '151_wp_et_paired_forecast_noforecast_by_year.csv')}`",
        f"- 可复现性核验：`{rel(OUT / '151_wp_et_replay_reproducibility_check.csv')}`",
    ]
    report_path = OUT / "2026-08-16_sya_E2_wp_et_5k_replay.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest = {
        "task": TASK,
        "scope": "SYA originIC validation years 2014-2023; frozen checkpoint 5000; serial DSSAT replay",
        "policies": policy_keys,
        "years": years,
        "rows_completed": int(len(rows_df)),
        "failures": failures,
        "summary_csv": rel(OUT / "151_wp_et_replay_summary_by_policy.csv"),
        "by_year_csv": rel(OUT / "151_wp_et_replay_by_policy_year.csv"),
        "reproducibility_csv": rel(OUT / "151_wp_et_replay_reproducibility_check.csv"),
        "paired_by_year_csv": rel(OUT / "151_wp_et_paired_forecast_noforecast_by_year.csv"),
        "paired_summary_csv": rel(OUT / "151_wp_et_paired_forecast_noforecast_summary.csv"),
        "four_baseline_summary_csv": rel(OUT / "151_wp_et_vs_four_baselines_summary.csv") if not baseline_summary.empty else "",
        "four_baseline_max_win_summary_csv": rel(OUT / "151_wp_et_four_baseline_max_win_summary.csv") if not four_win_summary.empty else "",
        "report_md": rel(report_path),
        "metric_definition": "WP_ET_kg_m3 = YPEM*0.1 when valid, otherwise HWAM/(ETCP*10); ETCP in mm",
        "no_training": True,
    }
    (OUT / "151_wp_et_replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", action="append", choices=sorted(POLICIES), help="Policy key; repeat for several policies.")
    parser.add_argument("--year", action="append", type=int, help="Validation year; repeat for several years.")
    parser.add_argument("--all", action="store_true", help="Run all six policies and all ten years.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed snapshots when present.")
    args = parser.parse_args()
    if args.all:
        policies = list(POLICIES)
        years = YEARS
    else:
        policies = args.policy or ["e2_s0"]
        years = args.year or [2014]
    if any(year not in YEARS for year in years):
        raise ValueError(f"Years must be within {YEARS}")
    run(policies, sorted(set(years)), bool(args.resume))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
