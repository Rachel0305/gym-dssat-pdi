"""178: complete WP_ET for the frozen HL/YC/FQ transfer checkpoints.

This is a read-only serial DSSAT replay.  It does not train or overwrite the
original transfer outputs.  The replay is deliberately limited to the
promising FQ 2K/5K pair and the YC 2K pair, and writes compact Summary.OUT
snapshots plus paired WP_ET/PFP-N tables.
"""

from __future__ import annotations

import contextlib
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
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_sya_originIC_expanded_action_maskableppo_046_10 as nof
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
import run_141E1_sya_actionable_weather_encoding_smoke2k as e1
import run_176_multisite_weather_framework_transfer as transfer  # registers the saved extractor
from ppo_evaluate import latest_observation_dict


TASK = "178_multisite_wp_et_replay"
OUT = ROOT / "benchmark_results" / TASK
REPORT_PATH = ROOT / "docs" / "178_multisite_wp_et_replay_record.md"

POLICIES: dict[str, dict[str, Any]] = {
    "fqa_forecast_2k": {
        "station": "FQA", "site": "FQ", "seed": 1, "checkpoint_step": 2000,
        "kind": "forecast", "cfg": ROOT / "configs/174FQA_fqa_originIC_dual_branch_weather_reader_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/174FQA_fqa_originIC_dual_branch_weather_reader_5k_seed1/evaluation/174FQA_checkpoint_validation_summary.csv",
    },
    "fqa_noforecast_2k": {
        "station": "FQA", "site": "FQ", "seed": 1, "checkpoint_step": 2000,
        "kind": "noforecast", "cfg": ROOT / "configs/175FQA_fqa_originIC_dual_branch_noforecast_control_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/175FQA_fqa_originIC_dual_branch_noforecast_control_5k_seed1/evaluation/175FQA_checkpoint_validation_summary.csv",
    },
    "fqa_forecast_5k": {
        "station": "FQA", "site": "FQ", "seed": 1, "checkpoint_step": 5000,
        "kind": "forecast", "cfg": ROOT / "configs/174FQA_fqa_originIC_dual_branch_weather_reader_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/174FQA_fqa_originIC_dual_branch_weather_reader_5k_seed1/evaluation/174FQA_checkpoint_validation_summary.csv",
    },
    "fqa_noforecast_5k": {
        "station": "FQA", "site": "FQ", "seed": 1, "checkpoint_step": 5000,
        "kind": "noforecast", "cfg": ROOT / "configs/175FQA_fqa_originIC_dual_branch_noforecast_control_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/175FQA_fqa_originIC_dual_branch_noforecast_control_5k_seed1/evaluation/175FQA_checkpoint_validation_summary.csv",
    },
    "yca_forecast_2k": {
        "station": "YCA", "site": "YC", "seed": 1, "checkpoint_step": 2000,
        "kind": "forecast", "cfg": ROOT / "configs/172YCA_yca_lowIC_dual_branch_weather_reader_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/172YCA_yca_lowIC_dual_branch_weather_reader_5k_seed1/evaluation/172YCA_checkpoint_validation_summary.csv",
    },
    "yca_noforecast_2k": {
        "station": "YCA", "site": "YC", "seed": 1, "checkpoint_step": 2000,
        "kind": "noforecast", "cfg": ROOT / "configs/173YCA_yca_lowIC_dual_branch_noforecast_control_5k_seed1.json",
        "inventory": ROOT / "benchmark_results/173YCA_yca_lowIC_dual_branch_noforecast_control_5k_seed1/evaluation/173YCA_checkpoint_validation_summary.csv",
    },
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_runtime(meta: dict[str, Any], policy_out: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    cfg = read_json(Path(meta["cfg"]))
    engine = nof.engine
    actions = cfg["actions"]
    engine.BINARY_IRRIGATION_LEVELS = list(map(float, actions["irrigation_levels_mm"]))
    engine.BINARY_NITROGEN_LEVELS = list(map(float, actions["nitrogen_levels_kg_ha"]))
    engine.LOWIC_INPUT_ROOT = nof.base04602.INPUT_PROFILES[str(cfg["input_profile"])]
    engine.STATION = str(meta["station"])
    engine.SITES = [str(meta["station"])]
    engine.base03222.load_config = engine.load_config
    engine.base03222.base.StressAwareDiscreteWrapper = engine.base04036.LateIrrigationReserveMaskWrapper
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
    runtime_cfg = json.loads(json.dumps(engine.load_config()))
    runtime_cfg["paths"]["output_root"] = rel(policy_out)
    split = engine.base03222.load_split()
    split = split[split["station_code"].eq(str(meta["station"]))].copy()
    selection = engine.base03222.build_selection(split)
    env_cfg = direct_ppo.build_env_config(runtime_cfg, selection)
    env_cfg["paths"]["output_root"] = rel(policy_out)
    (policy_out / "configs").mkdir(parents=True, exist_ok=True)
    direct_ppo.write_yaml(env_cfg, policy_out / "configs" / "resolved_env_config.yaml")
    return runtime_cfg, env_cfg, engine


def inventory(meta: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(meta["inventory"], keep_default_na=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce").astype(int)
    out = df[df["checkpoint_step"].eq(int(meta["checkpoint_step"]))].copy()
    if len(out) != 10:
        raise RuntimeError(f"{meta}: expected 10 rows, got {len(out)}")
    return out.sort_values("year").reset_index(drop=True)


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


def summary_metrics(snapshot: Path, final_yield: float, expected_i: float, expected_n: float) -> dict[str, Any]:
    rows = siteppo.parse_summary_out(snapshot / "Summary.OUT")
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        hwam = siteppo.num(row, "HWAM")
        if hwam is None:
            continue
        score = abs(float(hwam) - float(final_yield)) + abs(float(siteppo.num(row, "IRCM") or 0.0) - expected_i) + abs(float(siteppo.num(row, "NICM") or 0.0) - expected_n)
        candidates.append((score, -idx, row))
    if not candidates:
        raise RuntimeError(f"No usable Summary.OUT row in {snapshot}")
    score, neg_idx, row = min(candidates, key=lambda x: (x[0], x[1]))
    etcp = siteppo.num(row, "ETCP")
    ypem = siteppo.num(row, "YPEM")
    ypnam = siteppo.num(row, "YPNAM")
    ircm = siteppo.num(row, "IRCM")
    nicm = siteppo.num(row, "NICM")
    if etcp is None or float(etcp) <= 0:
        raise RuntimeError(f"Invalid ETCP in {snapshot}: {etcp}")
    wp = float(ypem) * 0.1 if ypem is not None and float(ypem) >= 0 else float(final_yield) / (float(etcp) * 10.0)
    pfp = float(ypnam) if nicm and float(nicm) > 0 and ypnam is not None and float(ypnam) >= 0 else math.nan
    return {
        "summary_row_index": int(-neg_idx), "summary_match_score": float(score),
        "summary_irrigation_total": float(ircm or 0.0), "summary_nitrogen_total": float(nicm or 0.0),
        "etcp_mm": float(etcp), "WP_ET_kg_m3": float(wp), "PFP_N_kg_kg": float(pfp),
    }


def action_sequence(daily: pd.DataFrame) -> str:
    dap = pd.to_numeric(daily["dap"], errors="coerce")
    irr = pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0)
    nit = pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0)
    return "; ".join(f"DAP{int(d)} I{float(i):g}/N{float(n):g}" for d, i, n in zip(dap, irr, nit) if float(i) > 1e-9 or float(n) > 1e-9)


def evaluate_one(meta: dict[str, Any], inv: pd.Series, runtime_cfg: dict[str, Any], env_cfg: dict[str, Any], engine: Any, policy_out: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks
    station, seed, step = str(meta["station"]), int(meta["seed"]), int(meta["checkpoint_step"])
    model_path = ROOT / str(inv["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    snapshot = policy_out / "snapshots" / station / str(int(inv["year"])) / f"ckpt{step}"
    daily_path = policy_out / "daily_outputs" / station / f"{station}_{int(inv['year'])}_seed{seed}_ckpt{step}_daily.csv"
    if (snapshot / "Summary.OUT").exists() and daily_path.exists():
        daily = pd.read_csv(daily_path)
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0).sum())
        total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0).sum())
        return {"policy": meta["key"], "station": station, "site": meta["site"], "year": int(inv["year"]), "seed": seed, "checkpoint_step": step, "run_status": "ok_existing", "final_grnwt": final_y, "total_irrigation": total_i, "total_n": total_n, "action_sequence": action_sequence(daily), **summary_metrics(snapshot, final_y, total_i, total_n)}

    model = MaskablePPO.load(str(model_path), device="cpu")
    old_make_env = engine.base03222.base.make_env
    cfg = read_json(Path(meta["cfg"]))
    e1.patch_contract()
    if meta["kind"] == "forecast":
        engine.base03222.base.make_env = transfer.forecast.make_forecast_env_factory(old_make_env, cfg)
    else:
        engine.base03222.base.make_env = transfer.make_zero_weather_factory(old_make_env)
    year = int(inv["year"])
    run_tag = f"{station}_{year}_{meta['key']}_ckpt{step}_wp_et_replay"
    env = engine.base03222.base.make_env(runtime_cfg, env_cfg, station, year, seed, run_tag, evaluation=True)
    records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        planting = pd.Timestamp(direct_ppo.find_year(env_cfg, station, year)["planting_date"])
        weather = direct_ppo.weather_for_daily(runtime_cfg)
        weather = weather[weather["station_code"].astype(str).eq(station)].copy()
        weather["date"] = pd.to_datetime(weather["date"])
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
            w = weather[weather["date"].eq(date)]
            wrow = w.iloc[0].to_dict() if len(w) else {}
            records.append({"station_code": station, "site": meta["site"], "year": year, "seed": seed, "checkpoint_step": step, "date": date.strftime("%Y-%m-%d"), "doy": int(date.dayofyear), "dap": dap, "rain": direct_ppo.scalar(wrow.get("rain"), np.nan), "srad": direct_ppo.scalar(wrow.get("srad"), np.nan), "tmax": direct_ppo.scalar(wrow.get("tmax"), np.nan), "tmin": direct_ppo.scalar(wrow.get("tmin"), np.nan), "swfac": direct_ppo.scalar(latest.get("swfac")), "nstres": direct_ppo.scalar(latest.get("nstres")), "topwt": direct_ppo.scalar(latest.get("topwt")), "grnwt": direct_ppo.scalar(latest.get("grnwt")), "xlai": direct_ppo.scalar(latest.get("xlai")), "reward": float(reward), **dict(getattr(env, "last_action_info", {})), "done": done, "info": json.dumps(info, ensure_ascii=False, default=str)})
            step_count += 1
        if not done:
            raise RuntimeError(f"Episode did not finish: {meta['key']} {station}{year}, steps={step_count}")
        snapshot_src = siteppo.snapshot_from_env(env)
        copied = copy_compact_snapshot(snapshot_src, snapshot)
    finally:
        env.close()
        engine.base03222.base.make_env = old_make_env
    daily = pd.DataFrame(records)
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
    total_i = float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").fillna(0.0).sum())
    total_n = float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").fillna(0.0).sum())
    return {"policy": meta["key"], "station": station, "site": meta["site"], "year": int(year), "seed": seed, "checkpoint_step": step, "run_status": "ok", "episode_length": len(daily), "model_path": rel(model_path), "daily_csv_path": rel(daily_path), "snapshot_path": rel(snapshot), "snapshot_files": copied, "final_grnwt": final_y, "total_irrigation": total_i, "total_n": total_n, "action_sequence": action_sequence(daily), **summary_metrics(snapshot, final_y, total_i, total_n)}


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for key, raw_meta in POLICIES.items():
        meta = {**raw_meta, "key": key}
        policy_out = OUT / "replays" / key
        policy_out.mkdir(parents=True, exist_ok=True)
        runtime_cfg, env_cfg, engine = configure_runtime(meta, policy_out)
        for _, inv in inventory(meta).iterrows():
            try:
                row = evaluate_one(meta, inv, runtime_cfg, env_cfg, engine, policy_out)
                all_rows.append(row)
                print(json.dumps({"policy": key, "year": int(inv["year"]), "WP_ET": row["WP_ET_kg_m3"], "status": row["run_status"]}, ensure_ascii=False), flush=True)
            except Exception:
                failure = {"policy": key, "year": int(inv["year"]), "error": traceback.format_exc()[-5000:]}
                failures.append(failure)
                print(json.dumps(failure, ensure_ascii=False), flush=True)
    if not all_rows:
        raise RuntimeError("No multisite WP_ET replay rows completed")
    rows = pd.DataFrame(all_rows).sort_values(["station", "checkpoint_step", "policy", "year"]).reset_index(drop=True)
    by_year = OUT / "178_wp_et_replay_by_policy_year.csv"
    summary = OUT / "178_wp_et_replay_summary_by_policy.csv"
    rows.to_csv(by_year, index=False, encoding="utf-8-sig")
    summary_df = rows.groupby(["station", "site", "policy", "checkpoint_step"], as_index=False).agg(years=("year", "nunique"), mean_yield=("final_grnwt", "mean"), mean_etcp_mm=("etcp_mm", "mean"), mean_WP_ET=("WP_ET_kg_m3", "mean"), total_yield=("final_grnwt", "sum"), total_etcp_mm=("etcp_mm", "sum"), mean_PFP_N=("PFP_N_kg_kg", "mean"), total_n=("total_n", "sum"))
    summary_df["weighted_WP_ET"] = summary_df["total_yield"] / (summary_df["total_etcp_mm"] * 10.0)
    summary_df["weighted_PFP_N"] = summary_df["total_yield"] / summary_df["total_n"].replace(0, np.nan)
    summary_df.to_csv(summary, index=False, encoding="utf-8-sig")
    pairs: list[pd.DataFrame] = []
    for station, checkpoint in [("FQA", 2000), ("FQA", 5000), ("YCA", 2000)]:
        sub = rows[(rows["station"].eq(station)) & (rows["checkpoint_step"].eq(checkpoint))].copy()
        f = sub[sub["policy"].str.contains("forecast") & ~sub["policy"].str.contains("noforecast")]
        n = sub[sub["policy"].str.contains("noforecast")]
        keep = ["year", "final_grnwt", "total_irrigation", "total_n", "etcp_mm", "WP_ET_kg_m3", "PFP_N_kg_kg"]
        if f.empty or n.empty:
            continue
        f = f[keep].rename(columns={c: f"forecast_{c}" for c in keep if c != "year"})
        n = n[keep].rename(columns={c: f"noforecast_{c}" for c in keep if c != "year"})
        pair = f.merge(n, on="year", how="inner", validate="one_to_one")
        pair.insert(0, "station", station)
        pair.insert(1, "checkpoint_step", checkpoint)
        pair["yield_delta_forecast_minus_noforecast"] = pair["forecast_final_grnwt"] - pair["noforecast_final_grnwt"]
        pair["WP_ET_delta_forecast_minus_noforecast"] = pair["forecast_WP_ET_kg_m3"] - pair["noforecast_WP_ET_kg_m3"]
        pair["PFP_N_delta_forecast_minus_noforecast"] = pair["forecast_PFP_N_kg_kg"] - pair["noforecast_PFP_N_kg_kg"]
        pairs.append(pair)
    paired = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    paired_path = OUT / "178_wp_et_paired_delta_by_year.csv"
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")
    paired_summary = pd.DataFrame()
    if not paired.empty:
        paired_summary = paired.groupby(["station", "checkpoint_step"], as_index=False).agg(years=("year", "nunique"), mean_yield_delta=("yield_delta_forecast_minus_noforecast", "mean"), mean_WP_ET_delta=("WP_ET_delta_forecast_minus_noforecast", "mean"), mean_PFP_N_delta=("PFP_N_delta_forecast_minus_noforecast", "mean"), WP_ET_win_years=("WP_ET_delta_forecast_minus_noforecast", lambda x: int((x > 0).sum())), yield_win_years=("yield_delta_forecast_minus_noforecast", lambda x: int((x > 0).sum())))
        paired_summary.to_csv(OUT / "178_wp_et_paired_summary.csv", index=False, encoding="utf-8-sig")
    report = OUT / "178_multisite_wp_et_replay_record.md"
    lines = ["# 178 HL/YC/FQ transfer WP_ET replay", "", "- 本轮只读串行回放冻结 checkpoint，不训练，不覆盖原始结果。", "- FQ：2K/5K forecast 与 no-forecast；YC：2K forecast 与 no-forecast。", "- WP_ET = YPEM × 0.1；若 YPEM 不可用，则用 HWAM / (ETCP × 10)，ETCP 单位为 mm。", "", "## policy summary", "", summary_df.round(4).to_markdown(index=False), "", "## paired deltas", "", paired_summary.round(4).to_markdown(index=False) if not paired_summary.empty else "无配对结果。", "", f"- 逐年明细：`{rel(by_year)}`", f"- 配对逐年差值：`{rel(paired_path)}`", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    manifest = {"task": TASK, "policies": list(POLICIES), "rows_completed": len(rows), "failures": failures, "summary_csv": rel(summary), "by_year_csv": rel(by_year), "paired_delta_csv": rel(paired_path), "paired_summary_csv": rel(OUT / "178_wp_et_paired_summary.csv") if not paired_summary.empty else "", "no_training": True, "metric_definition": "WP_ET_kg_m3 = YPEM*0.1 when valid, otherwise HWAM/(ETCP*10); ETCP in mm"}
    (OUT / "178_wp_et_replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text("# 178 HL/YC/FQ transfer WP_ET replay\n\n" + "\n".join(lines[2:]), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
