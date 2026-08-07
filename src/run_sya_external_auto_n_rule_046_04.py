"""046_04: DSSAT auto-irrigation plus external NSTRES-triggered N rule.

It is deliberately a no-learning baseline.  Its management modes are rendered
as IRRIG=A and FERTI=L: DSSAT handles irrigation, while the gym-DSSAT action
channel applies only the transparent nitrogen rule.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline
from build_relaxed_success_five_scenario_daily_evidence_027_05 import parse_management_events


DEFAULT_CONFIG = ROOT / "configs" / "046_02_sya_originIC_binary_timing_ppo.json"
PROMPT = ROOT / "prompts" / "046_04_sya_external_auto_n_rule.md"
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}
SCENARIO = "dssat_auto_irrigation_external_n_rule"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if str(cfg.get("station_code")) != "SYA":
        raise ValueError("046_04 仅适用于 SYA")
    if str(cfg.get("input_profile")) not in INPUT_PROFILES:
        raise ValueError("input_profile 必须为 originIC 或 lowIC")
    rule = cfg.get("external_auto_n_rule", {})
    for key in ["nitrogen_stress_threshold", "nitrogen_dose_kg_ha", "min_days_between_nitrogen", "season_nitrogen_cap_kg_ha", "fertilization_last_dap"]:
        if key not in rule:
            raise ValueError(f"external_auto_n_rule 缺少 {key}")
    return cfg


def run_suffix(run_id: str = "") -> str:
    run_id = str(run_id).strip()
    return f"_run_{run_id}" if run_id else ""


def output_root(cfg: dict[str, Any], run_id: str = "") -> Path:
    suffix = str(cfg.get("external_auto_n_rule", {}).get("output_suffix", "")).strip()
    name = f"046_04_sya_{cfg['input_profile']}_external_auto_n_rule"
    if suffix:
        name = f"{name}_{suffix}"
    name = f"{name}{run_suffix(run_id)}"
    return ROOT / "benchmark_results" / name


def executed_nitrogen_from_mgmt_event(snapshot: Path) -> float:
    events = parse_management_events(snapshot / "MgmtEvent.OUT")
    if events.empty:
        return 0.0
    fertilizer = events[events["operation"].astype(str).str.lower().str.startswith("fert")]
    return float(pd.to_numeric(fertilizer["amount"], errors="coerce").fillna(0.0).sum())


def selected_rows(years: list[int]) -> pd.DataFrame:
    split = pd.read_csv(baseline.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    out = split[split["station_code"].astype(str).eq("SYA") & split["year"].isin(years)].copy()
    if len(out) != len(years):
        raise RuntimeError("部分验证年份不在 split 表中")
    return out.sort_values("year").reset_index(drop=True)


def build_configs(out: Path, selected: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    run_config = direct_ppo.load_yaml(baseline.BASE_CONFIG)
    run_config = json.loads(json.dumps(run_config))
    run_config["seed"] = 0
    run_config["paths"]["output_root"] = rel(out)
    run_config["runtime"]["max_steps"] = 260
    env_config = baseline.build_env_config(run_config, selected)
    env_config["paths"]["output_root"] = rel(out)
    env_config["runtime"]["max_steps"] = 260
    env_config["seed"] = 0
    return run_config, env_config


def make_env_auto_irrigation_external_n(env_config: dict[str, Any], year: int, run_tag: str):
    baseline.ensure_project_on_path()
    import gym
    from sb3_wrapper import GymDssatWrapper

    info = direct_ppo.find_year(env_config, "SYA", int(year))
    env_args = baseline.build_env_args(
        station="SYA", year=int(year), planting_date=info["planting_date"], seed=0,
        config=env_config, run_tag=run_tag, evaluation=True, mode=env_config.get("runtime", {}).get("mode", "all"),
    )
    template = Path(env_args["fileX_template_path"])
    text = template.read_text(encoding="utf-8", errors="replace")
    text = baseline.set_auto_treatment_one(text, int(year))
    # The automatic MZX block remains for irrigation; FERTI=L makes the direct
    # gym action channel the sole source of nitrogen applications.
    text = baseline.set_management_for_treatment(text, 1, "A", "L")
    template.write_text(text, encoding="utf-8")
    return GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped), env_args, text


def evaluate_year(run_config: dict[str, Any], env_config: dict[str, Any], year: int, rule: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    env, env_args, rendered_text = make_env_auto_irrigation_external_n(env_config, year, f"SYA_{year}_046_04_external_auto_n")
    rows: list[dict[str, Any]] = []
    last_n_dap = -10_000
    total_requested_n = 0.0
    planting = pd.Timestamp(direct_ppo.find_year(env_config, "SYA", int(year))["planting_date"])
    weather = direct_ppo.weather_for_daily(run_config)
    try:
        obs, info = env.reset()
        done, steps = False, 0
        while not done and steps < 260:
            latest_pre = baseline.latest_observation_dict(env, obs, info)
            dap = int(round(float(baseline.scalar(latest_pre.get("dap", steps + 1), steps + 1))))
            nstres_pre = float(baseline.scalar(latest_pre.get("nstres"), 0.0))
            can_apply = (
                nstres_pre >= float(rule["nitrogen_stress_threshold"])
                and dap <= int(rule["fertilization_last_dap"])
                and dap - last_n_dap >= int(rule["min_days_between_nitrogen"])
                and total_requested_n < float(rule["season_nitrogen_cap_kg_ha"]) - 1e-9
            )
            amount = min(float(rule["nitrogen_dose_kg_ha"]), float(rule["season_nitrogen_cap_kg_ha"]) - total_requested_n) if can_apply else 0.0
            action = baseline.step_scheduled_action(env, {"amir": 0.0, "anfer": amount})
            obs, reward, terminated, truncated, info = env.step(action)
            if amount > 0:
                last_n_dap = dap
                total_requested_n += amount
            latest = baseline.latest_observation_dict(env, obs, info)
            date = planting + pd.Timedelta(days=max(dap - 1, 0))
            wrow = baseline.weather_row(weather, "SYA", date)
            rows.append({
                "station_code": "SYA", "site": "SY", "year": int(year), "scenario": SCENARIO,
                "date": date.strftime("%Y-%m-%d"), "dap": dap,
                "rain": baseline.scalar(wrow.get("rain"), np.nan), "tmax": baseline.scalar(wrow.get("tmax"), np.nan),
                "tmin": baseline.scalar(wrow.get("tmin"), np.nan), "srad": baseline.scalar(wrow.get("srad"), np.nan),
                "grnwt": baseline.scalar(latest.get("grnwt")), "topwt": baseline.scalar(latest.get("topwt")),
                "swfac": baseline.scalar(latest.get("swfac")), "nstres": baseline.scalar(latest.get("nstres")),
                "nstres_pre_action": nstres_pre, "nitrogen_requested_kg_ha": amount,
                "irrigation_requested_mm": 0.0, "external_action_note": "NSTRES_rule_external_N;DSSAT_native_auto_irrigation",
                "reward": float(reward), "done": bool(terminated or truncated),
            })
            done = bool(terminated or truncated)
            steps += 1
        daily = pd.DataFrame(rows)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"SYA{year} 未在 260 步内结束")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        final_b = float(pd.to_numeric(daily["topwt"], errors="coerce").iloc[-1])
        snapshot_tmp = baseline.siteppo.snapshot_from_env(env)
        snapshot = baseline.OUT / "snapshots" / "SYA" / str(year) / SCENARIO
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(snapshot_tmp, snapshot)
        metrics = baseline.metrics_from_snapshot(snapshot, final_y)
        summary_n = float(metrics["actual_nitrogen_kg_ha"])
        mgmt_event_n = executed_nitrogen_from_mgmt_event(snapshot)
        actual_n = mgmt_event_n if mgmt_event_n > 1e-9 else summary_n
        pfp_n = final_y / actual_n if actual_n > 1e-9 else np.nan
        n_closure_gap = actual_n - total_requested_n
        summary = {
            "station_code": "SYA", "site": "SY", "year": int(year), "scenario": SCENARIO,
            "grain_yield_kg_ha": final_y, "biomass_kg_ha": final_b,
            "actual_irrigation_mm": float(metrics["actual_irrigation_mm"]),
            "actual_nitrogen_kg_ha": actual_n,
            "summary_nitrogen_kg_ha": summary_n,
            "mgmt_event_nitrogen_kg_ha": mgmt_event_n,
            "requested_nitrogen_kg_ha": total_requested_n,
            "external_n_closure_gap_kg_ha": n_closure_gap,
            "external_n_closure_status": "ok" if abs(n_closure_gap) <= 1.0 else "requested_vs_mgmt_mismatch",
            "WP_ET_kg_m3": float(metrics["WP_ET_kg_m3"]), "PFP_N_kg_kg": pfp_n,
            "max_water_stress": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nitrogen_stress": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "input_profile": str(env_config.get("input_profile", "")), "snapshot_path": rel(snapshot),
        }
        rendered = {"management_mode_A_L_present": " 1 MA              R     A     L     R     M" in rendered_text, "rendered_template_path": rel(Path(env_args["fileX_template_path"]))}
        return daily, summary, rendered
    finally:
        env.close()


def run(cfg_path: Path, dry_run: bool, years_override: list[int] | None = None, run_id: str = "") -> dict[str, Any]:
    cfg = read_config(cfg_path)
    input_root = INPUT_PROFILES[str(cfg["input_profile"])]
    years = years_override or list(map(int, cfg["scope"]["validation_years"]))
    rule = dict(cfg["external_auto_n_rule"])
    out = output_root(cfg, run_id)
    pf = {"input_profile": cfg["input_profile"], "resolved_input_root": rel(input_root), "years": years, "rule": rule, "next_step_allowed": input_root.exists()}
    if dry_run:
        return {"task": "046_04_sya_external_auto_n_rule", "mode": "dry_run", **pf, "output_root": rel(out)}
    if not pf["next_step_allowed"]:
        raise RuntimeError("input root missing")
    for sub in ["configs", "evaluation", "snapshots"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    selected = selected_rows(years)
    run_config, env_config = build_configs(out, selected)
    env_config["input_profile"] = str(cfg["input_profile"])
    direct_ppo.write_yaml(env_config, out / "configs" / "046_04_resolved_env_config.yaml")
    originals = (baseline.MULTISITE_INPUT_ROOT, baseline.OUT, baseline.TASK_ID, baseline.MAX_STEPS, ppo_safe_rendering.MULTISITE_INPUT_ROOT)
    baseline.MULTISITE_INPUT_ROOT, baseline.OUT, baseline.TASK_ID, baseline.MAX_STEPS, ppo_safe_rendering.MULTISITE_INPUT_ROOT = input_root, out, "046_04", 260, input_root
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    rendered_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    started = time.time()
    try:
        for year in years:
            print(f"[046_04] SYA{year} {SCENARIO}", flush=True)
            try:
                daily, summary, rendered = evaluate_year(run_config, env_config, year, rule)
                daily_frames.append(daily); summary_rows.append(summary); rendered_rows.append({"year": year, **rendered})
            except Exception as exc:
                failure_rows.append({"year": year, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
    finally:
        baseline.MULTISITE_INPUT_ROOT, baseline.OUT, baseline.TASK_ID, baseline.MAX_STEPS, ppo_safe_rendering.MULTISITE_INPUT_ROOT = originals
    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    rendered_df, failures = pd.DataFrame(rendered_rows), pd.DataFrame(failure_rows)
    summary.to_csv(out / "evaluation" / "046_04_external_auto_n_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out / "evaluation" / "046_04_external_auto_n_daily.csv", index=False, encoding="utf-8-sig")
    rendered_df.to_csv(out / "evaluation" / "046_04_rendered_management_mode_audit.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(out / "evaluation" / "046_04_failures.csv", index=False, encoding="utf-8-sig")
    doc = ROOT / "docs" / f"046_04_sya_{cfg['input_profile']}_external_auto_n_rule_record.md"
    doc.write_text("\n".join([
        f"# 046_04 SYA {cfg['input_profile']} 外部规则型 auto-N 记录", "",
        "- 不训练网络；灌溉为 DSSAT native auto，氮肥为 gym-DSSAT 外部 NSTRES 规则。",
        f"- 规则：NSTRES >= {rule['nitrogen_stress_threshold']}，每次 {rule['nitrogen_dose_kg_ha']} kg/ha，间隔 {rule['min_days_between_nitrogen']} 天，季节上限 {rule['season_nitrogen_cap_kg_ha']} kg/ha，DAP{rule['fertilization_last_dap']} 后禁氮。",
        f"- 成功年份：{len(summary)}；失败年份：{len(failures)}；耗时：{time.time() - started:.1f} s。",
        "- 若实际施氮仍为 0，代表该规则阈值未被触发，不调整阈值以凑出施肥。",
    ]) + "\n", encoding="utf-8")
    result = {"task": "046_04_sya_external_auto_n_rule", "output_root": rel(out), "record_md": rel(doc), "successful_runs": int(len(summary)), "failed_runs": int(len(failures))}
    (out / "046_04_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--years", type=str, default="", help="可选 smoke 年份，例如 2014；空值表示全部验证年")
    parser.add_argument("--run-id", type=str, default="", help="Optional run batch id; pass the same timestamp to avoid overwriting earlier outputs")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()] or None
    print(json.dumps(run(path, args.dry_run, years, args.run_id), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
