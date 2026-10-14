from __future__ import annotations

import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as ppo_env
import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline_034
import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as siteppo
from ppo_safe_rendering import build_env_args, check_rendered_input, ensure_project_on_path
from run_fq_yc_new_cultivar_forward_screening_013_01 import set_management_for_treatment


TASK_ID = "034_02"
OUT = ROOT / "benchmark_results" / "034_02_ppo_external_action_linkage_rootcause_audit"
DOC = ROOT / "docs" / "034_02_ppo_external_action_linkage_rootcause_audit_record.md"
PROMPT = ROOT / "prompts" / "034_02_ppo_external_action_linkage_rootcause_audit.md"
SOURCE_SPLIT = ROOT / "benchmark_results" / "033_04_multisite_input_enabled_five_site_half_split_maskableppo_rerun" / "configs" / "033_04_available_weather_half_split_years.csv"


def ensure_dirs() -> None:
    for rel in ["configs", "evaluation", "snapshots", "logs", "rendered_inputs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def latest_observation_dict(env: Any, obs: Any | None = None, info: dict | None = None) -> dict[str, Any]:
    return direct_ppo.latest_observation_dict(env, obs, info)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def parse_mzx_modes(path: Path, treatment: int = 1) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    treatment_row = ""
    management_row = ""
    header = ""
    in_treatments = False
    in_management = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N R O C TNAME"):
            header = stripped
            in_treatments = True
            in_management = False
            continue
        if stripped.startswith("@N MANAGEMENT") and "IRRIG" in stripped and "FERTI" in stripped:
            in_management = True
            in_treatments = False
            continue
        if stripped.startswith("*"):
            in_treatments = False
            in_management = False
        if in_treatments and re.match(rf"^\s*{treatment}\s+", line):
            treatment_row = stripped
        if in_management and re.match(rf"^\s*{treatment}\s+MA\b", line):
            management_row = stripped
    parts = treatment_row.split()
    factors = ["CU", "FL", "SA", "IC", "MP", "MI", "MF", "MR", "MC", "MT", "ME", "MH", "SM"]
    factor_map = {}
    if len(parts) >= 5 + len(factors):
        factor_map = dict(zip(factors, parts[5 : 5 + len(factors)]))
    mparts = management_row.split()
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/") if path.exists() else str(path),
        "exists": path.exists(),
        "treatment": treatment,
        "treatment_header": header,
        "treatment_row": treatment_row,
        "management_row": management_row,
        "IC": factor_map.get("IC", ""),
        "MI": factor_map.get("MI", ""),
        "MF": factor_map.get("MF", ""),
        "PLANT": mparts[2] if len(mparts) >= 6 else "",
        "IRRIG": mparts[3] if len(mparts) >= 6 else "",
        "FERTI": mparts[4] if len(mparts) >= 6 else "",
    }


def overview_modes(snapshot: Path) -> str:
    path = snapshot / "OVERVIEW.OUT"
    if not path.exists():
        return ""
    lines = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "MANAGEMENT OPT" in line:
            lines.append(line.strip())
    return " | ".join(lines[:3])


def build_config() -> tuple[dict[str, Any], dict[str, Any]]:
    config = batch_ppo.load_config()
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    split = pd.read_csv(SOURCE_SPLIT, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    selection = batch_ppo.build_selection(split)
    direct_ppo.OUTPUT_ROOT = OUT
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "034_02_resolved_env_config.yaml")
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    return config, env_config


def action_index_for(env: Any, target_i: float, target_n: float) -> int:
    for idx, raw in enumerate(env.grid):
        if abs(float(raw.get("amir", 0.0)) - target_i) < 1e-9 and abs(float(raw.get("anfer", 0.0)) - target_n) < 1e-9:
            return idx
    raise RuntimeError(f"动作网格中找不到 I{target_i}/N{target_n}: {env.grid}")


def run_forced_case(config: dict[str, Any], env_config: dict[str, Any], *, mode: str) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    ensure_project_on_path()
    import gym
    from sb3_wrapper import GymDssatWrapper

    station = "FQA"
    year = 2014
    year_info = direct_ppo.find_year(env_config, station, year)
    run_tag = f"{station}_{year}_{TASK_ID}_{mode}"
    env_args = build_env_args(
        station=station,
        year=year,
        planting_date=year_info["planting_date"],
        seed=0,
        config=env_config,
        run_tag=run_tag,
        evaluation=True,
        mode=env_config.get("runtime", {}).get("mode", "all"),
    )
    template = Path(env_args["fileX_template_path"])
    if mode == "external_ll":
        text = template.read_text(encoding="utf-8", errors="replace")
        template.write_text(set_management_for_treatment(text, 1, "L", "L"), encoding="utf-8")
    elif mode != "external_rr":
        raise ValueError(mode)

    render_checks = check_rendered_input(template)
    base = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    env = ppo_env.StressAwareDiscreteWrapper(base, config)
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        action_index = action_index_for(env, 45.0, 80.0)
        records: list[dict[str, Any]] = []
        while not done and step_count < 260:
            latest_pre = latest_observation_dict(env, obs, info)
            dap_raw = scalar(latest_pre.get("dap", step_count + 1))
            dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step_count + 1
            action = action_index if step_count == 0 else 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            records.append(
                {
                    "station_code": station,
                    "year": year,
                    "scenario": mode,
                    "step": step_count + 1,
                    "dap": dap,
                    "grnwt": scalar(latest.get("grnwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": float(reward),
                    **dict(getattr(env, "last_action_info", {})),
                    "done": done,
                }
            )
            step_count += 1
        daily = pd.DataFrame(records)
        if daily.empty or not bool(daily["done"].iloc[-1]):
            raise RuntimeError(f"{mode} 未在 260 步内结束")
        final_y = float(pd.to_numeric(daily["grnwt"], errors="coerce").iloc[-1])
        snapshot = OUT / "snapshots" / station / str(year) / mode
        tmp = siteppo.snapshot_from_env(env)
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(tmp, snapshot)
        metrics = baseline_034.metrics_from_snapshot(snapshot, final_y)
        summary = {
            "station_code": station,
            "year": year,
            "scenario": mode,
            "forced_first_action": "DAP1 I45/N80",
            "mzx_irrig_mode": parse_mzx_modes(template)["IRRIG"],
            "mzx_ferti_mode": parse_mzx_modes(template)["FERTI"],
            "overview_management_opt": overview_modes(snapshot),
            "safe_action_irrigation_sum_mm": float(pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "safe_action_n_sum_kg_ha": float(pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
            "summary_irrigation_mm": float(metrics["actual_irrigation_mm"]),
            "summary_n_kg_ha": float(metrics["actual_nitrogen_kg_ha"]),
            "grain_yield_kg_ha": final_y,
            "max_swfac": float(pd.to_numeric(daily["swfac"], errors="coerce").max()),
            "max_nstres": float(pd.to_numeric(daily["nstres"], errors="coerce").max()),
            "snapshot_path": str(snapshot.relative_to(ROOT)).replace("\\", "/"),
        }
        return daily, summary, render_checks
    finally:
        env.close()


def write_record(mode_parse: pd.DataFrame, summary: pd.DataFrame, failures: pd.DataFrame, elapsed: float) -> None:
    diagnosis = "undetermined"
    if not summary.empty and set(summary["scenario"]) >= {"external_rr", "external_ll"}:
        rr = summary.set_index("scenario").loc["external_rr"]
        ll = summary.set_index("scenario").loc["external_ll"]
        if (
            float(rr["safe_action_irrigation_sum_mm"]) + float(rr["safe_action_n_sum_kg_ha"]) > 0
            and abs(float(rr["summary_irrigation_mm"])) < 1e-6
            and abs(float(rr["summary_n_kg_ha"])) < 1e-6
            and float(ll["summary_irrigation_mm"]) > 0
            and float(ll["summary_n_kg_ha"]) > 0
        ):
            diagnosis = "management_mode_rr_blocks_external_action_and_ll_enables_it"
        elif abs(float(ll["summary_irrigation_mm"])) < 1e-6 and abs(float(ll["summary_n_kg_ha"])) < 1e-6:
            diagnosis = "ll_still_not_applied_need_action_mapping_audit"
    lines = [
        "# 034_02 PPO 外部动作落地链路根因审计记录",
        "",
        "## 结论先说",
        "",
        f"- 判定：`{diagnosis}`。",
        "- 本任务不训练、不修改原始输入，只做 FQA2014 的最小外部动作复核。",
        "",
        "## 现有 034_01 快照模式解析",
        "",
        md_table(mode_parse, 40),
        "",
        "## 强制外部动作复核",
        "",
        md_table(summary, 20),
        "",
        "## 失败记录",
        "",
        md_table(failures, 20),
        "",
        "## 输出文件",
        "",
        "- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_mode_parse.csv`",
        "- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_forced_action_summary.csv`",
        "- `benchmark_results/034_02_ppo_external_action_linkage_rootcause_audit/evaluation/034_02_forced_action_daily.csv`",
        "",
        f"耗时：{elapsed:.1f} 秒",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / DOC.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    ensure_dirs()
    config, env_config = build_config()
    mode_rows: list[dict[str, Any]] = []
    for scenario in ["null_external_noop", "ppo_external_replay", "ppo_static_mzx_same_schedule"]:
        path = ROOT / "benchmark_results" / "034_01_ppo_action_dssat_effect_audit" / "snapshots" / "FQA" / "2014" / scenario / "fileX.MZX"
        row = parse_mzx_modes(path)
        row["source_scenario"] = f"034_01_{scenario}"
        snap = path.parent
        row["overview_management_opt"] = overview_modes(snap)
        mode_rows.append(row)
    linked_path = ROOT / "DSSAT_auto_validation" / "fq_n_timing_response_diagnostic_016_07" / "runs" / "2008" / "I60_N150_early" / "pdi_tmp_snapshot" / "fileX.MZX"
    row = parse_mzx_modes(linked_path, treatment=2)
    row["source_scenario"] = "known_linked_dqn_success_treatment2"
    row["overview_management_opt"] = overview_modes(linked_path.parent)
    mode_rows.append(row)
    mode_parse = pd.DataFrame(mode_rows)

    summaries: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for mode in ["external_rr", "external_ll"]:
        try:
            daily, summary, _checks = run_forced_case(config, env_config, mode=mode)
            daily_frames.append(daily)
            summaries.append(summary)
        except Exception:
            failures.append({"scenario": mode, "status": "failed", "traceback": traceback.format_exc()[-5000:]})
    summary_df = pd.DataFrame(summaries)
    daily_df = pd.concat(daily_frames, ignore_index=True, sort=False) if daily_frames else pd.DataFrame()
    failures_df = pd.DataFrame(failures)

    mode_parse.to_csv(OUT / "evaluation" / "034_02_mode_parse.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "evaluation" / "034_02_forced_action_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(OUT / "evaluation" / "034_02_forced_action_daily.csv", index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUT / "evaluation" / "034_02_failures.csv", index=False, encoding="utf-8-sig")
    write_record(mode_parse, summary_df, failures_df, time.time() - start)
    result = {
        "task": TASK_ID,
        "record_md": str(DOC.relative_to(ROOT)).replace("\\", "/"),
        "summary_csv": str((OUT / "evaluation" / "034_02_forced_action_summary.csv").relative_to(ROOT)).replace("\\", "/"),
        "failures": int(len(failures_df)),
    }
    (OUT / "034_02_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not summary_df.empty:
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
