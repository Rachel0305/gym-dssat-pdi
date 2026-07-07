from __future__ import annotations

import json
import multiprocessing as mp
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import parse_dssat_table, prepare_text_for_scenario


INPUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "LC"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "lc_pdi_initialization_rescue_017_10"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-06_017_10_lc_pdi_initialization_rescue_record.md"
MZX_NAME = "CNLC0801.MZX"
YEAR = 2008
TRNO = 1
ORIGINAL_SOIL_ID = "LC99001200"
FIXED_SOIL_ID = "LC990012007"


def apply_sdate_fix(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*1\s+GE\b", line):
            # Preserve DSSAT fixed-column spacing; only replace the first
            # treatment-1 SDATE token.
            line = line.replace("08150", "08121", 1)
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_text(scenario: str, soil_fix: bool, sdate_fix: bool) -> str:
    source = (INPUT_ROOT / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    if scenario == "null":
        source = prepare_text_for_scenario(source, TRNO, "null")
    elif scenario == "dssat_auto":
        source = prepare_text_for_scenario(source, TRNO, "dssat_auto")
    elif scenario != "recorded":
        raise ValueError(f"Unknown scenario: {scenario}")
    if soil_fix:
        source = source.replace(ORIGINAL_SOIL_ID, FIXED_SOIL_ID)
    if sdate_fix:
        source = apply_sdate_fix(source)
    return source


def prepare_case(case_name: str, scenario: str, soil_fix: bool, sdate_fix: bool, include_mza_mzt: bool = False) -> Path:
    run_dir = OUT_DIR / "cases" / case_name
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / MZX_NAME).write_text(prepare_text(scenario, soil_fix, sdate_fix), encoding="latin-1", errors="ignore")
    for src in INPUT_ROOT.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)

    aux_candidates = [
        input_dir / "CNLC0801.WTH",
        input_dir / "SOIL.SOL",
        input_dir / "MZCER048.CUL",
        input_dir / "CNLC.CLI",
        input_dir / "CNLC.PRM",
        input_dir / "CNLC.wdb",
    ]
    if include_mza_mzt:
        aux_candidates += [input_dir / "CNLC0801.MZA", input_dir / "CNLC0801.MZT"]
    aux = [str(p) for p in aux_candidates if p.exists()]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(input_dir / MZX_NAME),
        "experiment_number": TRNO,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def _worker(env_args: dict[str, Any], run_dir_str: str, mode: str, queue: mp.Queue) -> None:
    start = time.time()
    try:
        import gym
        from sb3_wrapper import GymDssatWrapper

        raw = gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped
        env = GymDssatWrapper(raw)
        result: dict[str, Any] = {"constructor_ok": True, "reset_ok": False, "step_ok": False}
        obs, info = env.reset()
        result["reset_ok"] = True
        rows: list[dict[str, Any]] = []
        max_steps = 1 if mode == "one_step" else 380
        for step in range(max_steps):
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, {"amir": 0.0, "anfer": 0.0})
            obs, reward, terminated, truncated, info = env.step(norm)
            result["step_ok"] = True
            latest = latest_observation_dict(env, obs, info)
            rows.append(
                {
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": scalar(latest.get("yrdoy")),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if terminated or truncated:
                break
        tmp = getattr(getattr(env, "_env", None), "_tmp_folder", None) or getattr(raw, "_tmp_folder", None)
        run_dir = Path(run_dir_str)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot_eval", dirs_exist_ok=True)
        pd.DataFrame(rows).to_csv(run_dir / "daily_probe.csv", index=False, encoding="utf-8-sig")
        env.close()
        result["elapsed_sec"] = time.time() - start
        result["n_steps"] = len(rows)
        result["completed"] = bool(rows and (rows[-1]["terminated"] or rows[-1]["truncated"]))
        queue.put({"status": "ok", **result})
    except Exception as exc:
        queue.put({"status": "error", "error": repr(exc), "elapsed_sec": time.time() - start})


def run_with_timeout(case_name: str, scenario: str, soil_fix: bool, sdate_fix: bool, include_mza_mzt: bool, mode: str, timeout_sec: int) -> dict[str, Any]:
    run_dir = prepare_case(case_name, scenario, soil_fix, sdate_fix, include_mza_mzt)
    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(env_args, str(run_dir), mode, queue))
    proc.start()
    proc.join(timeout_sec)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "case": case_name,
            "scenario": scenario,
            "soil_fix": soil_fix,
            "sdate_fix": sdate_fix,
            "include_mza_mzt": include_mza_mzt,
            "mode": mode,
            "status": "timeout",
            "timeout_sec": timeout_sec,
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        }
    result = queue.get() if not queue.empty() else {"status": "no_queue_result"}
    result.update(
        {
            "case": case_name,
            "scenario": scenario,
            "soil_fix": soil_fix,
            "sdate_fix": sdate_fix,
            "include_mza_mzt": include_mza_mzt,
            "mode": mode,
            "timeout_sec": timeout_sec,
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        }
    )
    return result


def summarize_case(run_dir: Path, scenario: str) -> dict[str, Any]:
    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot_eval" / "PlantGro.OUT")
    final_gwad = np.nan
    final_cwad = np.nan
    max_swfac = np.nan
    max_nstres = np.nan
    if not plantgro.empty:
        if "GWAD" in plantgro.columns and not plantgro["GWAD"].dropna().empty:
            final_gwad = float(plantgro["GWAD"].dropna().iloc[-1])
        if "CWAD" in plantgro.columns and not plantgro["CWAD"].dropna().empty:
            final_cwad = float(plantgro["CWAD"].dropna().iloc[-1])
        if "WSPD" in plantgro.columns:
            max_swfac = float(plantgro["WSPD"].dropna().max())
        elif "SWFAC" in plantgro.columns:
            max_swfac = float(plantgro["SWFAC"].dropna().max())
        if "NSTD" in plantgro.columns:
            max_nstres = float(plantgro["NSTD"].dropna().max())
        elif "NSTRES" in plantgro.columns:
            max_nstres = float(plantgro["NSTRES"].dropna().max())

    events_path = run_dir / "pdi_tmp_snapshot_eval" / "MgmtEvent.OUT"
    irrigation = 0.0
    fertilizer = 0.0
    if events_path.exists():
        for raw in events_path.read_text(encoding="latin-1", errors="ignore").splitlines():
            m = re.search(r"([-+]?\d+(?:\.\d*)?)\s*(mm|kg(?:\[[A-Za-z]+\])?/ha|kg)", raw)
            if not m:
                continue
            amount = float(m.group(1))
            if "Irrigation" in raw and m.group(2) == "mm":
                irrigation += amount
            if "Fertil" in raw or "Nitrogen" in raw:
                fertilizer += amount
    return {
        "scenario": scenario,
        "final_gwad": final_gwad,
        "final_cwad": final_cwad,
        "irrigation_total": irrigation,
        "fertilizer_total": fertilizer,
        "max_water_stress": max_swfac,
        "max_nitrogen_stress": max_nstres,
    }


def write_record(status_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# 017_10 LC PDI/gym 初始化问题诊断与抢救记录",
        "",
        "## 关键发现",
        "",
        f"- 原始 LC MZX 中 ID_SOIL 为 `{ORIGINAL_SOIL_ID}`。",
        f"- LC 的 SOIL.SOL 中实际 profile 为 `{FIXED_SOIL_ID}`。",
        "- 本阶段只修改临时副本，不修改原始输入包。",
        "",
        "## 初始化测试状态",
        "",
        status_df.to_csv(index=False),
        "",
        "## 基准运行结果",
        "",
        summary_df.to_csv(index=False) if not summary_df.empty else "未生成完整基准结果。",
        "",
        "## 判断",
        "",
    ]
    fixed_ok = status_df[(status_df["case"].str.contains("soilfix")) & (status_df["status"].eq("ok"))]
    orig_timeout = status_df[(status_df["case"].str.contains("original")) & (status_df["status"].eq("timeout"))]
    if not fixed_ok.empty and not orig_timeout.empty:
        lines += [
            "- LC 初始化问题高度可能来自 MZX 与 SOIL.SOL 的土壤 ID 不一致。",
            "- 修正临时 MZX 的 ID_SOIL 后，LC 可以进入 gym/PDI 运行链路。",
            "- 下一步可用修正后的临时输入进行 LC 年份筛选；在确认无副作用后，再决定是否把源输入包修正为一致版本。",
        ]
    else:
        lines += [
            "- 当前测试尚不能证明土壤 ID 是唯一原因。",
            "- 若 soilfix 仍失败，需要继续检查多 treatment、自动管理日期、WeatherMan 文件或 PDI 对 LC 文件结构的兼容性。",
        ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tests = [
        ("original_recorded_one_step", "recorded", False, False, False, "one_step", 45),
        ("soilfix_recorded_one_step", "recorded", True, False, False, "one_step", 45),
        ("soilfix_sdatefix_recorded_one_step", "recorded", True, True, False, "one_step", 60),
        ("soilfix_sdatefix_recorded_mza_mzt_one_step", "recorded", True, True, True, "one_step", 60),
    ]
    rows = [run_with_timeout(*test) for test in tests]
    status = pd.DataFrame(rows)

    summary_rows: list[dict[str, Any]] = []
    can_run = not status[(status["case"].eq("soilfix_sdatefix_recorded_one_step")) & (status["status"].eq("ok"))].empty
    if can_run:
        for scenario in ["null", "recorded", "dssat_auto"]:
            case = f"soilfix_{scenario}_full"
            row = run_with_timeout(case, scenario, True, True, False, "full", 120)
            rows.append(row)
            if row.get("status") == "ok":
                summary_rows.append(summarize_case(PROJECT_ROOT / row["run_dir"], scenario))
        status = pd.DataFrame(rows)

    summary = pd.DataFrame(summary_rows)
    status.to_csv(OUT_DIR / "017_10_lc_initialization_status.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "017_10_lc2008_soilfix_baseline_summary.csv", index=False, encoding="utf-8-sig")
    write_record(status, summary)
    print(status.to_string(index=False))
    if not summary.empty:
        print(summary.to_string(index=False))
    print(f"[done] {OUT_DIR}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
