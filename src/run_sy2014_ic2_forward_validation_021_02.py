from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    parse_dssat_table,
    parse_events,
    prepare_text_for_scenario,
)


INPUT_DIR = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY"
SOURCE_MZX = INPUT_DIR / "CNSY1201.MZX"
OUT_DIR = ROOT / "DSSAT_auto_validation" / "sy2014_ic2_forward_validation_021_02"
SCENARIOS = ("null", "recorded", "dssat_auto")
TRNO = 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ic2_audit(text: str) -> dict[str, object]:
    treatment_line = ""
    in_treatments = False
    for line in text.splitlines():
        if line.strip().startswith("@N R O C TNAME"):
            in_treatments = True
            continue
        if in_treatments and re.match(r"^\s*2\s+1\s+1\s+0\s+Sim2014\b", line):
            treatment_line = line
            break
        if in_treatments and line.strip().startswith("*"):
            break
    parts = treatment_line.split()
    ic_pointer = parts[8] if len(parts) >= 9 else None
    layer_rows = []
    for line in text.splitlines():
        m = re.match(r"^\s*2\s+(10|20|30|40|60|100)\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s*$", line)
        if m:
            layer_rows.append([float(x) for x in m.groups()])
    return {
        "treatment_line": treatment_line,
        "ic_pointer": ic_pointer,
        "ic2_layer_count": len(layer_rows),
        "ic2_layers": layer_rows,
        "valid": ic_pointer == "2" and len(layer_rows) == 6,
    }


def prepare_case(scenario: str, tag: str) -> Path:
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    run_dir = OUT_DIR / "runs" / tag
    if run_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有运行目录: {run_dir}")
    case_input = run_dir / "input"
    case_input.mkdir(parents=True)
    source = SOURCE_MZX.read_text(encoding="latin-1", errors="ignore")
    source_audit = ic2_audit(source)
    if not source_audit["valid"]:
        raise RuntimeError(f"源 MZX 未通过 IC=2 审计: {source_audit}")
    case_text = prepare_text_for_scenario(source, TRNO, scenario)
    case_audit = ic2_audit(case_text)
    if not case_audit["valid"]:
        raise RuntimeError(f"情景副本破坏了 IC=2: {case_audit}")
    filex = case_input / f"SY2014_{scenario}.MZX"
    filex.write_text(case_text, encoding="latin-1", errors="ignore")
    for src in INPUT_DIR.iterdir():
        if src.is_file() and src.name != SOURCE_MZX.name:
            shutil.copyfile(src, case_input / src.name)
    aux = [str(p) for p in case_input.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": TRNO,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "input_audit.json").write_text(
        json.dumps({"source_sha256": sha256(SOURCE_MZX), "source": source_audit, "case": case_audit}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def child_run(run_dir: Path, max_steps: int) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **args).unwrapped)
    rows = []
    done = False
    try:
        obs, info = env.reset()
        for step in range(max_steps):
            action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append({
                "step": step,
                "yrdoy": yrdoy,
                "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                "dap": scalar(latest.get("dap")),
                "topwt": scalar(latest.get("topwt")),
                "grnwt": scalar(latest.get("grnwt")),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "reward": scalar(reward),
                "done": bool(terminated or truncated),
            })
            done = bool(terminated or truncated)
            if done:
                break
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot", dirs_exist_ok=True)
        env.close()
    pd.DataFrame(rows).to_csv(run_dir / "gym_post_state_daily.csv", index=False, encoding="utf-8-sig")
    runtime_candidates = list((run_dir / "pdi_tmp_snapshot").glob("*.MZX"))
    runtime_audits = {p.name: ic2_audit(p.read_text(encoding="latin-1", errors="ignore")) for p in runtime_candidates}
    (run_dir / "runtime_input_audit.json").write_text(
        json.dumps({"completed": done, "steps": len(rows), "mzx": runtime_audits}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def summary_one(run_dir: Path, scenario: str) -> dict[str, object]:
    daily = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    events = parse_events(run_dir, "SY", 2014, scenario)
    completed = False
    runtime = run_dir / "runtime_input_audit.json"
    runtime_data = json.loads(runtime.read_text(encoding="utf-8")) if runtime.exists() else {}
    completed = bool(runtime_data.get("completed"))
    runtime_valid = bool(runtime_data.get("mzx")) and all(v.get("valid") for v in runtime_data.get("mzx", {}).values())
    irr = events[events["operation"].str.contains("Irrigation", case=False, na=False)] if not events.empty else events
    fert = events[events["operation"].str.contains("Fertil|Nitrogen", case=False, na=False, regex=True)] if not events.empty else events
    final = daily.iloc[-1] if not daily.empty else pd.Series(dtype=object)
    def final_num(key: str) -> float:
        value = pd.to_numeric(pd.Series([final.get(key)]), errors="coerce").iloc[0]
        return float(value) if pd.notna(value) else np.nan
    irrigation_total = float(irr["amount"].sum()) if not irr.empty else 0.0
    nitrogen_total = float(fert["amount"].sum()) if not fert.empty else 0.0
    return {
        "scenario": scenario,
        "source_sha256": sha256(SOURCE_MZX),
        "runtime_ic2_valid": runtime_valid,
        "completed": completed,
        # Summary.OUT contains blank identifier fields, so whitespace token parsing shifts
        # columns. Use the final PlantGro row for yield/biomass and MgmtEvent for inputs.
        "HWAM": final_num("GWAD"), "CWAM": final_num("CWAD"),
        "IRCM": irrigation_total, "NICM": nitrogen_total,
        "IR_events_total": irrigation_total,
        "N_events_total": nitrogen_total,
        "max_WSPD": float(pd.to_numeric(daily.get("WSPD"), errors="coerce").max()) if not daily.empty else np.nan,
        "max_NSTD": float(pd.to_numeric(daily.get("NSTD"), errors="coerce").max()) if not daily.empty else np.nan,
        "final_DAP": float(pd.to_numeric(daily.get("DAP"), errors="coerce").max()) if not daily.empty else np.nan,
        "run_dir": str(run_dir.relative_to(ROOT)),
    }


def collect() -> pd.DataFrame:
    rows = []
    daily_all = []
    events_all = []
    for scenario in SCENARIOS:
        run_dir = OUT_DIR / "runs" / scenario
        if not run_dir.exists():
            continue
        rows.append(summary_one(run_dir, scenario))
        daily = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
        if not daily.empty:
            daily.insert(0, "scenario", scenario)
            daily_all.append(daily)
        ev = parse_events(run_dir, "SY", 2014, scenario)
        if not ev.empty:
            events_all.append(ev)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "021_02_sy2014_ic2_forward_summary.csv", index=False, encoding="utf-8-sig")
    if daily_all:
        pd.concat(daily_all, ignore_index=True).to_csv(OUT_DIR / "021_02_sy2014_ic2_forward_daily.csv", index=False, encoding="utf-8-sig")
    if events_all:
        pd.concat(events_all, ignore_index=True).to_csv(OUT_DIR / "021_02_sy2014_ic2_forward_events.csv", index=False, encoding="utf-8-sig")
    return result


def plot_results() -> None:
    daily_path = OUT_DIR / "021_02_sy2014_ic2_forward_daily.csv"
    summary_path = OUT_DIR / "021_02_sy2014_ic2_forward_summary.csv"
    if not daily_path.exists() or not summary_path.exists():
        return
    # "null" is a scenario label, not a missing value.
    daily = pd.read_csv(daily_path, keep_default_na=False)
    summary = pd.read_csv(summary_path, keep_default_na=False)
    colors = {"null": "#222222", "recorded": "#C43C39", "dssat_auto": "#2F5DA8"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [1, 1, 0.9]})
    for scenario in SCENARIOS:
        sub = daily[daily["scenario"].eq(scenario)].sort_values("DAP")
        if sub.empty:
            continue
        axes[0].plot(sub["DAP"], sub["WSPD"], lw=2, color=colors[scenario], label=scenario)
        axes[1].plot(sub["DAP"], sub["NSTD"], lw=2, color=colors[scenario], label=scenario)
    x = np.arange(len(summary))
    axes[2].bar(x - 0.18, summary["HWAM"], 0.36, color="#555555", label="HWAM")
    axes[2].bar(x + 0.18, summary["CWAM"], 0.36, color="#BBBBBB", label="CWAM")
    axes[2].set_xticks(x, summary["scenario"])
    axes[0].set_ylabel("WSPD")
    axes[1].set_ylabel("NSTD")
    axes[2].set_ylabel("kg/ha")
    axes[1].set_xlabel("DAP")
    axes[0].legend(frameon=False, ncol=3)
    axes[2].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(True, color="#E5E5E5", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SY2014 IC=2 forward validation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "021_02_sy2014_ic2_forward_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", choices=SCENARIOS)
    parser.add_argument("--tag")
    parser.add_argument("--child", type=Path)
    parser.add_argument("--max-steps", type=int, default=380)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.child:
        child_run(args.child, args.max_steps)
    elif args.prepare:
        print(prepare_case(args.prepare, args.tag or args.prepare))
    elif args.collect:
        print(collect().to_string(index=False))
        plot_results()
    else:
        audit = ic2_audit(SOURCE_MZX.read_text(encoding="latin-1", errors="ignore"))
        print(json.dumps({"sha256": sha256(SOURCE_MZX), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
