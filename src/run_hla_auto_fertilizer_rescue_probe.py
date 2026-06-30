from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


SOURCE_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_new_cultivar_candidate_year_screening"
NEW_CUL = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "cultivar_calibration_HLA2004_480" / "input_corrected_package" / "MZCER048.CUL"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "hla_auto_fertilizer_rescue_probe"
YEARS = [2010, 2015]
VARIANTS = {
    "current_auto_irrig_only": {"mi": "0", "mf": "0", "irrig": "A", "ferti": "R", "ncode": "FE001"},
    "auto_irrig_fert_m0_fe001": {"mi": "0", "mf": "0", "irrig": "A", "ferti": "A", "ncode": "FE001"},
    "auto_irrig_fert_m1_fe001": {"mi": "1", "mf": "1", "irrig": "A", "ferti": "A", "ncode": "FE001"},
    "auto_irrig_fert_m1_fe005": {"mi": "1", "mf": "1", "irrig": "A", "ferti": "A", "ncode": "FE005"},
}


def set_management_line(text: str, irrig: str, ferti: str) -> str:
    return re.sub(
        r"(?m)^(\s*1\s+MA\s+R\s+)\S+(\s+)\S+(\s+R\s+M)\s*$",
        rf"\1{irrig}\2{ferti}\3",
        text,
        count=1,
    )


def set_treatment_mi_mf(text: str, mi: str, mf: str) -> str:
    out = []
    for line in text.splitlines():
        if re.match(r"^\s*1\s+1\s+1\s+0\s+\S+", line):
            parts = line.split()
            parts[10] = mi
            parts[11] = mf
            out.append(
                f" {parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]:<25} "
                f"{parts[5]:>2} {parts[6]:>2} {parts[7]:>2} {parts[8]:>2} {parts[9]:>2} {parts[10]:>2} {parts[11]:>2} "
                f"{parts[12]:>2} {parts[13]:>2} {parts[14]:>2} {parts[15]:>2} {parts[16]:>2} {parts[17] if len(parts)>17 else '1':>2}"
            )
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def set_ncode(text: str, ncode: str) -> str:
    return re.sub(
        r"(?m)^(\s*1\s+NI\s+\d+\s+\d+\s+\d+\s+)\S+(\s+\S+)\s*$",
        rf"\1{ncode}\2",
        text,
        count=1,
    )


def prepare_case(year: int, variant: str) -> Path:
    cfg = VARIANTS[variant]
    src_dir = SOURCE_ROOT / "auto_irrig" / str(year) / "input"
    case_dir = OUT_DIR / variant / str(year)
    input_dir = case_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_file():
            shutil.copyfile(src, input_dir / src.name)
    shutil.copyfile(NEW_CUL, input_dir / "MZCER048.CUL")
    filex = next(input_dir.glob("*.MZX"))
    text = filex.read_text(encoding="latin1", errors="ignore")
    text = set_treatment_mi_mf(text, cfg["mi"], cfg["mf"])
    text = set_management_line(text, cfg["irrig"], cfg["ferti"])
    text = set_ncode(text, cfg["ncode"])
    filex.write_text(text, encoding="latin1")
    aux = [str(p) for p in sorted(input_dir.iterdir()) if p.is_file() and p.name != filex.name]
    env_args = {
        "log_saving_path": str(case_dir / f"{variant}_{year}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_dir


def child_run(variant: str, year: int, max_steps: int = 360) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    case_dir = OUT_DIR / variant / str(year)
    env_args = json.loads((case_dir / "env_args.json").read_text(encoding="utf-8"))
    snapshot = case_dir / "pdi_tmp_snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows = []
    try:
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            done = bool(terminated or truncated)
            latest = latest_observation_dict(env, obs, info)
            rows.append({"step": step, "dap": scalar(latest.get("dap")), "reward": scalar(reward), "done": done})
            step += 1
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
    pd.DataFrame(rows).to_csv(case_dir / f"{variant}_{year}_post_state.csv", index=False, encoding="utf-8-sig")


def parse_events(path: Path) -> tuple[float, float, int, int]:
    irrigation = 0.0
    fert = 0.0
    i_count = 0
    f_count = 0
    if not path.exists():
        return irrigation, fert, i_count, f_count
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        if "Irrigation" in line:
            m = re.search(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*mm", line)
            if m:
                irrigation += float(m.group(1))
                i_count += 1
        if "Fertilizer" in line:
            m = re.search(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*kg", line)
            if m:
                fert += float(m.group(1))
                f_count += 1
    return irrigation, fert, i_count, f_count


def final_gwad(path: Path) -> float:
    if not path.exists():
        return float("nan")
    last = float("nan")
    header = None
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("@"):
            header = s.replace("@", "", 1).split()
            continue
        if header and re.match(r"^\d{4}\s+\d+", s):
            parts = s.split()
            if "GWAD" in header and len(parts) >= len(header):
                last = float(parts[header.index("GWAD")])
    return last


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child_run(sys.argv[2], int(sys.argv[3]))
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    statuses = []
    for variant in VARIANTS:
        for year in YEARS:
            case_dir = prepare_case(year, variant)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child", variant, str(year)]
            try:
                proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=150, capture_output=True, text=True)
                statuses.append({"variant": variant, "year": year, "returncode": proc.returncode, "timed_out": False, "stderr_tail": proc.stderr[-800:]})
            except subprocess.TimeoutExpired as exc:
                statuses.append({"variant": variant, "year": year, "returncode": None, "timed_out": True, "stderr_tail": str(exc)[-800:]})
    rows = []
    for variant in VARIANTS:
        for year in YEARS:
            snap = OUT_DIR / variant / str(year) / "pdi_tmp_snapshot"
            irrigation, fert, i_count, f_count = parse_events(snap / "MgmtEvent.OUT")
            rows.append(
                {
                    "variant": variant,
                    "year": year,
                    "final_gwad": final_gwad(snap / "PlantGro.OUT"),
                    "irrigation_total": irrigation,
                    "fertilizer_total": fert,
                    "irrigation_events": i_count,
                    "fertilizer_events": f_count,
                }
            )
    pd.DataFrame(statuses).to_csv(OUT_DIR / "run_status.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# HLA auto fertilizer rescue probe",
        "",
        "Low-cost DSSAT automatic nitrogen variants for 2010/2015. No PPO training.",
        "",
        "| " + " | ".join(summary.columns) + " |",
        "| " + " | ".join(["---"] * len(summary.columns)) + " |",
    ]
    for row in summary.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
