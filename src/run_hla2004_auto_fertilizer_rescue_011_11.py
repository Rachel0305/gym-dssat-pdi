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


BASE_MZX = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0408_DSSAT480_2004" / "CNHL0408.MZX"
WTH = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "CNHL0401.WTH"
SOIL = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "run_CNHL0404" / "SOIL.SOL"
NEW_CUL = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "cultivar_calibration_HLA2004_480" / "input_corrected_package" / "MZCER048.CUL"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "auto_fertilizer_rescue_2004_newcul_011_11"


VARIANTS = {
    # Positive process control: native auto irrigation only, no auto fertilizer.
    "auto_irrig_only_fertiR": {"irrig": "A", "ferti": "R", "nmdep": 30, "nmthr": 50, "namnt": 25, "ncode": "FE001", "naoff": "GS000"},
    # Same as base, but enable FERTI=A.
    "fertA_fe001_gs000": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 50, "namnt": 25, "ncode": "FE001", "naoff": "GS000"},
    # Material code used by earlier null/reported fertilizer rows.
    "fertA_fe005_gs000": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 50, "namnt": 25, "ncode": "FE005", "naoff": "GS000"},
    # More aggressive trigger: high threshold and larger amount.
    "fertA_fe001_thr99_amt50": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE001", "naoff": "GS000"},
    "fertA_fe005_thr99_amt50": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE005", "naoff": "GS000"},
    # Test whether GS000 is interpreted unexpectedly by auto-N stop stage.
    "fertA_fe001_gs999": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE001", "naoff": "GS999"},
    "fertA_fe005_gs999": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE005", "naoff": "GS999"},
    # Remove auto-irrigation to test whether stronger stress conditions can trigger native auto-N.
    "noirrig_fertA_fe001_thr99_amt50": {"irrig": "N", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE001", "naoff": "GS000"},
    "noirrig_fertA_fe005_thr99_amt50": {"irrig": "N", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE005", "naoff": "GS000"},
    # Keep initial water unchanged but force very low/zero mineral N to test
    # whether native auto-N can ever fire when soil N is clearly depleted.
    "lowN0_fertA_fe001_thr99_amt50": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE001", "naoff": "GS000", "initial_n": 0.0},
    "lowN0_fertA_fe005_thr99_amt50": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE005", "naoff": "GS000", "initial_n": 0.0},
    "lowN01_fertA_fe001_thr99_amt50": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE001", "naoff": "GS000", "initial_n": 0.1},
    "lowN01_fertA_fe005_thr99_amt50": {"irrig": "A", "ferti": "A", "nmdep": 30, "nmthr": 99, "namnt": 50, "ncode": "FE005", "naoff": "GS000", "initial_n": 0.1},
}


def patch_management(text: str, irrig: str, ferti: str) -> str:
    new, n = re.subn(
        r"(?m)^(\s*1\s+MA\s+R\s+)\S+(\s+)\S+(\s+R\s+M)\s*$",
        rf"\1{irrig}\2{ferti}\3",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not patch management line")
    return new


def patch_nitrogen(text: str, cfg: dict) -> str:
    new, n = re.subn(
        r"(?m)^(\s*1\s+NI\s+)\d+(\s+)\d+(\s+)\d+(\s+)\S+(\s+)\S+\s*$",
        rf"\g<1>{cfg['nmdep']:>10d}\g<2>{cfg['nmthr']:>5d}\g<3>{cfg['namnt']:>5d}\g<4>{cfg['ncode']}\g<5>{cfg['naoff']}",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Could not patch nitrogen line")
    return new


def patch_initial_soil_n(text: str, value: float | None) -> str:
    if value is None:
        return text
    lines = text.splitlines()
    out: list[str] = []
    in_ic_layers = False
    changed = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@C  ICBL") and "SH2O" in stripped and "SNH4" in stripped and "SNO3" in stripped:
            in_ic_layers = True
            out.append(line)
            continue
        if in_ic_layers:
            if re.match(r"^\s*1\s+\d+", line):
                parts = line.split()
                if len(parts) >= 5:
                    icbl = int(float(parts[1]))
                    sh2o = float(parts[2])
                    out.append(f" 1 {icbl:5d} {sh2o:5.2f} {value:5.1f} {value:5.1f}")
                    changed += 1
                    continue
            if stripped.startswith("*") or stripped.startswith("@"):
                in_ic_layers = False
        out.append(line)
    if changed == 0:
        raise RuntimeError("Could not patch initial soil SNH4/SNO3")
    return "\n".join(out) + "\n"


def prepare_case(variant: str) -> Path:
    cfg = VARIANTS[variant]
    case_dir = OUT_DIR / variant
    input_dir = case_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASE_MZX, input_dir / "CNHL04A_AUTO_N.MZX")
    shutil.copyfile(WTH, input_dir / "CNHL0401.WTH")
    shutil.copyfile(SOIL, input_dir / "SOIL.SOL")
    shutil.copyfile(NEW_CUL, input_dir / "MZCER048.CUL")

    filex = input_dir / "CNHL04A_AUTO_N.MZX"
    text = filex.read_text(encoding="latin1", errors="ignore")
    text = patch_management(text, cfg["irrig"], cfg["ferti"])
    text = patch_nitrogen(text, cfg)
    text = patch_initial_soil_n(text, cfg.get("initial_n"))
    filex.write_text(text, encoding="latin1")

    aux = [str(p) for p in sorted(input_dir.iterdir()) if p.is_file() and p.name != filex.name]
    env_args = {
        "log_saving_path": str(case_dir / f"{variant}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_dir


def child_run(variant: str, max_steps: int = 260) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    case_dir = OUT_DIR / variant
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
            rows.append(
                {
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": scalar(latest.get("yrdoy")),
                    "reward": scalar(reward),
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "done": done,
                }
            )
            step += 1
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, snapshot, dirs_exist_ok=True)
        env.close()
    pd.DataFrame(rows).to_csv(case_dir / f"{variant}_post_state.csv", index=False, encoding="utf-8-sig")


def parse_table_out(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header = None
    rows = []
    for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("@"):
            header = s.replace("@", "", 1).split()
            continue
        if header and re.match(r"^\d", s):
            parts = s.split()
            if len(parts) >= len(header):
                rows.append(parts[: len(header)])
    if not rows or not header:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def parse_events(path: Path) -> dict:
    irrigation = 0.0
    fertilizer = 0.0
    irrigation_events = 0
    fertilizer_events = 0
    if path.exists():
        for line in path.read_text(encoding="latin1", errors="ignore").splitlines():
            if "Irrigation" in line:
                m = re.search(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*mm", line)
                if m:
                    irrigation += float(m.group(1))
                    irrigation_events += 1
            if "Fertilizer" in line:
                m = re.search(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*kg", line)
                if m:
                    fertilizer += float(m.group(1))
                    fertilizer_events += 1
    return {
        "irrigation_total_mgmtevent": irrigation,
        "fertilizer_total_mgmtevent": fertilizer,
        "irrigation_events_mgmtevent": irrigation_events,
        "fertilizer_events_mgmtevent": fertilizer_events,
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if pd.isna(val) else f"{val:.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize_variant(variant: str, status: dict) -> dict:
    snap = OUT_DIR / variant / "pdi_tmp_snapshot"
    plant = parse_table_out(snap / "PlantGro.OUT")
    summary = parse_table_out(snap / "Summary.OUT")
    row = {"variant": variant, **VARIANTS[variant], **status, **parse_events(snap / "MgmtEvent.OUT")}
    if not plant.empty:
        last = plant.iloc[-1]
        row.update(
            {
                "final_dap": float(last.get("DAP", float("nan"))),
                "final_gwad": float(last.get("GWAD", float("nan"))),
                "final_cwad": float(last.get("CWAD", float("nan"))),
                "max_wspd": float(pd.to_numeric(plant.get("WSPD", pd.Series(dtype=float)), errors="coerce").max()),
                "max_nstd": float(pd.to_numeric(plant.get("NSTD", pd.Series(dtype=float)), errors="coerce").max()),
            }
        )
    if not summary.empty:
        s = summary.iloc[-1]
        for key in ["HWAM", "CWAM", "MDAT", "IR#M", "IRCM", "NI#M", "NICM"]:
            if key in s:
                row[key] = s[key]
    return row


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child_run(sys.argv[2])
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        prepare_case(variant)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", variant]
        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=120, capture_output=True, text=True)
            status = {"returncode": proc.returncode, "timed_out": False, "stderr_tail": proc.stderr[-600:]}
        except subprocess.TimeoutExpired as exc:
            status = {"returncode": None, "timed_out": True, "stderr_tail": str(exc)[-600:]}
        rows.append(summarize_variant(variant, status))
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 011_11 HLA 2004 native automatic fertilizer rescue",
        "",
        "Base input: Windows DSSAT480 `run_CNHL0408_DSSAT480_2004/CNHL0408.MZX` with updated cultivar copied from `cultivar_calibration_HLA2004_480/input_corrected_package/MZCER048.CUL`.",
        "",
        markdown_table(result),
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
