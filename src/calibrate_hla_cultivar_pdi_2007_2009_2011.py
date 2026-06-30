"""Calibrate HLA maize cultivar parameters with PDI/gym-DSSAT 4.8.0.024.

This is a low-cost forward-simulation calibration helper, not PPO training.

Inputs are kept in:
    DSSAT_auto_validation/HLA_2004/cultivar_calibration_HLA2004_480/input_review_package

The script copies those inputs into per-candidate run folders, patches only the
HY0006 line in a copied MZCER048.CUL, runs treatments 1/2/3 via the PDI runtime,
and compares against observed values in CNHL0701.MZA.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


BASE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004" / "cultivar_calibration_HLA2004_480"
INPUT_DIR = BASE_DIR / "input_corrected_package"
OUT_DIR = BASE_DIR / "pdi_calibration_runs"
FILEX_NAME = "CNHL0701_corrected_IC123.MZX"

PARAM_NAMES = ["P1", "P2", "P5", "G2", "G3", "PHINT"]
RANGES = {
    # Conservative first-pass ranges around the existing HLA cultivar.
    "P1": (190.0, 280.0),
    "P2": (0.20, 0.80),
    "P5": (560.0, 760.0),
    "G2": (180.0, 520.0),
    "G3": (8.0, 22.0),
    "PHINT": (32.0, 50.0),
}


def read_cultivar_params(cul_path: Path, cultivar_id: str = "HY0006") -> dict[str, float]:
    for raw in cul_path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if raw.startswith(f"{cultivar_id} "):
            parts = raw.split()
            if len(parts) < 11:
                raise RuntimeError(f"Malformed cultivar line for {cultivar_id}: {raw}")
            values = [float(x) for x in parts[-6:]]
            return dict(zip(PARAM_NAMES, values))
    raise RuntimeError(f"{cultivar_id} cultivar line not found in {cul_path}")


CURRENT = read_cultivar_params(INPUT_DIR / "MZCER048.CUL")


def parse_dssat_table(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    columns: list[str] | None = None
    with path.open("r", encoding="latin-1", errors="ignore") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                columns = stripped.split()
                if columns and columns[0] == "@":
                    columns = columns[1:]
                elif columns and columns[0].startswith("@"):
                    columns[0] = columns[0].lstrip("@")
                continue
            if columns is None or stripped.startswith("*") or stripped.startswith("!"):
                continue
            parts = stripped.split()
            if not parts or not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < len(columns):
                parts += [""] * (len(columns) - len(parts))
            elif len(parts) > len(columns):
                parts = parts[: len(columns)]
            rows.append(parts)
    df = pd.DataFrame(rows, columns=columns or [])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def read_observed() -> pd.DataFrame:
    obs = parse_dssat_table(INPUT_DIR / "CNHL0701.MZA")
    obs = obs.rename(columns={"TRNO": "trno", "ADAP": "obs_adap", "MDAP": "obs_mdap", "CWAM": "obs_cwam", "HWAM": "obs_hwam"})
    return obs[["trno", "obs_adap", "obs_mdap", "obs_cwam", "obs_hwam"]].copy()


def patch_cultivar(src: Path, dst: Path, params: dict[str, float]) -> None:
    lines = src.read_text(encoding="latin-1", errors="ignore").splitlines()
    out: list[str] = []
    changed = False
    for line in lines:
        if line.startswith("HY0006 "):
            out.append(
                "HY0006 Haiyu    No006       . IB0001 "
                f"{params['P1']:.1f} {params['P2']:.3f} {params['P5']:.1f} "
                f"{params['G2']:.1f} {params['G3']:.2f} {params['PHINT']:.2f}"
            )
            changed = True
        else:
            out.append(line)
    if not changed:
        raise RuntimeError("HY0006 cultivar line not found in MZCER048.CUL")
    dst.write_text("\n".join(out) + "\n", encoding="latin-1", errors="ignore")


def prepare_candidate(candidate_id: str, params: dict[str, float]) -> Path:
    run_dir = OUT_DIR / candidate_id
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for name in [FILEX_NAME, "CNHL0701.MZA", "CNHL0701.MZT", "CNHL0701.WTH", "CNHL0901.WTH", "CNHL1101.WTH", "SOIL.SOL"]:
        src = INPUT_DIR / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, input_dir / name)
    patch_cultivar(INPUT_DIR / "MZCER048.CUL", input_dir / "MZCER048.CUL", params)
    (run_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    return run_dir


def child_run(run_dir: Path, trno: int, max_steps: int = 360) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    input_dir = run_dir / "input"
    env_args = {
        "log_saving_path": str(run_dir / f"pdi_gym_trno{trno}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(input_dir / FILEX_NAME),
        "experiment_number": int(trno),
        "auxiliary_file_paths": [
            str(input_dir / "CNHL0701.WTH"),
            str(input_dir / "CNHL0901.WTH"),
            str(input_dir / "CNHL1101.WTH"),
            str(input_dir / "SOIL.SOL"),
            str(input_dir / "MZCER048.CUL"),
        ],
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    obs, info = env.reset()
    rows = []
    for step in range(max_steps):
        action = {name: 0.0 for name in env.formator.action_names}
        norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
        obs, reward, terminated, truncated, info = env.step(norm)
        latest = latest_observation_dict(env, obs, info)
        yrdoy = scalar(latest.get("yrdoy"))
        rows.append(
            {
                "step": step,
                "yrdoy": yrdoy,
                "dap": scalar(latest.get("dap")),
                "topwt": scalar(latest.get("topwt")),
                "grnwt": scalar(latest.get("grnwt")),
                "xlai": scalar(latest.get("xlai")),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "done": bool(terminated or truncated),
            }
        )
        if terminated or truncated:
            break
    pd.DataFrame(rows).to_csv(run_dir / f"gym_post_state_trno{trno}.csv", index=False, encoding="utf-8-sig")
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if tmp and Path(tmp).exists():
        shutil.copytree(tmp, run_dir / f"pdi_tmp_snapshot_trno{trno}", dirs_exist_ok=True)
    env.close()


def date_to_dap(date_value: float | int | str, pdat_value: float | int | str) -> float:
    try:
        date = int(float(date_value))
        pdat = int(float(pdat_value))
    except Exception:
        return np.nan
    if date <= 0 or pdat <= 0:
        return np.nan
    return float(date % 1000 - pdat % 1000)


def read_summary(run_dir: Path, trno: int) -> dict:
    overview = run_dir / f"pdi_tmp_snapshot_trno{trno}" / "OVERVIEW.OUT"
    text = overview.read_text(encoding="latin-1", errors="ignore")

    def grab(pattern: str) -> float:
        matches = re.findall(pattern, text)
        if not matches:
            return np.nan
        return float(matches[-1])

    sim_adap = grab(r"Anthesis day \(dap\)\s+([-+]?\d+(?:\.\d+)?)")
    sim_mdap = grab(r"Physiological maturity day \(dap\)\s+([-+]?\d+(?:\.\d+)?)")
    sim_hwam = grab(r"Yield at harvest maturity \(kg \[dm\]/ha\)\s+([-+]?\d+(?:\.\d+)?)")
    sim_cwam = grab(r"Tops weight at maturity \(kg \[dm\]/ha\)\s+([-+]?\d+(?:\.\d+)?)")
    sim_lai = grab(r"Leaf area index, maximum\s+([-+]?\d+(?:\.\d+)?)")

    # Seasonal water/N totals are still easiest to get from Summary.OUT. If the
    # whitespace parser drifts, keep these as NaN rather than failing scoring.
    row: dict = {}
    path = run_dir / f"pdi_tmp_snapshot_trno{trno}" / "Summary.OUT"
    try:
        df = parse_dssat_table(path)
        if not df.empty:
            row = df.iloc[0].to_dict()
    except Exception:
        row = {}
    return {
        "trno": trno,
        "sim_adap": sim_adap,
        "sim_mdap": sim_mdap,
        "sim_cwam": sim_cwam,
        "sim_hwam": sim_hwam,
        "sim_lai": sim_lai,
        "IRCM": float(row.get("IRCM", np.nan)),
        "NICM": float(row.get("NICM", np.nan)),
    }


def run_candidate(candidate_id: str, params: dict[str, float]) -> pd.DataFrame:
    run_dir = prepare_candidate(candidate_id, params)
    for trno in [1, 2, 3]:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--run-dir", str(run_dir), "--trno", str(trno)]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=160, capture_output=True, text=True)
        (run_dir / f"child_stdout_trno{trno}.txt").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        (run_dir / f"child_stderr_trno{trno}.txt").write_text(proc.stderr, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            raise RuntimeError(f"Candidate {candidate_id} TRNO {trno} failed; see {run_dir}")
    sim = pd.DataFrame([read_summary(run_dir, trno) for trno in [1, 2, 3]])
    obs = read_observed()
    merged = obs.merge(sim, on="trno", how="left")
    for key in ["adap", "mdap", "cwam", "hwam"]:
        merged[f"err_{key}"] = merged[f"sim_{key}"] - merged[f"obs_{key}"]
    merged["candidate_id"] = candidate_id
    for name, val in params.items():
        merged[name] = val
    merged.to_csv(run_dir / "candidate_observed_vs_simulated.csv", index=False, encoding="utf-8-sig")
    return merged


def score_candidate(df: pd.DataFrame) -> float:
    # Normalize errors to avoid yield completely swamping phenology.
    terms = []
    terms.extend((df["err_adap"] / 5.0).to_list())
    terms.extend((df["err_mdap"] / 7.0).to_list())
    terms.extend((df["err_hwam"] / 1000.0).to_list())
    terms.extend((df["err_cwam"] / 2000.0).to_list())
    return float(np.sqrt(np.nanmean(np.square(terms))))


def random_params(rng: random.Random) -> dict[str, float]:
    return {name: rng.uniform(lo, hi) for name, (lo, hi) in RANGES.items()}


def run_baseline() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_candidate("baseline_current_HY0006", CURRENT)
    df["score"] = score_candidate(df)
    df.to_csv(OUT_DIR / "baseline_current_HY0006_observed_vs_simulated.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print("score", score_candidate(df))


def run_search(n: int, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    all_rows = []
    candidates = [("baseline_current_HY0006", CURRENT)]
    for i in range(n):
        candidates.append((f"random_{i:03d}", random_params(rng)))
    for candidate_id, params in candidates:
        df = run_candidate(candidate_id, params)
        score = score_candidate(df)
        summary = {"candidate_id": candidate_id, "score": score, **params}
        for _, row in df.iterrows():
            summary[f"tr{int(row['trno'])}_sim_hwam"] = row["sim_hwam"]
            summary[f"tr{int(row['trno'])}_obs_hwam"] = row["obs_hwam"]
            summary[f"tr{int(row['trno'])}_sim_mdap"] = row["sim_mdap"]
            summary[f"tr{int(row['trno'])}_obs_mdap"] = row["obs_mdap"]
        all_rows.append(summary)
        pd.DataFrame(all_rows).sort_values("score").to_csv(OUT_DIR / "cultivar_search_summary.csv", index=False, encoding="utf-8-sig")
        print(candidate_id, score)
    print(pd.DataFrame(all_rows).sort_values("score").head(10).to_string(index=False))


def write_input_readme() -> None:
    obs = read_observed()
    lines = [
        "# HLA cultivar calibration input package",
        "",
        "Purpose: Hailun cultivar calibration with PDI/gym-DSSAT 4.8.0.024, using 2007/2009/2011 observed management and observations.",
        "",
        "Files:",
        "",
        f"- `{FILEX_NAME}`: corrected three-treatment file for 2007, 2009, 2011; cultivar `HY0006`; management `IRRIG=R, FERTI=R`; IC levels 1/2/3 match years.",
        "- `CNHL0701.MZA`: observed ADAP, MDAP, CWAM, HWAM.",
        "- `CNHL0701.MZT`: observed LAI/CWAD time-course, retained for later inspection but not used in first-pass score.",
        "- `CNHL0701.WTH`, `CNHL0901.WTH`, `CNHL1101.WTH`: year-specific weather files.",
        "- `SOIL.SOL`: soil file containing `HL99001200`.",
        "- `MZCER048.CUL`: cultivar file; calibration patches only the `HY0006` line in per-run copies.",
        "",
        "Observed targets from CNHL0701.MZA:",
        "",
        obs.to_csv(index=False),
        "",
        "Important: no original input file is modified during calibration; each candidate gets a copied `MZCER048.CUL`.",
    ]
    (INPUT_DIR / "README_input_package.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--trno", type=int)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--write-readme", action="store_true")
    args = parser.parse_args()
    if args.child:
        if args.run_dir is None or args.trno is None:
            raise SystemExit("--child requires --run-dir and --trno")
        child_run(args.run_dir, args.trno)
        return
    if args.write_readme:
        write_input_readme()
    if args.baseline:
        run_baseline()
    if args.search:
        run_search(args.n, args.seed)


if __name__ == "__main__":
    main()
