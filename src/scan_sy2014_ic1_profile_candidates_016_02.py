from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


INPUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY"
MZX_NAME = "CNSY1201.MZX"
OUT_ROOT = INPUT_ROOT / "sy2014_ic1_profile_scan_016_02"
MAX_STEPS = 450
WATER_ALPHAS = [0.0, 0.5, 1.0]
N_SCALES = [0.5, 1.0, 1.5]
DEPTHS = [10, 20, 30, 40, 60, 100]


def parse_soil_dul() -> dict[int, float]:
    text = (INPUT_ROOT / "SOIL.SOL").read_text(encoding="latin1", errors="ignore").splitlines()
    start = None
    for i, line in enumerate(text):
        if line.startswith("*SY990012007"):
            start = i
            break
    if start is None:
        raise RuntimeError("SY990012007 not found in SOIL.SOL")
    for j in range(start, min(start + 30, len(text))):
        if text[j].lstrip().startswith("@  SLB") or text[j].lstrip().startswith("@ SLB"):
            header = text[j].replace("@", "", 1).split()
            rows = []
            for k in range(j + 1, len(text)):
                raw = text[k].strip()
                if not raw or raw.startswith("*"):
                    break
                parts = raw.split()
                if len(parts) >= len(header):
                    rows.append(parts[: len(header)])
            df = pd.DataFrame(rows, columns=header)
            df["SLB"] = pd.to_numeric(df["SLB"])
            df["SDUL"] = pd.to_numeric(df["SDUL"])
            return {int(r.SLB): float(r.SDUL) for r in df.itertuples()}
    raise RuntimeError("SLB/SDUL block for SY990012007 not found")


def parse_base_ic() -> pd.DataFrame:
    text = (INPUT_ROOT / MZX_NAME).read_text(encoding="latin1", errors="ignore").splitlines()
    start = None
    for i, line in enumerate(text):
        if line.startswith("@C  ICBL"):
            start = i + 1
            break
    if start is None:
        raise RuntimeError("IC block not found")
    rows = []
    for j in range(start, len(text)):
        raw = text[j].strip()
        if not raw or raw.startswith("*"):
            break
        parts = raw.split()
        if len(parts) >= 5:
            rows.append(parts[:5])
    df = pd.DataFrame(rows, columns=["C", "ICBL", "SH2O", "SNH4", "SNO3"])
    for col in ["ICBL", "SH2O", "SNH4", "SNO3"]:
        df[col] = pd.to_numeric(df[col])
    df = df[df["ICBL"].isin(DEPTHS)].copy()
    return df


def patch_treatment_to_ic1(text: str) -> str:
    old = " 2 1 1 0 Sim2014                    1  2  0  0  2  0  2  0  0  0  0  0  2"
    new = " 2 1 1 0 Sim2014                    1  2  0  1  2  0  2  0  0  0  0  0  2"
    if old in text:
        return text.replace(old, new)
    return text


def patch_ic_profile(text: str, profile: pd.DataFrame) -> str:
    lines = text.splitlines()
    out = []
    in_ic_rows = False
    profile_lines = {
        int(r.ICBL): f" 1{int(r.ICBL):6d}{float(r.SH2O):7.3f}{float(r.SNH4):6.2f}{float(r.SNO3):6.2f}"
        for r in profile.itertuples()
    }
    for line in lines:
        if line.startswith("@C  ICBL"):
            in_ic_rows = True
            out.append(line)
            continue
        if in_ic_rows:
            stripped = line.strip()
            if not stripped or stripped.startswith("*"):
                in_ic_rows = False
                if stripped.startswith("*"):
                    out.append(line)
                continue
            parts = stripped.split()
            if len(parts) >= 4 and parts[0] == "1":
                depth = int(float(parts[1]))
                if depth in profile_lines:
                    out.append(profile_lines[depth])
                else:
                    out.append(line)
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_case(case_name: str, profile: pd.DataFrame) -> Path:
    case_dir = OUT_ROOT / case_name
    input_dir = case_dir / "input"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    text = (INPUT_ROOT / MZX_NAME).read_text(encoding="latin1", errors="ignore")
    text = patch_treatment_to_ic1(text)
    text = patch_ic_profile(text, profile)
    (input_dir / MZX_NAME).write_text(text, encoding="latin1", errors="ignore")

    for src in INPUT_ROOT.iterdir():
        if src.name == MZX_NAME:
            continue
        if src.suffix.upper() in {".WTH", ".SOL", ".CUL", ".CLI", ".WDB", ".PRM", ".MZA", ".MZT"}:
            shutil.copyfile(src, input_dir / src.name)
    profile.to_csv(case_dir / "patched_ic_profile.csv", index=False, encoding="utf-8-sig")
    return case_dir


def run_case(case_name: str, profile: pd.DataFrame) -> dict[str, Any]:
    import gym
    from sb3_wrapper import GymDssatWrapper

    case_dir = prepare_case(case_name, profile)
    input_dir = case_dir / "input"
    aux = [
        str(input_dir / "CNSY1401.WTH"),
        str(input_dir / "SOIL.SOL"),
        str(input_dir / "MZCER048.CUL"),
        str(input_dir / "CNSY.CLI"),
        str(input_dir / "CNSY.PRM"),
        str(input_dir / "CNSY.wdb"),
    ]
    env_args = {
        "log_saving_path": str(case_dir / f"{case_name}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(input_dir / MZX_NAME),
        "experiment_number": 2,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")

    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    obs, info = env.reset()
    rows = []
    terminated = truncated = False
    for step in range(MAX_STEPS):
        zero_action = {name: 0.0 for name in env.formator.action_names}
        norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, zero_action)
        obs, reward, terminated, truncated, info = env.step(norm)
        latest = latest_observation_dict(env, obs, info)
        rows.append(
            {
                "case": case_name,
                "step": step,
                "dap": scalar(latest.get("dap")),
                "topwt": scalar(latest.get("topwt")),
                "grnwt": scalar(latest.get("grnwt")),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "reward": scalar(reward),
                "done": bool(terminated or truncated),
            }
        )
        if terminated or truncated:
            break
    env.close()

    daily = pd.DataFrame(rows)
    daily.to_csv(case_dir / f"{case_name}_daily.csv", index=False, encoding="utf-8-sig")
    return {
        "case": case_name,
        "steps_completed": len(daily),
        "final_dap": float(daily["dap"].iloc[-1]) if not daily.empty else np.nan,
        "final_topwt": float(daily["topwt"].iloc[-1]) if not daily.empty else np.nan,
        "final_grnwt": float(daily["grnwt"].iloc[-1]) if not daily.empty else np.nan,
        "max_swfac": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nstres": float(daily["nstres"].max()) if not daily.empty else np.nan,
    }


def build_profile(base_ic: pd.DataFrame, dul_map: dict[int, float], water_alpha: float, n_scale: float) -> pd.DataFrame:
    prof = base_ic.copy()
    prof["SDUL"] = prof["ICBL"].map(dul_map)
    prof["SH2O"] = prof["SH2O"] + water_alpha * (prof["SDUL"] - prof["SH2O"])
    prof["SNH4"] = prof["SNH4"] * n_scale
    prof["SNO3"] = prof["SNO3"] * n_scale
    return prof[["C", "ICBL", "SH2O", "SNH4", "SNO3"]]


def plot_heatmap(df: pd.DataFrame) -> None:
    piv = df.pivot(index="n_scale", columns="water_alpha", values="final_grnwt").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
    ax.set_xlabel("water_alpha_to_DUL")
    ax.set_ylabel("nitrogen_scale")
    ax.set_title("SY2014 IC=1 candidate scan: final GRNWT")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f"{piv.values[i, j]:.0f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="final GRNWT")
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "sy2014_ic1_candidate_scan_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    best = summary.sort_values(["final_grnwt", "final_topwt"], ascending=[False, False]).iloc[0]
    lines = [
        "# 016_02 沈阳站 2014 专属 IC=1 候选扫描",
        "",
        "方法：",
        "- 仅针对 2014，强制使用 IC=1",
        "- SH2O 从当前剖面向土壤 DUL 插值，做 3 档 water_alpha",
        "- SNH4/SNO3 同步缩放，做 3 档 nitrogen_scale",
        "- 共 9 个 no-op 前向模拟，比较最终 GRNWT/TOPWT 与胁迫",
        "",
        "## 汇总表",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        f"推荐首个候选：`{best['case']}`，water_alpha={best['water_alpha']}, n_scale={best['n_scale']}。",
    ]
    (PROJECT_ROOT / "docs" / "2026-07-02_016_02_sy2014_ic1_candidate_scan_record.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dul_map = parse_soil_dul()
    base_ic = parse_base_ic()
    rows = []
    for water_alpha in WATER_ALPHAS:
        for n_scale in N_SCALES:
            case_name = f"wa{water_alpha:.1f}_ns{n_scale:.1f}".replace(".", "p")
            profile = build_profile(base_ic, dul_map, water_alpha, n_scale)
            summary = run_case(case_name, profile)
            summary["water_alpha"] = water_alpha
            summary["n_scale"] = n_scale
            rows.append(summary)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_ROOT / "sy2014_ic1_candidate_scan_summary.csv", index=False, encoding="utf-8-sig")
    plot_heatmap(summary_df)
    write_report(summary_df)
    print(OUT_ROOT / "sy2014_ic1_candidate_scan_summary.csv")


if __name__ == "__main__":
    main()
