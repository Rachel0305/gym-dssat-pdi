from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
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
OUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013" / "SY" / "sy_2012_2014_ic_diagnosis_016_01_runs"
MAX_STEPS = 450


@dataclass(frozen=True)
class Case:
    name: str
    experiment_number: int
    description: str
    patch_2014_ic: int | None = None
    icdat_override: int | None = None


CASES = [
    Case("sy2012_ic1", 1, "2012 treatment, keep original IC=1"),
    Case("sy2014_ic0", 2, "2014 treatment, keep original IC=0"),
    Case("sy2014_ic1", 2, "2014 treatment, patch treatment row back to IC=1", patch_2014_ic=1),
    Case(
        "sy2014_ic1_icdat14100",
        2,
        "2014 treatment, patch to IC=1 and set ICDAT=14100",
        patch_2014_ic=1,
        icdat_override=14100,
    ),
]


def parse_table_out(path: Path) -> pd.DataFrame:
    header: list[str] | None = None
    current_run: int | None = None
    current_treatment: str | None = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            run_match = re.match(r"\*RUN\s+(\d+)\s*:\s*(.*?)\s{2,}", line)
            if run_match:
                current_run = int(run_match.group(1))
                current_treatment = run_match.group(2).strip()
                continue
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if header and re.match(r"^\d{4}\s+\d+", stripped):
                parts = stripped.split()
                if len(parts) >= len(header):
                    row = dict(zip(header, parts[: len(header)]))
                    row["RUNNO"] = current_run
                    row["TNAM"] = current_treatment
                    rows.append(row)
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    if {"RUNNO", "YEAR", "DOY", "DAP"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["RUNNO", "YEAR", "DOY", "DAP"], keep="last").reset_index(drop=True)
    return df


def parse_summary_out(path: Path) -> pd.DataFrame:
    header = None
    rows = []
    with path.open("r", encoding="latin1", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                header = stripped.replace("@", "", 1).split()
                continue
            if header and re.match(r"^\d+", stripped):
                parts = stripped.split()
                if len(parts) >= len(header):
                    rows.append(parts[: len(header)])
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def patch_2014_ic_value(source: str, ic_value: int) -> str:
    old = " 2 1 1 0 Sim2014                    1  2  0  0  2  0  2  0  0  0  0  0  2"
    new = f" 2 1 1 0 Sim2014                    1  2  0  {ic_value}  2  0  2  0  0  0  0  0  2"
    if old in source:
        return source.replace(old, new)
    pattern = r"(?m)^(\s*2\s+1\s+1\s+0\s+Sim2014\s+1\s+2\s+0\s+)(\d+)(\s+2\s+0\s+2\s+0\s+0\s+0\s+0\s+0\s+2\s*)$"
    return re.sub(pattern, rf"\g<1>{ic_value}\g<3>", source)


def patch_icdat_value(source: str, icdat_value: int) -> str:
    pattern = r"(?m)^(\s*1\s+MZ\s+)(\d+)(\s+100\s+0\s+1\s+1\s+-99\s+1000\s+\.8\s+0\s+100\s+15\s+-99\s*)$"
    return re.sub(pattern, rf"\g<1>{icdat_value}\g<3>", source)


def prepare_case(case: Case) -> Path:
    case_dir = OUT_ROOT / case.name
    input_dir = case_dir / "input"
    if case_dir.exists():
        shutil.rmtree(case_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    mzx_src = INPUT_ROOT / MZX_NAME
    text = mzx_src.read_text(encoding="latin1", errors="ignore")
    if case.patch_2014_ic is not None:
        text = patch_2014_ic_value(text, case.patch_2014_ic)
    if case.icdat_override is not None:
        text = patch_icdat_value(text, case.icdat_override)
    (input_dir / MZX_NAME).write_text(text, encoding="latin1", errors="ignore")

    for src in INPUT_ROOT.iterdir():
        if src.name == MZX_NAME:
            continue
        if src.suffix.upper() in {".WTH", ".SOL", ".CUL", ".CLI", ".WDB", ".PRM", ".MZA", ".MZT"}:
            shutil.copyfile(src, input_dir / src.name)

    meta = {
        "case": case.name,
        "description": case.description,
        "experiment_number": case.experiment_number,
        "patch_2014_ic": case.patch_2014_ic,
        "icdat_override": case.icdat_override,
        "input_root": str(INPUT_ROOT),
    }
    (case_dir / "case_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return case_dir


def run_case(case: Case, max_steps: int = MAX_STEPS) -> dict[str, Any]:
    import gym
    from sb3_wrapper import GymDssatWrapper

    case_dir = prepare_case(case)
    input_dir = case_dir / "input"
    weather_code = {1: "CNSY1201.WTH", 2: "CNSY1401.WTH", 3: "CNSY1501.WTH"}[case.experiment_number]
    aux = [
        str(input_dir / weather_code),
        str(input_dir / "SOIL.SOL"),
        str(input_dir / "MZCER048.CUL"),
        str(input_dir / "CNSY.CLI"),
        str(input_dir / "CNSY.PRM"),
        str(input_dir / "CNSY.wdb"),
    ]
    env_args = {
        "log_saving_path": str(case_dir / f"{case.name}.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(input_dir / MZX_NAME),
        "experiment_number": case.experiment_number,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (case_dir / "env_args.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")

    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    obs, info = env.reset()
    records = []
    terminated = truncated = False
    for step in range(max_steps):
        zero_action = {name: 0.0 for name in env.formator.action_names}
        norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, zero_action)
        obs, reward, terminated, truncated, info = env.step(norm)
        latest = latest_observation_dict(env, obs, info)
        yrdoy = scalar(latest.get("yrdoy"))
        records.append(
            {
                "case": case.name,
                "step": step,
                "yrdoy": yrdoy,
                "year": int(yrdoy // 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                "dap": scalar(latest.get("dap")),
                "topwt": scalar(latest.get("topwt")),
                "grnwt": scalar(latest.get("grnwt")),
                "xlai": scalar(latest.get("xlai")),
                "swfac": scalar(latest.get("swfac")),
                "nstres": scalar(latest.get("nstres")),
                "reward": scalar(reward),
                "done": bool(terminated or truncated),
            }
        )
        if terminated or truncated:
            break
    daily = pd.DataFrame(records)
    daily.to_csv(case_dir / f"{case.name}_gym_post_state_daily.csv", index=False, encoding="utf-8-sig")

    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if tmp and Path(tmp).exists():
        shutil.copytree(tmp, case_dir / "pdi_tmp_snapshot", dirs_exist_ok=True)
    env.close()

    pg = parse_table_out(case_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    pg.to_csv(case_dir / f"{case.name}_PlantGro_parsed.csv", index=False, encoding="utf-8-sig")
    summary_out = parse_summary_out(case_dir / "pdi_tmp_snapshot" / "Summary.OUT")
    summary_out.to_csv(case_dir / f"{case.name}_Summary_parsed.csv", index=False, encoding="utf-8-sig")

    last_daily = daily.iloc[-1].to_dict() if not daily.empty else {}
    row0 = summary_out.iloc[0].to_dict() if not summary_out.empty else {}
    summary = {
        "case": case.name,
        "description": case.description,
        "experiment_number": case.experiment_number,
        "patch_2014_ic": case.patch_2014_ic,
        "icdat_override": case.icdat_override,
        "steps_completed": len(daily),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_dap_gym": last_daily.get("dap"),
        "final_topwt_gym": last_daily.get("topwt"),
        "final_grnwt_gym": last_daily.get("grnwt"),
        "max_swfac_gym": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nstres_gym": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "hwam_summary": row0.get("HWAM"),
        "cwam_summary": row0.get("CWAM"),
        "mdat_summary": row0.get("MDAT"),
        "edat_summary": row0.get("EDAT"),
    }
    pd.DataFrame([summary]).to_csv(case_dir / f"{case.name}_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def make_figure(all_daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    styles = {
        "sy2012_ic1": dict(color="black", linestyle="-", linewidth=2.0, label="SY2012 IC=1"),
        "sy2014_ic1": dict(color="red", linestyle="--", linewidth=2.0, label="SY2014 IC=1"),
        "sy2014_ic0": dict(color="blue", linestyle="-.", linewidth=2.0, label="SY2014 IC=0"),
    }
    panels = [
        ("topwt", "TOPWT"),
        ("grnwt", "GRNWT"),
        ("swfac", "SWFAC"),
        ("nstres", "NSTRES"),
    ]
    for ax, (col, label) in zip(axes, panels):
        for case, grp in all_daily.groupby("case"):
            grp = grp.sort_values("dap")
            ax.plot(grp["dap"], grp[col], **styles.get(case, {}))
        ax.set_ylabel(label)
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[-1].set_xlabel("DAP")
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("SY 2012/2014 gym-DSSAT process diagnosis", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "sy_2012_2014_ic_diagnosis_process.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    table_text = summary.to_string(index=False)
    lines = [
        "# 016_01 沈阳站 2012/2014 IC 诊断",
        "",
        "本诊断使用 gym-DSSAT/PDI 前向模拟，仅做 no-op（不追加动作）复现，比较三个 case：",
        "",
        "- SY2012 + IC=1",
        "- SY2014 + IC=1",
        "- SY2014 + IC=0",
        "",
        "目标：判断沈阳站 2014 产量异常下滑，究竟来自年份差异，还是来自 `IC=1` 初始条件。",
        "",
        "## 汇总",
        "",
        "```text",
        table_text,
        "```",
        "",
        "## 文件",
        "",
        f"- 汇总表：`{(OUT_ROOT / 'sy_2012_2014_ic_diagnosis_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- 日值总表：`{(OUT_ROOT / 'sy_2012_2014_ic_diagnosis_daily_all.csv').relative_to(PROJECT_ROOT)}`",
        f"- 对比图：`{(OUT_ROOT / 'sy_2012_2014_ic_diagnosis_process.png').relative_to(PROJECT_ROOT)}`",
    ]
    (PROJECT_ROOT / "docs" / "2026-07-02_016_01_sy2012_2014_ic_diagnosis_record.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    try:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    summaries = []
    all_daily = []
    for case in CASES:
        summary = run_case(case)
        summaries.append(summary)
        daily_path = OUT_ROOT / case.name / f"{case.name}_gym_post_state_daily.csv"
        if daily_path.exists():
            all_daily.append(pd.read_csv(daily_path))
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_ROOT / "sy_2012_2014_ic_diagnosis_summary.csv", index=False, encoding="utf-8-sig")
    if all_daily:
        all_daily_df = pd.concat(all_daily, ignore_index=True)
        all_daily_df.to_csv(OUT_ROOT / "sy_2012_2014_ic_diagnosis_daily_all.csv", index=False, encoding="utf-8-sig")
        make_figure(all_daily_df)
    write_report(summary_df)
    print(OUT_ROOT / "sy_2012_2014_ic_diagnosis_summary.csv")


if __name__ == "__main__":
    main()
