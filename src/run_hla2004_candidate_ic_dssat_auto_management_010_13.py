"""Run HLA 2004 candidate-IC DSSAT native automatic management check.

This is a forward-simulation diagnostic only. It does not train PPO.

Scenario:
    HLA 2004, candidate IC = water fraction 0.55 + mineral N scale 0.25.
    Change management from null/rule baseline to DSSAT native automatic
    irrigation and automatic fertilization (IRRIG=A, FERTI=A).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ppo_action_safety import normalize_action
from ppo_evaluate import latest_observation_dict, scalar


YEAR = 2004
SOURCE_RUN_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "candidate_ic055_n025_null_2004_2023"
    / "runs"
    / str(YEAR)
)
OUT_DIR = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "HLA_2004"
    / "candidate_ic055_n025_dssat_auto_management_010_13"
)
RUN_DIR = OUT_DIR / "run"


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
            if columns is None:
                continue
            if stripped.startswith("*") or stripped.startswith("!"):
                continue
            parts = stripped.split()
            if not parts or not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < len(columns):
                parts = parts + [""] * (len(columns) - len(parts))
            elif len(parts) > len(columns):
                parts = parts[: len(columns)]
            rows.append(parts)
    df = pd.DataFrame(rows, columns=columns or [])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def convert_to_dssat_auto_management(text: str) -> str:
    """Change only the management control line to IRRIG=A and FERTI=A."""
    out: list[str] = []
    in_management_table = False
    changed = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N MANAGEMENT") and "PLANT" in stripped and "IRRIG" in stripped and "FERTI" in stripped:
            in_management_table = True
            out.append(line)
            continue
        if in_management_table and re.match(r"^\s*1\s+MA\b", line):
            out.append(" 1 MA              R     A     A     R     M")
            changed = True
            in_management_table = False
            continue
        if in_management_table and (stripped.startswith("@") or stripped.startswith("*")):
            in_management_table = False
        out.append(line)
    if not changed:
        raise RuntimeError("Could not find management line to change to IRRIG=A/FERTI=A")
    return "\n".join(out) + "\n"


def prepare_run() -> None:
    input_dir = RUN_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_input = SOURCE_RUN_DIR / "input"
    mzx_files = sorted(source_input.glob("*.MZX"))
    if not mzx_files:
        raise FileNotFoundError(f"No source MZX found in {source_input}")

    source_mzx = mzx_files[0]
    text = source_mzx.read_text(encoding="latin-1", errors="ignore")
    text = convert_to_dssat_auto_management(text)
    filex = input_dir / "CNHL04DA.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    aux_paths = []
    for src in [*source_input.glob("*.WTH"), *source_input.glob("*.SOL"), *source_input.glob("*.CUL")]:
        dst = input_dir / src.name
        shutil.copyfile(src, dst)
        aux_paths.append(str(dst))

    env_args = {
        "log_saving_path": str(RUN_DIR / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": 1,
        "auxiliary_file_paths": aux_paths,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (RUN_DIR / "env_args.json").write_text(json.dumps(env_args, indent=2), encoding="utf-8")
    (RUN_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "scenario": "candidate_ic055_n025_dssat_auto_irrigation_auto_fertilization",
                "source_mzx": str(source_mzx),
                "changed_management_line": "IRRIG=A, FERTI=A",
                "automatic_irrigation": {
                    "IMDEP": 30,
                    "ITHRL": 50,
                    "ITHRU": 100,
                    "IROFF": "GS000",
                    "IMETH": "IR001",
                    "IRAMT": 10,
                    "IREFF": 1,
                },
                "automatic_nitrogen": {
                    "NMDEP": 30,
                    "NMTHR": 50,
                    "NAMNT": 25,
                    "NCODE": "FE001",
                    "NAOFF": "GS000",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def child_run(max_steps: int = 320) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((RUN_DIR / "env_args.json").read_text(encoding="utf-8"))
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
    pd.DataFrame(rows).to_csv(RUN_DIR / "gym_post_state_daily.csv", index=False, encoding="utf-8-sig")
    tmp = getattr(env.unwrapped, "_tmp_folder", None)
    if tmp and Path(tmp).exists():
        shutil.copytree(tmp, RUN_DIR / "pdi_tmp_snapshot", dirs_exist_ok=True)
    env.close()


def read_weather_rain() -> pd.DataFrame:
    wth_files = sorted((RUN_DIR / "input").glob("*.WTH"))
    if not wth_files:
        return pd.DataFrame(columns=["doy", "rain"])
    rows = []
    header = None
    for raw in wth_files[0].read_text(encoding="latin-1", errors="ignore").splitlines():
        stripped = raw.strip()
        if stripped.startswith("@") and "DATE" in stripped and "RAIN" in stripped:
            header = stripped.replace("@", "", 1).split()
            continue
        if not header or not stripped or not stripped[0].isdigit():
            continue
        parts = stripped.split()
        row = dict(zip(header, parts))
        date = int(row["DATE"])
        rows.append({"doy": date % 1000, "rain": float(row.get("RAIN", 0))})
    return pd.DataFrame(rows)


def standardize_plantgro() -> pd.DataFrame:
    plantgro = RUN_DIR / "pdi_tmp_snapshot" / "PlantGro.OUT"
    df = parse_dssat_table(plantgro)
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(df.get("YEAR"), errors="coerce"),
            "doy": pd.to_numeric(df.get("DOY"), errors="coerce"),
            "das": pd.to_numeric(df.get("DAS"), errors="coerce"),
            "dap": pd.to_numeric(df.get("DAP"), errors="coerce"),
            "wspd": pd.to_numeric(df.get("WSPD"), errors="coerce"),
            "nstd": pd.to_numeric(df.get("NSTD"), errors="coerce"),
            "gwad": pd.to_numeric(df.get("GWAD"), errors="coerce"),
            "cwad": pd.to_numeric(df.get("CWAD"), errors="coerce"),
            "lai": pd.to_numeric(df.get("LAID"), errors="coerce"),
        }
    )
    rain = read_weather_rain()
    if not rain.empty:
        out = out.merge(rain, on="doy", how="left")
    else:
        out["rain"] = 0.0
    out["rain"] = out["rain"].fillna(0.0)
    out = out.dropna(subset=["dap"]).drop_duplicates(subset=["dap"], keep="last")
    out.to_csv(OUT_DIR / "hla2004_candidate_ic_dssat_auto_daily_values.csv", index=False, encoding="utf-8-sig")
    return out


def parse_management_events() -> pd.DataFrame:
    path = RUN_DIR / "pdi_tmp_snapshot" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["dap", "operation", "amount", "unit", "raw"])
    pattern = re.compile(
        r"^\s*\d+\s+\w+\s+\d+,\s+\d{4}\s+\d+\s+\d+\s+(-?\d+)\s+\w+\s+(.+?)\s+([-+]?\d+(?:\.\d+)?)\s+(\S+)"
    )
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        match = pattern.match(raw)
        if match:
            rows.append(
                {
                    "dap": int(match.group(1)),
                    "operation": match.group(2).strip(),
                    "amount": float(match.group(3)),
                    "unit": match.group(4),
                    "raw": raw.rstrip(),
                }
            )
        else:
            rows.append({"dap": np.nan, "operation": "unparsed", "amount": np.nan, "unit": "", "raw": raw.rstrip()})
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.drop_duplicates(subset=["dap", "operation", "amount", "unit"], keep="first").reset_index(drop=True)
    events.to_csv(OUT_DIR / "hla2004_candidate_ic_dssat_auto_management_events.csv", index=False, encoding="utf-8-sig")
    return events


def parse_summary() -> dict[str, float | str | None]:
    path = RUN_DIR / "pdi_tmp_snapshot" / "Summary.OUT"
    if not path.exists():
        return {}
    lines = path.read_text(encoding="latin-1", errors="ignore").splitlines()
    header_line = next((line for line in lines if line.startswith("@")), None)
    data_lines = [line for line in lines if re.match(r"^\s+\d+\s+\d+\s+\d+", line)]
    if not header_line or not data_lines:
        return {}
    columns = header_line.replace("@", "", 1).split()
    parts = data_lines[-1].split()
    # HLA rows can leave XLAT/LONG/ELEV blank in Summary.OUT.  If we split by
    # whitespace, every field after SOIL_ID shifts left by three columns.
    # Insert explicit missing values after SOIL_ID so summary metrics align.
    if len(columns) - len(parts) == 3 and "SOIL_ID..." in columns:
        soil_idx = columns.index("SOIL_ID...")
        parts = parts[: soil_idx + 1] + ["-99", "-99", "-99"] + parts[soil_idx + 1 :]
    row = dict(zip(columns, parts))
    keys = ["HWAM", "CWAM", "MDAT", "ADAT", "IR#M", "IRCM", "NI#M", "NICM", "PRCP", "ETCP"]
    out: dict[str, float | str | None] = {}
    for key in keys:
        if key in row:
            val = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
            out[key] = None if pd.isna(val) else float(val)
    return out


def plot_daily(daily: pd.DataFrame, events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax = axes[0]
    ax.bar(daily["dap"], daily["rain"], color="#9AA3B2", alpha=0.55, width=1.0, label="Rain")
    irrig = events[events["operation"].str.contains("Irrigation", case=False, na=False)]
    fert = events[events["operation"].str.contains("Fertil|Nitrogen", case=False, na=False)]
    if not irrig.empty:
        ax.bar(irrig["dap"], irrig["amount"], color="#2176FF", alpha=0.8, width=2.5, label="Auto irrigation")
    if not fert.empty:
        ax.bar(fert["dap"], fert["amount"], color="#7D3C98", alpha=0.8, width=2.5, label="Auto fertilization")
    ax.set_ylabel("Water/N amount")
    ax.set_title("Rainfall and DSSAT automatic management events")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[1]
    ax.plot(daily["dap"], daily["wspd"], color="#D62728", linewidth=1.8, label="WSPD water stress")
    ax.plot(daily["dap"], daily["nstd"], color="#2CA02C", linewidth=1.8, label="NSTD nitrogen stress")
    ax.set_ylabel("Stress index")
    ax.set_title("DSSAT stress indices; lower value means stronger stress")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ax = axes[2]
    ax.plot(daily["dap"], daily["gwad"], color="#111111", linewidth=1.8, label="GWAD")
    ax.plot(daily["dap"], daily["cwad"], color="#666666", linewidth=1.5, linestyle="--", label="CWAD")
    ax.set_ylabel("kg/ha")
    ax.set_xlabel("DAP")
    ax.set_title("Growth outcome")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("HLA 2004 candidate IC: DSSAT native automatic irrigation + fertilization", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "hla2004_candidate_ic_dssat_auto_management_daily.png", dpi=220)
    plt.close(fig)


def write_report(daily: pd.DataFrame, events: pd.DataFrame, summary: dict[str, float | str | None]) -> None:
    irrig = events[events["operation"].str.contains("Irrigation", case=False, na=False)]
    fert = events[events["operation"].str.contains("Fertil|Nitrogen", case=False, na=False)]
    final = daily.sort_values("dap").iloc[-1].to_dict()
    rows = [
        "# 2026-06-27 HLA 2004 候选 IC 下 DSSAT 原生自动管理检查",
        "",
        "## 设置",
        "",
        "- 年份/站点：HLA 2004。",
        "- 初始条件：候选 `SH2O=0.55 可利用水分 + SNH4/SNO3=0.25N`。",
        "- 管理方式：`IRRIG=A, FERTI=A`。",
        "- 自动灌溉参数沿用输入文件：`IMDEP=30, ITHRL=50, ITHRU=100, IROFF=GS000, IMETH=IR001, IRAMT=10, IREFF=1`。",
        "- 自动施肥参数沿用输入文件：`NMDEP=30, NMTHR=50, NAMNT=25, NCODE=FE001, NAOFF=GS000`。",
        "- 不训练 PPO；仅 DSSAT/PDI forward simulation。",
        "",
        "## 结果摘要",
        "",
        f"- 最终 DAP：{final.get('dap')}",
        f"- 最终 GWAD：{final.get('gwad')} kg/ha",
        f"- 最终 CWAD：{final.get('cwad')} kg/ha",
        f"- Summary HWAM：{summary.get('HWAM')} kg/ha",
        f"- Summary CWAM：{summary.get('CWAM')} kg/ha",
        f"- Summary IR#M/IRCM：{summary.get('IR#M')} 次 / {summary.get('IRCM')} mm",
        f"- Summary NI#M/NICM：{summary.get('NI#M')} 次 / {summary.get('NICM')} kg/ha",
        f"- MgmtEvent 自动灌溉事件数：{len(irrig)}，合计：{irrig['amount'].sum() if not irrig.empty else 0:.2f} mm",
        f"- MgmtEvent 自动施肥事件数：{len(fert)}，合计：{fert['amount'].sum() if not fert.empty else 0:.2f}",
        "",
        "## 初步判读",
        "",
    ]
    if len(irrig) > 0 and len(fert) > 0:
        rows.append("- 自动灌溉和自动施肥都发生了，该情景可暂时作为 DSSAT 原生自动管理候选。")
    elif len(irrig) > 0 and len(fert) == 0:
        rows.append("- 自动灌溉发生了，但自动施肥没有触发；该情景不能直接称为完整水氮自动管理，只能称为 DSSAT 自动灌溉 + 未触发自动施肥。")
    elif len(irrig) == 0 and len(fert) > 0:
        rows.append("- 自动施肥发生了，但自动灌溉没有触发；该情景不是完整的 DSSAT 自动水氮管理。")
    else:
        rows.append("- 自动灌溉和自动施肥都没有触发；该情景不适合作为自动管理基线。")
    rows.extend(
        [
            "",
            "## 文件",
            "",
            "- `hla2004_candidate_ic_dssat_auto_daily_values.csv`",
            "- `hla2004_candidate_ic_dssat_auto_management_events.csv`",
            "- `hla2004_candidate_ic_dssat_auto_summary.csv`",
            "- `hla2004_candidate_ic_dssat_auto_management_daily.png`",
        ]
    )
    (PROJECT_ROOT / "docs" / "2026-06-27_hla2004_candidate_ic_dssat_auto_management_010_13.md").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def parent_run(rerun: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_run()
    plantgro = RUN_DIR / "pdi_tmp_snapshot" / "PlantGro.OUT"
    if rerun or not plantgro.exists():
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child"]
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=120, capture_output=True, text=True)
        (OUT_DIR / "child_stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="ignore")
        (OUT_DIR / "child_stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            raise RuntimeError(f"Child run failed with {proc.returncode}; see {OUT_DIR / 'child_stderr.txt'}")
    daily = standardize_plantgro()
    events = parse_management_events()
    summary = parse_summary()
    summary_df = pd.DataFrame([{**summary, "final_gwad_from_plantgro": daily.sort_values("dap").iloc[-1]["gwad"]}])
    summary_df.to_csv(OUT_DIR / "hla2004_candidate_ic_dssat_auto_summary.csv", index=False, encoding="utf-8-sig")
    plot_daily(daily, events)
    write_report(daily, events, summary)
    print(summary_df.to_string(index=False))
    print(events.to_string(index=False) if not events.empty else "No management events found")
    print(f"Wrote outputs to {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.child:
        child_run()
    else:
        parent_run(rerun=args.rerun)


if __name__ == "__main__":
    main()
