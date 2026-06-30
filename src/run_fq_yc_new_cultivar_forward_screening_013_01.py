from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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


INPUT_ROOT = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_forward_screening_013_01"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_013_01_fq_yc_new_cultivar_forward_screening_record.md"


SITE_CONFIG = {
    "FQ": {
        "label": "Fengqiu",
        "mzx": "CNFQ0801.MZX",
        "treatments": {2007: 1, 2008: 2, 2010: 3},
    },
    "YC": {
        "label": "Yucheng",
        "mzx": "CNYC0801.MZX",
        "treatments": {2008: 1, 2014: 2},
    },
}
SCENARIOS = ["null", "recorded", "dssat_auto"]


def parse_dssat_table(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    columns: list[str] | None = None
    if not path.exists():
        return pd.DataFrame()
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


def set_treatment_pointers(text: str, trno: int, mi: str, mf: str) -> str:
    out = []
    changed = False
    in_treatments = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N R O C TNAME"):
            in_treatments = True
            out.append(line)
            continue
        if in_treatments and re.match(rf"^\s*{trno}\s+\d+\s+\d+\s+\d+\s+\S+", line):
            parts = line.split()
            while len(parts) < 18:
                parts.append("0")
            # DSSAT treatment columns after TNAME:
            # CU FL SA IC MP MI MF MR MC MT ME MH SM
            parts[10] = mi
            parts[11] = mf
            out.append(
                f" {parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]:<25} "
                f"{parts[5]:>2} {parts[6]:>2} {parts[7]:>2} {parts[8]:>2} {parts[9]:>2} {parts[10]:>2} {parts[11]:>2} "
                f"{parts[12]:>2} {parts[13]:>2} {parts[14]:>2} {parts[15]:>2} {parts[16]:>2} {parts[17]:>2}"
            )
            changed = True
            continue
        if in_treatments and stripped.startswith("*"):
            in_treatments = False
        out.append(line)
    if not changed:
        raise RuntimeError(f"Could not update treatment pointers for TRNO {trno}")
    return "\n".join(out) + "\n"


def set_management_for_treatment(text: str, trno: int, irrig: str, ferti: str) -> str:
    out = []
    changed = False
    in_management = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@N MANAGEMENT") and "IRRIG" in stripped and "FERTI" in stripped:
            in_management = True
            out.append(line)
            continue
        if in_management and re.match(rf"^\s*{trno}\s+MA\b", line):
            out.append(f"{trno:2d} MA              R     {irrig}     {ferti}     R     M")
            changed = True
            continue
        if in_management and stripped.startswith("*"):
            in_management = False
        out.append(line)
    if not changed:
        raise RuntimeError(f"Could not update management line for TRNO {trno}")
    return "\n".join(out) + "\n"


def zero_target_reported_rows(text: str, trno: int) -> str:
    out = []
    in_ir = False
    in_fe = False
    wrote_ir_zero = False
    wrote_fe_zero = False
    for line in text.splitlines():
        stripped = line.strip()
        if line.startswith("@I IDATE"):
            in_ir = True
            in_fe = False
            out.append(line)
            continue
        if line.startswith("@F FDATE"):
            in_fe = True
            in_ir = False
            out.append(line)
            continue
        if in_ir:
            if stripped.startswith("@") or stripped.startswith("*"):
                if not wrote_ir_zero:
                    out.append(f" {trno} 01001 IR001     0")
                    wrote_ir_zero = True
                in_ir = False
                out.append(line)
                continue
            if re.match(rf"^\s*{trno}\s+\d{{5}}\b", line):
                if not wrote_ir_zero:
                    date = line.split()[1]
                    out.append(f" {trno} {date} IR001     0")
                    wrote_ir_zero = True
                continue
        if in_fe:
            if stripped.startswith("@") or stripped.startswith("*"):
                if not wrote_fe_zero:
                    out.append(f" {trno} 01001 FE005 AP002     0     0   -99   -99   -99   -99   -99 null")
                    wrote_fe_zero = True
                in_fe = False
                out.append(line)
                continue
            if re.match(rf"^\s*{trno}\s+\d{{5}}\b", line):
                if not wrote_fe_zero:
                    date = line.split()[1]
                    out.append(f" {trno} {date} FE005 AP002     0     0   -99   -99   -99   -99   -99 null")
                    wrote_fe_zero = True
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def prepare_text_for_scenario(source: str, trno: int, scenario: str) -> str:
    if scenario == "recorded":
        return source
    if scenario == "null":
        text = set_treatment_pointers(source, trno, "0", "0")
        text = set_management_for_treatment(text, trno, "N", "N")
        text = zero_target_reported_rows(text, trno)
        return text
    if scenario == "dssat_auto":
        text = set_management_for_treatment(source, trno, "A", "A")
        return text
    raise ValueError(scenario)


def prepare_run(site: str, year: int, scenario: str) -> Path:
    cfg = SITE_CONFIG[site]
    trno = cfg["treatments"][year]
    input_src = INPUT_ROOT / site
    run_dir = OUT_DIR / "runs" / site / f"{year}_{scenario}"
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    source = (input_src / cfg["mzx"]).read_text(encoding="latin-1", errors="ignore")
    text = prepare_text_for_scenario(source, trno, scenario)
    filex = input_dir / f"{site}{year}_{scenario}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")
    for src in input_src.iterdir():
        if src.is_file() and src.name != cfg["mzx"]:
            shutil.copyfile(src, input_dir / src.name)
    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
    env_args = {
        "log_saving_path": str(run_dir / "pdi_gym.log"),
        "mode": "all",
        "seed": 0,
        "random_weather": False,
        "evaluation": True,
        "fileX_template_path": str(filex),
        "experiment_number": trno,
        "auxiliary_file_paths": aux,
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
    }
    (run_dir / "env_args.json").write_text(json.dumps(env_args, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps({"site": site, "year": year, "trno": trno, "scenario": scenario}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def child_run(run_dir: Path, max_steps: int = 380) -> None:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows = []
    try:
        obs, info = env.reset()
        for step in range(max_steps):
            action = {name: 0.0 for name in env.formator.action_names}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": meta["site"],
                    "requested_year": meta["year"],
                    "scenario": meta["scenario"],
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
    finally:
        tmp = getattr(env.unwrapped, "_tmp_folder", None)
        if tmp and Path(tmp).exists():
            shutil.copytree(tmp, run_dir / "pdi_tmp_snapshot", dirs_exist_ok=True)
        env.close()
    pd.DataFrame(rows).to_csv(run_dir / "gym_post_state_daily.csv", index=False, encoding="utf-8-sig")


def parse_summary(run_dir: Path) -> dict[str, Any]:
    df = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "Summary.OUT")
    if df.empty:
        return {}
    row = df.iloc[-1]
    out: dict[str, Any] = {}
    for key in ["HWAM", "CWAM", "ADAT", "MDAT", "IR#M", "IRCM", "NI#M", "NICM", "PRCP", "ETCP"]:
        val = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        out[key] = None if pd.isna(val) else float(val)
    return out


def parse_plantgro(run_dir: Path, site: str, year: int, scenario: str) -> pd.DataFrame:
    df = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "site": site,
            "requested_year": year,
            "scenario": scenario,
            "doy": pd.to_numeric(df.get("DOY"), errors="coerce"),
            "dap": pd.to_numeric(df.get("DAP"), errors="coerce"),
            "wspd": pd.to_numeric(df.get("WSPD"), errors="coerce"),
            "nstd": pd.to_numeric(df.get("NSTD"), errors="coerce"),
            "gwad": pd.to_numeric(df.get("GWAD"), errors="coerce"),
            "cwad": pd.to_numeric(df.get("CWAD"), errors="coerce"),
            "lai": pd.to_numeric(df.get("LAID"), errors="coerce"),
        }
    )
    return out.dropna(subset=["dap"]).drop_duplicates(["site", "requested_year", "scenario", "dap"], keep="last")


def parse_events(run_dir: Path, site: str, year: int, scenario: str) -> pd.DataFrame:
    path = run_dir / "pdi_tmp_snapshot" / "MgmtEvent.OUT"
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["site", "requested_year", "scenario", "dap", "operation", "amount", "unit", "raw"])
    for raw in path.read_text(encoding="latin-1", errors="ignore").splitlines():
        if "Irrigation" not in raw and "Fertil" not in raw and "Nitrogen" not in raw:
            continue
        parts = raw.split()
        dap = np.nan
        if len(parts) >= 7:
            try:
                dap = int(parts[6])
            except ValueError:
                pass
        amount = 0.0
        unit = ""
        m = re.search(r"([-+]?\d+(?:\.\d*)?)\s*(mm|kg(?:\[[A-Za-z]+\])?/ha|kg)", raw)
        if m:
            amount = float(m.group(1))
            unit = m.group(2)
        rows.append(
            {
                "site": site,
                "requested_year": year,
                "scenario": scenario,
                "dap": dap,
                "operation": raw.strip(),
                "amount": amount,
                "unit": unit,
                "raw": raw,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["site", "requested_year", "scenario", "dap", "operation", "amount", "unit"])
    return out


def parse_weather(site: str, year: int) -> pd.DataFrame:
    yy = year % 100
    wth = INPUT_ROOT / site / f"CN{site}{yy:02d}01.WTH"
    if not wth.exists():
        return pd.DataFrame(columns=["doy", "rain"])
    rows = []
    in_data = False
    header = []
    for line in wth.read_text(encoding="latin-1", errors="ignore").splitlines():
        if line.startswith("@"):
            header = line.replace("@", "", 1).split()
            in_data = "DATE" in header
            continue
        if in_data and line.strip() and not line.startswith("*") and not line.startswith("!"):
            parts = line.split()
            if len(parts) >= len(header):
                rec = dict(zip(header, parts[: len(header)]))
                date = rec.get("DATE", "")
                try:
                    doy = int(date[-3:])
                    rain = float(rec.get("RAIN", 0))
                except ValueError:
                    continue
                rows.append({"doy": doy, "rain": rain})
    return pd.DataFrame(rows)


def run_cases(timeout: int, smoke: bool = False) -> pd.DataFrame:
    statuses = []
    for site, cfg in SITE_CONFIG.items():
        years = list(cfg["treatments"])
        if smoke:
            years = years[:1]
        for year in years:
            for scenario in SCENARIOS:
                run_dir = prepare_run(site, year, scenario)
                cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(run_dir)]
                try:
                    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout, capture_output=True, text=True)
                    statuses.append(
                        {
                            "site": site,
                            "year": year,
                            "scenario": scenario,
                            "returncode": proc.returncode,
                            "timed_out": False,
                            "stdout_tail": proc.stdout[-1000:],
                            "stderr_tail": proc.stderr[-1000:],
                            "run_dir": str(run_dir),
                        }
                    )
                except subprocess.TimeoutExpired as exc:
                    statuses.append(
                        {
                            "site": site,
                            "year": year,
                            "scenario": scenario,
                            "returncode": None,
                            "timed_out": True,
                            "stdout_tail": "",
                            "stderr_tail": str(exc)[-1000:],
                            "run_dir": str(run_dir),
                        }
                    )
    status = pd.DataFrame(statuses)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status.to_csv(OUT_DIR / ("013_01_smoke_status.csv" if smoke else "013_01_run_status.csv"), index=False, encoding="utf-8-sig")
    return status


def collect_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    daily_frames = []
    event_frames = []
    for site, cfg in SITE_CONFIG.items():
        for year in cfg["treatments"]:
            for scenario in SCENARIOS:
                run_dir = OUT_DIR / "runs" / site / f"{year}_{scenario}"
                summary = {"site": site, "station": cfg["label"], "year": year, "scenario": scenario}
                summary.update(parse_summary(run_dir))
                daily = parse_plantgro(run_dir, site, year, scenario)
                if not daily.empty:
                    summary["max_wspd"] = float(daily["wspd"].max())
                    summary["max_nstd"] = float(daily["nstd"].max())
                    summary["final_gwad_daily"] = float(daily["gwad"].dropna().iloc[-1])
                    summary["final_cwad_daily"] = float(daily["cwad"].dropna().iloc[-1])
                    summary["final_dap"] = float(daily["dap"].dropna().iloc[-1])
                    daily_frames.append(daily)
                events = parse_events(run_dir, site, year, scenario)
                if not events.empty:
                    event_frames.append(events)
                    irr = events[events["operation"].str.contains("Irrigation", case=False, na=False)]
                    fert = events[events["operation"].str.contains("Fertil", case=False, na=False)]
                    summary["event_irrigation_total"] = float(irr["amount"].sum()) if not irr.empty else 0.0
                    summary["event_fertilizer_total"] = float(fert["amount"].sum()) if not fert.empty else 0.0
                summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    event_df = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    summary_df.to_csv(OUT_DIR / "013_01_fq_yc_forward_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(OUT_DIR / "013_01_fq_yc_forward_daily.csv", index=False, encoding="utf-8-sig")
    event_df.to_csv(OUT_DIR / "013_01_fq_yc_forward_events.csv", index=False, encoding="utf-8-sig")
    return summary_df, daily_df, event_df


def plot_one(site: str, year: int, daily: pd.DataFrame, events: pd.DataFrame) -> None:
    sub = daily[(daily["site"].eq(site)) & (daily["requested_year"].eq(year))].copy()
    if sub.empty:
        return
    rain = parse_weather(site, year)
    if not rain.empty:
        pdate_doy = int(sub["doy"].min() - sub["dap"].min())
        rain["dap"] = rain["doy"] - pdate_doy
        rain = rain[(rain["dap"] >= 0) & (rain["dap"] <= sub["dap"].max() + 5)]
    scenarios = [s for s in SCENARIOS if s in set(sub["scenario"])]
    colors = {"null": "#464C55", "recorded": "#CC6F47", "dssat_auto": "#5477C4"}
    labels = {"null": "Null", "recorded": "Recorded", "dssat_auto": "DSSAT auto"}
    fig, axes = plt.subplots(5, 1, figsize=(15.5, 13.0), sharex=True, gridspec_kw={"height_ratios": [0.8, 1, 1, 1, 1.1]})
    fig.suptitle(f"{site} {year} new-cultivar forward screening", x=0.08, ha="left", fontsize=15, fontweight="bold")
    if not rain.empty:
        axes[0].bar(rain["dap"], rain["rain"], width=1.0, color="#C5CAD3", edgecolor="#7A828F", linewidth=0.35)
    axes[0].set_ylabel("Rain\n(mm)")
    for scenario in scenarios:
        s = sub[sub["scenario"].eq(scenario)].sort_values("dap")
        axes[1].plot(s["dap"], s["wspd"], color=colors[scenario], lw=2, label=labels[scenario])
        axes[2].plot(s["dap"], s["nstd"], color=colors[scenario], lw=2, label=labels[scenario])
        axes[4].plot(s["dap"], s["gwad"], color=colors[scenario], lw=2)
        axes[4].plot(s["dap"], s["cwad"], color=colors[scenario], lw=1.6, ls="--", alpha=0.7)
    axes[1].set_ylabel("Water\nstress")
    axes[2].set_ylabel("Nitrogen\nstress")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")
    ev = events[(events["site"].eq(site)) & (events["requested_year"].eq(year))]
    for scenario in scenarios:
        sev = ev[ev["scenario"].eq(scenario)]
        color = colors[scenario]
        irr = sev[sev["operation"].str.contains("Irrigation", case=False, na=False)]
        fert = sev[sev["operation"].str.contains("Fertil", case=False, na=False)]
        if not irr.empty:
            axes[3].vlines(irr["dap"], 0, irr["amount"], color=color, lw=2.2)
        if not fert.empty:
            axes[3].scatter(fert["dap"], fert["amount"], marker="^", color=color, edgecolor="white", s=45, zorder=4)
    axes[3].set_ylabel("Mgmt\namount")
    axes[4].set_ylabel("kg/ha")
    axes[4].set_xlabel("DAP")
    axes[4].set_title("solid = grain; dashed = biomass", loc="left", fontsize=10)
    for ax in axes:
        ax.grid(True, color="#E6E8F0", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = OUT_DIR / "figures" / f"{site}_{year}_forward_process.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(daily: pd.DataFrame, events: pd.DataFrame) -> None:
    for site, cfg in SITE_CONFIG.items():
        for year in cfg["treatments"]:
            plot_one(site, year, daily, events)


def write_doc(summary: pd.DataFrame, status: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_无数据_"
        safe = df.copy()
        safe = safe.fillna("")
        cols = list(safe.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in safe.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# 013_01 封丘/禹城新参数前向筛选记录",
        "",
        "## 目的",
        "",
        "本轮只做前向模拟，不训练 RL。目标是检查 FQ/YC 新品种参数和输入包是否能在 PDI/gym-DSSAT 中正常运行，并初步判断校准年份是否存在水氮管理优化空间。",
        "",
        "## 运行状态",
        "",
        md_table(status[["site", "year", "scenario", "returncode", "timed_out"]]),
        "",
        "## 汇总结果",
        "",
    ]
    cols = [
        "site",
        "year",
        "scenario",
        "HWAM",
        "CWAM",
        "IR#M",
        "IRCM",
        "NI#M",
        "NICM",
        "max_wspd",
        "max_nstd",
        "final_dap",
    ]
    present = [c for c in cols if c in summary.columns]
    lines.append(md_table(summary[present]))
    lines.extend(
        [
            "",
            "## 初步说明",
            "",
            "- 这些年份是参数校准/验证年份，不等同于后续 RL 必选年份。",
            "- 若 recorded 情景已经接近高产且胁迫较低，则后续需要把专家策略迁移到其他年份筛选优化空间。",
            "- DSSAT auto 中 auto-N 若不触发，只记录现象，不把它作为是否可训练 RL 的前提。",
        ]
    )
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if args.child:
        child_run(args.child)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUT_DIR / ("013_01_smoke_status.csv" if args.smoke else "013_01_run_status.csv")
    status = pd.read_csv(status_path, keep_default_na=False) if args.collect_only and status_path.exists() else run_cases(args.timeout, smoke=args.smoke)
    print(status[["site", "year", "scenario", "returncode", "timed_out"]].to_string(index=False))
    if not args.smoke:
        summary, daily, events = collect_outputs()
        make_plots(daily, events)
        write_doc(summary, status)
        print(summary.to_string(index=False))
        print(OUT_DIR)


if __name__ == "__main__":
    main()
