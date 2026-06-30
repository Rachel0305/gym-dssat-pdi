from __future__ import annotations

import datetime as dt
import json
import shutil
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
from run_fq_yc_new_cultivar_forward_screening_013_01 import (
    INPUT_ROOT,
    SITE_CONFIG,
    parse_dssat_table,
    parse_events,
)


OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_yc2014_fixed_water_n_scan_013_02"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-06-30_013_02_yc2014_fixed_water_n_scan_record.md"

SITE = "YC"
YEAR = 2014
TRNO = SITE_CONFIG[SITE]["treatments"][YEAR]
MZX_NAME = SITE_CONFIG[SITE]["mzx"]

IRRIGATION_TOTALS = [0.0, 60.0, 120.0]
N_TOTALS = [0.0, 150.0, 300.0, 375.0]

PLANTING_YYDOY = 14168
IRRIGATION_DAPS = [43, 50, 57]
N_DAPS = [0, 43, 50]
N_SPLIT_PRIMARY = [96 / 374, 278 / 374]
IRRIGATION_EVENT_CAP = 50.0
N_EVENT_CAP = 200.0


def scenario_name(irrigation_total: float, n_total: float) -> str:
    return f"I{int(irrigation_total)}_N{int(n_total)}"


def yy_doy_to_date(yy_doy: int) -> dt.date:
    yy = yy_doy // 1000
    doy = yy_doy % 1000
    year_full = 2000 + yy
    return dt.date(year_full, 1, 1) + dt.timedelta(days=doy - 1)


def date_to_yy_doy(day: dt.date) -> int:
    yy = day.year % 100
    doy = day.timetuple().tm_yday
    return yy * 1000 + doy


def yydoy_from_dap(dap: int) -> int:
    return date_to_yy_doy(yy_doy_to_date(PLANTING_YYDOY) + dt.timedelta(days=int(dap)))


def allocate_irrigation(total: float) -> dict[int, float]:
    remaining = float(total)
    out: dict[int, float] = {}
    for dap in IRRIGATION_DAPS:
        if remaining <= 1e-9:
            break
        amount = min(IRRIGATION_EVENT_CAP, remaining)
        out[dap] = amount
        remaining -= amount
    return out


def allocate_nitrogen(total: float) -> dict[int, float]:
    total = float(total)
    if total <= 1e-9:
        return {}
    first = min(N_EVENT_CAP, total * N_SPLIT_PRIMARY[0])
    second = min(N_EVENT_CAP, total * N_SPLIT_PRIMARY[1])
    used = first + second
    third = max(0.0, total - used)
    out: dict[int, float] = {}
    if first > 0:
        out[N_DAPS[0]] = first
    if second > 0:
        out[N_DAPS[1]] = second
    if third > 0:
        out[N_DAPS[2]] = min(N_EVENT_CAP, third)
    return out


def build_irrigation_lines(irrigation_total: float) -> list[str]:
    alloc = allocate_irrigation(irrigation_total)
    if not alloc:
        return [f" {TRNO} {yydoy_from_dap(43):05d} IR003     0"]
    lines = []
    for dap, amount in alloc.items():
        lines.append(f" {TRNO} {yydoy_from_dap(dap):05d} IR003 {amount:>5.0f}")
    return lines


def build_fertilizer_lines(n_total: float) -> list[str]:
    alloc = allocate_nitrogen(n_total)
    if not alloc:
        return [f" {TRNO} {yydoy_from_dap(0):05d} FE005 AP001     5     0     0     0   -99   -99   -99 null"]
    lines = []
    for dap, amount in alloc.items():
        lines.append(
            f" {TRNO} {yydoy_from_dap(dap):05d} FE005 AP001     5 {amount:>5.0f}     0     0   -99   -99   -99 N{int(round(amount))}"
        )
    return lines


def replace_section_rows(text: str, header_prefix: str, trno: int, new_rows: list[str]) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        out.append(line)
        if line.startswith(header_prefix):
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].startswith("@") and not lines[i].startswith("*"):
                block.append(lines[i])
                i += 1
            kept = [row for row in block if not row.strip().startswith(f"{trno} ")]
            has_target = len(kept) != len(block)
            out.extend(kept)
            if has_target:
                out.extend(new_rows)
                replaced = True
            continue
        i += 1
    if not replaced:
        raise RuntimeError(f"Could not replace rows for TRNO {trno} under {header_prefix}")
    return "\n".join(out) + "\n"


def prepare_run(irrigation_total: float, n_total: float) -> Path:
    run_name = scenario_name(irrigation_total, n_total)
    run_dir = OUT_DIR / "runs" / run_name
    input_dir = run_dir / "input"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    input_src = INPUT_ROOT / SITE
    source = (input_src / MZX_NAME).read_text(encoding="latin-1", errors="ignore")
    text = replace_section_rows(source, "@I IDATE", TRNO, build_irrigation_lines(irrigation_total))
    text = replace_section_rows(text, "@F FDATE", TRNO, build_fertilizer_lines(n_total))
    filex = input_dir / f"{run_name}.MZX"
    filex.write_text(text, encoding="latin-1", errors="ignore")

    for src in input_src.iterdir():
        if src.is_file() and src.name != MZX_NAME:
            shutil.copyfile(src, input_dir / src.name)

    aux = [str(p) for p in input_dir.iterdir() if p.suffix.upper() in {".CUL", ".SOL", ".WTH", ".MZA", ".MZT"}]
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
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {"site": SITE, "year": YEAR, "scenario": run_name, "irrigation_total": irrigation_total, "n_total": n_total},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def child_run(run_dir: Path, max_steps: int = 380) -> dict:
    import gym
    from sb3_wrapper import GymDssatWrapper

    env_args = json.loads((run_dir / "env_args.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    env = GymDssatWrapper(gym.make("gym_dssat_pdi:GymDssatPdi-v0", **env_args).unwrapped)
    rows = []
    try:
        obs, info = env.reset()
        for step in range(max_steps):
            action = {"amir": 0.0, "anfer": 0.0}
            norm = normalize_action(env.formator.action_names, env.formator.action_space_dict, action)
            obs, reward, terminated, truncated, info = env.step(norm)
            latest = latest_observation_dict(env, obs, info)
            yrdoy = scalar(latest.get("yrdoy"))
            rows.append(
                {
                    "site": SITE,
                    "requested_year": YEAR,
                    "scenario": meta["scenario"],
                    "step": step,
                    "dap": scalar(latest.get("dap")),
                    "yrdoy": yrdoy,
                    "year": int(yrdoy // 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "doy": int(yrdoy % 1000) if np.isfinite(yrdoy) and yrdoy > 0 else np.nan,
                    "grnwt": scalar(latest.get("grnwt")),
                    "topwt": scalar(latest.get("topwt")),
                    "swfac": scalar(latest.get("swfac")),
                    "nstres": scalar(latest.get("nstres")),
                    "reward": reward,
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

    daily = pd.DataFrame(rows)
    daily.to_csv(run_dir / f"{meta['scenario']}_daily.csv", index=False, encoding="utf-8-sig")

    plantgro = parse_dssat_table(run_dir / "pdi_tmp_snapshot" / "PlantGro.OUT")
    events = parse_events(run_dir, SITE, YEAR, meta["scenario"])
    summary_row = {
        "site": SITE,
        "year": YEAR,
        "scenario": meta["scenario"],
        "irrigation_total_target": meta["irrigation_total"],
        "nitrogen_total_target": meta["n_total"],
        "event_irrigation_total": float(events.loc[events["unit"].eq("mm"), "amount"].sum()) if not events.empty else 0.0,
        "event_fertilizer_total": float(events.loc[events["unit"].str.contains("kg", na=False), "amount"].sum()) if not events.empty else 0.0,
        "final_gwad_daily": float(plantgro["GWAD"].dropna().iloc[-1]) if "GWAD" in plantgro.columns and plantgro["GWAD"].notna().any() else np.nan,
        "final_cwad_daily": float(plantgro["CWAD"].dropna().iloc[-1]) if "CWAD" in plantgro.columns and plantgro["CWAD"].notna().any() else np.nan,
        "max_wspd": float(daily["swfac"].max()) if not daily.empty else np.nan,
        "max_nstd": float(daily["nstres"].max()) if not daily.empty else np.nan,
        "maturity_dap": float(plantgro["DAP"].dropna().iloc[-1]) if "DAP" in plantgro.columns and plantgro["DAP"].notna().any() else np.nan,
        "plantgro_rows": int(len(plantgro)),
    }
    return {"summary": summary_row, "daily": daily}


def pivot_metric(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        summary.pivot(index="nitrogen_total_target", columns="irrigation_total_target", values=metric)
        .sort_index()
        .sort_index(axis=1)
    )


def plot_heatmap(table: pd.DataFrame, title: str, out_path: Path, fmt: str = ".0f", cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    data = table.values.astype(float)
    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([f"I{int(x)}" for x in table.columns])
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels([f"N{int(y)}" for y in table.index])
    ax.set_xlabel("Irrigation total (mm)")
    ax.set_ylabel("Nitrogen total (kg N/ha)")
    ax.set_title(title)
    threshold = np.nanmean(data) if np.isfinite(data).any() else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = format(val, fmt) if np.isfinite(val) else "NA"
            color = "white" if np.isfinite(val) and val > threshold else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.3f}" if not float(val).is_integer() else f"{int(val)}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_doc(summary: pd.DataFrame) -> None:
    best_yield = summary.sort_values("final_gwad_daily", ascending=False).iloc[0]
    best_low_n = summary[summary["nitrogen_total_target"] <= 150].sort_values("final_gwad_daily", ascending=False).iloc[0]
    best_low_water = summary[summary["irrigation_total_target"] <= 60].sort_values("final_gwad_daily", ascending=False).iloc[0]
    top = summary.sort_values("final_gwad_daily", ascending=False).head(6).copy()
    top = top[
        [
            "scenario",
            "irrigation_total_target",
            "nitrogen_total_target",
            "final_gwad_daily",
            "final_cwad_daily",
            "max_wspd",
            "max_nstd",
        ]
    ]
    lines = [
        "# 013_02 YC2014 固定水氮扫描记录",
        "",
        "## 目的",
        "",
        "- 不训练 PPO，只做固定组合前向模拟；",
        "- 判断 YC 2014 在当前新参数下是否存在明显节水节氮优化空间；",
        "- 为后续是否值得做 RL 提供依据。",
        "",
        "## 网格设置",
        "",
        "- 灌溉总量：0 / 60 / 120 mm",
        "- 施氮总量：0 / 150 / 300 / 375 kg N/ha",
        "- 灌溉在 DAP 43 / 50 / 57 分次写入 reported 管理表",
        "- 施氮优先沿用原记录 DAP 0 与 DAP 43 的比例；超过单次 200 kg N/ha 的部分顺延到 DAP 50",
        "",
        "## 关键结论",
        "",
        f"- 最高 GWAD 组合：`{best_yield['scenario']}`，GWAD={best_yield['final_gwad_daily']:.1f} kg/ha。",
        f"- 低氮(<=150)最优组合：`{best_low_n['scenario']}`，GWAD={best_low_n['final_gwad_daily']:.1f} kg/ha。",
        f"- 低水(<=60)最优组合：`{best_low_water['scenario']}`，GWAD={best_low_water['final_gwad_daily']:.1f} kg/ha。",
        "- 如果低水或低氮组合已经接近高投入组合，说明后续 RL 有真实节本空间；反之则更像高投入追产场景。",
        "",
        "## 产量前六组合",
        "",
        md_table(top),
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    summaries = []
    daily_frames = []
    for irrigation_total in IRRIGATION_TOTALS:
        for n_total in N_TOTALS:
            name = scenario_name(irrigation_total, n_total)
            print(f"running {name}", flush=True)
            run_dir = prepare_run(irrigation_total, n_total)
            result = child_run(run_dir)
            summaries.append(result["summary"])
            daily_frames.append(result["daily"])

    summary_df = pd.DataFrame(summaries).sort_values(["nitrogen_total_target", "irrigation_total_target"]).reset_index(drop=True)
    daily_df = pd.concat(daily_frames, ignore_index=True, sort=False)
    summary_df.to_csv(OUT_DIR / "013_02_yc2014_fixed_scan_summary.csv", index=False, encoding="utf-8-sig")
    daily_df.to_csv(OUT_DIR / "013_02_yc2014_fixed_scan_daily.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "final_gwad_daily": ("YC 2014 GWAD heatmap (kg/ha)", "viridis", ".0f"),
        "final_cwad_daily": ("YC 2014 CWAD heatmap (kg/ha)", "plasma", ".0f"),
        "max_wspd": ("YC 2014 max water stress heatmap", "magma_r", ".3f"),
        "max_nstd": ("YC 2014 max nitrogen stress heatmap", "cividis_r", ".3f"),
    }
    for metric, (title, cmap, fmt) in metrics.items():
        table = pivot_metric(summary_df, metric)
        plot_heatmap(table, title, OUT_DIR / "figures" / f"{metric}_heatmap.png", fmt=fmt, cmap=cmap)

    write_doc(summary_df)
    print(
        summary_df[
            [
                "scenario",
                "irrigation_total_target",
                "nitrogen_total_target",
                "event_irrigation_total",
                "event_fertilizer_total",
                "final_gwad_daily",
                "final_cwad_daily",
                "max_wspd",
                "max_nstd",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
