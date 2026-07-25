from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import gym
import gym_dssat_pdi.envs  # noqa: F401  # register GymDssatPdi-v0 in old gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_free_timing_stress_aware_ppo_dqn_smoke_032_00 as stress_env
from ppo_safe_rendering import SITE_INFO, build_env_args


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sb3_wrapper import GymDssatWrapper
PROMPT = ROOT / "prompts" / "033_00_initial_soil_water_sensitivity_wspd_audit.md"
CONFIG = ROOT / "experiments" / "ppo_observed_years" / "config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
POOL = ROOT / "Leave_One_experiments" / "all_year_weather_calibration_validation" / "scenario_pool" / "all_year_weather_scenario_pool.csv"
OUT = ROOT / "benchmark_results" / "033_00_initial_soil_water_sensitivity_wspd_audit"
TAB = OUT / "tables"
FIG = OUT / "figures"
DOC = ROOT / "docs" / "033_00_initial_soil_water_sensitivity_wspd_audit_record.md"

CASES = [
    {"station": "LCA", "site": "LC", "year": 2019, "reason": "LC2019 五情景图中 WSPD 全程为 0"},
    {"station": "HLA", "site": "HL", "year": 2015, "reason": "HLA 站点 032_22 PPO 输出中 WSPD 普遍为 0"},
]
FRACTIONS = [0.55, 0.30, 0.15]
IC_BRANCHES = [
    {"ic_branch": "current_ic_factor", "force_ic1": False},
    {"ic_branch": "force_ic1", "force_ic1": True},
]
SEED = 0


def ensure_dirs() -> None:
    for path in [TAB, FIG, OUT / "configs", DOC.parent]:
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    shutil.copy2(CONFIG, OUT / "configs" / CONFIG.name)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_selection(config: dict[str, Any]) -> pd.DataFrame:
    pool = pd.read_csv(ROOT / config["paths"]["scenario_pool_csv"])
    pieces = []
    for case in CASES:
        station = case["station"]
        year = int(case["year"])
        row = pool[(pool["station_code"].eq(station)) & (pool["year"].astype(int).eq(year))].copy()
        if len(row) != 1:
            raise RuntimeError(f"Expected exactly one scenario_pool row for {station}{year}, found {len(row)}")
        row["selected_for_train"] = False
        row["selected_for_eval"] = True
        row["selection_reason"] = "033_00_initial_soil_water_sensitivity_wspd_audit"
        pieces.append(row)
    return pd.concat(pieces, ignore_index=True)


def parse_soil_layers(station: str, soil_id: str) -> dict[int, dict[str, float]]:
    soil_path = ROOT / "my_data" / SITE_INFO[station]["soil"]
    lines = soil_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"*{soil_id}"):
            start = i
            break
    if start is None:
        # Some LC templates use LC99001200 while the soil file id is LC990012007.
        for i, line in enumerate(lines):
            if line.strip().startswith("*") and soil_id[:8] in line:
                start = i
                break
    if start is None:
        raise KeyError(f"soil profile {soil_id} not found in {soil_path}")
    header_idx = None
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("@") and "SLLL" in lines[i] and "SDUL" in lines[i]:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"SLLL/SDUL header not found for {soil_id}")
    cols = lines[header_idx].replace("@", " ").split()
    layers: dict[int, dict[str, float]] = {}
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            continue
        if line.lstrip().startswith("*") or line.lstrip().startswith("@") or line.lstrip().startswith("!"):
            break
        parts = line.split()
        if len(parts) < len(cols):
            continue
        data = dict(zip(cols, parts))
        try:
            slb = int(float(data["SLB"]))
            layers[slb] = {"SLLL": float(data["SLLL"]), "SDUL": float(data["SDUL"])}
        except Exception:
            continue
    if not layers:
        raise RuntimeError(f"no soil layers parsed for {soil_id}")
    return layers


def extract_soil_id(template_text: str) -> str:
    for line in template_text.splitlines():
        if "ID_SOIL" in line:
            continue
        if re.match(r"\s*1\s+\S+\s+\S+", line) and "CN" in line:
            parts = line.split()
            if len(parts) >= 12:
                return parts[11]
    raise RuntimeError("could not extract ID_SOIL from rendered template")


def replace_initial_sh2o(template_path: Path, station: str, fraction: float) -> pd.DataFrame:
    text = template_path.read_text(encoding="utf-8", errors="replace")
    soil_id = extract_soil_id(text)
    layers = parse_soil_layers(station, soil_id)
    rows: list[dict[str, Any]] = []
    out_lines: list[str] = []
    in_ic_table = False
    for line in text.splitlines():
        if line.lstrip().startswith("@C  ICBL") and "SH2O" in line:
            in_ic_table = True
            out_lines.append(line)
            continue
        if in_ic_table:
            if line.lstrip().startswith("*") or line.lstrip().startswith("@") or not line.strip():
                in_ic_table = False
                out_lines.append(line)
                continue
            parts = line.split()
            if len(parts) >= 5:
                trt = parts[0]
                icbl = int(float(parts[1]))
                old_sh2o = float(parts[2])
                snh4 = parts[3]
                sno3 = parts[4]
                if icbl not in layers:
                    raise KeyError(f"ICBL {icbl} not found in soil layers for {station} {soil_id}")
                slll = layers[icbl]["SLLL"]
                sdul = layers[icbl]["SDUL"]
                new_sh2o = round(slll + fraction * (sdul - slll), 3)
                out_lines.append(f"{int(trt):2d}{icbl:6d}{new_sh2o:7.3f}{float(snh4):6.1f}{float(sno3):6.1f}")
                rows.append(
                    {
                        "station": station,
                        "soil_id": soil_id,
                        "fraction": fraction,
                        "ICBL_cm": icbl,
                        "SLLL": slll,
                        "SDUL": sdul,
                        "old_SH2O": old_sh2o,
                        "new_SH2O": new_sh2o,
                        "SNH4": float(snh4),
                        "SNO3": float(sno3),
                    }
                )
                continue
        out_lines.append(line)
    template_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)


def force_treatment_ic_factor_one(template_path: Path) -> dict[str, Any]:
    text = template_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    changed = False
    before = ""
    after = ""
    for i, line in enumerate(lines):
        if line.startswith("@N R O C TNAME") and i + 1 < len(lines):
            before = lines[i + 1]
            parts = before.split()
            if len(parts) >= 9:
                # @N R O C TNAME CU FL SA IC ...
                parts[8] = "1"
                after = f"{int(parts[0]):2d} {parts[1]} {parts[2]} {parts[3]} {parts[4]:<26s} " + "  ".join(parts[5:])
                lines[i + 1] = after
                changed = before != after
            break
    template_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ic_forced": changed, "treatment_line_before": before, "treatment_line_after": after}


def find_attr(obj: Any, attr: str) -> Any | None:
    seen: set[int] = set()
    cur = obj
    for _ in range(12):
        if id(cur) in seen:
            return None
        seen.add(id(cur))
        if hasattr(cur, attr):
            try:
                return getattr(cur, attr)
            except Exception:
                return None
        nxt = None
        for name in ["env", "unwrapped", "current_env"]:
            try:
                cand = getattr(cur, name)
                if cand is not cur:
                    nxt = cand
                    break
            except Exception:
                pass
        if nxt is None:
            return None
        cur = nxt
    return None


def run_noop_case(
    config: dict[str, Any],
    env_config: dict[str, Any],
    case: dict[str, Any],
    fraction: float,
    ic_branch: str,
    force_ic1: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None, dict[str, Any]]:
    station = str(case["station"])
    year = int(case["year"])
    year_info = direct_ppo.find_year(env_config, station, year)
    run_tag = f"033_00_{station}_{year}_{ic_branch}_f{str(fraction).replace('.', 'p')}_null_noop"
    env_args = build_env_args(
        station=station,
        year=year,
        planting_date=str(year_info["planting_date"]),
        seed=SEED,
        config=config,
        run_tag=run_tag,
        evaluation=True,
        mode=config.get("runtime", {}).get("mode", "all"),
    )
    template_path = Path(env_args["fileX_template_path"])
    sh2o_df = replace_initial_sh2o(template_path, station, fraction)
    ic_info = {"ic_forced": False, "treatment_line_before": "", "treatment_line_after": ""}
    if force_ic1:
        ic_info = force_treatment_ic_factor_one(template_path)
    sh2o_df["ic_branch"] = ic_branch
    sh2o_df["force_ic1"] = force_ic1
    env = None
    records: list[dict[str, Any]] = []
    snapshot_path: Path | None = None
    try:
        base_env = GymDssatWrapper(gym.make("GymDssatPdi-v0", **env_args).unwrapped)
        env = stress_env.StressAwareDiscreteWrapper(base_env, config)
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < int(config["runtime"]["max_steps"]):
            obs, reward, terminated, truncated, info = env.step(0)
            done = bool(terminated or truncated)
            latest = stress_env.latest_observation_dict(env, obs, info)
            action = dict(getattr(env, "last_action_info", {}))
            records.append(
                {
                    "station": station,
                    "site": case["site"],
                    "year": year,
                    "ic_branch": ic_branch,
                    "force_ic1": force_ic1,
                    "fraction": fraction,
                    "step": step + 1,
                    "dap": stress_env.scalar(latest.get("dap"), np.nan),
                    "grnwt": stress_env.scalar(latest.get("grnwt"), np.nan),
                    "topwt": stress_env.scalar(latest.get("topwt"), np.nan),
                    "wspd": stress_env.scalar(latest.get("swfac"), np.nan),
                    "nstd": stress_env.scalar(latest.get("nstres"), np.nan),
                    "safe_action_amir": float(action.get("safe_action_amir", 0.0)),
                    "safe_action_anfer": float(action.get("safe_action_anfer", 0.0)),
                    "reward": float(reward),
                    "done": done,
                }
            )
            step += 1
        tmp = find_attr(env, "_tmp_folder")
        if tmp is not None:
            src = Path(tmp)
            snapshot_path = OUT / "snapshots" / station / str(year) / ic_branch / f"f{str(fraction).replace('.', 'p')}_null_noop"
            snapshot_path.mkdir(parents=True, exist_ok=True)
            for name in ["Weather.OUT", "PlantGro.OUT", "SoilWat.OUT", "MgmtEvent.OUT", "Summary.OUT", "fileX.MZX"]:
                if (src / name).exists():
                    shutil.copy2(src / name, snapshot_path / name)
    finally:
        if env is not None:
            env.close()
    return pd.DataFrame(records), sh2o_df, snapshot_path, ic_info


def summarize_daily(daily: pd.DataFrame, snapshot_path: Path | None) -> dict[str, Any]:
    wspd = pd.to_numeric(daily["wspd"], errors="coerce")
    nstd = pd.to_numeric(daily["nstd"], errors="coerce")
    return {
        "station": daily["station"].iloc[0],
        "site": daily["site"].iloc[0],
        "year": int(daily["year"].iloc[0]),
        "ic_branch": str(daily["ic_branch"].iloc[0]),
        "force_ic1": bool(daily["force_ic1"].iloc[0]),
        "fraction": float(daily["fraction"].iloc[0]),
        "final_grnwt": float(pd.to_numeric(daily["grnwt"], errors="coerce").max()),
        "final_topwt": float(pd.to_numeric(daily["topwt"], errors="coerce").max()),
        "max_wspd": float(wspd.max()),
        "mean_wspd": float(wspd.mean()),
        "wspd_days_gt_0": int((wspd > 1e-8).sum()),
        "wspd_days_gt_005": int((wspd > 0.05).sum()),
        "max_nstd": float(nstd.max()),
        "mean_nstd": float(nstd.mean()),
        "total_irrigation": float(pd.to_numeric(daily["safe_action_amir"], errors="coerce").sum()),
        "total_nitrogen": float(pd.to_numeric(daily["safe_action_anfer"], errors="coerce").sum()),
        "snapshot_path": snapshot_path.relative_to(ROOT).as_posix() if snapshot_path else "",
    }


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_空表_"
    show = df.copy()
    for col in show.select_dtypes(include=["float", "int"]).columns:
        show[col] = pd.to_numeric(show[col], errors="coerce").round(4)
    show = show.astype(object).where(pd.notna(show), "")
    header = "| " + " | ".join(map(str, show.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(show.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in show.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def plot_summary(summary: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    for (station, year, ic_branch), group in summary.groupby(["station", "year", "ic_branch"]):
        group = group.sort_values("fraction", ascending=False)
        fig, ax1 = plt.subplots(figsize=(7, 4.2))
        ax1.plot(group["fraction"], group["max_wspd"], marker="o", label="max WSPD")
        ax1.plot(group["fraction"], group["mean_wspd"], marker="s", label="mean WSPD")
        ax1.set_xlabel("Initial available water fraction f")
        ax1.set_ylabel("WSPD")
        ax1.set_title(f"{station}{int(year)} initial soil water sensitivity ({ic_branch})")
        ax1.invert_xaxis()
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(group["fraction"], group["final_grnwt"], color="#2E8B57", marker="^", label="final grain")
        ax2.set_ylabel("Final grain yield (kg/ha)")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout()
        path = FIG / f"033_00_{station.lower()}{int(year)}_{ic_branch}_initial_water_sensitivity.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    ensure_dirs()
    config = load_yaml(CONFIG)
    config["paths"]["output_root"] = "benchmark_results/033_00_initial_soil_water_sensitivity_wspd_audit"
    selection = build_selection(config)
    env_config = direct_ppo.build_env_config(config, selection)
    direct_ppo.write_yaml(env_config, OUT / "configs" / "033_00_resolved_env_config.yaml")

    daily_frames: list[pd.DataFrame] = []
    sh2o_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for case in CASES:
        for branch in IC_BRANCHES:
            for fraction in FRACTIONS:
                try:
                    daily, sh2o, snapshot, ic_info = run_noop_case(
                        config,
                        env_config,
                        case,
                        fraction,
                        str(branch["ic_branch"]),
                        bool(branch["force_ic1"]),
                    )
                    daily_frames.append(daily)
                    sh2o_frames.append(sh2o)
                    row = summarize_daily(daily, snapshot)
                    row.update(ic_info)
                    summary_rows.append(row)
                except Exception as exc:
                    failures.append(
                        {
                            "station": case["station"],
                            "year": case["year"],
                            "ic_branch": branch["ic_branch"],
                            "fraction": fraction,
                            "error": repr(exc),
                        }
                    )

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    sh2o_all = pd.concat(sh2o_frames, ignore_index=True) if sh2o_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["station", "year", "ic_branch", "fraction"], ascending=[True, True, True, False])
    failures_df = pd.DataFrame(failures)

    daily_path = TAB / "033_00_initial_water_sensitivity_daily.csv"
    sh2o_path = TAB / "033_00_initial_water_sensitivity_sh2o_settings.csv"
    summary_path = TAB / "033_00_initial_water_sensitivity_summary.csv"
    failures_path = TAB / "033_00_failures.csv"
    daily_all.to_csv(daily_path, index=False, encoding="utf-8-sig")
    sh2o_all.to_csv(sh2o_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    failures_df.to_csv(failures_path, index=False, encoding="utf-8-sig")
    figures = plot_summary(summary) if not summary.empty else []

    lines = [
        "# 033_00 初始土壤水分敏感性与 WSPD 审计记录",
        "",
        "## 结论先说",
        "",
        f"- 状态：{'完成' if failures_df.empty else '部分失败'}。",
        "- 本任务未训练 PPO/DQN，只做 DSSAT 前向 no-op/null 回放。",
        "- 原始 `my_data/UFGA8201-*.jinja2` 和 `.SOL` 文件没有修改；所有 SH2O/IC 因子改动只发生在本任务输出目录下的派生模板中。",
        "- 由于当前渲染模板的 treatment `IC` 因子可能为 0，本任务同时记录 `current_ic_factor` 和 `force_ic1` 两条分支。",
        "",
        "## 汇总结果",
        "",
        md_table(summary),
        "",
        "## SH2O 派生设定",
        "",
        md_table(sh2o_all),
        "",
        "## 失败记录",
        "",
        md_table(failures_df),
        "",
        "## 输出文件",
        "",
        f"- 日值表：`{daily_path.relative_to(ROOT).as_posix()}`",
        f"- SH2O 设定表：`{sh2o_path.relative_to(ROOT).as_posix()}`",
        f"- 汇总表：`{summary_path.relative_to(ROOT).as_posix()}`",
        f"- 失败表：`{failures_path.relative_to(ROOT).as_posix()}`",
        "",
        "## 图件",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{fig.relative_to(ROOT).as_posix()}`")
    lines += [
        "",
        "## 判读边界",
        "",
        "- 如果降低初始水分后 WSPD 升高，只能说明初始水分设定对水分胁迫有影响；不能直接说明主实验必须改初始水分。",
        "- 如果 WSPD 仍不升高，说明低 WSPD 不太可能由初始 SH2O 单独解释，需要继续查土壤蓄水能力、天气过程、DSSAT 水分胁迫变量定义或解析链条。",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")

    result = {
        "task": "033_00_initial_soil_water_sensitivity_wspd_audit",
        "record_md": DOC.relative_to(ROOT).as_posix(),
        "summary_csv": summary_path.relative_to(ROOT).as_posix(),
        "daily_csv": daily_path.relative_to(ROOT).as_posix(),
        "figures": [fig.relative_to(ROOT).as_posix() for fig in figures],
        "failures": failures,
    }
    (OUT / "033_00_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
