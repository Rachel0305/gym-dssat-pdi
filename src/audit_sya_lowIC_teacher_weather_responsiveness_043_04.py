from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TASK_ID = "043_04"
TASK_NAME = "sya_lowIC_teacher_weather_responsiveness_audit"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TABLES = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / "043_04_sya_lowIC_weather_responsive_training_mechanism.md"

TEACHER_DIR = ROOT / "benchmark_results" / "041_02_sya_lowIC_layered_teacher_imitation_dataset"
TEACHER_SELECTED = TEACHER_DIR / "tables" / "041_02_selected_teacher_trajectories.csv"
TEACHER_ACTION_DAYS = TEACHER_DIR / "tables" / "041_02_imitation_action_days.csv"
TEACHER_DAILY = TEACHER_DIR / "tables" / "041_02_imitation_daily_dataset.csv"

PPO_04302_EVAL = (
    ROOT
    / "benchmark_results"
    / "043_02_sya_lowIC_binary_timing_forecast_normalized_maskableppo"
    / "evaluation"
    / "043_02_checkpoint_validation_summary.csv"
)
ENV_CONFIG = (
    ROOT
    / "benchmark_results"
    / "041_03_sya_lowIC_teacher_warmstart_maskableppo_smoke2k"
    / "configs"
    / "041_03_env_config.json"
)


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty CSV: {path}")
    return pd.read_csv(path, keep_default_na=False)


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_dssat_wth(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    in_table = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            in_table = line.upper().startswith("@  DATE")
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        yrdoy = parts[0]
        if len(yrdoy) < 7 or not yrdoy[:7].isdigit():
            continue
        year = int(yrdoy[:4])
        doy = int(yrdoy[4:7])
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        rows.append(
            {
                "date": date,
                "weather_year": year,
                "doy": doy,
                "srad": float(parts[1]),
                "tmax": float(parts[2]),
                "tmin": float(parts[3]),
                "rain": float(parts[4]),
            }
        )
    if not rows:
        raise ValueError(f"No weather rows parsed from {path}")
    return pd.DataFrame(rows)


def load_weather_and_planting() -> tuple[pd.DataFrame, dict[int, pd.Timestamp], dict[int, Path]]:
    if not ENV_CONFIG.exists():
        raise FileNotFoundError(ENV_CONFIG)
    config = json.loads(ENV_CONFIG.read_text(encoding="utf-8"))
    planting: dict[int, pd.Timestamp] = {}
    weather_files: dict[int, Path] = {}
    frames: list[pd.DataFrame] = []
    for item in config.get("observed_years", {}).get("SYA", []):
        year = int(item["year"])
        planting[year] = pd.Timestamp(item["planting_date"])
        weather_path = ROOT / str(item["weather_file"])
        weather_files[year] = weather_path
        frame = parse_dssat_wth(weather_path)
        frame["year"] = year
        frames.append(frame)
    if not frames:
        raise RuntimeError("No SYA weather files found in env config.")
    return pd.concat(frames, ignore_index=True), planting, weather_files


def weather_features(weather: pd.DataFrame, date: pd.Timestamp) -> dict[str, float]:
    today = weather[weather["date"].eq(date)]
    past = weather[(weather["date"] >= date - pd.Timedelta(days=6)) & (weather["date"] <= date)]
    future = weather[(weather["date"] >= date) & (weather["date"] <= date + pd.Timedelta(days=6))]
    tmean_future7 = np.nan
    if not future.empty:
        tmean_future7 = float(((future["tmax"] + future["tmin"]) / 2.0).mean())
    return {
        "rain_today_mm": float(today["rain"].iloc[0]) if not today.empty else np.nan,
        "tmin_today_c": float(today["tmin"].iloc[0]) if not today.empty else np.nan,
        "tmax_today_c": float(today["tmax"].iloc[0]) if not today.empty else np.nan,
        "srad_today": float(today["srad"].iloc[0]) if not today.empty else np.nan,
        "rain_past7_mm": float(past["rain"].fillna(0).sum()) if not past.empty else np.nan,
        "rain_future7_mm": float(future["rain"].fillna(0).sum()) if not future.empty else np.nan,
        "tmean_future7_c": tmean_future7,
    }


def enrich_events_with_weather(events: pd.DataFrame) -> pd.DataFrame:
    weather, planting, _ = load_weather_and_planting()
    rows: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        year = int(row.year)
        dap = int(round(float(row.dap_before_action)))
        date = planting[year] + pd.Timedelta(days=max(dap, 0))
        item = row._asdict()
        item["date_reconstructed"] = date.date().isoformat()
        item.update(weather_features(weather[weather["year"].eq(year)], date))
        rows.append(item)
    return pd.DataFrame(rows)


def build_teacher_year_summary(selected: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, sub in events.groupby("year"):
        sub = sub.sort_values("dap_before_action")
        irr = sub[pd.to_numeric(sub["safe_irrigation_mm_action"], errors="coerce").fillna(0) > 0]
        nit = sub[pd.to_numeric(sub["safe_nitrogen_kg_ha_action"], errors="coerce").fillna(0) > 0]
        all_events = []
        for r in sub.itertuples(index=False):
            i = float(r.safe_irrigation_mm_action)
            n = float(r.safe_nitrogen_kg_ha_action)
            all_events.append(f"DAP{int(round(float(r.dap_before_action)))} I{i:.0f}/N{n:.0f}")
        selected_row = selected[selected["year"].astype(int).eq(int(year))].iloc[0].to_dict()
        rows.append(
            {
                "year": int(year),
                "teacher_tier": selected_row.get("teacher_tier", ""),
                "candidate_id": selected_row.get("candidate_id", ""),
                "action_signature": "; ".join(all_events),
                "irrigation_signature": "; ".join(
                    f"DAP{int(round(float(r.dap_before_action)))}:{float(r.safe_irrigation_mm_action):.0f}"
                    for r in irr.itertuples(index=False)
                ),
                "nitrogen_signature": "; ".join(
                    f"DAP{int(round(float(r.dap_before_action)))}:{float(r.safe_nitrogen_kg_ha_action):.0f}"
                    for r in nit.itertuples(index=False)
                ),
                "irrigation_events": int(len(irr)),
                "nitrogen_events": int(len(nit)),
                "first_irrigation_dap": float(irr["dap_before_action"].min()) if not irr.empty else np.nan,
                "first_nitrogen_dap": float(nit["dap_before_action"].min()) if not nit.empty else np.nan,
                "total_irrigation": float(selected_row.get("safe_irrigation_mm", np.nan)),
                "total_n": float(selected_row.get("safe_nitrogen_kg_ha", np.nan)),
                "grain_yield_kg_ha": float(selected_row.get("grain_yield_kg_ha", np.nan)),
                "WP_ET_kg_m3": float(selected_row.get("WP_ET_kg_m3", np.nan)),
                "PFP_N_kg_kg": float(selected_row.get("PFP_N_kg_kg", np.nan)),
                "max_swfac": float(selected_row.get("max_swfac", np.nan)),
                "max_nstres": float(selected_row.get("max_nstres", np.nan)),
                "win_count": int(float(selected_row.get("win_count", 0))),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def build_year_weather_context(year_summary: pd.DataFrame) -> pd.DataFrame:
    weather, planting, _ = load_weather_and_planting()
    rows: list[dict[str, Any]] = []
    for row in year_summary.itertuples(index=False):
        year = int(row.year)
        start = planting[year]
        season = weather[(weather["year"].eq(year)) & (weather["date"] >= start) & (weather["date"] <= start + pd.Timedelta(days=140))]
        early = season[(season["date"] >= start) & (season["date"] <= start + pd.Timedelta(days=60))]
        mid = season[(season["date"] >= start + pd.Timedelta(days=61)) & (season["date"] <= start + pd.Timedelta(days=100))]
        late = season[(season["date"] >= start + pd.Timedelta(days=101)) & (season["date"] <= start + pd.Timedelta(days=140))]
        rows.append(
            {
                "year": year,
                "planting_date": start.date().isoformat(),
                "season_rain_0_140_mm": float(season["rain"].sum()),
                "early_rain_0_60_mm": float(early["rain"].sum()),
                "mid_rain_61_100_mm": float(mid["rain"].sum()),
                "late_rain_101_140_mm": float(late["rain"].sum()),
                "mean_tmax_0_140_c": float(season["tmax"].mean()),
                "mean_tmin_0_140_c": float(season["tmin"].mean()),
                "teacher_total_irrigation": float(row.total_irrigation),
                "teacher_total_n": float(row.total_n),
                "teacher_first_irrigation_dap": float(row.first_irrigation_dap) if pd.notna(row.first_irrigation_dap) else np.nan,
                "teacher_first_nitrogen_dap": float(row.first_nitrogen_dap) if pd.notna(row.first_nitrogen_dap) else np.nan,
                "teacher_max_swfac": float(row.max_swfac),
                "teacher_max_nstres": float(row.max_nstres),
            }
        )
    return pd.DataFrame(rows)


def corr(x: pd.Series, y: pd.Series) -> float:
    work = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(work) < 3 or work["x"].nunique() < 2 or work["y"].nunique() < 2:
        return np.nan
    return float(work["x"].corr(work["y"]))


def load_ppo_04302_template() -> pd.DataFrame:
    if not PPO_04302_EVAL.exists():
        return pd.DataFrame()
    ppo = read_csv(PPO_04302_EVAL)
    numeric(ppo, ["year", "checkpoint_step", "final_grnwt", "total_irrigation", "total_n", "max_swfac", "max_nstres"])
    return ppo[ppo["checkpoint_step"].isin([50000, 75000])].copy()


def build_comparison_summary(year_summary: pd.DataFrame, context: pd.DataFrame, ppo: pd.DataFrame) -> pd.DataFrame:
    teacher_unique_actions = int(year_summary["action_signature"].nunique())
    teacher_unique_irrigation = int(year_summary["irrigation_signature"].nunique())
    teacher_unique_nitrogen = int(year_summary["nitrogen_signature"].nunique())
    teacher_irrigation_sd = float(pd.to_numeric(year_summary["total_irrigation"], errors="coerce").std(ddof=0))
    teacher_n_sd = float(pd.to_numeric(year_summary["total_n"], errors="coerce").std(ddof=0))
    teacher_first_i_sd = float(pd.to_numeric(year_summary["first_irrigation_dap"], errors="coerce").std(ddof=0))
    teacher_first_n_sd = float(pd.to_numeric(year_summary["first_nitrogen_dap"], errors="coerce").std(ddof=0))

    ppo_unique_50k = np.nan
    ppo_unique_75k = np.nan
    if not ppo.empty:
        ppo_unique_50k = int(ppo[ppo["checkpoint_step"].eq(50000)]["action_sequence"].nunique())
        ppo_unique_75k = int(ppo[ppo["checkpoint_step"].eq(75000)]["action_sequence"].nunique())

    rows = [
        {
            "metric": "teacher_unique_action_signatures",
            "value": teacher_unique_actions,
            "interpretation": "teacher逐年完整动作序列唯一数；越高越非模板化",
        },
        {
            "metric": "teacher_unique_irrigation_signatures",
            "value": teacher_unique_irrigation,
            "interpretation": "teacher逐年灌溉序列唯一数",
        },
        {
            "metric": "teacher_unique_nitrogen_signatures",
            "value": teacher_unique_nitrogen,
            "interpretation": "teacher逐年施氮序列唯一数",
        },
        {
            "metric": "teacher_total_irrigation_sd",
            "value": teacher_irrigation_sd,
            "interpretation": "teacher逐年总灌溉量标准差",
        },
        {
            "metric": "teacher_total_n_sd",
            "value": teacher_n_sd,
            "interpretation": "teacher逐年总施氮量标准差",
        },
        {
            "metric": "teacher_first_irrigation_dap_sd",
            "value": teacher_first_i_sd,
            "interpretation": "teacher首次灌溉DAP标准差",
        },
        {
            "metric": "teacher_first_nitrogen_dap_sd",
            "value": teacher_first_n_sd,
            "interpretation": "teacher首次施氮DAP标准差",
        },
        {
            "metric": "corr_total_irrigation_vs_max_swfac",
            "value": corr(year_summary["total_irrigation"], year_summary["max_swfac"]),
            "interpretation": "总灌溉与水分胁迫强度相关性；正值说明胁迫重年份用水更多",
        },
        {
            "metric": "corr_total_n_vs_max_nstres",
            "value": corr(year_summary["total_n"], year_summary["max_nstres"]),
            "interpretation": "总施氮与氮胁迫强度相关性",
        },
        {
            "metric": "corr_total_irrigation_vs_season_rain",
            "value": corr(context["teacher_total_irrigation"], context["season_rain_0_140_mm"]),
            "interpretation": "总灌溉与季节降雨相关性；负值通常更符合补水直觉",
        },
        {
            "metric": "ppo_04302_50k_unique_action_signatures",
            "value": ppo_unique_50k,
            "interpretation": "043_02 50K PPO验证年动作序列唯一数",
        },
        {
            "metric": "ppo_04302_75k_unique_action_signatures",
            "value": ppo_unique_75k,
            "interpretation": "043_02 75K PPO验证年动作序列唯一数",
        },
    ]
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def branch_from_summary(summary: pd.DataFrame) -> str:
    vals = {str(r.metric): r.value for r in summary.itertuples(index=False)}
    teacher_unique = float(vals.get("teacher_unique_action_signatures", 0))
    ppo_unique = max(float(vals.get("ppo_04302_50k_unique_action_signatures", 0)), float(vals.get("ppo_04302_75k_unique_action_signatures", 0)))
    irrigation_sd = float(vals.get("teacher_total_irrigation_sd", 0))
    n_sd = float(vals.get("teacher_total_n_sd", 0))
    if teacher_unique >= 3 and teacher_unique > ppo_unique and (irrigation_sd > 0 or n_sd > 0):
        return "A_teacher_has_more_weather_responsive_signal_than_04302_ppo"
    if teacher_unique > ppo_unique:
        return "B_teacher_non_template_but_weather_link_weak"
    return "C_teacher_not_better_than_ppo_template"


def write_record(
    branch: str,
    year_summary: pd.DataFrame,
    context: pd.DataFrame,
    comparison: pd.DataFrame,
    events: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    lines = [
        "# 043_04 SYA lowIC teacher 天气响应性审计记录",
        "",
        "## 一句话结论",
        "",
        f"- 分支：`{branch}`",
        "- 本任务不训练 PPO、不运行 DSSAT，只读取已有 041_02 lowIC teacher 与 043_02 PPO 输出。",
        "- 目的：判断已有 lowIC teacher 是否比 043_02 PPO 模板更适合作为“天气/胁迫敏感性”的训练信号。",
        "",
        "## 核心对比指标",
        "",
        md_table(comparison, 40),
        "",
        "## Teacher 逐年措施摘要",
        "",
        md_table(
            year_summary[
                [
                    "year",
                    "teacher_tier",
                    "candidate_id",
                    "total_irrigation",
                    "total_n",
                    "grain_yield_kg_ha",
                    "WP_ET_kg_m3",
                    "PFP_N_kg_kg",
                    "max_swfac",
                    "max_nstres",
                    "irrigation_signature",
                    "nitrogen_signature",
                ]
            ],
            20,
        ),
        "",
        "## Teacher 年份天气/胁迫背景",
        "",
        md_table(context, 20),
        "",
        "## 非零动作日天气样例",
        "",
        md_table(
            events[
                [
                    "year",
                    "dap_before_action",
                    "safe_irrigation_mm_action",
                    "safe_nitrogen_kg_ha_action",
                    "rain_today_mm",
                    "rain_past7_mm",
                    "rain_future7_mm",
                    "swfac",
                    "nstres",
                    "teacher_tier",
                ]
            ],
            30,
        ),
        "",
        "## 输出文件",
        "",
    ]
    for name, path in paths.items():
        lines.append(f"- {name}: `{path.relative_to(ROOT).as_posix()}`")
    lines.extend(
        [
            "",
            "## 对下一步的含义",
            "",
            "- 如果分支 A：已有 lowIC teacher 至少比 043_02 PPO 模板更非模板化，可进入 warm-start 训练设计，但还要检查 teacher 是否真的与天气/胁迫存在可解释关系。",
            "- 如果分支 B：teacher 比 PPO 非模板化，但天气关联弱；可以作为 warm-start 候选，但需要额外天气响应 reward 或分层采样。",
            "- 如果分支 C：teacher 也不能提供足够天气响应信号，不应继续 teacher warm-start，应改做天气响应 reward/采样机制。",
        ]
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    selected = read_csv(TEACHER_SELECTED)
    action_days = read_csv(TEACHER_ACTION_DAYS)
    numeric(
        selected,
        [
            "year",
            "safe_irrigation_mm",
            "safe_nitrogen_kg_ha",
            "grain_yield_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "max_swfac",
            "max_nstres",
            "win_count",
        ],
    )
    numeric(
        action_days,
        [
            "year",
            "step",
            "dap_before_action",
            "safe_irrigation_mm_action",
            "safe_nitrogen_kg_ha_action",
            "swfac",
            "nstres",
        ],
    )
    events = enrich_events_with_weather(action_days)
    year_summary = build_teacher_year_summary(selected, events)
    context = build_year_weather_context(year_summary)
    ppo = load_ppo_04302_template()
    comparison = build_comparison_summary(year_summary, context, ppo)
    branch = branch_from_summary(comparison)

    paths = {
        "teacher_year_summary": TABLES / "043_04_teacher_year_summary.csv",
        "teacher_year_weather_context": TABLES / "043_04_teacher_year_weather_context.csv",
        "teacher_event_weather": TABLES / "043_04_teacher_event_weather.csv",
        "teacher_vs_04302_ppo_summary": TABLES / "043_04_teacher_vs_04302_ppo_summary.csv",
    }
    year_summary.to_csv(paths["teacher_year_summary"], index=False, encoding="utf-8-sig")
    context.to_csv(paths["teacher_year_weather_context"], index=False, encoding="utf-8-sig")
    events.to_csv(paths["teacher_event_weather"], index=False, encoding="utf-8-sig")
    comparison.to_csv(paths["teacher_vs_04302_ppo_summary"], index=False, encoding="utf-8-sig")
    write_record(branch, year_summary, context, comparison, events, paths)

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "training_run": False,
        "dssat_run": False,
        "teacher_selected": TEACHER_SELECTED.relative_to(ROOT).as_posix(),
        "teacher_action_days": TEACHER_ACTION_DAYS.relative_to(ROOT).as_posix(),
        "ppo_04302_eval": PPO_04302_EVAL.relative_to(ROOT).as_posix(),
        "outputs": {k: v.relative_to(ROOT).as_posix() for k, v in paths.items()},
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "043_04_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
