from __future__ import annotations

import itertools
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

import run_all_year_direct_action_safe_ppo as direct_ppo
import run_sya_lowIC_teacher_warmstart_maskableppo_041_03 as base04103
import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as base04040


TASK_ID = "042_04"
TASK_NAME = "sya_lowIC_04202_weather_responsiveness_audit"
SRC = ROOT / "benchmark_results" / "042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TABLES = OUT / "tables"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"

SUMMARY = SRC / "evaluation" / "041_03_checkpoint_validation_summary.csv"
DAILY_DIR = SRC / "daily_outputs" / "SYA"
STATION = "SYA"
SITE = "SY"
YEARS = list(range(2014, 2024))
CHECKPOINTS = [0, 25_000, 50_000, 75_000, 100_000]


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty CSV: {path}")
    return pd.read_csv(path, keep_default_na=False)


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


def load_weather_and_planting() -> tuple[pd.DataFrame, dict[int, pd.Timestamp]]:
    config = base04040.load_config()
    _config2, env_config = base04103.build_config_and_env_config(YEARS)
    weather = direct_ppo.weather_for_daily(config)
    weather = weather[weather["station_code"].astype(str).eq(STATION)].copy()
    weather["date"] = pd.to_datetime(weather["date"])
    for col in ["rain", "tmax", "tmin", "srad"]:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather = weather.sort_values("date").reset_index(drop=True)

    planting = {}
    for year in YEARS:
        row = direct_ppo.find_year(env_config, STATION, int(year))
        planting[int(year)] = pd.Timestamp(row["planting_date"])
    return weather, planting


def enrich_weather(daily: pd.DataFrame) -> pd.DataFrame:
    weather, planting = load_weather_and_planting()
    rows = []
    for r in daily.itertuples(index=False):
        year = int(r.year)
        dap = float(r.dap) if pd.notna(r.dap) else np.nan
        # Some 041_03 daily rows record DAP0 in early days. Treat DAP0 as planting day.
        dap_for_date = max(int(round(dap)), 0) if np.isfinite(dap) else int(getattr(r, "step", 0))
        date = planting[year] + pd.Timedelta(days=dap_for_date)
        w_today = weather[weather["date"].eq(date)]
        past = weather[(weather["date"] >= date - pd.Timedelta(days=6)) & (weather["date"] <= date)]
        future = weather[(weather["date"] >= date) & (weather["date"] <= date + pd.Timedelta(days=6))]
        tmean_future = np.nan
        if not future.empty and {"tmax", "tmin"}.issubset(future.columns):
            tmean_future = float(((future["tmax"] + future["tmin"]) / 2).mean())
        item = r._asdict()
        item.update(
            {
                "date_reconstructed": date.date().isoformat(),
                "rain_today_reconstructed": float(w_today["rain"].iloc[0]) if not w_today.empty and "rain" in w_today else np.nan,
                "tmin_today_reconstructed": float(w_today["tmin"].iloc[0]) if not w_today.empty and "tmin" in w_today else np.nan,
                "past7_rain_reconstructed": float(past["rain"].fillna(0).sum()) if not past.empty and "rain" in past else np.nan,
                "future7_rain_reconstructed": float(future["rain"].fillna(0).sum()) if not future.empty and "rain" in future else np.nan,
                "future7_tmean_reconstructed": tmean_future,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def action_signature(df: pd.DataFrame) -> str:
    events = []
    for r in df.itertuples(index=False):
        i = float(r.irrigation_mm_action)
        n = float(r.nitrogen_kg_ha_action)
        if i > 0 or n > 0:
            events.append(f"DAP{int(round(float(r.dap)))}:I{i:.0f}/N{n:.0f}")
    return "; ".join(events)


def pairwise_hamming(group: pd.DataFrame) -> dict[str, float]:
    seqs = []
    for _, sub in group.groupby("year"):
        s = sub.sort_values("step")["action_index"].astype(int).tolist()
        seqs.append(s)
    if len(seqs) < 2:
        return {"pairwise_hamming_mean": np.nan, "pairwise_hamming_max": np.nan}
    vals = []
    for a, b in itertools.combinations(seqs, 2):
        m = min(len(a), len(b))
        vals.append(sum(1 for i in range(m) if a[i] != b[i]) + abs(len(a) - len(b)))
    return {"pairwise_hamming_mean": float(np.mean(vals)), "pairwise_hamming_max": float(np.max(vals))}


def standardized_diff(x1: pd.Series, x0: pd.Series) -> float:
    x1 = pd.to_numeric(x1, errors="coerce").dropna()
    x0 = pd.to_numeric(x0, errors="coerce").dropna()
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    pooled = np.sqrt(((len(x1) - 1) * x1.var(ddof=1) + (len(x0) - 1) * x0.var(ddof=1)) / max(len(x1) + len(x0) - 2, 1))
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((x1.mean() - x0.mean()) / pooled)


def association_rows(daily: pd.DataFrame) -> pd.DataFrame:
    features = [
        "dap",
        "past7_rain_reconstructed",
        "future7_rain_reconstructed",
        "rain_today_reconstructed",
        "tmin_today_reconstructed",
        "future7_tmean_reconstructed",
        "swfac",
        "nstres",
        "grnwt",
        "topwt",
    ]
    rows = []
    for ck, sub in daily.groupby("checkpoint_step"):
        for action_name, flag_col in [("irrigation", "has_irrigation"), ("nitrogen", "has_nitrogen")]:
            pos = sub[sub[flag_col]]
            neg = sub[~sub[flag_col]]
            for feature in features:
                if feature not in sub.columns:
                    continue
                rows.append(
                    {
                        "checkpoint_step": int(ck),
                        "action_type": action_name,
                        "feature": feature,
                        "n_action_days": int(len(pos)),
                        "n_no_action_days": int(len(neg)),
                        "mean_on_action_days": float(pd.to_numeric(pos[feature], errors="coerce").mean()) if len(pos) else np.nan,
                        "mean_on_no_action_days": float(pd.to_numeric(neg[feature], errors="coerce").mean()) if len(neg) else np.nan,
                        "standardized_difference": standardized_diff(pos[feature], neg[feature]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    summary = require_csv(SUMMARY)
    for col in [
        "checkpoint_step",
        "grain_yield_kg_ha",
        "WP_ET_kg_m3",
        "PFP_N_kg_kg",
        "max_swfac",
        "max_nstres",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    daily_frames = []
    for p in sorted(DAILY_DIR.glob("*_daily.csv")):
        df = require_csv(p)
        if "checkpoint_step" not in df.columns:
            continue
        daily_frames.append(df)
    if not daily_frames:
        raise RuntimeError(f"No daily csv found under {DAILY_DIR}")
    daily = pd.concat(daily_frames, ignore_index=True)
    for col in ["checkpoint_step", "year", "step", "dap", "action_index", "irrigation_mm_action", "nitrogen_kg_ha_action", "rain", "swfac", "nstres", "grnwt", "topwt"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")
    daily["has_irrigation"] = daily["irrigation_mm_action"].fillna(0) > 0
    daily["has_nitrogen"] = daily["nitrogen_kg_ha_action"].fillna(0) > 0
    daily = enrich_weather(daily)

    seq_rows = []
    for (ck, year), sub in daily.groupby(["checkpoint_step", "year"]):
        sub = sub.sort_values("step")
        seq_rows.append(
            {
                "checkpoint_step": int(ck),
                "year": int(year),
                "action_signature": action_signature(sub),
                "irrigation_events": int(sub["has_irrigation"].sum()),
                "nitrogen_events": int(sub["has_nitrogen"].sum()),
                "total_irrigation": float(sub["irrigation_mm_action"].fillna(0).sum()),
                "total_nitrogen": float(sub["nitrogen_kg_ha_action"].fillna(0).sum()),
                "first_irrigation_dap": float(sub.loc[sub["has_irrigation"], "dap"].iloc[0]) if sub["has_irrigation"].any() else np.nan,
                "first_nitrogen_dap": float(sub.loc[sub["has_nitrogen"], "dap"].iloc[0]) if sub["has_nitrogen"].any() else np.nan,
            }
        )
    seq = pd.DataFrame(seq_rows).sort_values(["checkpoint_step", "year"])
    seq_path = TABLES / "042_04_action_sequence_by_year_checkpoint.csv"
    seq.to_csv(seq_path, index=False, encoding="utf-8-sig")

    diversity_rows = []
    for ck, sub in daily.groupby("checkpoint_step"):
        sig = seq[seq["checkpoint_step"].eq(int(ck))]
        h = pairwise_hamming(sub)
        diversity_rows.append(
            {
                "checkpoint_step": int(ck),
                "years": int(sig["year"].nunique()),
                "unique_action_signatures": int(sig["action_signature"].nunique()),
                "mean_total_irrigation": float(sig["total_irrigation"].mean()),
                "sd_total_irrigation": float(sig["total_irrigation"].std(ddof=0)),
                "mean_total_nitrogen": float(sig["total_nitrogen"].mean()),
                "sd_total_nitrogen": float(sig["total_nitrogen"].std(ddof=0)),
                "mean_irrigation_events": float(sig["irrigation_events"].mean()),
                "sd_irrigation_events": float(sig["irrigation_events"].std(ddof=0)),
                "mean_nitrogen_events": float(sig["nitrogen_events"].mean()),
                "sd_nitrogen_events": float(sig["nitrogen_events"].std(ddof=0)),
                **h,
            }
        )
    diversity = pd.DataFrame(diversity_rows).sort_values("checkpoint_step")
    diversity_path = TABLES / "042_04_action_diversity_by_checkpoint.csv"
    diversity.to_csv(diversity_path, index=False, encoding="utf-8-sig")

    assoc = association_rows(daily)
    assoc_path = TABLES / "042_04_weather_stress_action_association.csv"
    assoc.to_csv(assoc_path, index=False, encoding="utf-8-sig")

    # A compact diagnostic: among candidate drivers, what has the largest
    # standardized difference between action and no-action days?
    top_assoc = (
        assoc.assign(abs_std_diff=lambda x: x["standardized_difference"].abs())
        .sort_values(["checkpoint_step", "action_type", "abs_std_diff"], ascending=[True, True, False])
        .groupby(["checkpoint_step", "action_type"], as_index=False)
        .head(5)
    )
    top_assoc_path = TABLES / "042_04_top_action_association_features.csv"
    top_assoc.to_csv(top_assoc_path, index=False, encoding="utf-8-sig")

    # Merge diversity with performance for quick read.
    perf_cols = [
        "checkpoint_step",
        "mean_yield",
        "mean_wp_et",
        "mean_pfp_n",
        "any_metric_win_years",
        "all3_win_years",
        "max_swfac",
        "max_nstres",
    ]
    perf = summary.groupby("checkpoint_step", as_index=False).agg(
        mean_yield=("grain_yield_kg_ha", "mean"),
        mean_wp_et=("WP_ET_kg_m3", "mean"),
        mean_pfp_n=("PFP_N_kg_kg", "mean"),
        any_metric_win_years=("any_metric_win_vs_four_max", lambda s: sum(str(x).lower() == "true" for x in s)),
        all3_win_years=("all3_win_vs_four_max", lambda s: sum(str(x).lower() == "true" for x in s)),
        max_swfac=("max_swfac", "max"),
        max_nstres=("max_nstres", "max"),
    )
    performance_diversity = diversity.merge(perf[perf_cols], on="checkpoint_step", how="left")
    perf_div_path = TABLES / "042_04_performance_plus_diversity.csv"
    performance_diversity.to_csv(perf_div_path, index=False, encoding="utf-8-sig")

    row75 = diversity[diversity["checkpoint_step"].eq(75000)].iloc[0].to_dict()
    assoc75 = top_assoc[top_assoc["checkpoint_step"].eq(75000)].copy()
    unique75 = int(row75["unique_action_signatures"])
    hmean75 = float(row75["pairwise_hamming_mean"])
    branch = "A_75k_nonzero_management_is_template_like" if unique75 <= 1 else "B_75k_nonzero_management_has_year_specific_variation"

    lines = [
        "# 042_04 SYA lowIC 042_02 天气/胁迫响应性审计记录",
        "",
        "## 任务性质",
        "",
        "- 只读已有 `042_02_rerun100k` daily/evaluation 结果。",
        "- 不训练、不运行 DSSAT、不修改 checkpoint。",
        "- 目的：判断较好 checkpoint，尤其 75K，是否真的根据天气/胁迫做出年份差异化响应。",
        "",
        "## 数据限制",
        "",
        "- `041_03` 评估 daily CSV 没有直接保存 tmin/future7 weather observation，因此本任务用站点天气表和 planting date 重建 today/past7/future7 天气特征。",
        "- 部分 daily 行 DAP 早期记录为 0；本任务把 DAP0 视为播种日起点用于日期重建。",
        "- 本审计只能说明相关性和模板化程度，不是动作必要性的因果反事实。",
        "",
        "## 动作多样性汇总",
        "",
        md_table(diversity, max_rows=20),
        "",
        "## 性能 + 多样性合并表",
        "",
        md_table(performance_diversity, max_rows=20),
        "",
        "## 75K 最强关联特征",
        "",
        md_table(assoc75[["checkpoint_step", "action_type", "feature", "n_action_days", "mean_on_action_days", "mean_on_no_action_days", "standardized_difference"]], max_rows=20),
        "",
        "## 判定",
        "",
        f"- 分支：`{branch}`",
        f"- 75K unique action signatures：{unique75}",
        f"- 75K pairwise Hamming mean：{hmean75:.4f}",
        "- 说明：pairwise Hamming 包含所有日步和后期 no-op 序列，受不同年份季节长度影响；本任务判定以非零措施 signature 为主。",
        "- 如果分支为 A，说明 75K 的好指标更接近固定管理模板带来的结果，而不是明确的天气/胁迫响应策略。",
        "- 下一步若继续优化，应把“动作多样性/天气响应性”写入 checkpoint selection guardrail，而不是只按 endpoint 指标选点。",
        "",
        "## 输出",
        "",
        f"- action sequence: `{seq_path.relative_to(ROOT)}`",
        f"- diversity: `{diversity_path.relative_to(ROOT)}`",
        f"- association: `{assoc_path.relative_to(ROOT)}`",
        f"- top associations: `{top_assoc_path.relative_to(ROOT)}`",
        f"- performance + diversity: `{perf_div_path.relative_to(ROOT)}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "source": str(SRC.relative_to(ROOT)),
        "checkpoint_count": int(daily["checkpoint_step"].nunique()),
        "year_count": int(daily["year"].nunique()),
        "checkpoint_75k_unique_action_signatures": unique75,
        "checkpoint_75k_pairwise_hamming_mean": hmean75,
        "outputs": {
            "record_md": str(DOC.relative_to(ROOT)),
            "action_sequence": str(seq_path.relative_to(ROOT)),
            "diversity": str(diversity_path.relative_to(ROOT)),
            "association": str(assoc_path.relative_to(ROOT)),
            "top_associations": str(top_assoc_path.relative_to(ROOT)),
            "performance_plus_diversity": str(perf_div_path.relative_to(ROOT)),
        },
    }
    (OUT / "042_04_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
