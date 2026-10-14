from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


TASK_ID = "042_05"
TASK_NAME = "sya_lowIC_04202_checkpoint_weather_response_guardrail"
SRC = ROOT / "benchmark_results" / "042_02_sya_lowIC_normalized_weather_teacher_warmstart_stress_response_rerun100k"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"

SUMMARY = SRC / "evaluation" / "041_03_checkpoint_validation_summary.csv"
DAILY_DIR = SRC / "daily_outputs" / "SYA"
ENV_CONFIG = SRC / "configs" / "041_03_env_config.json"


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
    rows: list[dict[str, float | str]] = []
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
        year = int(yrdoy[:4])
        doy = int(yrdoy[4:])
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        rows.append(
            {
                "date": date,
                "srad": float(parts[1]),
                "tmax": float(parts[2]),
                "tmin": float(parts[3]),
                "rain": float(parts[4]),
            }
        )
    if not rows:
        raise ValueError(f"No weather rows parsed from {path}")
    return pd.DataFrame(rows)


def load_weather_and_planting() -> tuple[pd.DataFrame, dict[int, pd.Timestamp]]:
    config = json.loads(ENV_CONFIG.read_text(encoding="utf-8"))
    planting: dict[int, pd.Timestamp] = {}
    weather_frames: list[pd.DataFrame] = []
    for item in config.get("observed_years", {}).get("SYA", []):
        year = int(item["year"])
        planting[year] = pd.Timestamp(item["planting_date"])
        weather_path = ROOT / str(item["weather_file"])
        frame = parse_dssat_wth(weather_path)
        frame["year"] = year
        weather_frames.append(frame)
    if not weather_frames:
        raise RuntimeError(f"No SYA weather files found in {ENV_CONFIG}")
    return pd.concat(weather_frames, ignore_index=True), planting


def enrich_weather(daily: pd.DataFrame) -> pd.DataFrame:
    weather, planting = load_weather_and_planting()
    rows = []
    for row in daily.itertuples(index=False):
        year = int(row.year)
        dap = float(row.dap) if pd.notna(row.dap) else np.nan
        dap_for_date = max(int(round(dap)), 0) if np.isfinite(dap) else int(getattr(row, "step", 0))
        date = planting[year] + pd.Timedelta(days=dap_for_date)
        w_today = weather[weather["date"].eq(date)]
        past = weather[(weather["date"] >= date - pd.Timedelta(days=6)) & (weather["date"] <= date)]
        future = weather[(weather["date"] >= date) & (weather["date"] <= date + pd.Timedelta(days=6))]
        tmean_future = np.nan
        if not future.empty:
            tmean_future = float(((future["tmax"] + future["tmin"]) / 2).mean())
        item = row._asdict()
        item.update(
            {
                "date_reconstructed": date.date().isoformat(),
                "rain_today_reconstructed": float(w_today["rain"].iloc[0]) if not w_today.empty else np.nan,
                "tmin_today_reconstructed": float(w_today["tmin"].iloc[0]) if not w_today.empty else np.nan,
                "past7_rain_reconstructed": float(past["rain"].fillna(0).sum()) if not past.empty else np.nan,
                "future7_rain_reconstructed": float(future["rain"].fillna(0).sum()) if not future.empty else np.nan,
                "future7_tmean_reconstructed": tmean_future,
            }
        )
        rows.append(item)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
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


def load_daily() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(DAILY_DIR.glob("*_daily.csv")):
        df = read_csv(path)
        if "checkpoint_step" in df.columns:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No daily CSV found under {DAILY_DIR}")
    daily = pd.concat(frames, ignore_index=True)
    numeric(
        daily,
        [
            "checkpoint_step",
            "year",
            "step",
            "dap",
            "action_index",
            "irrigation_mm_action",
            "nitrogen_kg_ha_action",
            "rain",
            "swfac",
            "nstres",
            "grnwt",
            "topwt",
        ],
    )
    daily["has_irrigation"] = daily["irrigation_mm_action"].fillna(0) > 0
    daily["has_nitrogen"] = daily["nitrogen_kg_ha_action"].fillna(0) > 0
    return enrich_weather(daily)


def action_signature(sub: pd.DataFrame) -> str:
    events: list[str] = []
    for row in sub.sort_values("step").itertuples(index=False):
        i = float(row.irrigation_mm_action)
        n = float(row.nitrogen_kg_ha_action)
        if i > 0 or n > 0:
            events.append(f"DAP{int(round(float(row.dap)))}:I{i:.0f}/N{n:.0f}")
    return "; ".join(events)


def build_sequence_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ck, year), sub in daily.groupby(["checkpoint_step", "year"]):
        sub = sub.sort_values("step")
        rows.append(
            {
                "checkpoint_step": int(ck),
                "year": int(year),
                "action_signature": action_signature(sub),
                "irrigation_signature": "; ".join(
                    f"DAP{int(round(float(r.dap)))}:{float(r.irrigation_mm_action):.0f}"
                    for r in sub[sub["has_irrigation"]].itertuples(index=False)
                ),
                "nitrogen_signature": "; ".join(
                    f"DAP{int(round(float(r.dap)))}:{float(r.nitrogen_kg_ha_action):.0f}"
                    for r in sub[sub["has_nitrogen"]].itertuples(index=False)
                ),
                "irrigation_events": int(sub["has_irrigation"].sum()),
                "nitrogen_events": int(sub["has_nitrogen"].sum()),
                "total_irrigation": float(sub["irrigation_mm_action"].fillna(0).sum()),
                "total_nitrogen": float(sub["nitrogen_kg_ha_action"].fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["checkpoint_step", "year"])


def event_position_variability(daily: pd.DataFrame, action_type: str) -> pd.DataFrame:
    flag = "has_irrigation" if action_type == "irrigation" else "has_nitrogen"
    amount = "irrigation_mm_action" if action_type == "irrigation" else "nitrogen_kg_ha_action"
    rows = []
    events = daily[daily[flag]].copy()
    if events.empty:
        return pd.DataFrame()
    events["event_order"] = events.groupby(["checkpoint_step", "year"]).cumcount() + 1
    for (ck, event_order), sub in events.groupby(["checkpoint_step", "event_order"]):
        rows.append(
            {
                "checkpoint_step": int(ck),
                "action_type": action_type,
                "event_order": int(event_order),
                "n_years_with_event": int(sub["year"].nunique()),
                "mean_dap": float(sub["dap"].mean()),
                "sd_dap": float(sub["dap"].std(ddof=0)),
                "min_dap": float(sub["dap"].min()),
                "max_dap": float(sub["dap"].max()),
                "mean_amount": float(sub[amount].mean()),
                "sd_amount": float(sub[amount].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["checkpoint_step", "action_type", "event_order"])


def standardized_diff(pos: pd.Series, neg: pd.Series) -> float:
    pos = pd.to_numeric(pos, errors="coerce").dropna()
    neg = pd.to_numeric(neg, errors="coerce").dropna()
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    pooled = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / max(len(pos) + len(neg) - 2, 1))
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled)


def build_association_table(daily: pd.DataFrame) -> pd.DataFrame:
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
        for action_type, flag in [("irrigation", "has_irrigation"), ("nitrogen", "has_nitrogen")]:
            pos = sub[sub[flag]]
            neg = sub[~sub[flag]]
            for feature in features:
                if feature not in sub.columns:
                    continue
                rows.append(
                    {
                        "checkpoint_step": int(ck),
                        "action_type": action_type,
                        "feature": feature,
                        "n_action_days": int(len(pos)),
                        "n_no_action_days": int(len(neg)),
                        "mean_action_days": float(pd.to_numeric(pos[feature], errors="coerce").mean()) if len(pos) else np.nan,
                        "mean_no_action_days": float(pd.to_numeric(neg[feature], errors="coerce").mean()) if len(neg) else np.nan,
                        "standardized_difference": standardized_diff(pos[feature], neg[feature]),
                        "abs_standardized_difference": abs(standardized_diff(pos[feature], neg[feature])),
                    }
                )
    return pd.DataFrame(rows)


def build_perf_summary() -> pd.DataFrame:
    summary = read_csv(SUMMARY)
    if "total_irrigation" not in summary.columns and "summary_irrigation_total" in summary.columns:
        summary["total_irrigation"] = summary["summary_irrigation_total"]
    if "total_nitrogen" not in summary.columns and "summary_nitrogen_total" in summary.columns:
        summary["total_nitrogen"] = summary["summary_nitrogen_total"]
    numeric(
        summary,
        [
            "checkpoint_step",
            "grain_yield_kg_ha",
            "WP_ET_kg_m3",
            "PFP_N_kg_kg",
            "max_swfac",
            "max_nstres",
            "total_irrigation",
            "total_nitrogen",
        ],
    )
    def count_true(series: pd.Series) -> int:
        return int(sum(str(x).lower() == "true" for x in series))

    return (
        summary.groupby("checkpoint_step", as_index=False)
        .agg(
            years=("year", "nunique"),
            mean_yield=("grain_yield_kg_ha", "mean"),
            mean_wp_et=("WP_ET_kg_m3", "mean"),
            mean_pfp_n=("PFP_N_kg_kg", "mean"),
            mean_irrigation=("total_irrigation", "mean"),
            mean_nitrogen=("total_nitrogen", "mean"),
            any_metric_win_years=("any_metric_win_vs_four_max", count_true),
            all3_win_years=("all3_win_vs_four_max", count_true),
            max_swfac=("max_swfac", "max"),
            max_nstres=("max_nstres", "max"),
        )
        .sort_values("checkpoint_step")
    )


def build_guardrail_summary(seq: pd.DataFrame, event_var: pd.DataFrame, assoc: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ck, sub in seq.groupby("checkpoint_step"):
        ck = int(ck)
        ivar = event_var[(event_var["checkpoint_step"].eq(ck)) & (event_var["action_type"].eq("irrigation"))]
        nvar = event_var[(event_var["checkpoint_step"].eq(ck)) & (event_var["action_type"].eq("nitrogen"))]
        i_assoc = assoc[(assoc["checkpoint_step"].eq(ck)) & (assoc["action_type"].eq("irrigation"))]
        n_assoc = assoc[(assoc["checkpoint_step"].eq(ck)) & (assoc["action_type"].eq("nitrogen"))]

        def feature_abs(frame: pd.DataFrame, feature: str) -> float:
            hit = frame[frame["feature"].eq(feature)]
            if hit.empty:
                return np.nan
            return float(hit["abs_standardized_difference"].iloc[0])

        rows.append(
            {
                "checkpoint_step": ck,
                "unique_nonzero_action_signatures": int(sub["action_signature"].nunique()),
                "unique_irrigation_signatures": int(sub["irrigation_signature"].nunique()),
                "unique_nitrogen_signatures": int(sub["nitrogen_signature"].nunique()),
                "sd_total_irrigation": float(sub["total_irrigation"].std(ddof=0)),
                "sd_total_nitrogen": float(sub["total_nitrogen"].std(ddof=0)),
                "sd_irrigation_events": float(sub["irrigation_events"].std(ddof=0)),
                "sd_nitrogen_events": float(sub["nitrogen_events"].std(ddof=0)),
                "mean_irrigation_event_dap_sd": float(ivar["sd_dap"].mean()) if not ivar.empty else 0.0,
                "mean_nitrogen_event_dap_sd": float(nvar["sd_dap"].mean()) if not nvar.empty else 0.0,
                "irrigation_future7_rain_abs_std_diff": feature_abs(i_assoc, "future7_rain_reconstructed"),
                "irrigation_past7_rain_abs_std_diff": feature_abs(i_assoc, "past7_rain_reconstructed"),
                "irrigation_swfac_abs_std_diff": feature_abs(i_assoc, "swfac"),
                "nitrogen_nstres_abs_std_diff": feature_abs(n_assoc, "nstres"),
                "nitrogen_dap_abs_std_diff": feature_abs(n_assoc, "dap"),
            }
        )
    guard = pd.DataFrame(rows).merge(perf, on="checkpoint_step", how="left")
    guard["endpoint_guardrail_pass"] = (guard["any_metric_win_years"].eq(10)) & (guard["all3_win_years"].ge(5))
    guard["non_template_guardrail_pass"] = guard["unique_nonzero_action_signatures"].ge(3)
    guard["irrigation_response_guardrail_pass"] = (
        guard["mean_irrigation_event_dap_sd"].ge(3)
        | guard["irrigation_future7_rain_abs_std_diff"].ge(0.2)
    )
    guard["nitrogen_response_guardrail_pass"] = (
        guard["mean_nitrogen_event_dap_sd"].ge(3)
        | guard["nitrogen_nstres_abs_std_diff"].ge(0.2)
    )
    guard["overall_response_guardrail_pass"] = (
        guard["endpoint_guardrail_pass"]
        & guard["non_template_guardrail_pass"]
        & guard["irrigation_response_guardrail_pass"]
        & guard["nitrogen_response_guardrail_pass"]
    )
    return guard.sort_values("checkpoint_step")


def plot_guardrail(guard: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ck = guard["checkpoint_step"].astype(int).astype(str)
    axes[0, 0].bar(ck, guard["all3_win_years"], color="#4C78A8")
    axes[0, 0].set_title("All-3 metric wins by checkpoint")
    axes[0, 0].set_ylabel("years / 10")
    axes[0, 0].axhline(5, color="grey", linestyle="--", linewidth=1)

    axes[0, 1].bar(ck, guard["unique_nonzero_action_signatures"], color="#F58518")
    axes[0, 1].set_title("Nonzero action signature diversity")
    axes[0, 1].set_ylabel("unique signatures")
    axes[0, 1].axhline(3, color="grey", linestyle="--", linewidth=1)

    axes[1, 0].bar(ck, guard["mean_irrigation_event_dap_sd"], color="#54A24B")
    axes[1, 0].set_title("Irrigation timing variation")
    axes[1, 0].set_ylabel("mean DAP SD")
    axes[1, 0].axhline(3, color="grey", linestyle="--", linewidth=1)

    axes[1, 1].bar(ck, guard["mean_nitrogen_event_dap_sd"], color="#B279A2")
    axes[1, 1].set_title("Nitrogen timing variation")
    axes[1, 1].set_ylabel("mean DAP SD")
    axes[1, 1].axhline(3, color="grey", linestyle="--", linewidth=1)

    for ax in axes.ravel():
        ax.set_xlabel("checkpoint")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)

    if not PROMPT.exists():
        raise FileNotFoundError(f"Prompt missing: {PROMPT}")

    daily = load_daily()
    seq = build_sequence_table(daily)
    event_var = pd.concat(
        [
            event_position_variability(daily, "irrigation"),
            event_position_variability(daily, "nitrogen"),
        ],
        ignore_index=True,
    )
    assoc = build_association_table(daily)
    perf = build_perf_summary()
    guard = build_guardrail_summary(seq, event_var, assoc, perf)

    seq_path = TABLES / "042_05_action_sequence_by_year_checkpoint.csv"
    event_var_path = TABLES / "042_05_event_position_variability.csv"
    assoc_path = TABLES / "042_05_action_feature_association.csv"
    guard_path = TABLES / "042_05_checkpoint_response_guardrail_summary.csv"
    fig_path = FIGURES / "042_05_checkpoint_metric_response_guardrail.png"

    seq.to_csv(seq_path, index=False, encoding="utf-8-sig")
    event_var.to_csv(event_var_path, index=False, encoding="utf-8-sig")
    assoc.to_csv(assoc_path, index=False, encoding="utf-8-sig")
    guard.to_csv(guard_path, index=False, encoding="utf-8-sig")
    plot_guardrail(guard, fig_path)

    passed = guard[guard["overall_response_guardrail_pass"]]
    row75 = guard[guard["checkpoint_step"].eq(75000)]
    if passed.empty:
        branch = "C_no_checkpoint_passes_endpoint_plus_response_guardrail"
    elif 75000 in passed["checkpoint_step"].tolist():
        branch = "A_75k_passes_endpoint_plus_response_guardrail"
    else:
        branch = "B_other_checkpoint_passes_but_75k_does_not"

    lines = [
        "# 042_05 SYA lowIC 042_02 checkpoint 天气响应性 guardrail 审计记录",
        "",
        "## 任务性质",
        "",
        "- 只读取 042_02 rerun100k 的 daily/evaluation CSV。",
        "- 不训练 PPO，不运行 DSSAT，不修改 checkpoint。",
        "- 目的：把“终点指标好”和“动作是否根据天气/胁迫变化”拆开审计。",
        "",
        "## 预注册 guardrail",
        "",
        "- 终点指标：`any_metric_win_years == 10` 且 `all3_win_years >= 5`。",
        "- 非模板化：`unique_nonzero_action_signatures >= 3`。",
        "- 灌溉响应：`mean_irrigation_event_dap_sd >= 3` 或 `irrigation_future7_rain_abs_std_diff >= 0.2`。",
        "- 施肥响应：`mean_nitrogen_event_dap_sd >= 3` 或 `nitrogen_nstres_abs_std_diff >= 0.2`。",
        "",
        "## checkpoint guardrail 总表",
        "",
        md_table(
            guard[
                [
                    "checkpoint_step",
                    "any_metric_win_years",
                    "all3_win_years",
                    "unique_nonzero_action_signatures",
                    "mean_irrigation_event_dap_sd",
                    "irrigation_future7_rain_abs_std_diff",
                    "mean_nitrogen_event_dap_sd",
                    "nitrogen_nstres_abs_std_diff",
                    "overall_response_guardrail_pass",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## 75K checkpoint 单独摘录",
        "",
        md_table(row75, max_rows=5),
        "",
        "## 判定",
        "",
        f"- 分支：`{branch}`",
        f"- 通过 checkpoint：{passed['checkpoint_step'].astype(int).tolist() if not passed.empty else []}",
        "- 若分支为 C，说明当前 042_02 虽有高指标 checkpoint，但没有 checkpoint 同时满足“高指标 + 非模板化天气响应”审计门槛。",
        "- 下一步应改 checkpoint selection guardrail 或训练目标，使策略不仅终点指标好，而且不同年份的管理动作能随天气/胁迫输入变化。",
        "",
        "## 输出文件",
        "",
        f"- guardrail summary: `{guard_path.relative_to(ROOT)}`",
        f"- action sequence: `{seq_path.relative_to(ROOT)}`",
        f"- event variability: `{event_var_path.relative_to(ROOT)}`",
        f"- feature association: `{assoc_path.relative_to(ROOT)}`",
        f"- figure: `{fig_path.relative_to(ROOT)}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "branch": branch,
        "source": str(SRC.relative_to(ROOT)),
        "checkpoints": guard["checkpoint_step"].astype(int).tolist(),
        "passed_checkpoints": passed["checkpoint_step"].astype(int).tolist() if not passed.empty else [],
        "checkpoint_75k": row75.to_dict(orient="records")[0] if not row75.empty else None,
        "outputs": {
            "record_md": str(DOC.relative_to(ROOT)),
            "guardrail_summary": str(guard_path.relative_to(ROOT)),
            "action_sequence": str(seq_path.relative_to(ROOT)),
            "event_position_variability": str(event_var_path.relative_to(ROOT)),
            "action_feature_association": str(assoc_path.relative_to(ROOT)),
            "figure": str(fig_path.relative_to(ROOT)),
        },
    }
    (OUT / "042_05_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
