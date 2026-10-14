"""Read-only YC weather reporting from the verified 004_17 manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery"
MANIFEST = ARCHIVE / "weather_archive_manifest_verified.csv"
TRACE = ROOT / "results/yc_random_weather_ppo/004_05/evaluation_step_level_all_models.csv"
OUT = ROOT / "results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting"
DOC = ROOT / "docs"
STAGES = [(0, 30, "DAP0_30"), (31, 60, "DAP31_60"), (61, 90, "DAP61_90"), (91, 999, "DAP_gt90")]
COLORS = {"training": "#367A9B", "heldout": "#D18836", "observed": "#B54A58"}
METRICS = ["rainfall_total", "rainfall_days", "wet_day_intensity", "longest_dry_spell", "rx1day", "rx5day", "tmax_mean", "tmax_max", "hot_days_gt30", "hot_days_gt32", "hot_days_gt35", "longest_hot_spell", "tmin_mean", "tmin_min", "srad_mean", "srad_min", "srad_max"]
PRESENT = ["rainfall_total", "longest_dry_spell", "rx1day", "rx5day", "tmax_mean", "tmax_max", "hot_days_gt32", "tmin_mean", "srad_mean"]
LABELS = {"rainfall_total": "Seasonal rainfall (mm)", "longest_dry_spell": "Longest dry spell (days)", "rx1day": "Rx1day (mm)", "rx5day": "Rx5day (mm)", "tmax_mean": "Mean Tmax (C)", "tmax_max": "Max Tmax (C)", "hot_days_gt32": "Days Tmax >32C", "tmin_mean": "Mean Tmin (C)", "srad_mean": "Mean SRAD (MJ/m2/day)"}
CAVEAT = "Descriptive climate-position analysis; formal coverage verdict pending provenance closure / sensitivity review."


def sha(path: Path) -> str:
    with open(long_path(path), "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest().upper()


def long_path(path: Path) -> str:
    value = str(path.absolute())
    return "\\\\?\\" + value if os.name == "nt" else value


def season_window(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.sort_values("DATE", kind="stable").reset_index(drop=True)
    dap = pd.to_numeric(df["DAP"], errors="raise").astype(int)
    zeros = np.flatnonzero(dap.to_numpy() == 0)
    if len(zeros) == 0:
        raise ValueError("No DAP=0 planting anchor")
    start = int(zeros[-1])
    season = df.iloc[start:].copy().reset_index(drop=True)
    season["DAP"] = pd.to_numeric(season["DAP"], errors="raise").astype(int)
    if season["DAP"].iloc[0] != 0 or season["DAP"].duplicated().any() or not season["DAP"].is_monotonic_increasing:
        raise ValueError("DAP alignment not unique/monotonic after final DAP0")
    for col in ("RAIN", "SRAD", "TMAX", "TMIN"):
        season[col] = pd.to_numeric(season[col], errors="raise")
        if not np.isfinite(season[col]).all():
            raise ValueError(f"Non-finite {col}")
    if (season["DAP"].diff().dropna() != 1).any():
        raise ValueError("Nonconsecutive DAP series")
    if (season["RAIN"] < 0).any():
        raise ValueError("Negative rainfall")
    return season, start


def longest(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def metrics(df: pd.DataFrame) -> dict[str, float | int]:
    rain = df["RAIN"].to_numpy(dtype=float)
    tmax = df["TMAX"].to_numpy(dtype=float)
    tmin = df["TMIN"].to_numpy(dtype=float)
    srad = df["SRAD"].to_numpy(dtype=float)
    wet = rain > 0
    return {
        "n_days": len(df), "rainfall_total": float(rain.sum()), "rainfall_days": int(wet.sum()),
        "wet_day_intensity": float(rain[wet].mean()) if wet.any() else np.nan,
        "longest_dry_spell": longest(~wet), "rx1day": float(rain.max()),
        "rx5day": float(pd.Series(rain).rolling(5).sum().max()) if len(rain) >= 5 else np.nan,
        "tmax_mean": float(tmax.mean()), "tmax_max": float(tmax.max()),
        "hot_days_gt30": int((tmax > 30).sum()), "hot_days_gt32": int((tmax > 32).sum()),
        "hot_days_gt35": int((tmax > 35).sum()), "longest_hot_spell": longest(tmax > 32),
        "tmin_mean": float(tmin.mean()), "tmin_min": float(tmin.min()),
        "srad_mean": float(srad.mean()), "srad_min": float(srad.min()), "srad_max": float(srad.max()),
    }


def observed_rows() -> dict[int, pd.DataFrame]:
    cols = ["training_regime", "ppo_seed", "evaluation_weather_type", "evaluation_weather_year", "dap"]
    parts = []
    for chunk in pd.read_csv(TRACE, usecols=cols, chunksize=50000, low_memory=False):
        mask = (chunk["training_regime"] == "RANDOM_WEATHER_WGEN") & (chunk["ppo_seed"] == 5) & (chunk["evaluation_weather_type"] == "observed_weather")
        parts.append(chunk.loc[mask].copy())
    data = pd.concat(parts, ignore_index=True)
    data["evaluation_weather_year"] = data["evaluation_weather_year"].astype(int)
    output = {}
    for year, group in data.groupby("evaluation_weather_year"):
        year = int(year)
        folder = ROOT / f"results/yc_random_weather_ppo/004_05/evaluation/runs/observed_weather/random_weather/ppo_seed_5/rendered_inputs/YCA/{year}"
        wths = list(folder.rglob("*.WTH"))
        filexs = list(folder.rglob("*.jinja2"))
        if len(wths) != 1 or len(filexs) != 1:
            raise ValueError(f"Expected one historical WTH and FileX for {year}")
        with open(long_path(filexs[0]), encoding="utf-8", errors="replace") as stream:
            lines = stream.read().splitlines()
        pdate = None
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("@P PDATE"):
                for candidate in lines[i + 1:i + 5]:
                    fields = candidate.split()
                    if len(fields) >= 2 and fields[0] == "1" and re.fullmatch(r"\d{5}", fields[1]):
                        pdate = datetime(year, 1, 1) + timedelta(days=int(fields[1][-3:]) - 1)
                        break
                break
        if pdate is None:
            raise ValueError(f"No treatment-1 PDATE for {year}")
        max_dap = int(group.dap.max())
        rows = []
        started = False
        with open(long_path(wths[0]), encoding="utf-8", errors="replace") as stream:
            weather_lines = stream.read().splitlines()
        for line in weather_lines:
            if re.match(r"^\s*@\s*DATE\s+SRAD\s+TMAX\s+TMIN\s+RAIN", line, re.I):
                started = True
                continue
            fields = line.split()
            if not started or len(fields) < 5 or not re.fullmatch(r"\d{7}", fields[0]):
                continue
            dt = datetime.strptime(fields[0], "%Y%j")
            dap = (dt - pdate).days
            if 0 <= dap <= max_dap:
                rows.append({"DATE": dt.strftime("%Y-%m-%d"), "DAP": dap,
                             "SRAD": float(fields[1]), "TMAX": float(fields[2]),
                             "TMIN": float(fields[3]), "RAIN": float(fields[4])})
        output[year] = pd.DataFrame(rows)
        if len(output[year]) != max_dap + 1:
            raise ValueError(f"Observed WTH missing dates for {year}: {len(output[year])}/{max_dap+1}")
    if sorted(output) != list(range(2014, 2024)):
        raise ValueError("Observed year set not 2014-2023")
    return output


def describe(values: pd.Series) -> dict[str, float]:
    return {"mean": values.mean(), "median": values.median(), "p5": values.quantile(.05), "p25": values.quantile(.25), "p75": values.quantile(.75), "p95": values.quantile(.95), "min": values.min(), "max": values.max()}


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / "figures" / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DFE4E7", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def distribution_panels(df: pd.DataFrame, specs: list[tuple[str, str]], filename: str) -> None:
    fig, axes = plt.subplots(1, len(specs), figsize=(6.0, 4.2) if len(specs) == 1 else (3.4 * len(specs), 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(418)
    for ax, (metric, title) in zip(axes, specs):
        for x, dataset in enumerate(("training", "heldout", "observed")):
            vals = df.loc[df.dataset == dataset, metric].dropna().to_numpy()
            ax.boxplot(vals, positions=[x], widths=.48, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": COLORS[dataset], "alpha": .24, "edgecolor": COLORS[dataset]},
                       medianprops={"color": COLORS[dataset], "linewidth": 1.7},
                       whiskerprops={"color": COLORS[dataset]}, capprops={"color": COLORS[dataset]})
            ax.scatter(x + rng.uniform(-.17, .17, len(vals)), vals, s=9 if dataset != "observed" else 18,
                       color=COLORS[dataset], alpha=.65, zorder=3)
            if dataset == "observed" and filename == "figure2_seasonal_rainfall.png":
                labeled = sorted(zip(df.loc[df.dataset == dataset, "crop_year"], vals), key=lambda item: item[1])
                gap = max(25.0, (max(vals) - min(vals)) * .055)
                previous = -np.inf
                for year, value in labeled:
                    label_y = max(value, previous + gap)
                    ax.annotate(str(year), xy=(x + .04, value), xytext=(x + .34, label_y),
                                fontsize=8, color="#7D2634", va="center",
                                arrowprops={"arrowstyle": "-", "color": "#B87B84", "linewidth": .6})
                    previous = label_y
                ax.set_xlim(-.5, 2.85)
        ax.set_xticks(range(3), ["Train 80", "Held-out 20", "Observed 10"])
        style(ax, title)
    save_figure(fig, filename)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    manifest = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    if len(manifest) != 100 or set(manifest.weather_seed) != set(range(1001, 1101)) or manifest.weather_seed.duplicated().any():
        raise ValueError("Verified 100-seed manifest mismatch")
    records, stage_records, windows, qc = [], [], {}, []
    for row in manifest.itertuples(index=False):
        path = ARCHIVE / row.canonical_csv_path
        expected = str(row.canonical_series_sha256).upper()
        if sha(path) != expected:
            raise ValueError(f"Archive file hash mismatch for seed {row.weather_seed}")
        data = pd.read_csv(path)
        window, dropped = season_window(data)
        dataset = "training" if row.set == "training" else "heldout"
        seed = int(row.weather_seed)
        status = "EXACT" if row.verification_level == "EXACT_MATCH" else "PROVISIONAL"
        base = {"weather_seed": seed, "dataset": dataset, "verification_status": status,
                "crop_year": int(row.crop_year), "context": str(row.season_start),
                "raw_n_days": len(data), "preseason_rows_excluded": dropped,
                "max_dap": int(window.DAP.max()), "canonical_series_sha256": expected}
        records.append({**base, **metrics(window)})
        windows[(dataset, seed)] = window
        qc.append({"weather_seed": seed, "dataset": dataset, "raw_n_days": len(data),
                   "preseason_rows_excluded": dropped, "season_n_days": len(window), "source_path": str(path.relative_to(ROOT)), "source_file_sha256": sha(path)})
        for lo, hi, label in STAGES:
            sub = window[window.DAP.between(lo, hi)]
            stage_records.append({**base, "stage": label,
                                  **(metrics(sub) if len(sub) else {"n_days": 0, **{metric: np.nan for metric in METRICS}})})
    for year, data in observed_rows().items():
        window, dropped = season_window(data)
        base = {"weather_seed": np.nan, "dataset": "observed", "verification_status": "OFFICIAL_WTH_FILEX",
                "crop_year": year, "context": f"observed:{year}", "raw_n_days": len(data),
                "preseason_rows_excluded": dropped, "max_dap": int(window.DAP.max()), "canonical_series_sha256": ""}
        records.append({**base, **metrics(window)})
        windows[("observed", year)] = window
        wth_path = next((ROOT / f"results/yc_random_weather_ppo/004_05/evaluation/runs/observed_weather/random_weather/ppo_seed_5/rendered_inputs/YCA/{year}").rglob("*.WTH"))
        qc.append({"weather_seed": "", "dataset": "observed", "crop_year": year, "raw_n_days": len(data),
                   "preseason_rows_excluded": dropped, "season_n_days": len(window), "source_path": str(wth_path.relative_to(ROOT)), "source_file_sha256": sha(wth_path)})
        for lo, hi, label in STAGES:
            sub = window[window.DAP.between(lo, hi)]
            stage_records.append({**base, "stage": label,
                                  **(metrics(sub) if len(sub) else {"n_days": 0, **{metric: np.nan for metric in METRICS}})})
    all_df = pd.DataFrame(records).sort_values(["dataset", "crop_year", "weather_seed"])
    stage_df = pd.DataFrame(stage_records).sort_values(["dataset", "crop_year", "weather_seed", "stage"])
    all_df.to_csv(OUT / "weather_realization_summary.csv", index=False)
    stage_df.to_csv(OUT / "weather_stage_summary.csv", index=False)
    pd.DataFrame(qc).to_csv(OUT / "season_alignment_qc.csv", index=False)
    observed = all_df[all_df.dataset == "observed"].copy().sort_values("crop_year")
    observed.to_csv(OUT / "observed_weather_2014_2023.csv", index=False)
    train80 = all_df[all_df.dataset == "training"]
    train78 = train80[train80.verification_status == "EXACT"]
    heldout = all_df[all_df.dataset == "heldout"]
    if (len(train80), len(train78), len(heldout), len(observed)) != (80, 78, 20, 10):
        raise ValueError("Dataset/provenance counts mismatch")
    summary = []
    for metric in PRESENT:
        t, h, o = train80[metric], heldout[metric], observed[metric]
        summary.append({"Metric": metric, "Training median": t.median(), "Training P5": t.quantile(.05), "Training P95": t.quantile(.95),
                        "Training min": t.min(), "Training max": t.max(), "Heldout median": h.median(),
                        "Heldout min": h.min(), "Heldout max": h.max(), "Observed min": o.min(), "Observed max": o.max()})
    pd.DataFrame(summary).to_csv(OUT / "weather_summary_for_presentation.csv", index=False)
    sensitivity, percentiles = [], []
    for scope, frame in (("season", all_df), ("stage", stage_df)):
        for stage in ([None] if scope == "season" else [s[2] for s in STAGES]):
            sub = frame if stage is None else frame[frame.stage == stage]
            t80 = sub[sub.dataset == "training"]
            t78 = t80[t80.verification_status == "EXACT"]
            obs = sub[sub.dataset == "observed"]
            for metric in METRICS:
                a, b = t80[metric].dropna(), t78[metric].dropna()
                da, db = describe(a), describe(b)
                for year, value in zip(obs.crop_year, obs[metric]):
                    if pd.isna(value) or a.empty or b.empty:
                        percentiles.append({"scope": scope, "stage": stage or "ALL", "metric": metric,
                                            "year": int(year), "observed_value": value,
                                            "full80_percentile": np.nan, "exact78_percentile": np.nan,
                                            "percentile_delta_pp": np.nan, "full80_tail_p5_p95": np.nan,
                                            "exact78_tail_p5_p95": np.nan, "full80_outside_support": np.nan,
                                            "exact78_outside_support": np.nan, "material_change": False})
                        continue
                    p80 = 100 * float((a <= value).mean())
                    p78 = 100 * float((b <= value).mean())
                    tail80 = value < da["p5"] or value > da["p95"]
                    tail78 = value < db["p5"] or value > db["p95"]
                    outside80 = value < da["min"] or value > da["max"]
                    outside78 = value < db["min"] or value > db["max"]
                    material = abs(p80 - p78) >= 10 or tail80 != tail78 or outside80 != outside78
                    common = {"scope": scope, "stage": stage or "ALL", "metric": metric, "year": int(year), "observed_value": value,
                              "full80_percentile": p80, "exact78_percentile": p78, "percentile_delta_pp": p80 - p78,
                              "full80_tail_p5_p95": tail80, "exact78_tail_p5_p95": tail78,
                              "full80_outside_support": outside80, "exact78_outside_support": outside78,
                              "material_change": material}
                    percentiles.append(common)
                sensitivity.append({"scope": scope, "stage": stage or "ALL", "metric": metric,
                                    **{f"full80_{k}": v for k, v in da.items()},
                                    **{f"exact78_{k}": v for k, v in db.items()},
                                    **{f"delta_{k}": db[k] - da[k] for k in da}})
    sens_df = pd.DataFrame(sensitivity)
    pct_df = pd.DataFrame(percentiles)
    sens_df.to_csv(OUT / "full80_vs_exact78_sensitivity.csv", index=False)
    pct_df.to_csv(OUT / "observed_training_percentiles.csv", index=False)
    # Nine compact, consistent presentation figures.
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    labels, values, colors = ["Training", "Held-out", "Observed years", "Historical exact", "Provisional"], [80, 20, 10, 98, 2], [COLORS["training"], COLORS["heldout"], COLORS["observed"], "#4F8269", "#9A7040"]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=.62)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xlim(0, 110)
    style(ax, "YC random-weather dataset and provenance")
    save_figure(fig, "figure1_dataset_overview.png")
    distribution_panels(all_df, [("rainfall_total", "Seasonal rainfall")], "figure2_seasonal_rainfall.png")
    distribution_panels(all_df, [("longest_dry_spell", "Dry spell"), ("rx1day", "Rx1day"), ("rx5day", "Rx5day")], "figure3_rainfall_extremes.png")
    distribution_panels(all_df, [("tmax_mean", "Mean Tmax"), ("tmax_max", "Max Tmax"), ("hot_days_gt32", "Hot days >32C")], "figure4_temperature_exposure.png")
    distribution_panels(all_df, [("srad_mean", "Mean SRAD")], "figure5_srad_distribution.png")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    for ax, (_, _, stage) in zip(axes, STAGES):
        part = stage_df[stage_df.stage == stage]
        for i, dataset in enumerate(("training", "heldout", "observed")):
            vals = part.loc[part.dataset == dataset, "rainfall_total"]
            ax.boxplot(vals, positions=[i], widths=.5, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": COLORS[dataset], "alpha": .3, "edgecolor": COLORS[dataset]},
                       medianprops={"color": COLORS[dataset]})
            if dataset == "observed": ax.scatter(np.full(len(vals), i), vals, color=COLORS[dataset], s=14)
        ax.set_xticks(range(3), ["Train", "Held", "Obs"])
        style(ax, f"{stage} rainfall")
    save_figure(fig, "figure6_stage_rainfall.png")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5), constrained_layout=True)
    for j, (_, _, stage) in enumerate(STAGES):
        part = stage_df[stage_df.stage == stage]
        for i, metric in enumerate(("tmax_mean", "hot_days_gt32")):
            ax = axes[i, j]
            for x, dataset in enumerate(("training", "heldout", "observed")):
                vals = part.loc[part.dataset == dataset, metric]
                ax.boxplot(vals, positions=[x], widths=.5, patch_artist=True, showfliers=False,
                           boxprops={"facecolor": COLORS[dataset], "alpha": .3, "edgecolor": COLORS[dataset]},
                           medianprops={"color": COLORS[dataset]})
                if dataset == "observed": ax.scatter(np.full(len(vals), x), vals, color=COLORS[dataset], s=12)
            ax.set_xticks(range(3), ["Train", "Held", "Obs"])
            style(ax, f"{stage} {metric}")
    save_figure(fig, "figure7_stage_temperature.png")
    hm_metrics = ["rainfall_total", "longest_dry_spell", "rx1day", "rx5day", "tmax_mean", "tmax_max", "hot_days_gt32", "tmin_mean", "srad_mean"]
    heat = pct_df[(pct_df.scope == "season") & pct_df.metric.isin(hm_metrics)].pivot(index="metric", columns="year", values="full80_percentile").reindex(hm_metrics)
    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    im = ax.imshow(heat.values, vmin=0, vmax=100, cmap="RdYlBu_r", aspect="auto")
    ax.set_xticks(range(10), heat.columns.astype(str), rotation=45)
    ax.set_yticks(range(len(heat)), [LABELS.get(m, m) for m in heat.index])
    for i in range(len(heat)):
        for j in range(10): ax.text(j, i, f"{heat.iloc[i,j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Empirical percentile in FULL80")
    ax.set_title("Observed 2014-2023 climate position", loc="left", fontsize=11)
    fig.subplots_adjust(left=.20, right=.90, bottom=.26, top=.90)
    fig.text(.5, .035, CAVEAT, ha="center", fontsize=7, color="#555555")
    save_figure(fig, "figure8_observed_percentile_heatmap.png")
    key_years = [2014, 2019, 2017, 2023]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("rainfall_total", "longest_dry_spell", "tmax_mean", "srad_mean")):
        vals = [float(observed.loc[observed.crop_year == y, metric].iloc[0]) for y in key_years]
        ax.bar([str(y) for y in key_years], vals, color=["#B54A58", "#B54A58", "#4F8269", "#4F8269"])
        ax.axhspan(train80[metric].quantile(.05), train80[metric].quantile(.95), color=COLORS["training"], alpha=.13, label="Train P5-P95")
        style(ax, LABELS.get(metric, metric)); ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Seed5 key years: 2014/2019 vs 2017/2023 (weather only)", fontsize=11)
    save_figure(fig, "figure9_seed5_key_year_weather.png")
    material = pct_df[pct_df.material_change]
    verdict = "DESCRIPTIVE_RESULTS_ROBUST_TO_TWO_PROVISIONAL_REALIZATIONS" if material.empty else "SENSITIVITY_CHANGES_IDENTIFIED"
    report = f"""# 004_18A YC 随机天气汇报统计\n\n## 范围与来源\n使用 `004_17/weather_archive_manifest_verified.csv` 所指向的100套 canonical daily CSV；Observed 使用 004_05 seed5 正式 evaluation 的原始 `CNYCxxxx.WTH`，其 FileX treatment-1 PDATE 定义 DAP0，step trace 的最大 DAP 界定季末。004_05 step trace 天气四列全空，未以缺失值代替天气。历史 exact：training 78/80、held-out 20/20；seed1003、1036 暂定。未运行 WGEN、DSSAT 或 PPO。\n\n## 对齐与质量控制\nWGEN runtime DATE 为统一参考日期，不等同于 crop year；crop year/context 取 manifest。每套 WGEN 序列保留最后一条 DAP=0 及其后记录，排除前置 DAP=0；每套裁剪数见 `season_alignment_qc.csv`。该裁剪是描述性可比口径，不更改归档或历史 hash。Observed 的 WTH 按真实日期、FileX PDATE、trace 末日对齐。80/20/10 季已通过 DAP 唯一、连续和气象值有效性检查。各组季节长度中位数分别为 {train80.n_days.median():.0f}/{heldout.n_days.median():.0f}/{observed.n_days.median():.0f} 天；季长差异会影响整季累计量。\n\n## FULL80 vs EXACT78\n`{verdict}`。预设判据：percentile 差至少10个百分点或 P5/P95 tail、min/max support 标记翻转。触发条目 {len(material)}/{len(pct_df)}；详情见 `observed_training_percentiles.csv`，所有分布量见 `full80_vs_exact78_sensitivity.csv`。这只是统计敏感性，不等于对 provisional 两套完成历史验证。\n\n## 描述性结果\n整季汇报表见 `weather_summary_for_presentation.csv`，逐年见 `observed_weather_2014_2023.csv`；阶段数据见 `weather_stage_summary.csv`。Observed 逐年在 FULL80 的位置见 `observed_training_percentiles.csv` 和 Figure 8。Figure 9 对照 seed5 的2014/2019和已有五情景报告中产量相对较好的2017/2023；只描述天气，不推断产量原因。\n\n## 证据边界\n{CAVEAT} 004_17 两条 provisional 历史 hash 尚未闭合；正式 climate coverage verdict 留待004_18B/004_19。\n\n## 图表\n"""
    report += "\n".join(f"- Figure {i}: `results/yc_random_weather_ppo/004_18A_yc_weather_summary_for_reporting/figures/{name}`" for i, name in enumerate(sorted(p.name for p in (OUT / "figures").glob("*.png")), 1)) + "\n"
    if not material.empty:
        report += "\n### 敏感项\n" + material[["scope", "stage", "metric", "year", "percentile_delta_pp", "full80_tail_p5_p95", "exact78_tail_p5_p95", "full80_outside_support", "exact78_outside_support"]].to_markdown(index=False) + "\n"
        report += "\n尤其整季降雨：2014、2016、2019 在 FULL80 中仅靠 provisional 低降雨样本保持在 min/max 内，EXACT78 则落在 min 以下。故不能称两条 provisional 对 tail identification 无影响。\n"
    (DOC / "yc_random_weather_004_18A_weather_summary_for_reporting.md").write_text(report, encoding="utf-8")
    lines = ["# YC 随机天气组会汇报摘要", "", "- 生成：由2004–2013 frozen fitting weather 拟合 CNYC.CLI，经历史 DSSAT/WGEN runtime 按 seed/context 生成；本轮只读取 004_17 已归档的逐日状态天气。", "- 设计：training seeds1001–1080 (80套)，held-out seeds1081–1100 (20套，严格不入训练)；Observed 2014–2023 (10年)。", "- 每日变量：DATE、DAP、RAIN、SRAD、TMAX、TMIN；WGEN DATE 是运行时参考日期，crop year 见 manifest；observed DATE 来自正式 WTH。", f"- Training 整季降雨中位数 {train80.rainfall_total.median():.1f} mm (P5–P95 {train80.rainfall_total.quantile(.05):.1f}–{train80.rainfall_total.quantile(.95):.1f})；held-out {heldout.rainfall_total.median():.1f} mm；observed {observed.rainfall_total.min():.1f}–{observed.rainfall_total.max():.1f} mm。", f"- Training Tmax 均值中位数 {train80.tmax_mean.median():.1f}°C，SRAD 均值中位数 {train80.srad_mean.median():.1f} MJ/m²/day；held-out 分别 {heldout.tmax_mean.median():.1f}°C / {heldout.srad_mean.median():.1f}；observed 范围分别 {observed.tmax_mean.min():.1f}–{observed.tmax_mean.max():.1f}°C / {observed.srad_mean.min():.1f}–{observed.srad_mean.max():.1f}。", f"- FULL80 vs EXACT78：{verdict}；触发预设敏感性判据 {len(material)} 项。2014、2016、2019 整季降雨在 FULL80 范围内、但在 EXACT78 范围外，不能称 tail identification 完全稳健。", "- 2014/2019 与2017/2023的天气并列展示，不解释产量因果。", "- 1003、1036 为 provisional，98/100 历史逐日 hash 精确匹配；100/100 归档文件完整。", "- 仍需004_18B解决两条 provenance，随后004_19做正式 climate coverage audit，才能决定是否扩天气池。", "", "## 组会建议图", "", "Figure 1 设计与核验数；Figure 2 降雨；Figure 3 干旱/极值；Figure 6 生育阶段降雨；Figure 8 observed 百分位；Figure 9 关键年份。", "", CAVEAT, ""]
    (DOC / "yc_random_weather_weather_summary_for_presentation.md").write_text("\n".join(lines), encoding="utf-8")
    decision = {"artifact_status": "COMPLETE", "historical_exact": 98, "provisional_seeds": [1003, 1036],
                "sensitivity_verdict": verdict, "sensitivity_changed_rows": len(material),
                "formal_climate_coverage_verdict": "PENDING", "season_window": "last DAP=0 through final DAP"}
    (OUT / "summary_status.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False))


if __name__ == "__main__":
    main()
