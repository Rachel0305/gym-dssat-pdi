"""Complete the 15 seed galleries after exact PPO ETCP replay."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
from hl_fq_8seed_cross_site.plot_hl_fq_five_scenario_05503_style import (
    BASELINE_FILES, OUT_ROOT, ROOT, SEEDS_BY_SITE, YEARS, num,
    plot_daily_year, plot_management_year, plot_metric_bars, read_csv, scenario_key,
    to_number,
)

BACKUP = ROOT / "backups/hl_fq_05503_20260930/pre_wp_et_finalize"


def backup(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    target = BACKUP / path.relative_to(OUT_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)


def seed0_metrics(site: str) -> dict[int, dict]:
    source = read_csv(BASELINE_FILES[site]["summary"])
    rows = source.loc[source.scenario.eq("rl_candidate")]
    if set(rows.year.astype(int)) != set(YEARS) or len(rows) != 10:
        raise ValueError(f"Incomplete existing seed0 Summary.OUT metrics: {site}")
    return {int(row.year): row.to_dict() for _, row in rows.iterrows()}


def main() -> None:
    gallery = [
        "# HL/FQ 历史天气 PPO：15 个 seed 逐年五情景图集", "",
        "每个 seed 包含 2014–2023 的 10 张五情景逐日过程图、10 张逐年灌溉/N 柱状图，以及跨年产量/WP_ET/PFP_N、管理投入、reward 图。", "",
        "FQ 2018 原始 .WTH 存在已记录的 Tmin 符号异常，五情景原始输入结果产量均为 0；图中保留该年并标注异常。隔离修正的 2018 结果见 docs/088 和 docs/089，未混入这里。", "",
    ]
    manifest = []
    for site, seeds in SEEDS_BY_SITE.items():
        original_seed0 = seed0_metrics(site)
        for seed in seeds:
            base = OUT_ROOT / site / f"best_seed_seed{seed}"
            figures = base / "figures"
            tables = base / "tables"
            prefix = f"{site.lower()}_seed{seed}"
            metrics_path = tables / f"{prefix}_yearly_five_scenario_metrics.csv"
            wide_path = tables / f"{prefix}_yearly_wp_et_pfp_n_table.csv"
            image_path = figures / f"{prefix}_five_scenario_metrics.png"
            readme_path = base / "README.md"
            for target in [metrics_path, wide_path, image_path, readme_path]:
                backup(target)
            metrics = read_csv(metrics_path)
            metrics["year"] = to_number(metrics.year).astype(int)
            ppo = metrics.scenario.eq(scenario_key(seed))
            if int(ppo.sum()) != 10:
                raise ValueError(f"Incomplete PPO metric rows: {site} seed{seed}")
            wp = {}
            for year in YEARS:
                row = metrics.loc[ppo & metrics.year.eq(year)].iloc[0]
                if seed == 0:
                    old = original_seed0[year]
                    if abs(float(old["grain_yield_kg_ha"]) - float(row.grain_yield_kg_ha)) > 1.0:
                        raise ValueError(f"Seed0 historical snapshot yield mismatch: {site} {year}")
                    if abs(float(old["actual_irrigation_mm"]) - float(row.total_irrigation_mm)) > 0.51:
                        raise ValueError(f"Seed0 historical snapshot irrigation mismatch: {site} {year}")
                    if abs(float(old["actual_nitrogen_kg_ha"]) - float(row.total_n_kg_ha)) > 0.51:
                        raise ValueError(f"Seed0 historical snapshot N mismatch: {site} {year}")
                    etcp, wp_et = float(old["etcp_mm"]), float(old["WP_ET_kg_m3"])
                    source = "Existing 051_03/054_03 exact seed0 Summary.OUT replay; endpoint and I/N verified"
                else:
                    item = OUT_ROOT / "wp_et_replay" / site / f"seed{seed}" / f"{year}_wp_et.json"
                    evidence = json.loads(item.read_text(encoding="utf-8"))
                    if abs(float(evidence["saved_yield_kg_ha"]) - float(row.grain_yield_kg_ha)) > 0.01:
                        raise ValueError(f"Replay selected endpoint mismatch: {site} seed{seed} {year}")
                    etcp, wp_et = float(evidence["etcp_mm"]), float(evidence["WP_ET_kg_m3"])
                    source = evidence["metric_source"]
                if not math.isfinite(etcp) or etcp <= 0 or not math.isfinite(wp_et):
                    raise ValueError(f"Invalid ETCP/WP_ET: {site} seed{seed} {year}")
                index = metrics.index[ppo & metrics.year.eq(year)][0]
                metrics.at[index, "etcp_mm"] = etcp
                metrics.at[index, "WP_ET_kg_m3"] = wp_et
                metrics.at[index, "WUE"] = wp_et
                metrics.at[index, "WUE_status"] = "WP_ET from exact ETCP; WUE interpreted as ET-based water productivity"
                metrics.at[index, "source_status"] = source
                wp[year] = wp_et
            metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
            wide = read_csv(wide_path)
            column = f"PPO seed {seed}"
            if len(wide) != 10 or column not in wide.columns:
                raise ValueError(f"Invalid wide table: {wide_path}")
            pfp = metrics.loc[ppo].set_index("year")["PFP_N_kg_grain_per_kg_N"]
            for i, year in enumerate(YEARS):
                val = num(pfp.loc[year])
                wide.at[i, column] = f"{wp[year]:.2f} / {val:.1f}" if math.isfinite(val) else f"{wp[year]:.2f} / —"
            wide.to_csv(wide_path, index=False, encoding="utf-8-sig")
            plot_metric_bars(site, seed, metrics, figures)
            if site == "FQ":
                daily_2018_path = figures / f"{prefix}_2018_five_scenario_daily.png"
                backup(daily_2018_path)
                daily = read_csv(tables / f"{prefix}_five_scenario_daily.csv")
                daily["year"] = to_number(daily.year).astype(int)
                audit = json.loads((tables / f"{prefix}_action_sequence_audit.json").read_text(encoding="utf-8"))
                weather_match = bool(audit["weather_input_match_by_calendar_date"]["2018"]["weather_values_match_by_calendar_date"])
                plot_daily_year(site, seed, 2018, daily, figures, weather_match)
            for year in YEARS:
                destination = figures / f"{prefix}_{year}_five_scenario_management_bars.png"
                if destination.exists():
                    raise FileExistsError(f"Refusing to overwrite existing per-year bar figure: {destination}")
                plot_management_year(site, seed, year, metrics, figures)
            existing = readme_path.read_text(encoding="utf-8")
            existing = existing.replace(
                "- WP_ET / WUE: unavailable for PPO because the selected PPO validation artifacts do not contain an exact Summary.OUT + ETCP replay.",
                "- PPO WP_ET (ET-based WUE) is available for all ten years from verified Summary.OUT ETCP: prior seed0 exact replay or new deterministic checkpoint replay."
            )
            existing += "\n## 逐年管理用量柱状图\n\n" + "\n".join(
                f"- [{year}](figures/{prefix}_{year}_five_scenario_management_bars.png)" for year in YEARS
            ) + "\n"
            readme_path.write_text(existing, encoding="utf-8")
            gallery += [f"## {site} seed {seed}", "", f"[跨年产量 / WP_ET / PFP_N](./{site}/best_seed_seed{seed}/figures/{prefix}_five_scenario_metrics.png) · [跨年灌溉 / N](./{site}/best_seed_seed{seed}/figures/{prefix}_five_scenario_management.png) · [数据表](./{site}/best_seed_seed{seed}/tables/{prefix}_yearly_five_scenario_metrics.csv)", ""]
            for year in YEARS:
                gallery.append(f"- {year}: [五情景逐日过程](./{site}/best_seed_seed{seed}/figures/{prefix}_{year}_five_scenario_daily.png) · [逐年灌溉/N柱状图](./{site}/best_seed_seed{seed}/figures/{prefix}_{year}_five_scenario_management_bars.png)")
            gallery.append("")
            manifest.append({"site": site, "seed": seed, "validation_years": YEARS, "png_count": len(list(figures.glob("*.png"))), "ppo_wp_et_years": len(wp)})
            print(f"DONE {site} seed{seed}: {len(wp)} PPO WP_ET, {len(list(figures.glob('*.png')))} PNG", flush=True)
    (OUT_ROOT / "GALLERY.md").write_text("\n".join(gallery), encoding="utf-8")
    (OUT_ROOT / "figure_completion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme = OUT_ROOT / "README.md"
    backup(readme)
    original = readme.read_text(encoding="utf-8")
    original = original.replace(
        "This is a read-only post-processing of completed 100K checkpoints and existing four-baseline results. No training, DSSAT replay, WGEN weather generation, or configuration edit was performed.",
        "Figures use completed 100K checkpoints and frozen four-baseline results. PPO WP_ET was completed by deterministic, endpoint-verified DSSAT replays of the existing checkpoints; no training, WGEN generation, or configuration edit was performed."
    )
    original = original.replace(
        "The yearly metric tables contain yield / irrigation / N and WP_ET / PFP_N. PPO WP_ET and NUE remain blank/marked unavailable because exact ETCP and plant N uptake results are absent from the selected validation artifacts. PFP_N is not NUE.",
        "The yearly tables contain yield / irrigation / N and WP_ET / PFP_N. PPO WP_ET is populated from endpoint-verified Summary.OUT ETCP replays. Plant N uptake is unavailable, so NUE remains blank; PFP_N is not NUE."
    )
    original = original.replace("— 13 PNG figures", "— 23 PNG figures")
    original += "\n[Open the complete 15-seed yearly figure gallery](GALLERY.md). Each seed has 23 figures (10 daily, 10 per-year irrigation/N bars, 3 across-year).\n"
    readme.write_text(original, encoding="utf-8")
    print(json.dumps({"seeds": len(manifest), "png_total": sum(x["png_count"] for x in manifest), "gallery": str(OUT_ROOT / "GALLERY.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
