from __future__ import annotations

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03"
AUDIT_DIR = BASE_DIR / "018_05_multisite_dqn_vs_extension_audit"
SY_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "sy2014_seed1_minimal_reproduction_018_08"
OUT_DIR = BASE_DIR / "018_09_multisite_status_refresh"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-09_018_09_multisite_status_refresh_record.md"


def classify_row(row: pd.Series) -> str:
    site = str(row["site"])
    year = int(row["year"])
    if site == "SY" and year == 2014:
        return "stable_success_across_seed"
    if site == "LC" and year == 2010:
        return "yield_stable_resource_unstable"
    return str(row["current_evidence_level"])


def interpretation_row(row: pd.Series) -> str:
    site = str(row["site"])
    year = int(row["year"])
    if site == "SY" and year == 2014:
        return "SY2014 已完成 seed0/seed1 复现；两粒种子都达到高产，且 seed1 在保持 300 kg/ha 施氮下比 seed0 少用 30 mm 灌溉，可作为当前最完整的稳定成功案例。"
    if site == "LC" and year == 2010:
        return "LC2010 两个 seed 都能追平 auto/extension 产量，但 seed1 会回到 N300，不满足稳定节氮，因此目前只能算产量稳定、资源效率不稳定。"
    return str(row["interpretation"])


def build_seed_status_table(site_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in site_audit.sort_values(["site", "year"]).iterrows():
        site = str(row["site"])
        year = int(row["year"])
        if site == "SY" and year == 2014:
            rows.append(
                {
                    "site": site,
                    "year": year,
                    "seed_status": "seed0+seed1 verified",
                    "best_seed0": "ckpt15000, GWAD 11216, I120, N300",
                    "best_seed1": "ckpt10000, GWAD 11227, I90, N300",
                    "overall_judgement": "stable_success_across_seed",
                }
            )
        elif site == "LC" and year == 2010:
            rows.append(
                {
                    "site": site,
                    "year": year,
                    "seed_status": "seed0+seed1 verified",
                    "best_seed0": "ckpt5000, GWAD 8739, I90, N0",
                    "best_seed1": "ckpt5000, GWAD 8739, I120, N300",
                    "overall_judgement": "yield_stable_resource_unstable",
                }
            )
        else:
            rows.append(
                {
                    "site": site,
                    "year": year,
                    "seed_status": "only seed0 evidence",
                    "best_seed0": f"GWAD {row['dqn_yield']:.3f}, I{row['dqn_irrigation']:.1f}, N{row['dqn_nitrogen']:.1f}",
                    "best_seed1": "",
                    "overall_judgement": classify_row(row),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    site_audit = pd.read_csv(AUDIT_DIR / "018_05_site_level_audit.csv")
    pairwise = pd.read_csv(AUDIT_DIR / "018_05_pairwise_dqn_advantage.csv")
    sy_seed = pd.read_csv(SY_DIR / "018_08_seed0_vs_seed1_comparison.csv")

    refreshed = site_audit.copy()
    refreshed["current_evidence_level"] = refreshed.apply(classify_row, axis=1)
    refreshed["interpretation"] = refreshed.apply(interpretation_row, axis=1)
    refreshed["next_priority_rank"] = refreshed["site"].map({"SY": 0, "HLA": 1, "YC": 1, "FQ": 1, "LC": 2}).fillna(9)
    refreshed = refreshed.sort_values(["next_priority_rank", "site", "year"]).reset_index(drop=True)
    refreshed.to_csv(OUT_DIR / "018_09_site_level_audit_refreshed.csv", index=False, encoding="utf-8-sig")

    seed_status = build_seed_status_table(refreshed)
    seed_status.to_csv(OUT_DIR / "018_09_seed_status_summary.csv", index=False, encoding="utf-8-sig")

    compact = refreshed[
        [
            "site",
            "station",
            "year",
            "dqn_yield",
            "dqn_irrigation",
            "dqn_nitrogen",
            "yield_diff_vs_dssat_auto",
            "yield_diff_vs_extension",
            "current_evidence_level",
            "interpretation",
        ]
    ].copy()
    compact.to_csv(OUT_DIR / "018_09_multisite_compact_status_table.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 018_09 多站点状态刷新记录",
        "",
        "## 目的",
        "",
        "- 用 018_08 的 SY2014 seed1 新证据刷新 018_05 的多站点状态判断。",
        "- 形成当前可直接汇报的总表：哪些站点年份已经稳定、哪些只是 promising、哪些还需补 seed。",
        "",
        "## SY2014 新增证据",
        "",
        markdown_table(sy_seed),
        "",
        "## 刷新后的站点总表",
        "",
        markdown_table(compact),
        "",
        "## seed 状态总表",
        "",
        markdown_table(seed_status),
        "",
        "## 当前口径",
        "",
        "- stable_success_across_seed：至少 seed0/seed1 都已补证，且结论方向一致。",
        "- yield_stable_resource_unstable：跨 seed 产量稳定，但节水/节氮方向不稳定。",
        "- promising_but_needs_seed：当前只有 seed0 或等价单粒种子证据，暂不能称稳定成功。",
        "",
        "## 输出",
        "",
        f"- `{(OUT_DIR / '018_09_site_level_audit_refreshed.csv').relative_to(PROJECT_ROOT)}`",
        f"- `{(OUT_DIR / '018_09_seed_status_summary.csv').relative_to(PROJECT_ROOT)}`",
        f"- `{(OUT_DIR / '018_09_multisite_compact_status_table.csv').relative_to(PROJECT_ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
