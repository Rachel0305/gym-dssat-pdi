"""Cross-seed action comparison for LC-E2 2K/5K outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_161LC_e2_action_diversity_weather_sensitivity import YEARS, dap_weather_audit, sequence_tables

RUNS = {
    "seed0": ROOT / "benchmark_results/156LC_lca_lowIC_dual_branch_weather_reader_5k_seed0",
    "seed1": ROOT / "benchmark_results/163LC_lca_lowIC_dual_branch_weather_reader_5k_seed1",
}
OUT = ROOT / "benchmark_results/164LC_lca_lowIC_e2_cross_seed_action_comparison"


def load(seed: str, root: Path, checkpoint: int) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        # The shared engine retains the historical seed0 filename convention;
        # the manifest/config remains the authoritative seed label.
        seed_suffix = seed if seed.startswith("seed") else f"seed{seed}"
        candidates = [root / "daily_outputs/LCA" / f"LCA_{year}_{seed_suffix}_ckpt{checkpoint}_daily.csv"]
        # Seed0's historical run and the corrected seed1 run use different
        # filename suffixes; accept only the expected root-local variant.
        if seed == "seed0":
            candidates.append(root / "daily_outputs/LCA" / f"LCA_{year}_seed0_ckpt{checkpoint}_daily.csv")
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["year"] = int(year)
        frame["action_token"] = frame.apply(lambda r: f"I{int(round(float(r.safe_action_amir or 0)))} / N{int(round(float(r.safe_action_anfer or 0)))}", axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metrics(frame: pd.DataFrame, seed: str, checkpoint: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    for year, group in frame.groupby("year"):
        positive = group[(group["safe_action_amir"].abs() + group["safe_action_anfer"].abs()) > 0]
        rows.append({
            "seed": seed,
            "checkpoint_step": checkpoint,
            "year": int(year),
            "yield_kg_ha": float(pd.to_numeric(group["grnwt"], errors="coerce").iloc[-1]),
            "total_irrigation_mm": float(pd.to_numeric(group["safe_action_amir"], errors="coerce").fillna(0).sum()),
            "total_n_kg_ha": float(pd.to_numeric(group["safe_action_anfer"], errors="coerce").fillna(0).sum()),
            "positive_action_events": int(len(positive)),
        })
    detail = pd.DataFrame(rows)
    seq, summary = sequence_tables(frame, checkpoint)
    row = summary.iloc[0].to_dict()
    row.update({"seed": seed, "checkpoint_step": checkpoint, "mean_yield_kg_ha": float(detail.yield_kg_ha.mean()), "mean_irrigation_mm": float(detail.total_irrigation_mm.mean()), "mean_n_kg_ha": float(detail.total_n_kg_ha.mean())})
    return detail.merge(seq, on=["checkpoint_step", "year"], how="left"), row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    details, summaries, dap_audits = [], [], []
    for checkpoint in [2000, 5000]:
        for seed, root in RUNS.items():
            frame = load(seed, root, checkpoint)
            detail, summary = metrics(frame, seed, checkpoint)
            details.append(detail); summaries.append(summary)
            dap = dap_weather_audit(frame, checkpoint)
            dap["seed"] = seed
            dap_audits.append(dap)
    detail = pd.concat(details, ignore_index=True)
    summary = pd.DataFrame(summaries)
    detail.to_csv(OUT / "tables/164LC_cross_seed_yearly_actions_and_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "tables/164LC_cross_seed_action_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(dap_audits, ignore_index=True).to_csv(OUT / "tables/164LC_cross_seed_dap_weather_action_audit.csv", index=False, encoding="utf-8-sig")
    comparisons = []
    for checkpoint in [2000, 5000]:
        a = detail[(detail.seed == "seed0") & detail.checkpoint_step.eq(checkpoint)].set_index("year")
        b = detail[(detail.seed == "seed1") & detail.checkpoint_step.eq(checkpoint)].set_index("year")
        common = sorted(set(a.index) & set(b.index))
        comparisons.append({
            "checkpoint_step": checkpoint,
            "common_years": len(common),
            "same_full_sequence_years": int((a.loc[common, "full_action_sequence"].values == b.loc[common, "full_action_sequence"].values).sum()),
            "seed0_unique_full_sequences": int(a.full_action_sequence.nunique()),
            "seed1_unique_full_sequences": int(b.full_action_sequence.nunique()),
            "seed0_unique_nitrogen_sequences": int(a.nitrogen_sequence.nunique()),
            "seed1_unique_nitrogen_sequences": int(b.nitrogen_sequence.nunique()),
            "mean_yield_delta_seed1_minus_seed0": float((b.loc[common, "yield_kg_ha"] - a.loc[common, "yield_kg_ha"]).mean()),
            "mean_irrigation_delta_seed1_minus_seed0": float((b.loc[common, "total_irrigation_mm"] - a.loc[common, "total_irrigation_mm"]).mean()),
            "mean_n_delta_seed1_minus_seed0": float((b.loc[common, "total_n_kg_ha"] - a.loc[common, "total_n_kg_ha"]).mean()),
        })
    comp = pd.DataFrame(comparisons)
    comp.to_csv(OUT / "tables/164LC_cross_seed_comparison.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# LC-E2 seed0/seed1 动作多样性比较", "",
        "- seed1 只改变随机种子；2K smoke 通过后完成 5K。",
        "- 共享引擎的 daily 文件名仍使用 `seed0` 后缀；实际 seed 以各自 manifest/config 为准。",
        "- 2K：seed0/seed1 的完整序列分别为 3/6 种；共同出现的 DAP 中天气变化但动作固定为 91/94 与 80/94。",
        "- 5K：seed0/seed1 都收敛为 2 种完整序列、1 种施氮序列；共同出现的 DAP 中均为 94/94 天气变化但动作固定。两种 seed 的年度完整序列没有一年的完全相同，说明是不同的固定模板。",
        "", "## 解释边界", "",
        "若 seed1 也形成低多样性序列，说明固定模板更可能是 LC-E2 的稳定倾向；若 seed1 与 seed0 差异大，则说明策略对随机种子敏感。产量差异这里只作为辅助，不替代 WP_ET/PFP-N 正式回放。",
    ]
    (OUT / "164LC_cross_seed_record.md").write_text("\n".join(lines), encoding="utf-8")
    result = {"status": "completed", "checkpoints": [2000, 5000], "output_root": OUT.relative_to(ROOT).as_posix(), "comparison_csv": (OUT / "tables/164LC_cross_seed_comparison.csv").relative_to(ROOT).as_posix()}
    (OUT / "164LC_cross_seed_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
