#!/usr/bin/env python3
"""Complete FQ2016 MaskablePPO seeds 1 and 2 without repeating seed0."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_yc_fq_lc_site_specific_stage_maskable_ppo_027_07 as ppo


OUT = ROOT / "benchmark_results" / "028_10_fq2016_missing_seed1_seed2"
DOC = ROOT / "docs" / "2026-07-18_028_10_fq2016_missing_seed1_seed2.md"
OLD = ROOT / "benchmark_results" / "027_07_site_specific_stage_maskable_ppo_attempt2" / "FQ"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    site_root = OUT / "FQ"
    site_root.mkdir(parents=True)
    shutil.copytree(OLD / "readiness", site_root / "readiness")
    payloads = []
    for seed in (1, 2):
        print(f"[028_10] START FQ2016 seed{seed}", flush=True)
        payloads.append(ppo.run_seed(ppo.SPECS["FQ"], site_root, seed))
    new_rows = [ppo.selected_summary(payload) for payload in payloads]
    old_payload = json.loads((OLD / "seed0" / "seed_result.json").read_text(encoding="utf-8"))
    all_rows = [ppo.selected_summary(old_payload), *new_rows]
    summary = pd.DataFrame(all_rows)
    summary.insert(0, "site", "FQ")
    summary.to_csv(OUT / "028_10_fq2016_three_seed_selected_summary.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "completed",
        "new_seeds_run": [1, 2],
        "seed0_reused_not_rerun": True,
        "advisor_any_metric_winner_count": int(summary["advisor_any_metric_strict_winner"].sum()),
        "primary_pass_count": int(summary["primary_pass"].sum()),
        "training_steps_new": 480,
        "selected_results": summary.to_dict("records"),
    }
    (OUT / "028_10_result.json").write_text(ppo.json_text(result, indent=2), encoding="utf-8")
    lines = [
        "# 028_10 FQ2016 缺失 seed1/2 补齐记录", "",
        "状态：`completed`", "",
        "- seed0 复用 027_07，没有重跑。",
        "- seed1/2 各训练 240 stage steps，配置与 027_07 完全一致。",
        f"- 导师至少一项严格领先规则：{result['advisor_any_metric_winner_count']}/3 seed。",
        f"- 旧 primary：{result['primary_pass_count']}/3 seed。",
        "- checkpoint0 不参加训练后选模，也不冒充训练成功。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(ppo.json_text(result, indent=2))


if __name__ == "__main__":
    main()
