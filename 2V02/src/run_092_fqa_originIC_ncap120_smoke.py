"""092_00: isolated FQ N-cap-120 2K smoke with a strict cross-year gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CONFIG = ROOT / "configs" / "092_00_fqa_originIC_ncap120_smoke.json"
PROMPT = ROOT / "prompts" / "092_00_fqa_originIC_ncap120_smoke.md"
NCAP = 120.0
BASELINE_CAP = 250.0


def load_051_runner():
    path = ROOT / "src" / "051_fqa_originIC_site_transfer" / "run_051_00_fqa_originIC_expanded_action_maskableppo.py"
    spec = importlib.util.spec_from_file_location("fqa_051_runner_for_092", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_n_cap(runner):
    original = runner.engine.base04036.load_config

    def capped_config():
        cfg = original()
        safety = cfg.get("action_safety", {})
        actual = float(safety.get("season_n_soft_limit", float("nan")))
        if not np.isclose(actual, BASELINE_CAP):
            raise RuntimeError(f"unexpected inherited season_n_soft_limit={actual}; expected {BASELINE_CAP}")
        safety["season_n_soft_limit"] = NCAP
        cfg["action_safety"] = safety
        return cfg

    runner.engine.base04036.load_config = capped_config
    return original


def strict_gate(runner, out: Path, smoke_cfg: dict) -> tuple[pd.DataFrame, dict]:
    audit, base_gate = runner.audit_actions(out, smoke_cfg)
    final = audit[audit["checkpoint_step"].eq(2000)].copy()
    signatures = set(final["nonzero_action_pairs"].fillna("").astype(str))
    all_pairs = set()
    for text in signatures:
        all_pairs.update(piece for piece in text.split("; ") if piece)
    max_n = 0.0
    summary_path = out / "evaluation" / "092_00_checkpoint_validation_summary.csv"
    summary = pd.read_csv(summary_path, keep_default_na=False)
    for daily_rel in summary.loc[summary["checkpoint_step"].eq(2000), "daily_csv_path"].astype(str):
        daily_path = ROOT / daily_rel
        if not daily_path.exists():
            continue
        daily = pd.read_csv(daily_path)
        max_n = max(max_n, float(pd.to_numeric(daily["season_cumulative_n"], errors="coerce").max()))
    gate = {
        "checkpoint": 2000,
        "base_action_gate": base_gate,
        "validation_rows": int(len(final)),
        "expected_validation_rows": int(len(smoke_cfg["scope"]["validation_years"])),
        "max_cumulative_n_kg_ha": max_n,
        "cap_compliant": bool(max_n <= NCAP + 1e-6),
        "nonzero_action_pair_count": int(len(all_pairs)),
        "action_diversity_pass": bool(len(all_pairs) >= 3),
        "positive_action_rows_after_dap1": int(final["positive_action_rows_after_dap1"].sum()) if len(final) else 0,
        "post_dap1_pass": bool(len(final) and final["positive_action_rows_after_dap1"].sum() > 0),
        "cross_year_distinct_signatures": int(len(signatures)),
        "cross_year_signature_pass": bool(len(signatures) > 1),
        "transmission_pass": bool(len(final) and final["transmission_mismatch_rows"].sum() == 0),
    }
    gate["next_step_allowed"] = bool(
        gate["validation_rows"] == gate["expected_validation_rows"]
        and gate["cap_compliant"]
        and gate["action_diversity_pass"]
        and gate["post_dap1_pass"]
        and gate["cross_year_signature_pass"]
        and gate["transmission_pass"]
    )
    return audit, gate


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    runner = load_051_runner()
    runner.PROMPT = PROMPT
    original = patch_n_cap(runner)
    try:
        result = runner.run(CONFIG, dry_run=args.dry_run, smoke=True, formal=False)
    finally:
        runner.engine.base04036.load_config = original
    if args.dry_run:
        result["single_factor_diff"] = json.loads(CONFIG.read_text(encoding="utf-8"))["single_factor_diff"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    out = ROOT / result["output_root"]
    smoke_cfg = runner.prepare_config(json.loads(CONFIG.read_text(encoding="utf-8")), smoke=True)
    audit, gate = strict_gate(runner, out, smoke_cfg)
    audit_path = out / "audits" / "092_00_strict_ncap120_action_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    gate_path = out / "092_00_ncap120_strict_gate.json"
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    doc = ROOT / "docs" / "092_00_fqa_originIC_ncap120_smoke_record.md"
    lines = [
        "# 092_00 FQ N-cap 120 单因素 2K smoke",
        "",
        "- 唯一干预：season_n_soft_limit 250 -> 120 kg/ha；其他配置保持 051/046_10 不变。",
        "- 无 external-N、无天气预报、无长训练；WP_ET 未由 2K evaluator 导出时不推断。",
        "",
        "## 严格闸门",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in gate.items())
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 该 smoke 只检验低氮约束下的动作机制与跨年签名，不是正式 PPO 性能结论。",
        "- 只有严格闸门全部通过，才允许另行批准更长 checkpoint；否则停止该分支。",
        "",
    ])
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({**result, "strict_gate": str(gate_path.relative_to(ROOT)), "next_step_allowed": gate["next_step_allowed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
