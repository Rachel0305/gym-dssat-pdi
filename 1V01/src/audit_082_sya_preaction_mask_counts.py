"""Reconstruct the pre-action legal-action count from existing SY daily CSVs.

Read-only audit: no DSSAT/PPO loading and no result overwrite.  The daily
export records the post-step mask count, so this script reconstructs the mask
at the actual action-selection state using previous cumulative resources and
the previous positive-action DAP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID = [(i, n) for i in (0.0, 15.0, 30.0, 45.0) for n in (0.0, 40.0, 80.0, 120.0)]


def legal_count(dap: int, prev_i: float, prev_n: float, last_i_dap: int | None, last_n_dap: int | None, gap: int) -> int:
    count = 0
    for i, n in GRID:
        if i == 0.0 and n == 0.0:
            count += 1
            continue
        if i > 0 and (dap < 1 or dap > 120 or i > 160.0 - prev_i + 1e-9 or (last_i_dap is not None and dap - last_i_dap < gap)):
            continue
        if n > 0 and (dap < 1 or dap > 90 or n > 250.0 - prev_n + 1e-9 or (last_n_dap is not None and dap - last_n_dap < gap)):
            continue
        count += 1
    return count


def audit_run(root: Path, label: str, gap: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "daily_outputs" / "SYA").glob("SYA_*_seed*_ckpt*_daily.csv")):
        name = path.name
        if "ckpt25000" not in name and "ckpt100000" not in name and "ckpt2000" not in name:
            continue
        df = pd.read_csv(path, low_memory=False).sort_values(["dap", "date"]).reset_index(drop=True)
        last_i_dap: int | None = None
        last_n_dap: int | None = None
        for r in df.itertuples(index=False):
            dap = int(round(float(r.dap)))
            prev_i = float(r.previous_cumulative_irrigation)
            prev_n = float(r.previous_cumulative_n)
            pre = legal_count(dap, prev_i, prev_n, last_i_dap, last_n_dap, gap)
            post = int(r.mask_valid_action_count)
            rows.append({
                "run": label,
                "daily_file": str(path),
                "year": int(r.year),
                "checkpoint_step": int(r.checkpoint_step),
                "dap": dap,
                "pre_action_mask_count": pre,
                "exported_post_action_mask_count": post,
                "pre_count_is_one": pre == 1,
                "action_index": int(r.discrete_action_index),
                "safe_i": float(r.safe_action_amir),
                "safe_n": float(r.safe_action_anfer),
                "prev_cumulative_i": prev_i,
                "prev_cumulative_n": prev_n,
                "gap_days": gap,
            })
            if float(r.safe_action_amir) > 0:
                last_i_dap = dap
            if float(r.safe_action_anfer) > 0:
                last_n_dap = dap
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (run, ckpt), g in detail.groupby(["run", "checkpoint_step"], sort=True):
        rows.append({
            "run": run,
            "checkpoint_step": int(ckpt),
            "rows": int(len(g)),
            "pre_action_mask_one_fraction": float(g.pre_count_is_one.mean()),
            "pre_action_mask_mean": float(g.pre_action_mask_count.mean()),
            "pre_action_mask_min": int(g.pre_action_mask_count.min()),
            "pre_action_mask_max": int(g.pre_action_mask_count.max()),
            "post_action_mask_one_fraction": float((g.exported_post_action_mask_count == 1).mean()),
            "pre_vs_post_exact_fraction": float((g.pre_action_mask_count == g.exported_post_action_mask_count).mean()),
            "pre_action_count_distribution": ";".join(f"{int(k)}:{int(v)}" for k, v in g.pre_action_mask_count.value_counts().sort_index().items()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--root", action="append", nargs=3, metavar=("LABEL", "ROOT", "GAP"), required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    details = [audit_run(Path(root), label, int(gap)) for label, root, gap in args.root]
    detail = pd.concat(details, ignore_index=True)
    summary = summarize(detail)
    detail.to_csv(args.output / "sya_preaction_mask_detail.csv", index=False)
    summary.to_csv(args.output / "sya_preaction_mask_summary.csv", index=False)
    payload = {"purpose": "Reconstruct action-selection-time legal-mask counts", "summary": summary.to_dict(orient="records"), "inputs": args.root}
    (args.output / "sya_preaction_mask_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
