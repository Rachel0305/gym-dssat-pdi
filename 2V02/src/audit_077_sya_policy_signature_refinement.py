"""Read-only refinement of the SY cross-year policy-collapse audit.

The historical gate used the set of positive (irrigation, nitrogen) pairs as
the cross-year signature.  That is useful, but it does not establish that the
timing or sequence of actions is identical.  This script keeps the old metric
and adds timing-aware signatures without loading DSSAT or a PPO model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CKPTS = (25_000, 50_000, 75_000, 100_000)
FILE_RE = re.compile(r"SYA_(?P<year>\d+)_seed(?P<seed>\d+)_ckpt(?P<ckpt>\d+)_daily\.csv$")


def _num(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _signature(df: pd.DataFrame) -> dict[str, object]:
    # Use safe actions when present; fall back to requested actions only for
    # compatibility with an older daily export.  Positive events exclude the
    # legal no-op action and therefore capture actual management interventions.
    i_col = "safe_action_amir" if "safe_action_amir" in df else "raw_action_amir"
    n_col = "safe_action_anfer" if "safe_action_anfer" in df else "raw_action_anfer"
    i = pd.to_numeric(df[i_col], errors="coerce").fillna(0.0)
    n = pd.to_numeric(df[n_col], errors="coerce").fillna(0.0)
    dap = pd.to_numeric(df.get("dap", pd.Series(range(len(df)))), errors="coerce")
    positive = (i.abs() > 1e-9) | (n.abs() > 1e-9)
    events = []
    for d, iv, nv in zip(dap[positive], i[positive], n[positive]):
        events.append(f"DAP{float(d):g}:I{float(iv):g}/N{float(nv):g}")
    pair_set = sorted({x.split(":", 1)[1] for x in events})
    event_sig = "|".join(events) if events else "NONE"
    pair_sig = ";".join(pair_set) if pair_set else "NONE"
    entropy = 0.0
    if pair_set:
        counts = pd.Series([x.split(":", 1)[1] for x in events]).value_counts().to_numpy(dtype=float)
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log2(probs)).sum())
    return {
        "pair_set_signature": pair_sig,
        "timed_event_signature": event_sig,
        "event_count": int(len(events)),
        "pair_set_size": int(len(pair_set)),
        "pair_entropy_bits": entropy,
        "total_irrigation": float(i.sum()),
        "total_nitrogen": float(n.sum()),
        "first_nonzero_dap": float(dap[positive].iloc[0]) if positive.any() else float("nan"),
    }


def collect(root: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((root / "daily_outputs" / "SYA").glob("SYA_*_seed*_ckpt*_daily.csv")):
        m = FILE_RE.search(path.name)
        if not m or int(m.group("ckpt")) not in CKPTS:
            continue
        df = pd.read_csv(path, low_memory=False)
        metrics = _signature(df)
        rows.append(
            {
                "run": label,
                "seed": int(m.group("seed")),
                "year": int(m.group("year")),
                "checkpoint_step": int(m.group("ckpt")),
                "daily_file": str(path),
                **metrics,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No SY daily CSVs found under {root}")
    return pd.DataFrame(rows).sort_values(["run", "checkpoint_step", "year"]).reset_index(drop=True)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, object]] = []
    for (run, seed, ckpt), g in detail.groupby(["run", "seed", "checkpoint_step"], sort=True):
        out.append(
            {
                "run": run,
                "seed": int(seed),
                "checkpoint_step": int(ckpt),
                "years": int(g["year"].nunique()),
                "pair_set_distinct": int(g["pair_set_signature"].nunique()),
                "timed_event_signature_distinct": int(g["timed_event_signature"].nunique()),
                "timed_event_signature_collapsed": bool(g["timed_event_signature"].nunique() == 1),
                "event_count_mean": float(g["event_count"].mean()),
                "event_count_min": int(g["event_count"].min()),
                "event_count_max": int(g["event_count"].max()),
                "pair_entropy_mean_bits": float(g["pair_entropy_bits"].mean()),
                "total_irrigation_mean": float(g["total_irrigation"].mean()),
                "total_nitrogen_mean": float(g["total_nitrogen"].mean()),
                "first_nonzero_dap_distinct": int(g["first_nonzero_dap"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(out).sort_values(["run", "checkpoint_step"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed1-root", type=Path, required=True)
    ap.add_argument("--seed3-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    detail = pd.concat(
        [collect(args.seed1_root, "seed1_070"), collect(args.seed3_root, "seed3_076")],
        ignore_index=True,
    )
    summary = summarize(detail)
    detail.to_csv(args.output / "sya_policy_signature_detail.csv", index=False)
    summary.to_csv(args.output / "sya_policy_signature_summary.csv", index=False)
    payload = {
        "purpose": "Separate pair-set signature from timing-aware action sequence signature",
        "inputs": {"seed1_070": str(args.seed1_root), "seed3_076": str(args.seed3_root)},
        "summary_rows": summary.to_dict(orient="records"),
        "interpretation_rule": "pair_set_distinct=1 is not sufficient evidence of complete trajectory collapse; timed_event_signature_distinct=1 is the stronger deterministic-event test.",
    }
    (args.output / "sya_policy_signature_refinement.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
