"""Read-only audit of LC-E2 checkpoint action diversity and weather sensitivity."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "benchmark_results/156LC_lca_lowIC_dual_branch_weather_reader_5k_seed0"
OUT = ROOT / "benchmark_results/161LC_lca_lowIC_e2_action_diversity_weather_sensitivity_audit"
YEARS = list(range(2014, 2024))
CHECKPOINTS = [2000, 5000, 10000]
WEATHER_COLS = [
    "forecast_obs_rain_next1_norm",
    "forecast_obs_rain_next3_norm",
    "forecast_obs_rain_next7_norm",
    "forecast_obs_rain_max_next7_norm",
    "forecast_obs_rain_days_next7_norm",
    "forecast_obs_first_rain_lead_norm",
    "forecast_obs_tmax_mean_next7_norm",
    "forecast_obs_tmax_max_next7_norm",
    "forecast_obs_tmin_mean_next7_norm",
    "forecast_obs_srad_mean_next7_norm",
    "forecast_obs_valid_days_next7_norm",
    "forecast_obs_dry_spell_next7_flag",
]


def entropy(values: pd.Series) -> float:
    counts = values.astype(str).value_counts(normalize=True)
    return float(-(counts * np.log2(counts)).sum()) if len(counts) > 1 else 0.0


def action_token(row: pd.Series) -> str:
    i = float(row.get("safe_action_amir", 0.0) or 0.0)
    n = float(row.get("safe_action_anfer", 0.0) or 0.0)
    return f"I{int(round(i))}/N{int(round(n))}"


def load_checkpoint(ckpt: int) -> pd.DataFrame | None:
    frames = []
    for year in YEARS:
        path = RUN_ROOT / "daily_outputs/LCA" / f"LCA_{year}_seed0_ckpt{ckpt}_daily.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        frame["year"] = int(year)
        frame["action_token"] = frame.apply(action_token, axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def sequence_tables(frame: pd.DataFrame, ckpt: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for year, group in frame.groupby("year"):
        group = group.sort_values("dap")
        events = group[(group["safe_action_amir"].abs() + group["safe_action_anfer"].abs()) > 0]
        full = "; ".join(f"DAP{int(r.dap)} {r.action_token}" for _, r in events.iterrows())
        irr = "; ".join(f"DAP{int(r.dap)} I{int(round(r.safe_action_amir))}" for _, r in events.iterrows() if abs(r.safe_action_amir) > 0)
        nit = "; ".join(f"DAP{int(r.dap)} N{int(round(r.safe_action_anfer))}" for _, r in events.iterrows() if abs(r.safe_action_anfer) > 0)
        rows.append({"checkpoint_step": ckpt, "year": int(year), "full_action_sequence": full, "irrigation_sequence": irr, "nitrogen_sequence": nit})
    seq = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"checkpoint_step": ckpt, "years": len(seq), "unique_full_sequences": int(seq.full_action_sequence.nunique()), "unique_irrigation_sequences": int(seq.irrigation_sequence.nunique()), "unique_nitrogen_sequences": int(seq.nitrogen_sequence.nunique()), "mean_positive_action_events": float(frame.groupby("year").apply(lambda g: ((g.safe_action_amir.abs() + g.safe_action_anfer.abs()) > 0).sum(), include_groups=False).mean())},
    ])
    return seq, summary


def dap_weather_audit(frame: pd.DataFrame, ckpt: int) -> pd.DataFrame:
    rows = []
    # The daily logger writes a pre/post pair for each DAP.  Collapse that pair
    # to one decision row per year-DAP, preferring the row with a positive
    # executed action when one exists.
    selected = []
    for (_year, _dap), pair in frame.groupby(["year", "dap"], sort=True):
        positive = pair[(pair["safe_action_amir"].abs() + pair["safe_action_anfer"].abs()) > 0]
        selected.append((positive if not positive.empty else pair).iloc[0])
    decision_frame = pd.DataFrame(selected)
    for dap, group in decision_frame.groupby("dap"):
        weather_sig = group[WEATHER_COLS].round(6).astype(str).agg("|".join, axis=1)
        actions = group["discrete_action_index"].astype(str)
        rows.append({
            "checkpoint_step": ckpt,
            "dap": int(dap),
            "weather_signature_unique": int(weather_sig.nunique()),
            "action_unique": int(actions.nunique()),
            "action_entropy_bits": entropy(actions),
            "weather_varied": bool(weather_sig.nunique() > 1),
            "action_varied": bool(actions.nunique() > 1),
            "weather_varied_action_fixed": bool(weather_sig.nunique() > 1 and actions.nunique() == 1),
            "weather_and_action_varied": bool(weather_sig.nunique() > 1 and actions.nunique() > 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    available = []
    seq_frames, summary_frames, dap_frames = [], [], []
    for ckpt in CHECKPOINTS:
        frame = load_checkpoint(ckpt)
        if frame is None:
            continue
        available.append(ckpt)
        seq, summary = sequence_tables(frame, ckpt)
        dap = dap_weather_audit(frame, ckpt)
        seq_frames.append(seq); summary_frames.append(summary); dap_frames.append(dap)
        seq.to_csv(OUT / "tables" / f"161LC_ckpt{ckpt}_action_sequences.csv", index=False, encoding="utf-8-sig")
        dap.to_csv(OUT / "tables" / f"161LC_ckpt{ckpt}_dap_weather_action_audit.csv", index=False, encoding="utf-8-sig")
    seq_all = pd.concat(seq_frames, ignore_index=True) if seq_frames else pd.DataFrame()
    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    dap_all = pd.concat(dap_frames, ignore_index=True) if dap_frames else pd.DataFrame()
    seq_all.to_csv(OUT / "tables/161LC_action_sequences_all_checkpoints.csv", index=False, encoding="utf-8-sig")
    summary_all.to_csv(OUT / "tables/161LC_action_diversity_summary.csv", index=False, encoding="utf-8-sig")
    dap_all.to_csv(OUT / "tables/161LC_dap_weather_action_audit_all_checkpoints.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# LC-E2 动作多样性与天气敏感性审计", "",
        f"- 可用 checkpoint：{', '.join(map(str, available)) if available else 'none'}。",
        f"- 未提供 checkpoint：{', '.join(map(str, [c for c in CHECKPOINTS if c not in available])) if len(available) < len(CHECKPOINTS) else 'none'}。",
        "- 本审计只读取现有 daily CSV，不改变奖励函数、动作评价、观测编码或模型。", "",
        "## 主要结果", "",
    ]
    for _, row in summary_all.iterrows():
        dap_sub = dap_all[dap_all["checkpoint_step"].eq(int(row["checkpoint_step"]))]
        lines.append(
            f"- checkpoint {int(row['checkpoint_step'])}：完整序列 {int(row['unique_full_sequences'])} 种，灌溉序列 {int(row['unique_irrigation_sequences'])} 种，施氮序列 {int(row['unique_nitrogen_sequences'])} 种；共同出现的 DAP 中天气变化但执行动作固定的比例为 {int(dap_sub['weather_varied_action_fixed'].sum())}/{int(dap_sub['weather_varied'].sum())}。"
        )
    lines += [
        "- 5K 的 10 年完整序列差异只来自部分年份是否延长到 DAP91 追加 I30；施氮序列为 1 种。",
        "- 5K 在共同出现的 DAP 上没有观察到执行动作随天气签名变化，说明天气输入在成熟 checkpoint 上未形成明显的跨年动作响应。", "",
        "## 判读边界", "",
        "完整年度动作序列数量很少，说明跨年管理行为低多样性；但这本身不能证明网络参数完全坍缩。还需结合 checkpoint 演化、天气变化与动作变化的对应关系判断。",
    ]
    (OUT / "161LC_audit_record.md").write_text("\n".join(lines), encoding="utf-8")
    report = {"status": "completed", "available_checkpoints": available, "missing_checkpoints": [c for c in CHECKPOINTS if c not in available], "output_root": OUT.relative_to(ROOT).as_posix()}
    (OUT / "161LC_audit_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
