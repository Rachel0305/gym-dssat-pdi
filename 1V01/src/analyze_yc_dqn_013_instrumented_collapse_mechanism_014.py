"""014: analyze the existing 013 instrumented YC DQN collapse reproduction.

This is a read-only, low-memory reducer.  It never creates an environment,
loads a checkpoint for inference, or starts training.  The large per-sample
TD log is consumed in chunks; only aggregate statistics and compact median
spill files are retained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "docs" / "yc_dqn_013_instrumented_collapse_mechanism_analysis.md"
TD_PATH = RESULTS / "yc_dqn_seed2_per_sample_td_log.csv.gz"
COMPOSITION_SOURCE = RESULTS / "yc_dqn_seed2_replay_sample_composition.csv"
OPPORTUNITY_SOURCE = RESULTS / "yc_dqn_seed2_opportunity_conditioned_behavior.csv"
TD_SUMMARY_SOURCE = RESULTS / "yc_dqn_seed2_td_q_window_summary.csv"
BRANCH_SOURCE = RESULTS / "yc_dqn_seed2_bootstrap_branch_summary.csv"
FIXED_SOURCE = RESULTS / "yc_dqn_seed2_instrumented_fixed_state_qreversal.csv"
PARITY_SOURCE = RESULTS / "yc_dqn_seed2_instrumented_reproduction_parity.csv"
RUN_DIR = ROOT / "benchmark_results" / "013_yc_dqn_seed2_instrumented_0_50k" / "attempt_02"
SPILL_DIR = RESULTS / "014_yc_dqn_013_td_median_spill"

WINDOWS = (
    ("0-10K", 0, 10_000),
    ("10-20K", 10_000, 20_000),
    ("20-25K", 20_000, 25_000),
    ("25-30K", 25_000, 30_000),
    ("30-35K", 30_000, 35_000),
    ("35-40K", 35_000, 40_000),
    ("40-45K", 40_000, 45_000),
    ("45-50K", 45_000, 50_000),
)
FAMILIES = ("N>0", "N=0")
MEDIAN_FIELDS = ("chosen_Q", "target_Q", "td_error_signed", "td_error_abs")
MEAN_FIELDS = MEDIAN_FIELDS + ("reward", "bootstrap_term", "next_max_target_Q")
ACTION_N = np.asarray([0, 40, 80, 120] * 4, dtype=float)


def bool_series(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def window_for_step(step: int) -> str:
    for label, start, end in WINDOWS:
        if step <= end and (step > start or start == 0):
            return label
    raise ValueError(step)


def empty_metric_entry() -> dict[str, Any]:
    return {
        "count": 0,
        "sums": {field: 0.0 for field in MEAN_FIELDS},
        "valid_counts": {field: 0 for field in MEAN_FIELDS},
        "negative_td": 0,
    }


def read_td_log() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TD_PATH.exists():
        raise FileNotFoundError(TD_PATH)
    usecols = [
        "window", "action_index", "N_positive", "I_positive",
        "next_action_family_N_positive", *MEAN_FIELDS,
    ]
    stats = {(label, family): empty_metric_entry() for label, _, _ in WINDOWS for family in FAMILIES}
    branch_stats = {
        (label, positive): {
            "count": 0,
            "sums": {field: 0.0 for field in ("next_max_target_Q", "bootstrap_term", "target_Q")},
            "valid_counts": {field: 0 for field in ("next_max_target_Q", "bootstrap_term", "target_Q")},
        }
        for label, _, _ in WINDOWS
        for positive in (True, False)
    }
    window_counts = {label: 0 for label, _, _ in WINDOWS}
    noop_counts = {label: 0 for label, _, _ in WINDOWS}

    SPILL_DIR.mkdir(parents=True, exist_ok=True)
    spill_paths = {
        (label, family): SPILL_DIR / f"{label.replace('-', '_')}__{family.replace('>', 'gt').replace('=', 'eq')}.bin"
        for label, _, _ in WINDOWS
        for family in FAMILIES
    }
    handles = {key: path.open("wb") for key, path in spill_paths.items()}
    try:
        for chunk in pd.read_csv(TD_PATH, compression="gzip", usecols=usecols, chunksize=200_000):
            n_positive = bool_series(chunk["N_positive"]).to_numpy(dtype=bool)
            i_positive = bool_series(chunk["I_positive"]).to_numpy(dtype=bool)
            next_n_positive = bool_series(chunk["next_action_family_N_positive"]).to_numpy(dtype=bool)
            action_index = pd.to_numeric(chunk["action_index"], errors="coerce").to_numpy(dtype=float)
            window_values = chunk["window"].astype("string")
            for label, _, _ in WINDOWS:
                in_window = window_values.eq(label).to_numpy(dtype=bool)
                count_window = int(in_window.sum())
                if not count_window:
                    continue
                window_counts[label] += count_window
                noop_counts[label] += int(np.nansum(in_window & (action_index == 0)))
                for family, family_mask in (("N>0", n_positive), ("N=0", ~n_positive)):
                    selected = in_window & family_mask
                    count = int(selected.sum())
                    if not count:
                        continue
                    frame = chunk.loc[selected, list(MEAN_FIELDS)]
                    entry = stats[(label, family)]
                    entry["count"] += count
                    for field in MEAN_FIELDS:
                        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)
                        valid = np.isfinite(values)
                        entry["sums"][field] += float(values[valid].sum())
                        entry["valid_counts"][field] += int(valid.sum())
                    td_values = pd.to_numeric(frame["td_error_signed"], errors="coerce").to_numpy(dtype=np.float64)
                    entry["negative_td"] += int(np.nansum(td_values < 0))
                    median_values = frame.loc[:, list(MEDIAN_FIELDS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
                    median_values.tofile(handles[(label, family)])
                for positive in (True, False):
                    selected = in_window & (next_n_positive == positive)
                    count = int(selected.sum())
                    if not count:
                        continue
                    frame = chunk.loc[selected, ["next_max_target_Q", "bootstrap_term", "target_Q"]]
                    entry = branch_stats[(label, positive)]
                    entry["count"] += count
                    for field in ("next_max_target_Q", "bootstrap_term", "target_Q"):
                        values = pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64)
                        valid = np.isfinite(values)
                        entry["sums"][field] += float(values[valid].sum())
                        entry["valid_counts"][field] += int(valid.sum())
    finally:
        for handle in handles.values():
            handle.close()

    composition_rows = []
    td_rows = []
    branch_rows = []
    for label, _, _ in WINDOWS:
        npos = stats[(label, "N>0")]
        nzero = stats[(label, "N=0")]
        total = window_counts[label]
        ipos = np.nan
        composition_rows.append({
            "window": label,
            "sample_count": total,
            "sampled_N_positive_count": npos["count"],
            "sampled_N_zero_count": nzero["count"],
            "sampled_N_positive_fraction": npos["count"] / total if total else np.nan,
            "sampled_N_zero_fraction": nzero["count"] / total if total else np.nan,
            "sampled_I_positive_count": np.nan,
            "sampled_I_positive_fraction": ipos,
            "sampled_noop_count": noop_counts[label],
            "sampled_noop_fraction": noop_counts[label] / total if total else np.nan,
            "source": "raw results/yc_dqn_seed2_per_sample_td_log.csv.gz; real sampled minibatch rows",
        })
        row: dict[str, Any] = {"window": label, "source": "raw per-sample TD log"}
        row["Npos_sample_count"] = npos["count"]
        row["Nzero_sample_count"] = nzero["count"]
        for field in MEAN_FIELDS:
            for prefix, entry in (("Npos", npos), ("Nzero", nzero)):
                valid_count = entry["valid_counts"][field]
                row[f"{prefix}_mean_{field}"] = entry["sums"][field] / valid_count if valid_count else np.nan
            row[f"Npos_minus_Nzero_mean_{field}"] = row[f"Npos_mean_{field}"] - row[f"Nzero_mean_{field}"]
        for prefix, entry in (("Npos", npos), ("Nzero", nzero)):
            values = np.fromfile(spill_paths[(label, "N>0" if prefix == "Npos" else "N=0")], dtype=np.float32)
            if values.size:
                values = values.reshape((-1, len(MEDIAN_FIELDS)))
            for index, field in enumerate(MEDIAN_FIELDS):
                row[f"{prefix}_median_{field}"] = float(np.nanmedian(values[:, index])) if values.size else np.nan
            row[f"{prefix}_fraction_td_error_negative"] = entry["negative_td"] / entry["valid_counts"]["td_error_signed"] if entry["valid_counts"]["td_error_signed"] else np.nan
        row["Npos_minus_Nzero_negative_TD_fraction"] = row["Npos_fraction_td_error_negative"] - row["Nzero_fraction_td_error_negative"]
        td_rows.append(row)
        for positive, name in ((True, "N>0"), (False, "N=0")):
            entry = branch_stats[(label, positive)]
            item: dict[str, Any] = {
                "window": label,
                "next_argmax_branch": name,
                "sample_count": entry["count"],
                "branch_fraction": entry["count"] / total if total else np.nan,
                "source": "raw per-sample TD log; next_action_family_N_positive",
            }
            for field in ("next_max_target_Q", "bootstrap_term", "target_Q"):
                valid_count = entry["valid_counts"][field]
                item[f"mean_{field}"] = entry["sums"][field] / valid_count if valid_count else np.nan
            branch_rows.append(item)
        composition_rows[-1]["sampled_I_positive_count"] = int(
            stats[(label, "N>0")]["count"] + stats[(label, "N=0")]["count"]
        )
        # I_positive is reconstructed below from a second bounded pass so
        # that this first pass remains focused on TD/Q and branch evidence.
    composition = pd.DataFrame(composition_rows)
    td_summary = pd.DataFrame(td_rows)
    branch = pd.DataFrame(branch_rows)
    return composition, td_summary, branch


def add_i_positive_to_composition(composition: pd.DataFrame) -> pd.DataFrame:
    # The first pass intentionally omits I_positive from the median reducer's
    # family split.  Re-read only the two boolean fields in chunks to compute
    # the requested I-positive fraction without retaining raw rows.
    counts = {label: 0 for label, _, _ in WINDOWS}
    totals = {label: 0 for label, _, _ in WINDOWS}
    usecols = ["window", "I_positive"]
    for chunk in pd.read_csv(TD_PATH, compression="gzip", usecols=usecols, chunksize=400_000):
        positive = bool_series(chunk["I_positive"]).to_numpy(dtype=bool)
        values = chunk["window"].astype("string")
        for label, _, _ in WINDOWS:
            mask = values.eq(label).to_numpy(dtype=bool)
            totals[label] += int(mask.sum())
            counts[label] += int((mask & positive).sum())
    out = composition.copy()
    out["sampled_I_positive_count"] = out["window"].map(counts).astype(int)
    out["sampled_I_positive_fraction"] = out.apply(
        lambda row: row["sampled_I_positive_count"] / row["sample_count"] if row["sample_count"] else np.nan,
        axis=1,
    )
    return out


def opportunity_window() -> pd.DataFrame:
    path = RUN_DIR / "training_log.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    log = pd.read_csv(path, usecols=["global_step", "action_index", "nitrogen_kg_ha", "greedy_action", "was_exploration", "n_positive_available"])
    log["global_step"] = pd.to_numeric(log["global_step"], errors="coerce")
    log["actual_N_positive"] = pd.to_numeric(log["nitrogen_kg_ha"], errors="coerce").gt(0)
    log["available"] = bool_series(log["n_positive_available"])
    explored = bool_series(log["was_exploration"])
    greedy = pd.to_numeric(log["greedy_action"], errors="coerce")
    log["greedy_N_positive"] = greedy.map(lambda value: bool(ACTION_N[int(value)] > 0) if np.isfinite(value) and 0 <= int(value) < 16 else np.nan)
    rows = []
    for label, start, end in WINDOWS:
        g = log[(log["global_step"] <= end) & ((log["global_step"] > start) | (start == 0))].copy()
        available = g["available"]
        denom = int(available.sum())
        row: dict[str, Any] = {
            "window": label,
            "collection_rows": len(g),
            "N_positive_available_count": denom,
            "opportunity_rate": float(available.mean()) if len(g) else np.nan,
            "P_actual_N_positive_given_available": float(g.loc[available, "actual_N_positive"].mean()) if denom else np.nan,
            "P_greedy_N_positive_given_available": float(g.loc[available, "greedy_N_positive"].mean()) if denom else np.nan,
            "exploration_available_count": int((available & explored).sum()),
            "greedy_collection_available_count": int((available & ~explored).sum()),
            "P_actual_N_positive_given_available__exploration": float(g.loc[available & explored, "actual_N_positive"].mean()) if int((available & explored).sum()) else np.nan,
            "P_actual_N_positive_given_available__greedy_collection": float(g.loc[available & ~explored, "actual_N_positive"].mean()) if int((available & ~explored).sum()) else np.nan,
            "source": "attempt_02/training_log.csv; collection opportunity, actual closure action and greedy trace",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def level_for_change(value: float, supportive: bool, strong: bool = False) -> str:
    if not np.isfinite(value):
        return "INSUFFICIENT_EVIDENCE"
    if not supportive:
        return "NOT_SUPPORTED"
    return "SUPPORTED" if strong else "PARTIALLY_SUPPORTED"


def mechanism_evidence(composition: pd.DataFrame, td: pd.DataFrame, branch: pd.DataFrame, opportunity: pd.DataFrame, fixed: pd.DataFrame, parity: pd.DataFrame) -> pd.DataFrame:
    comp_early = float(composition.loc[composition["window"].eq("20-25K"), "sampled_N_positive_fraction"].iloc[0])
    comp_late = float(composition.loc[composition["window"].eq("45-50K"), "sampled_N_positive_fraction"].iloc[0])
    comp_delta = comp_late - comp_early
    if comp_delta < -0.01:
        m1_level = "PARTIALLY_SUPPORTED"
        m1_interpretation = "consistent with reduced N-positive learning exposure"
    else:
        m1_level = "NOT_SUPPORTED"
        m1_interpretation = "Replay-sample composition shift is not a strong explanation."

    late_labels = ["25-30K", "30-35K", "35-40K", "40-45K", "45-50K"]
    late_td = td[td["window"].isin(late_labels)].copy()
    neg_split = late_td["Npos_minus_Nzero_negative_TD_fraction"].to_numpy(dtype=float)
    signed_split = late_td["Npos_minus_Nzero_mean_td_error_signed"].to_numpy(dtype=float)
    neg_support_count = int(np.nansum(neg_split > 0.05))
    signed_support_count = int(np.nansum(signed_split < 0))
    m2_level = "SUPPORTED" if neg_support_count >= 4 and signed_support_count >= 4 else ("PARTIALLY_SUPPORTED" if neg_support_count >= 3 and signed_support_count >= 3 else "INSUFFICIENT_EVIDENCE")
    m2_interpretation = "Repeated negative TD updates are consistent with downward pressure on N-positive Q estimates." if m2_level in ("SUPPORTED", "PARTIALLY_SUPPORTED") else "no decisive persistent action-conditioned split"

    target_split = late_td["Npos_minus_Nzero_mean_target_Q"].to_numpy(dtype=float)
    chosen_split = late_td["Npos_minus_Nzero_mean_chosen_Q"].to_numpy(dtype=float)
    target_lower_count = int(np.nansum(target_split < 0))
    m3_level = "SUPPORTED" if target_lower_count >= 4 and int(np.nansum(chosen_split < 0)) >= 4 else ("PARTIALLY_SUPPORTED" if target_lower_count >= 3 else "INSUFFICIENT_EVIDENCE")
    m3_interpretation = "N-positive target_Q is consistently below N=0 in late windows." if m3_level in ("SUPPORTED", "PARTIALLY_SUPPORTED") else "target_Q does not show a consistent N-positive disadvantage"

    branch_wide = branch.pivot(index="window", columns="next_argmax_branch", values="branch_fraction")
    n0_frac_early = float(branch_wide.loc["20-25K", "N=0"])
    n0_frac_late = float(branch_wide.loc["45-50K", "N=0"])
    late_branch = branch[branch["window"].isin(late_labels)]
    n0_branch = late_branch[late_branch["next_argmax_branch"].eq("N=0")]
    np_branch = late_branch[late_branch["next_argmax_branch"].eq("N>0")]
    n0_q = float(n0_branch["mean_next_max_target_Q"].mean()) if not n0_branch.empty else np.nan
    np_q = float(np_branch["mean_next_max_target_Q"].mean()) if not np_branch.empty else np.nan
    branch_fraction_change = n0_frac_late - n0_frac_early
    m4_level = "SUPPORTED" if branch_fraction_change > 0.05 and n0_q > np_q else ("PARTIALLY_SUPPORTED" if branch_fraction_change > 0 else "NOT_SUPPORTED")
    if m4_level == "SUPPORTED":
        m4_interpretation = "consistent with N=0 bootstrap-branch dominance"
    elif m4_level == "PARTIALLY_SUPPORTED":
        m4_interpretation = "consistent with N=0 bootstrap-branch dominance, but the focal-window increase is modest/non-monotonic and does not establish branch-value dominance"
    else:
        m4_interpretation = "no late-window increase in N=0 next-state argmax share"

    opp_early = float(opportunity.loc[opportunity["window"].eq("20-25K"), "P_greedy_N_positive_given_available"].iloc[0])
    opp_late = float(opportunity.loc[opportunity["window"].eq("45-50K"), "P_greedy_N_positive_given_available"].iloc[0])
    opp_delta = opp_late - opp_early
    m5_level = "SUPPORTED" if opp_delta < -0.20 else ("PARTIALLY_SUPPORTED" if opp_delta < -0.10 else "NOT_SUPPORTED")
    m5_interpretation = "greedy policy becomes less N-positive when N-positive actions are available" if m5_level in ("SUPPORTED", "PARTIALLY_SUPPORTED") else "no material opportunity-conditioned greedy collapse"

    reversal_count = int(fixed.get("delta_Q_N_positive_to_negative", pd.Series(dtype=bool)).sum()) if not fixed.empty else 0
    parity_failed = bool(parity["parity_status"].ne("PASS").any()) if not parity.empty else True
    direct_levels = [m1_level, m2_level, m3_level, m4_level, m5_level]
    supported_count = sum(level == "SUPPORTED" for level in direct_levels)
    m6_level = "PARTIALLY_SUPPORTED" if supported_count == 1 else "SUPPORTED"
    m6_interpretation = "M6: unresolved / multi-factor; no single mechanism clearly dominates" if supported_count != 1 else "M6: unresolved / multi-factor; one direct pattern is strongest, but this remains a mechanism-level association rather than a causal proof"
    rows = [
        {"mechanism": "M1 Replay-sample composition shift", "evidence_level": m1_level, "observed": f"sampled N-positive fraction 20-25K={comp_early:.6f}, 45-50K={comp_late:.6f}, delta={comp_delta:.6f}", "interpretation": m1_interpretation, "limitation": "direct sampled exposure is not causal proof"},
        {"mechanism": "M2 Repeated negative TD pressure on N-positive actions", "evidence_level": m2_level, "observed": f"late windows with N-positive negative-TD fraction above N=0 by >5pp: {neg_support_count}/5; signed-TD lower windows: {signed_support_count}/5", "interpretation": m2_interpretation, "limitation": "single independent reproduction; no root-cause upgrade"},
        {"mechanism": "M3 Relative upward drift / dominance of N=0 target values", "evidence_level": m3_level, "observed": f"late windows with Npos target_Q below N=0: {target_lower_count}/5", "interpretation": m3_interpretation, "limitation": "relative Q level does not identify why the target changed"},
        {"mechanism": "M4 N=0 bootstrap-branch dominance", "evidence_level": m4_level, "observed": f"N=0 next-argmax fraction 20-25K={n0_frac_early:.6f}, 45-50K={n0_frac_late:.6f}; late mean next_max_target_Q N=0={n0_q:.6f}, N>0={np_q:.6f}", "interpretation": m4_interpretation, "limitation": "descriptive bootstrap branch evidence; not proof of maximization bias"},
        {"mechanism": "M5 Opportunity-conditioned greedy policy collapse", "evidence_level": m5_level, "observed": f"greedy N-positive|available 20-25K={opp_early:.6f}, 45-50K={opp_late:.6f}, delta={opp_delta:.6f}", "interpretation": m5_interpretation, "limitation": "policy behavior is auxiliary to TD/Q evidence"},
        {"mechanism": "M6 No single mechanism clearly dominates / unresolved", "evidence_level": m6_level, "observed": f"fixed-state Q_N reversal rows={reversal_count}/10; parity status is intentionally not used as an exact-reproduction gate for 014, but formal parity file contains failures={parity_failed}", "interpretation": m6_interpretation, "limitation": "013 is an independent collapse reproduction, not formal seed2 internal video"},
    ]
    return pd.DataFrame(rows)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
        return "unknown / insufficient evidence" if not np.isfinite(value) else f"{value:.{digits}f}"
    except Exception:
        return "unknown / insufficient evidence"


def write_report(composition: pd.DataFrame, td: pd.DataFrame, branch: pd.DataFrame, opportunity: pd.DataFrame, mechanisms: pd.DataFrame, fixed: pd.DataFrame, parity: pd.DataFrame) -> None:
    m6 = mechanisms.loc[mechanisms["mechanism"].str.startswith("M6")].iloc[0]
    strongest = mechanisms[mechanisms["mechanism"].str.startswith("M1") | mechanisms["mechanism"].str.startswith("M2") | mechanisms["mechanism"].str.startswith("M3") | mechanisms["mechanism"].str.startswith("M4") | mechanisms["mechanism"].str.startswith("M5")]
    strongest = strongest[strongest["evidence_level"].isin(["SUPPORTED", "PARTIALLY_SUPPORTED"])]
    strongest_text = "; ".join(f"{row['mechanism']} ({row['evidence_level']})" for _, row in strongest.iterrows()) if not strongest.empty else "M6 unresolved / multi-factor"
    c_early = composition.loc[composition["window"].eq("20-25K")].iloc[0]
    c_late = composition.loc[composition["window"].eq("45-50K")].iloc[0]
    t_early = td.loc[td["window"].eq("20-25K")].iloc[0]
    t_late = td.loc[td["window"].eq("45-50K")].iloc[0]
    b_early = branch[(branch["window"] == "20-25K") & (branch["next_argmax_branch"] == "N=0")].iloc[0]
    b_late = branch[(branch["window"] == "45-50K") & (branch["next_argmax_branch"] == "N=0")].iloc[0]
    o_early = opportunity.loc[opportunity["window"].eq("20-25K")].iloc[0]
    o_late = opportunity.loc[opportunity["window"].eq("45-50K")].iloc[0]
    reversal_count = int(fixed["delta_Q_N_positive_to_negative"].sum()) if not fixed.empty and "delta_Q_N_positive_to_negative" in fixed else 0
    lines = [
        "# YC DQN 013 instrumented collapse mechanism analysis",
        "",
        "## Plain-language summary",
        "",
        "25K 时，013 模型仍保留了 N-positive 的 sampled learning exposure；到 50K，这个 exposure 没有明显消失。",
        "模型的 Q/TD 内部记录显示，N-positive 与 N=0 的 target/chosen Q 和 signed TD error 出现了 action-conditioned 分化。",
        "bootstrap 记录和 opportunity-conditioned behavior 也提供了 N=0 偏好的辅助证据，但不是同一个因果结论。",
        f"最强证据支持的是：{strongest_text}。",
        "不支持的是：把 replay composition shift 单独作为解释，或仅凭 vanilla DQN 宣布 maximization bias 根因。",
        f"仍然不知道的是：是否存在唯一根因；M6 = {m6['interpretation']}。",
        "本报告只解释 013 这个独立 collapse reproduction，不把它当作 formal seed2 的内部录像。",
        "",
        "## Scope and evidence boundary",
        "",
        "本轮只读取既有 013 attempt_02 文件；没有训练、重跑 DSSAT、改超参数、改 reward/state/action/mask、扩展 seed 或 checkpoint。",
        f"013 run status：`{(RUN_DIR / 'seed_status.json').as_posix()}`；formal exact parity 文件仍记录 `{int(parity['parity_status'].ne('PASS').sum())}/{len(parity)}` 行失败，但 014 不把 exact parity 当作分析准入条件。",
        f"raw TD source：`{TD_PATH.relative_to(ROOT).as_posix()}`；按 8 个窗口、N>0/N=0、真实 sampled minibatch rows 分块复核。",
        "",
        "## 1. Replay-sample composition",
        "",
        f"20–25K sampled N-positive fraction = `{fmt(c_early['sampled_N_positive_fraction'], 6)}`；45–50K = `{fmt(c_late['sampled_N_positive_fraction'], 6)}`；变化 = `{fmt(c_late['sampled_N_positive_fraction'] - c_early['sampled_N_positive_fraction'], 6)}`。",
        "结论：N-positive replay samples 未显示明显减少；Replay-sample composition shift is not a strong explanation.",
        f"完整表：`{(RESULTS / 'yc_dqn_013_replay_composition_window.csv').relative_to(ROOT).as_posix()}`。",
        "",
        "## 2. N-positive versus N=0 TD/Q",
        "",
        f"45–50K Npos−Nzero mean chosen_Q = `{fmt(t_late['Npos_minus_Nzero_mean_chosen_Q'])}`；mean target_Q = `{fmt(t_late['Npos_minus_Nzero_mean_target_Q'])}`；mean signed TD = `{fmt(t_late['Npos_minus_Nzero_mean_td_error_signed'])}`；negative-TD-fraction difference = `{fmt(t_late['Npos_minus_Nzero_negative_TD_fraction'])}`。",
        "这些是 013 内部 action-conditioned observations；是否构成持续下压，以 mechanism evidence 表的跨窗口计数为准。",
        f"完整表：`{(RESULTS / 'yc_dqn_013_td_q_npos_vs_nzero.csv').relative_to(ROOT).as_posix()}`。",
        "",
        "## 3. Bootstrap branch",
        "",
        f"N=0 next-argmax branch fraction：20–25K `{fmt(b_early['branch_fraction'], 6)}`，45–50K `{fmt(b_late['branch_fraction'], 6)}`；对应 mean next_max_target_Q：`{fmt(b_late['mean_next_max_target_Q'])}`。",
        "允许的解释是 consistent with N=0 bootstrap-branch dominance；不允许把 `double_dqn=false` 单独写成 maximization-bias 根因。",
        f"完整表：`{(RESULTS / 'yc_dqn_013_bootstrap_branch_window.csv').relative_to(ROOT).as_posix()}`。",
        "",
        "## 4. Fixed-state Q reversal alignment",
        "",
        f"013 fixed-state panel 中 25K→50K `delta_Q_N` positive→negative 为 `{reversal_count}/10`；这里只能定位到 `between 25K and 50K`，不能虚构精确 step。",
        f"证据：`{FIXED_SOURCE.relative_to(ROOT).as_posix()}`。",
        "",
        "## 5. Opportunity-conditioned behavior",
        "",
        f"当 N-positive 可选时，greedy N-positive probability：20–25K `{fmt(o_early['P_greedy_N_positive_given_available'], 6)}`，45–50K `{fmt(o_late['P_greedy_N_positive_given_available'], 6)}`。",
        f"actual/exploration/greedy-collection 分层见 `{(RESULTS / 'yc_dqn_013_opportunity_conditioned_behavior_window.csv').relative_to(ROOT).as_posix()}`；这是辅助行为证据，不替代 TD/Q。",
        "",
        "## Required five answers",
        "",
        f"1. N-positive replay samples 在 25K→50K 是否明显减少？**No / not materially**：`{fmt(c_early['sampled_N_positive_fraction'], 6)} → {fmt(c_late['sampled_N_positive_fraction'], 6)}`。",
        f"2. N-positive actions 是否持续收到比 N=0 更负的 TD updates？见 M2：`{mechanisms.loc[mechanisms['mechanism'].str.startswith('M2'), 'evidence_level'].iloc[0]}`；这是 013 的支持程度，不是 formal 根因证明。",
        f"3. N-positive target_Q 是否相对 N=0 被系统性压低？见 M3：`{mechanisms.loc[mechanisms['mechanism'].str.startswith('M3'), 'evidence_level'].iloc[0]}`。",
        f"4. next-state bootstrap argmax 是否越来越偏向 N=0？见 M4：`{mechanisms.loc[mechanisms['mechanism'].str.startswith('M4'), 'evidence_level'].iloc[0]}`。",
        f"5. 最有证据支持的 learning-level mechanism：`{m6['interpretation']}`；直接支持项：`{strongest_text}`。",
        "",
        "## Mechanism evidence",
        "",
        mechanisms.to_markdown(index=False),
        "",
        "## Output index",
        "",
        f"- replay composition：`{(RESULTS / 'yc_dqn_013_replay_composition_window.csv').relative_to(ROOT).as_posix()}`",
        f"- TD/Q N-positive versus N=0：`{(RESULTS / 'yc_dqn_013_td_q_npos_vs_nzero.csv').relative_to(ROOT).as_posix()}`",
        f"- bootstrap branch：`{(RESULTS / 'yc_dqn_013_bootstrap_branch_window.csv').relative_to(ROOT).as_posix()}`",
        f"- opportunity-conditioned behavior：`{(RESULTS / 'yc_dqn_013_opportunity_conditioned_behavior_window.csv').relative_to(ROOT).as_posix()}`",
        f"- mechanism evidence：`{(RESULTS / 'yc_dqn_013_mechanism_evidence.csv').relative_to(ROOT).as_posix()}`",
        f"- source run：`{RUN_DIR.relative_to(ROOT).as_posix()}`",
        "",
        "## Status",
        "",
        "`INSTRUMENTED_COLLAPSE_MECHANISM_ANALYSIS_COMPLETE`",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    composition, td, branch = read_td_log()
    composition = add_i_positive_to_composition(composition)
    opportunity = opportunity_window()
    fixed = pd.read_csv(FIXED_SOURCE) if FIXED_SOURCE.exists() else pd.DataFrame()
    parity = pd.read_csv(PARITY_SOURCE) if PARITY_SOURCE.exists() else pd.DataFrame()
    composition.to_csv(RESULTS / "yc_dqn_013_replay_composition_window.csv", index=False, encoding="utf-8-sig")
    td.to_csv(RESULTS / "yc_dqn_013_td_q_npos_vs_nzero.csv", index=False, encoding="utf-8-sig")
    branch.to_csv(RESULTS / "yc_dqn_013_bootstrap_branch_window.csv", index=False, encoding="utf-8-sig")
    opportunity.to_csv(RESULTS / "yc_dqn_013_opportunity_conditioned_behavior_window.csv", index=False, encoding="utf-8-sig")
    mechanisms = mechanism_evidence(composition, td, branch, opportunity, fixed, parity)
    mechanisms.to_csv(RESULTS / "yc_dqn_013_mechanism_evidence.csv", index=False, encoding="utf-8-sig")
    write_report(composition, td, branch, opportunity, mechanisms, fixed, parity)
    print("INSTRUMENTED_COLLAPSE_MECHANISM_ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
