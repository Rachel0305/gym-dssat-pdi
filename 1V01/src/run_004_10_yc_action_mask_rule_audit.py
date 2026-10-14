from __future__ import annotations

import ast
import hashlib
import json
import math
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/yc_random_weather_ppo/004_10_action_mask_rule_audit"
REPORT = ROOT / "docs/yc_random_weather_004_10_action_mask_rule_audit.md"
TRACE = ROOT / "results/yc_random_weather_ppo/004_09_observation_mask_action_audit/decision_trace_seed345.csv"
FORMAL = TRACE.parent / "formal_manifest.json"
REVISION = TRACE.parent / "analysis_revision_manifest.json"
CONFIG = ROOT / "results/yc_random_weather_ppo/004_05/config/canonical_yc_ppo_config.json"
PROMPT = ROOT / "prompt_02/004_10_yc_action_mask_rule_audit.md"
BASE_SOURCE = ROOT / "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py"
STAGED_SOURCE = ROOT / "src/run_sya_lowIC_ppo_i240_staged_reserve_040_26.py"
LATE_SOURCE = ROOT / "src/run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py"
SOURCE_PATHS = [
    BASE_SOURCE, STAGED_SOURCE, LATE_SOURCE,
    ROOT / "src/run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py",
    ROOT / "src/run_sya_lowIC_binary_timing_maskableppo_042_10.py",
    ROOT / "src/ppo_action_safety.py",
    ROOT / "results/yc_random_weather_ppo/004_03/run_controlled_pilot.py",
    ROOT / "src/run_004_09_yc_observation_mask_action_audit.py",
]
SEEDS = (3, 4, 5)
YEARS = tuple(range(2014, 2024))
ACTION_DIM = 16
RULE_ORDER = (
    "I_DAP_WINDOW", "I_SEASON_BUDGET", "I_MIN_INTERVAL",
    "N_DAP_WINDOW", "N_SEASON_BUDGET", "N_MIN_INTERVAL",
    "I_EARLY_CAP", "I_MID_CAP", "I_LATE_RESERVE",
)
RULE_LABEL = {
    "I_DAP_WINDOW": "灌溉 DAP 时窗", "I_SEASON_BUDGET": "季节灌溉上限",
    "I_MIN_INTERVAL": "灌溉最小间隔", "N_DAP_WINDOW": "施氮 DAP 时窗",
    "N_SEASON_BUDGET": "季节施氮上限", "N_MIN_INTERVAL": "施氮最小间隔",
    "I_EARLY_CAP": "早期灌溉上限", "I_MID_CAP": "中期灌溉上限",
    "I_LATE_RESERVE": "晚期灌溉预留",
}
RULE_COLOR = {
    "I_DAP_WINDOW": "#2878B5", "I_SEASON_BUDGET": "#154F7A",
    "I_MIN_INTERVAL": "#71A8D1", "N_DAP_WINDOW": "#4D8D40",
    "N_SEASON_BUDGET": "#23682F", "N_MIN_INTERVAL": "#92BC72",
    "I_EARLY_CAP": "#D9822B", "I_MID_CAP": "#E5AF58",
    "I_LATE_RESERVE": "#B85736",
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def literal_constants(path: Path, names: tuple[str, ...]) -> dict[str, float]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                found[name] = float(ast.literal_eval(node.value))
    if set(found) != set(names):
        raise RuntimeError(f"Missing source constants in {relative(path)}: {sorted(set(names) - set(found))}")
    return found


def validate_inputs() -> tuple[pd.DataFrame, dict, dict, list[tuple[float, float]]]:
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    revision = json.loads(REVISION.read_text(encoding="utf-8"))
    if formal.get("passed") is not True or formal.get("episode_count") != 30 or formal.get("step_count") != 3048:
        raise RuntimeError("004_09 formal replay did not pass the 30-episode/3048-step gate")
    if revision.get("passed") is not True or revision.get("source_step_count") != 3048:
        raise RuntimeError("004_09 revised analysis did not pass its source-coverage gate")

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    safety = cfg["run_config"]["action_safety"]
    if safety != cfg["action_safety"] or not safety["enabled"]:
        raise RuntimeError("Frozen action-safety fields disagree or are disabled")
    actions = cfg["run_config"]["discrete_actions"]
    irrigation = [float(value) for value in actions["irrigation_levels"]]
    nitrogen = [float(value) for value in actions["nitrogen_levels"]]
    grid = [(i, n) for i in irrigation for n in nitrogen]
    if len(grid) != ACTION_DIM or grid[0] != (0.0, 0.0):
        raise RuntimeError(f"Frozen action grid is not the expected 16-action grid: {grid}")
    if max(irrigation) > float(safety["daily_irrigation_max"]) or max(nitrogen) > float(safety["daily_n_max"]):
        raise RuntimeError("Action grid includes a level above the configured daily maximum")

    trace = pd.read_csv(TRACE)
    required = [
        "seed", "year", "timestep", "dap", "obs__dap", "action_index", "raw_argmax_action",
        "raw_argmax_masked_out", "mask_legal_count", "irrigation_action_mm", "fertilizer_action_kgN_ha",
        "action_matches_formal", "predict_distribution_action_match", "raw_equals_policy_input",
    ]
    required += [f"mask__a{index:02d}" for index in range(ACTION_DIM)]
    required += [f"raw_prob__a{index:02d}" for index in range(ACTION_DIM)]
    missing = sorted(set(required) - set(trace.columns))
    if missing or len(trace) != 3048 or trace.duplicated(["seed", "year", "timestep"]).any():
        raise RuntimeError(f"004_09 trace coverage/schema failure: missing={missing}, rows={len(trace)}")
    if set(trace.seed) != set(SEEDS) or set(trace.year) != set(YEARS):
        raise RuntimeError("004_09 seed/year coverage differs from 3/4/5 x 2014-2023")
    if (trace.groupby(["seed", "year"]).size() <= 0).any() or len(trace.groupby(["seed", "year"])) != 30:
        raise RuntimeError("004_09 lacks one or more seed/year episodes")
    for name in ("action_matches_formal", "predict_distribution_action_match", "raw_equals_policy_input"):
        if not trace[name].eq(True).all():
            raise RuntimeError(f"004_09 trace contains failed {name} checks")
    if not np.allclose(trace.dap, trace["obs__dap"], atol=1e-6):
        raise RuntimeError("Recorded DAP differs from pre-action observation DAP")
    return trace.sort_values(["seed", "year", "timestep"]).reset_index(drop=True), cfg, safety, grid


def action_type(irrigation: float, nitrogen: float) -> str:
    if irrigation > 0 and nitrogen > 0:
        return "I+N"
    if irrigation > 0:
        return "I-only"
    if nitrogen > 0:
        return "N-only"
    return "no-op"


def effective_dap(raw_dap: float) -> int:
    # Mirrors StressAwareDiscreteWrapper._dap(), including its DAP 0 -> 1 rule.
    return int(round(raw_dap)) if math.isfinite(raw_dap) and raw_dap > 0 else 1


def action_rules(
    irrigation: float, nitrogen: float, dap: int, state: dict,
    safety: dict, staged: dict, late: dict,
) -> list[str]:
    if irrigation == 0 and nitrogen == 0:
        return []
    rules = []
    if irrigation > 0:
        lo, hi = safety["irrigation_allowed_dap_range"]
        if dap < int(lo) or dap > int(hi):
            rules.append("I_DAP_WINDOW")
        if irrigation - (float(safety["season_irrigation_soft_limit"]) - state["cum_i"]) > 1e-9:
            rules.append("I_SEASON_BUDGET")
        if state["last_i"] is not None and dap - state["last_i"] < int(safety["min_days_between_irrigation"]):
            rules.append("I_MIN_INTERVAL")
    if nitrogen > 0:
        lo, hi = safety["fertilization_allowed_dap_range"]
        if dap < int(lo) or dap > int(hi):
            rules.append("N_DAP_WINDOW")
        if nitrogen - (float(safety["season_n_soft_limit"]) - state["cum_n"]) > 1e-9:
            rules.append("N_SEASON_BUDGET")
        if state["last_n"] is not None and dap - state["last_n"] < int(safety["min_days_between_fertilization"]):
            rules.append("N_MIN_INTERVAL")
    if irrigation > 0:
        after = state["cum_i"] + irrigation
        if dap <= staged["EARLY_DAP_END"] and after - staged["EARLY_CUM_IRRIGATION_CAP"] > 1e-9:
            rules.append("I_EARLY_CAP")
        if dap <= staged["MID_DAP_END"] and after - staged["MID_CUM_IRRIGATION_CAP"] > 1e-9:
            rules.append("I_MID_CAP")
        if dap <= late["LATE_RESERVE_DAP_END"] and after - late["PRE_LATE_CUM_IRRIGATION_CAP"] > 1e-9:
            rules.append("I_LATE_RESERVE")
    return rules


def audit(trace: pd.DataFrame, safety: dict, grid: list[tuple[float, float]], staged: dict, late: dict):
    decisions = []
    blocked = []
    hits = []
    discrepancies = []
    mask_columns = [f"mask__a{index:02d}" for index in range(ACTION_DIM)]
    for (seed, year), episode in trace.groupby(["seed", "year"], sort=True):
        state = {"cum_i": 0.0, "cum_n": 0.0, "last_i": None, "last_n": None}
        for row in episode.itertuples(index=False):
            values = row._asdict()
            seed_i, year_i = int(seed), int(year)
            timestep = int(values["timestep"])
            dap_raw = float(values["dap"])
            dap = effective_dap(float(values["obs__dap"]))
            predicted_mask = []
            reasons_by_action = []
            for index, (irrigation, nitrogen) in enumerate(grid):
                reasons = action_rules(irrigation, nitrogen, dap, state, safety, staged, late)
                reasons_by_action.append(reasons)
                predicted_mask.append(not reasons)
                actual = bool(values[mask_columns[index]])
                if actual != predicted_mask[-1]:
                    discrepancies.append({
                        "seed": seed_i, "year": year_i, "timestep": timestep,
                        "dap": dap_raw, "effective_dap": dap, "action_index": index,
                        "recorded_legal": actual, "reconstructed_legal": predicted_mask[-1],
                        "reconstructed_rules_json": json.dumps(reasons),
                    })
            chosen = int(values["action_index"])
            raw_index = int(values["raw_argmax_action"])
            if not (0 <= chosen < ACTION_DIM and 0 <= raw_index < ACTION_DIM):
                raise RuntimeError(f"Invalid action index at seed={seed_i}, year={year_i}, step={timestep}")
            chosen_i, chosen_n = grid[chosen]
            applied_i = float(values["irrigation_action_mm"])
            applied_n = float(values["fertilizer_action_kgN_ha"])
            if not predicted_mask[chosen] or abs(chosen_i - applied_i) > 1e-6 or abs(chosen_n - applied_n) > 1e-6:
                raise RuntimeError(f"Chosen action/applied doses do not close at seed={seed_i}, year={year_i}, step={timestep}")
            recorded_blocked = str(values["raw_argmax_masked_out"]).lower() == "true"
            actual_blocked = not bool(values[mask_columns[raw_index]])
            if recorded_blocked != actual_blocked or int(values["mask_legal_count"]) != sum(predicted_mask):
                raise RuntimeError(f"Stored blocker/legal count mismatch at seed={seed_i}, year={year_i}, step={timestep}")
            raw_i, raw_n = grid[raw_index]
            reasons = reasons_by_action[raw_index]
            rec = {
                "seed": seed_i, "year": year_i, "timestep": timestep,
                "dap": dap_raw, "dap_occurrence": int(values["dap_occurrence"]),
                "effective_mask_dap": dap,
                "raw_argmax_action": raw_index, "raw_argmax_type": action_type(raw_i, raw_n),
                "raw_argmax_irrigation_mm": raw_i, "raw_argmax_nitrogen_kg_ha": raw_n,
                "raw_argmax_probability": float(values[f"raw_prob__a{raw_index:02d}"]),
                "chosen_action": chosen, "chosen_irrigation_mm": chosen_i,
                "chosen_nitrogen_kg_ha": chosen_n,
                "legal_action_count": int(sum(predicted_mask)),
                "has_legal_alternative": sum(predicted_mask) > 1,
                "raw_argmax_blocked": actual_blocked,
                "primary_rule": reasons[0] if reasons else "",
                "primary_rule_label": RULE_LABEL[reasons[0]] if reasons else "",
                "active_rule_count": len(reasons), "active_rules_json": json.dumps(reasons),
                "cumulative_irrigation_before_mm": state["cum_i"],
                "cumulative_n_before_kg_ha": state["cum_n"],
                "last_irrigation_effective_dap": state["last_i"],
                "last_fertilization_effective_dap": state["last_n"],
                "remaining_season_irrigation_mm": float(safety["season_irrigation_soft_limit"]) - state["cum_i"],
                "remaining_season_n_kg_ha": float(safety["season_n_soft_limit"]) - state["cum_n"],
                "mask_reconstruction_exact": all(
                    bool(values[mask_columns[index]]) == predicted_mask[index] for index in range(ACTION_DIM)
                ),
            }
            decisions.append(rec)
            if actual_blocked:
                blocked.append(rec)
                for position, rule in enumerate(reasons):
                    hits.append({
                        "seed": seed_i, "year": year_i, "timestep": timestep, "dap": dap_raw,
                        "effective_mask_dap": dap, "raw_argmax_action": raw_index,
                        "raw_argmax_type": rec["raw_argmax_type"], "rule": rule,
                        "rule_label": RULE_LABEL[rule], "is_first_executed_rule": position == 0,
                        "note": "Later conditions are independent offline checks; source short-circuits at the first failing rule.",
                    })
            if applied_i > 0:
                state["last_i"] = dap
            if applied_n > 0:
                state["last_n"] = dap
            state["cum_i"] += applied_i
            state["cum_n"] += applied_n
    return pd.DataFrame(decisions), pd.DataFrame(blocked), pd.DataFrame(hits), pd.DataFrame(discrepancies)


def summarize(decisions: pd.DataFrame, blocked: pd.DataFrame, hits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    seed_year_rows = []
    for (seed, year), group in decisions.groupby(["seed", "year"], sort=True):
        choice = group[group.has_legal_alternative]
        seed_year_rows.append({
            "seed": seed, "year": year, "decision_steps": len(group),
            "raw_argmax_blocked_steps": int(group.raw_argmax_blocked.sum()),
            "raw_argmax_blocked_rate_all_steps": float(group.raw_argmax_blocked.mean()),
            "choice_capable_steps": len(choice),
            "blocked_when_choice_capable": int(choice.raw_argmax_blocked.sum()),
            "blocked_rate_when_choice_capable": float(choice.raw_argmax_blocked.mean()) if len(choice) else np.nan,
            "forced_single_action_steps": int(group.legal_action_count.eq(1).sum()),
        })
    seed_year = pd.DataFrame(seed_year_rows)
    dap_rule = blocked.groupby(["seed", "dap", "primary_rule", "raw_argmax_type"], as_index=False).size().rename(columns={"size": "blocked_steps"})
    action_types = blocked.groupby(["seed", "raw_argmax_type", "primary_rule"], as_index=False).size().rename(columns={"size": "blocked_steps"})
    primary_counts = blocked.primary_rule.value_counts().reindex(RULE_ORDER, fill_value=0)
    all_counts = hits.rule.value_counts().reindex(RULE_ORDER, fill_value=0)
    by_seed = {}
    for seed in SEEDS:
        group = decisions[decisions.seed.eq(seed)]
        choice = group[group.has_legal_alternative]
        by_seed[str(seed)] = {
            "steps": len(group), "blocked_all": int(group.raw_argmax_blocked.sum()),
            "choice_capable_steps": len(choice),
            "blocked_choice_capable": int(choice.raw_argmax_blocked.sum()),
            "blocked_rate_when_choice_capable": float(choice.raw_argmax_blocked.mean()),
            "forced_single_steps": int(group.legal_action_count.eq(1).sum()),
        }
    summary = {
        "coverage": {"seeds": list(SEEDS), "years": list(YEARS), "episodes": len(seed_year), "steps": len(decisions), "mask_cells_checked": len(decisions) * ACTION_DIM},
        "blocked_raw_argmax_steps": len(blocked),
        "blocked_choice_capable_steps": int(blocked.has_legal_alternative.sum()),
        "forced_single_action_steps": int(decisions.legal_action_count.eq(1).sum()),
        "primary_rule_counts": {name: int(primary_counts[name]) for name in RULE_ORDER},
        "all_active_rule_counts": {name: int(all_counts[name]) for name in RULE_ORDER},
        "multiple_active_rule_steps": int(blocked.active_rule_count.gt(1).sum()),
        "dap_zero_decisions": int(decisions.dap.eq(0).sum()),
        "dap_zero_blocked": int(blocked.dap.eq(0).sum()),
        "by_seed": by_seed,
    }
    seed5_choice = blocked[blocked.seed.eq(5) & blocked.has_legal_alternative]
    stage_count = int(seed5_choice.primary_rule.isin(["I_EARLY_CAP", "I_MID_CAP"]).sum())
    summary["seed5_choice_capable_stage_cap_blocking"] = {
        "blocked_by_inherited_early_or_mid_cap": stage_count,
        "blocked_choice_capable_total": len(seed5_choice),
        "share": stage_count / len(seed5_choice) if len(seed5_choice) else None,
    }
    return seed_year, dap_rule, action_types, summary


def figures(blocked: pd.DataFrame, decisions: pd.DataFrame) -> list[Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#667085", "axes.linewidth": 0.8,
        "grid.color": "#D9DEE3", "grid.linewidth": 0.65,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    folder = OUT / "figures"
    folder.mkdir(exist_ok=False)
    paths = []
    for suffix, scope in (("choice_capable", blocked[blocked.has_legal_alternative]), ("all_blocked", blocked)):
        counts = scope.groupby(["seed", "primary_rule"]).size().unstack(fill_value=0).reindex(index=SEEDS, columns=RULE_ORDER, fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
        bottom = np.zeros(len(SEEDS))
        for rule in RULE_ORDER:
            values = counts[rule].to_numpy()
            if values.sum() == 0:
                continue
            ax.bar(np.arange(len(SEEDS)), values, bottom=bottom, color=RULE_COLOR[rule], width=0.60, label=rule)
            bottom += values
        ax.set_xticks(np.arange(len(SEEDS)), [f"Seed {seed}" for seed in SEEDS])
        ax.set_ylabel("Blocked raw argmax decisions")
        ax.set_title("YC | Action-mask first-rule attribution", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=7)
        path = folder / f"primary_rule_by_seed_{suffix}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    matrix = np.full((len(SEEDS), 121), np.nan)
    for r, seed in enumerate(SEEDS):
        for dap, part in decisions[decisions.seed.eq(seed) & decisions.dap.between(0, 120)].groupby("dap"):
            choice = part[part.has_legal_alternative]
            if len(choice):
                matrix[r, int(dap)] = float(choice.raw_argmax_blocked.mean())
    fig, ax = plt.subplots(figsize=(12, 3.1), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Oranges", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(SEEDS)), [f"S{seed}" for seed in SEEDS])
    ax.set_xticks(np.arange(0, 121, 10), [str(dap) for dap in range(0, 121, 10)])
    ax.set_xlabel("Observed DAP (2014–2023)")
    ax.set_ylabel("PPO seed")
    ax.set_title("Raw argmax blocked when legal alternatives exist", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.024, pad=0.02, label="Blocked share")
    path = folder / "blocked_choice_rate_by_dap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def write_report(summary: dict, decisions: pd.DataFrame, blocked: pd.DataFrame, hits: pd.DataFrame, safety: dict, staged: dict, late: dict, plots: list[Path]) -> None:
    if REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite report: {relative(REPORT)}")
    top = sorted(summary["primary_rule_counts"].items(), key=lambda item: -item[1])
    top_rows = [f"| {rule} | {RULE_LABEL[rule]} | {count} | {summary['all_active_rule_counts'][rule]} |" for rule, count in top]
    seed_rows = [
        f"| {seed} | {item['blocked_all']}/{item['steps']} | {item['blocked_choice_capable']}/{item['choice_capable_steps']} ({item['blocked_rate_when_choice_capable']:.1%}) | {item['forced_single_steps']} |"
        for seed, item in summary["by_seed"].items()
    ]
    action_type_counts = blocked.raw_argmax_type.value_counts()
    type_rows = [f"| {name} | {int(action_type_counts.get(name, 0))} |" for name in ("I-only", "N-only", "I+N", "no-op")]
    seed5_stage = summary["seed5_choice_capable_stage_cap_blocking"]
    key_days = blocked[blocked.dap.isin([0, 8, 15, 31, 61])]
    key_rows = [
        f"| {int(dap)} | {len(part)} | {len(part[part.has_legal_alternative])} | {', '.join(f'{rule}:{count}' for rule, count in part.primary_rule.value_counts().items())} |"
        for dap, part in key_days.groupby("dap", sort=True)
    ]
    lines = [
        "# YC 004_10：action-mask 规则归因审计", "",
        "## 结论", "",
        f"- `mask_reconstruction = {summary['verdicts']['mask_reconstruction']}`：{summary['coverage']['mask_cells_checked']} 个 action-mask 单元逐个与 004_09 实录一致，差异 {summary['mask_mismatch_cells']} 个。",
        f"- `blocked_raw_argmax_attribution = {summary['verdicts']['blocked_raw_argmax_attribution']}`：{len(blocked)} 个被屏蔽 raw argmax 均有至少一个明确规则；其中 {summary['multiple_active_rule_steps']} 步同时满足多条禁用条件。",
        f"- `mask_rule_reasonableness = {summary['verdicts']['mask_rule_reasonableness']}`：mask 按源码和冻结限制一致执行；这些来自 SYA 的阶段上限在 YC 上是否最合适，现有结果不能验证。",
        f"- Seed 5 的 {seed5_stage['blocked_choice_capable_total']} 个有合法替代动作且 raw 首选被挡的决策中，{seed5_stage['blocked_by_inherited_early_or_mid_cap']} 个（{seed5_stage['share']:.1%}）由继承的早/中期灌溉上限首先触发。这些上限实质参与形成了当前用水策略。",
        f"- `next_step = {summary['verdicts']['next_step']}`。", "",
        "## 有效规则", "",
        "- 实际调用链为 `004_03 ScheduledEpisodeEnv -> LateIrrigationReserveMaskWrapper -> I240SwfacGuardrailWrapper -> I240StagedIrrigationReserveMaskWrapper -> StressAwareDiscreteWrapper`。040_28 的 SWFAC guardrail 修改 reward，不新增 mask 条件。",
        f"- 基础限制：灌溉 DAP {safety['irrigation_allowed_dap_range']}、施氮 DAP {safety['fertilization_allowed_dap_range']}；I/N 季节上限 {safety['season_irrigation_soft_limit']:g} mm / {safety['season_n_soft_limit']:g} kg/ha；最小间隔 I/N 各 {safety['min_days_between_irrigation']} / {safety['min_days_between_fertilization']} 天。",
        f"- 继承阶段限制：有效 DAP ≤{staged['EARLY_DAP_END']:g} 时累计 I ≤{staged['EARLY_CUM_IRRIGATION_CAP']:g} mm；≤{staged['MID_DAP_END']:g} 时 ≤{staged['MID_CUM_IRRIGATION_CAP']:g} mm；≤{late['LATE_RESERVE_DAP_END']:g} 时 ≤{late['PRE_LATE_CUM_IRRIGATION_CAP']:g} mm。",
        "- `action_safety` 冻结 JSON 明列了季节上限、间隔、时窗和晚期 195 mm 预留，但没有直接列出早期 75 mm / 中期 150 mm 阈值；它们在 040_26 继承类中生效。源码是有效契约的一部分，配置单独阅读会遗漏这两条。",
        "- 每日 I/N 上限用于安全裁剪；此 16 动作网格最高剂量刚好等于上限，不构成 `_action_is_legal_without_clipping` 的单独禁用规则。no-op 始终合法。", "",
        "## 逐 seed 统计", "",
        "| Seed | 被屏蔽/全步 | 被屏蔽/有其他合法动作 | 仅一个合法动作的 step |",
        "|---:|---:|---:|---:|", *seed_rows, "",
        f"- 重建的决策前累计 I/N 与 policy observation 的 `totir/cumsumfert` 最大绝对差为 {summary['state_observation_closure']['totir_max_abs_error_mm']:.6g} mm / {summary['state_observation_closure']['cumsumfert_max_abs_error_kg_ha']:.6g} kg/ha。", "",
        "## 首个禁用规则", "",
        "| 规则 | 含义 | 源码首个禁用 | 离线独立同时触发 |",
        "|---|---|---:|---:|", *top_rows, "",
        f"- 被挡 raw argmax 总计 {len(blocked)} 步，其中仍有合法替代动作 {summary['blocked_choice_capable_steps']} 步。`blocked_rule_hits.csv` 的并发条件是离线独立判定；源码会在首个失败条件处返回，不能把后续条件描述为运行时逐条执行。",
        "- 晚期预留可作为并发条件出现，哪怕它不是首个返回的禁用原因；首规则计数为 0 不表示该条件从未满足。", "",
        "## 被挡动作类型", "",
        "| raw argmax 类型 | 被挡次数 |", "|---|---:|", *type_rows, "",
        "- 每步完整 seed/year/timestep、原始与有效 DAP、动作剂量/类型、预算和上次施用 DAP、首规则和所有独立触发条件见 `blocked_raw_argmax_decisions.csv`；逐 seed/year 见 `seed_year_rule_summary.csv`。", "",
        "## 关键 DAP", "",
        "| 原始 DAP | 被挡总步数 | 其中有合法替代 | 首规则:次数 |",
        "|---:|---:|---:|---|", *key_rows, "",
        "- Seed 5 的 DAP 15 在全部十年都出现同一个关键归因：raw 首选 action 9（I30/N40），先前累计 I 已是 75 mm，增加 30 mm 会超过有效早期 75 mm 上限；该步实际选择 action 2（I0/N80）。这里有多个合法动作，阻断不是 forced no-op。",
        f"- 原始 DAP 0 共 {summary['dap_zero_decisions']} 个 step，其中 {summary['dap_zero_blocked']} 个 raw argmax 被挡。源码 `_dap()` 将非正值映射为有效 DAP 1；因此配置的时窗起点 1 并不会禁止记录为 DAP 0 的首日施用。DAP 0 当日后续 step 使用同一个有效 DAP，最小间隔会阻止再次施用。这是明确的时序契约差异，需要文档化；现有证据不能单独认定为错误。", "",
        "## 合理性判断", "",
        "- 机械层面：所有 mask 均可用现行源码和决策前累计状态解释，记录的动作合法且剂量与离散网格一致；没有未知规则造成的额外屏蔽。",
        "- 约束目的：时窗、最小间隔、季节上限和灌溉预留具有明确的资源管理意图；大量 raw 首选被挡说明它们实质改变了策略输出，但无法据此推断放开后会提高产量或 reward。",
        "- YC 适用性：早/中期预留阈值沿用 SYA 链路，冻结配置没有单独展开；它们挡住了 seed 5 多数有替代选择时的 raw 灌溉首选。因此 165 mm 年用水不能仅解释为 actor 自发偏好。DAP 0→1 的实现语义也与结果表面 DAP 不同。本次没有同状态的替代动作结果，结论是条件合理，尚不足以给出农艺最优保证。先把有效 mask 契约和 DAP 口径写清，再决定是否需要受控 evaluation-only 规则敏感性测试。", "",
        "## 图表和数据", "",
    ]
    for path in plots:
        lines.append(f"- `{relative(path)}`")
    lines.extend([
        "- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/decision_mask_reconciliation.csv`：全 3048 个决策的状态重建和闭合。",
        "- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/blocked_raw_argmax_decisions.csv`：被屏蔽的逐步原始偏好。",
        "- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/blocked_rule_hits.csv`：每条同时成立的禁用条件。",
        "- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/seed_year_rule_summary.csv`、`dap_rule_summary.csv`、`action_type_summary.csv`：汇总。",
        "- `results/yc_random_weather_ppo/004_10_action_mask_rule_audit/analysis_summary.json`、`provenance.json`、`manifest.json`：分母、结论及哈希。", "",
        "## 局限", "",
        "- 004_09 未提供 calendar date/DOY，故用原始 DAP 和 timestep；本审计不推测日历日期。",
        "- 只复算当前规则在已发生轨迹上的布尔值；不更改任何规则，也没有未选动作的 DSSAT 产量、水氮效率或 reward 结果。",
        "- 未 commit、未 push。", "",
    ])
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("004_10 output already exists; refusing to overwrite research evidence")
    OUT.mkdir(parents=True, exist_ok=False)
    manifest_path = OUT / "manifest.json"
    write_json(manifest_path, {"status": "running", "passed": False, "started_utc": pd.Timestamp.utcnow().isoformat()})
    try:
        trace, cfg, safety, grid = validate_inputs()
        staged = literal_constants(STAGED_SOURCE, ("EARLY_DAP_END", "MID_DAP_END", "EARLY_CUM_IRRIGATION_CAP", "MID_CUM_IRRIGATION_CAP"))
        late = literal_constants(LATE_SOURCE, ("LATE_RESERVE_DAP_END", "PRE_LATE_CUM_IRRIGATION_CAP"))
        configured_late = safety["late_irrigation_reserve_mask"]
        if not configured_late["enabled"] or float(configured_late["dap_end"]) != late["LATE_RESERVE_DAP_END"] or float(configured_late["pre_late_cum_irrigation_cap_mm"]) != late["PRE_LATE_CUM_IRRIGATION_CAP"]:
            raise RuntimeError("Frozen late reserve differs from the effective source constants")
        decisions, blocked, hits, discrepancies = audit(trace, safety, grid, staged, late)
        totir_error = float(np.max(np.abs(trace["obs__totir"].to_numpy(float) - decisions.cumulative_irrigation_before_mm.to_numpy(float))))
        fertilizer_error = float(np.max(np.abs(trace["obs__cumsumfert"].to_numpy(float) - decisions.cumulative_n_before_kg_ha.to_numpy(float))))
        if totir_error > 1e-4 or fertilizer_error > 1e-4:
            raise RuntimeError(f"Reconstructed safety state disagrees with pre-action observation: I={totir_error}, N={fertilizer_error}")
        if discrepancies.empty:
            discrepancies = pd.DataFrame(columns=["seed", "year", "timestep", "dap", "effective_dap", "action_index", "recorded_legal", "reconstructed_legal", "reconstructed_rules_json"])
        discrepancies.to_csv(OUT / "mask_discrepancies.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        decisions.to_csv(OUT / "decision_mask_reconciliation.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        blocked.to_csv(OUT / "blocked_raw_argmax_decisions.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        hits.to_csv(OUT / "blocked_rule_hits.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        if len(discrepancies) or not decisions.mask_reconstruction_exact.all() or blocked.primary_rule.eq("").any():
            raise RuntimeError(f"Mask reconstruction/attribution failed: mismatch cells={len(discrepancies)}")
        seed_year, dap_rule, action_types, summary = summarize(decisions, blocked, hits)
        seed_year.to_csv(OUT / "seed_year_rule_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        dap_rule.to_csv(OUT / "dap_rule_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        action_types.to_csv(OUT / "action_type_summary.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
        summary["mask_mismatch_cells"] = len(discrepancies)
        summary["state_observation_closure"] = {"totir_max_abs_error_mm": totir_error, "cumsumfert_max_abs_error_kg_ha": fertilizer_error}
        summary["effective_rule_parameters"] = {"action_safety": safety, "staged_source_constants": staged, "late_source_constants": late, "grid": grid}
        summary["verdicts"] = {
            "mask_reconstruction": "EXACT", "blocked_raw_argmax_attribution": "COMPLETE",
            "mask_rule_reasonableness": "CONDITIONAL",
            "next_step": "document effective YC mask contract and DAP0 semantics before any rule change",
        }
        write_json(OUT / "analysis_summary.json", summary)
        source_files = [PROMPT, Path(__file__), TRACE, FORMAL, REVISION, CONFIG, *SOURCE_PATHS]
        provenance = {"created_utc": pd.Timestamp.utcnow().isoformat(), "source_sha256": {relative(path): digest(path) for path in source_files}, "additional_replay": False, "checkpoint": 100000}
        write_json(OUT / "provenance.json", provenance)
        plots = figures(blocked, decisions)
        write_report(summary, decisions, blocked, hits, safety, staged, late, plots)
        write_json(manifest_path, {
            "status": "passed", "passed": True, "completed_utc": pd.Timestamp.utcnow().isoformat(),
            "steps": len(decisions), "mask_cells": len(decisions) * ACTION_DIM,
            "blocked_raw_argmax_steps": len(blocked), "mask_mismatch_cells": len(discrepancies),
            "verdicts": summary["verdicts"], "report": relative(REPORT), "additional_replay": False,
        })
        print(f"[004_10 pass] steps={len(decisions)} mask_cells={len(decisions) * ACTION_DIM} blocked={len(blocked)} report={relative(REPORT)}", flush=True)
    except Exception as exc:
        write_json(manifest_path, {
            "status": "failed", "passed": False, "completed_utc": pd.Timestamp.utcnow().isoformat(),
            "error": repr(exc), "traceback": traceback.format_exc(), "additional_replay": False,
        })
        raise


if __name__ == "__main__":
    main()
