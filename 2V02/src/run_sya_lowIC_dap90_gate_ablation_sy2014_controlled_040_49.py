"""040_49: SY2014-only DAP90 gate ablation, controlled (two-arm) version.

Why this replaces 040_48
-------------------------
040_48 compared:
  - "baseline" = 040_40 checkpoint100000, trained on MANY years (random-year
    sampling across the full SYA train split), gate ON
  - "ablated"  = a NEW model trained on ONLY 2014, gate OFF

That confounds two changes at once (gate ON/OFF, AND many-years-training vs
one-year-training), so 040_48's result (ablated model used far less water,
yielded much worse) cannot be attributed to the gate specifically -- it may
simply reflect under-training / lack of exposure to varied conditions from
only ever training on one year.

This task fixes that by training TWO models, both restricted to SY2014-only
training, identical seed/reward/discrete-action-levels/PPO hyperparameters,
differing ONLY in whether the DAP<=90 / 195mm gate is active:
  - Arm "control_gate_on":  base04036 defaults untouched (gate ON, as in
    040_36/040_40)
  - Arm "treatment_gate_off": LATE_RESERVE_DAP_END monkeypatched to 0 (gate
    effectively OFF), same mechanism as 040_48

Only the gate differs between the two arms now. If irrigation timing still
comes out identical between them, the gate is not the cause of the fixed
schedule; if it differs, the gate is at least a contributing cause -- and
either way, both arms are now equally "new to single-year training", so
that confound is gone.

Scope discipline
-----------------
  - Two new training runs only (control + treatment), both single
    station-year (SYA 2014), same seed.
  - Does NOT change reward, discrete action levels, PPO hyperparameters.
  - Does NOT touch 040_26's DAP<=30/DAP<=60 caps (see ABLATE_EARLY_MID_CAPS_TOO
    in 040_48 for why; same caveat applies here, left off by default).
  - This is a structural mask change -- treat as requiring the same
    re-authorization your project already applies to changes like 027_04,
    not a routine checkpoint eval. 040_48's single-arm result should NOT be
    treated as a valid finding; this controlled version supersedes it.

Usage
-----
    python run_sya_lowIC_dap90_gate_ablation_sy2014_controlled_040_49.py --dry-run
    python run_sya_lowIC_dap90_gate_ablation_sy2014_controlled_040_49.py --timesteps 100000
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_lowIC_ppo_yield_guardrail_v3_040_40 as base04040
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
from ppo_evaluate import latest_observation_dict  # same helper used in 040_46/040_47/040_48

TASK = "040_49_sya_lowIC_dap90_gate_ablation_sy2014_controlled"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = base04040.STATION
SITE = "SY"
YEAR = 2014
SEED = base03222.SEED
DEFAULT_TOTAL_TIMESTEPS = 100_000

ARMS = ["control_gate_on", "treatment_gate_off"]
GATE_DAP_END_BY_ARM = {
    "control_gate_on": 90,   # unchanged from 040_36/040_40 default
    "treatment_gate_off": 0,  # dap<=0 never true for dap>=1 -> gate never fires
}

# For context only -- NOT used as the comparison baseline anymore, since it
# is confounded (multi-year training vs this task's single-year training).
# Kept here only so the record can show why 040_48 is superseded.
CONTEXT_040_44_MULTIYEAR_BASELINE_SY2014 = {
    "irrigation_sequence": "DAP1:I45; DAP8:I30; DAP31:I45; DAP38:I30; DAP61:I45; DAP91:I45",
    "final_grnwt_kg_ha": 10210.0989,
    "note": "040_40 checkpoint100000, trained on MANY years, gate ON -- not a clean comparison for this task.",
}


def ensure_dirs() -> None:
    for rel in ["models", "tables", "logs", "configs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def apply_gate(dap_end: int) -> dict[str, Any]:
    before = base04036.LATE_RESERVE_DAP_END
    base04036.LATE_RESERVE_DAP_END = int(dap_end)
    return {"before": before, "after": base04036.LATE_RESERVE_DAP_END}


def build_env_config_for_station(config: dict[str, Any]) -> dict[str, Any]:
    """Same read-only selection -> env_config construction used in 040_47/040_48."""
    split = pd.read_csv(base03222.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    split = split[split["station_code"].eq(STATION)].sort_values("year").reset_index(drop=True)
    pool = pd.read_csv(base03222.POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selection = pool.merge(split[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selection["selected_for_train"] = selection["split"].eq("train")
    selection["selected_for_eval"] = True
    selection["selection_reason"] = "040_49_dap90_gate_ablation_sy2014_controlled"
    return base03222.direct_ppo.build_env_config(config, selection)


def dap_for_logging(raw_dap: float) -> int:
    """Match DirectActionSafeGrowthRewardWrapper's own convention: dap=0 at
    reset is logged as dap=1 (planting day), since DAP0 is not a real
    in-season day. Keeping this consistent with the rest of the codebase
    avoids the DAP0-vs-DAP1 off-by-one seen when comparing 040_48 against
    040_44's sequences."""
    return int(round(raw_dap)) if np.isfinite(raw_dap) and raw_dap > 0 else 1


def rollout(model: Any, env: Any) -> dict[str, Any]:
    obs, info = env.reset()
    done = False
    irrigation_events: list[str] = []
    nitrogen_events: list[str] = []
    total_irrigation = 0.0
    total_n = 0.0
    final_grnwt = np.nan
    max_swfac = 0.0
    while not done:
        named = latest_observation_dict(env, obs, info)
        dap = dap_for_logging(float(named.get("dap", 0) or 0))
        action_masks = env.action_masks() if hasattr(env, "action_masks") else None
        if action_masks is not None:
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        last_info = getattr(env, "last_action_info", {}) or {}
        i_amt = float(last_info.get("safe_action_amir", 0.0) or 0.0)
        n_amt = float(last_info.get("safe_action_anfer", 0.0) or 0.0)
        swfac = float(named.get("swfac", 0.0) or 0.0) if named.get("swfac") is not None else 0.0
        max_swfac = max(max_swfac, swfac)
        if i_amt > 1e-9:
            irrigation_events.append(f"DAP{dap}:I{i_amt:g}")
            total_irrigation += i_amt
        if n_amt > 1e-9:
            nitrogen_events.append(f"DAP{dap}:N{n_amt:g}")
            total_n += n_amt
        latest_named = latest_observation_dict(env, obs, info)
        g = latest_named.get("grnwt")
        if g is not None and np.isfinite(float(g)):
            final_grnwt = float(g)
    return {
        "irrigation_sequence": "; ".join(irrigation_events),
        "nitrogen_sequence": "; ".join(nitrogen_events),
        "total_irrigation_mm": total_irrigation,
        "total_n_kg_ha": total_n,
        "final_grnwt_kg_ha": final_grnwt,
        "max_swfac": max_swfac,
    }


def train_and_eval_arm(arm: str, config: dict[str, Any], env_config: dict[str, Any], total_timesteps: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    gate_state = apply_gate(GATE_DAP_END_BY_ARM[arm])

    run_tag = f"{STATION}_{YEAR}_040_49_{arm}_train"
    train_env = base04040.make_env_with_yield_guardrail(config, env_config, STATION, YEAR, SEED, run_tag, evaluation=False)
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=SEED,
        tensorboard_log=str(OUT / "logs" / "tensorboard" / arm),
        **base03222.base.ppo_kwargs(config),
    )
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True, progress_bar=False)
    if hasattr(train_env, "close"):
        train_env.close()

    model_path = OUT / "models" / f"{STATION}_{YEAR}_{arm}_seed{SEED}_ckpt{total_timesteps}.zip"
    model.save(str(model_path))

    eval_run_tag = f"{STATION}_{YEAR}_040_49_{arm}_eval"
    eval_env = base04040.make_env_with_yield_guardrail(config, env_config, STATION, YEAR, SEED, eval_run_tag, evaluation=True)
    eval_result = rollout(model, eval_env)
    if hasattr(eval_env, "close"):
        eval_env.close()

    return {
        "arm": arm,
        "gate_state": gate_state,
        "model_path": model_path.relative_to(ROOT).as_posix(),
        **eval_result,
    }


def write_record(results_by_arm: dict[str, dict[str, Any]], total_timesteps: int) -> None:
    control = results_by_arm["control_gate_on"]
    treatment = results_by_arm["treatment_gate_off"]
    same_sequence = control["irrigation_sequence"].strip() == treatment["irrigation_sequence"].strip()

    lines = [
        f"# {TASK} 记录",
        "",
        "## 任务目的",
        "",
        "修正040_48的对照缺陷：040_48把“多年训练+gate开”和“单年训练+gate关”放在一起比，",
        "混入了训练年份范围这个额外变量，无法单独归因于gate。本任务改为同时训练两个模型，",
        "两者都只在SY2014一年上训练、其余奖励/档位/超参数/seed完全一致，唯一区别是",
        "DAP<=90/195mm这条gate开还是关，从而干净地检验gate本身的影响。",
        "",
        "## 边界",
        "",
        "- 两个模型都只用SY2014一年训练，seed相同，只有gate开关不同。",
        "- 不改reward、不改离散动作档位、不改PPO超参数。",
        "- 不涉及DAP<=30/DAP<=60分段配额（040_26源码未核实，本任务不动）。",
        "- 040_48的单臂结果因存在训练年份混淆，不作为有效结论，本任务取代它。",
        "- 这是对动作合法性mask的结构性修改，按项目预注册习惯应视为新的预注册任务。",
        "",
        "## 结论先说",
        "",
        f"- 训练步数：`{total_timesteps}`（两个arm相同）。",
        f"- control（gate开）灌溉序列：`{control['irrigation_sequence']}`",
        f"- treatment（gate关）灌溉序列：`{treatment['irrigation_sequence']}`",
        f"- 两者完全相同：`{same_sequence}`",
        f"- control终产量：`{control['final_grnwt_kg_ha']}` kg/ha；treatment终产量：`{treatment['final_grnwt_kg_ha']}` kg/ha",
        f"- control总灌溉/总施氮：`{control['total_irrigation_mm']}` mm / `{control['total_n_kg_ha']}` kg/ha；"
        f"treatment总灌溉/总施氮：`{treatment['total_irrigation_mm']}` mm / `{treatment['total_n_kg_ha']}` kg/ha",
        "",
        "## 附：040_48已作废的单臂结果（仅作背景，不可比）",
        "",
        f"- {CONTEXT_040_44_MULTIYEAR_BASELINE_SY2014['note']}",
        f"- 灌溉序列：`{CONTEXT_040_44_MULTIYEAR_BASELINE_SY2014['irrigation_sequence']}`；"
        f"终产量：`{CONTEXT_040_44_MULTIYEAR_BASELINE_SY2014['final_grnwt_kg_ha']}` kg/ha",
        "",
        "## 解释边界",
        "",
        "- 若control和treatment灌溉序列相同，说明在“只训练单年”这个前提下gate本身",
        "  不改变时机选择，嫌疑应转回policy本身或reward对时机的敏感度。",
        "- 若两者不同，说明gate至少在单年单seed设定下是有影响的成因之一；仍不能直接推广到",
        "  多年多seed的正式训练设置，需要另开预注册任务在多年多seed下重复验证。",
        "- control本身的表现（相对于040_44多年训练的checkpoint）也提供了额外信息：",
        "  如果control（单年训练+gate开）本身就明显弱于040_44多年训练的checkpoint，说明",
        "  单年训练本身确实会显著拖累效果——这正是040_48暴露出的混淆来源，在这里可以直接量化。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(total_timesteps: int) -> None:
    ensure_dirs()
    config = base04040.load_config()
    result = {
        "task": TASK,
        "mode": "dry_run",
        "station": STATION,
        "year": YEAR,
        "seed": SEED,
        "total_timesteps": total_timesteps,
        "arms": ARMS,
        "gate_dap_end_by_arm": GATE_DAP_END_BY_ARM,
        "reward": config["reward"],
        "action_safety": config["action_safety"],
        "discrete_actions": config["discrete_actions"],
        "prompt_exists": PROMPT.exists(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(total_timesteps: int) -> None:
    ensure_dirs()
    config = base04040.load_config()
    env_config = build_env_config_for_station(config)

    results_by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        results_by_arm[arm] = train_and_eval_arm(arm, config, env_config, total_timesteps)

    rows = [{**v, "station_code": STATION, "year": YEAR, "checkpoint_step": total_timesteps} for v in results_by_arm.values()]
    pd.DataFrame(rows).to_csv(OUT / "tables" / "040_49_two_arm_eval.csv", index=False, encoding="utf-8-sig")

    write_record(results_by_arm, total_timesteps)

    control = results_by_arm["control_gate_on"]
    treatment = results_by_arm["treatment_gate_off"]
    result = {
        "task": TASK,
        "station": STATION,
        "year": YEAR,
        "seed": SEED,
        "total_timesteps": total_timesteps,
        "results_by_arm": results_by_arm,
        "sequences_identical_between_arms": control["irrigation_sequence"].strip() == treatment["irrigation_sequence"].strip(),
        "supersedes": "040_48_sya_lowIC_dap90_gate_ablation_sy2014 (confounded single-arm result)",
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_49_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.timesteps)
    else:
        run_training(args.timesteps)


if __name__ == "__main__":
    main()
