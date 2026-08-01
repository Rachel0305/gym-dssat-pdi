"""040_48: SY2014-only ablation of the DAP<=90 late-irrigation-reserve gate.

Motivation
----------
040_44 found that the 040_40 checkpoint100000 policy uses the exact same
irrigation DAP sequence in all 10 validation years (DAP1/8/31/38/61/91),
which pins its last irrigation event right at DAP91 -- exactly where
040_36's added constraint unlocks the reserved water. 040_47 then showed the
policy DOES receive year-varying weather/stress signal in its observation
(srad, dtt, sw, rtdep, nstres all carry cross-year variation), which argues
against "the policy can't see the weather" and toward "the mask geometry
itself makes near-identical timing the reliably optimal choice regardless
of weather."

This task tests that second hypothesis directly, on a single station-year
(SYA 2014) to keep cost low, per the discussion:
  - Keep everything else identical to 040_40: same reward (incl. terminal
    yield guardrail and swfac guardrail), same discrete action levels
    ([0,30,45] mm irrigation / [0,80,120] kg/ha N), same PPO hyperparameters,
    same seed, same total timesteps.
  - Ablate ONLY the constraint 040_36 itself adds:
        if DAP <= 90: cumulative_irrigation_after_action <= 195 mm
    by monkeypatching the two module-level constants that
    LateIrrigationReserveMaskWrapper._action_is_legal_without_clipping reads
    at call time (LATE_RESERVE_DAP_END, PRE_LATE_CUM_IRRIGATION_CAP) on the
    040_36 module object itself. This works because that method looks these
    names up in its defining module's globals at call time, the same
    mechanism this codebase already relies on elsewhere (e.g. 040_40's own
    patch_base_module reassigning base03222.load_config /
    base03222.base.make_env before training).

Scope discipline
-----------------
  - Single station-year (SYA 2014), one new training run only.
  - Does NOT change reward, discrete action levels, PPO hyperparameters,
    or seed.
  - Does NOT touch 040_26/040_28's earlier DAP<=30 / DAP<=60 windowed caps
    by default (ABLATE_EARLY_MID_CAPS_TOO=False) -- this script's author did
    not have 040_26's source available and cannot verify its internals from
    outside; see the flag below and verify attribute names against your
    actual 040_26 file before turning it on.
  - This changes action *legality* (the mask), which per your project's
    preregistration discipline is a structural change, not just an
    evaluation -- treat it like 027_04 needed re-authorization before it
    counted as an official result, not as a routine checkpoint eval.

Usage
-----
    python run_sya_lowIC_dap90_gate_ablation_sy2014_040_48.py --dry-run
    python run_sya_lowIC_dap90_gate_ablation_sy2014_040_48.py --timesteps 100000
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
from ppo_evaluate import latest_observation_dict  # same helper used in 040_46/040_47

TASK = "040_48_sya_lowIC_dap90_gate_ablation_sy2014"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = base04040.STATION
SITE = "SY"
YEAR = 2014
SEED = base03222.SEED
DEFAULT_TOTAL_TIMESTEPS = 100_000

# The ablation: dap<=0 is never true for dap>=1, so the DAP90 gate branch in
# LateIrrigationReserveMaskWrapper._action_is_legal_without_clipping never
# fires. PRE_LATE_CUM_IRRIGATION_CAP becomes irrelevant once the gate is off,
# left untouched for traceability in the record.
LATE_RESERVE_DAP_END_ABLATED = 0

# Off by default -- flip only after you have confirmed against your actual
# 040_26 source that these attribute names and semantics are correct.
ABLATE_EARLY_MID_CAPS_TOO = False

# Reference point from 040_44 (SYA 2014, 040_40 checkpoint100000, gate ON).
BASELINE_040_44_SY2014 = {
    "irrigation_sequence": "DAP1:I45; DAP8:I30; DAP31:I45; DAP38:I30; DAP61:I45; DAP91:I45",
    "nitrogen_sequence": "DAP43:N120; DAP50:N120",
    "total_irrigation_mm": 240.0,
    "total_n_kg_ha": 240.0,
    "final_grnwt_kg_ha": 10210.0989,
}


def ensure_dirs() -> None:
    for rel in ["models", "tables", "logs", "configs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def apply_ablation() -> dict[str, Any]:
    before = {
        "LATE_RESERVE_DAP_END": base04036.LATE_RESERVE_DAP_END,
        "PRE_LATE_CUM_IRRIGATION_CAP": base04036.PRE_LATE_CUM_IRRIGATION_CAP,
    }
    base04036.LATE_RESERVE_DAP_END = LATE_RESERVE_DAP_END_ABLATED
    after: dict[str, Any] = {
        "LATE_RESERVE_DAP_END": base04036.LATE_RESERVE_DAP_END,
        "PRE_LATE_CUM_IRRIGATION_CAP": base04036.PRE_LATE_CUM_IRRIGATION_CAP,
    }
    if ABLATE_EARLY_MID_CAPS_TOO:
        import run_sya_lowIC_ppo_i240_staged_reserve_040_26 as base04026

        required = ["EARLY_CUM_IRRIGATION_CAP", "MID_CUM_IRRIGATION_CAP", "SEASON_IRRIGATION_CAP"]
        missing = [a for a in required if not hasattr(base04026, a)]
        if missing:
            raise AttributeError(
                f"ABLATE_EARLY_MID_CAPS_TOO=True but base04026 is missing {missing}; "
                "verify attribute names against your actual 040_26 source before enabling this."
            )
        before.update(
            {
                "EARLY_CUM_IRRIGATION_CAP": base04026.EARLY_CUM_IRRIGATION_CAP,
                "MID_CUM_IRRIGATION_CAP": base04026.MID_CUM_IRRIGATION_CAP,
            }
        )
        base04026.EARLY_CUM_IRRIGATION_CAP = base04026.SEASON_IRRIGATION_CAP
        base04026.MID_CUM_IRRIGATION_CAP = base04026.SEASON_IRRIGATION_CAP
        after.update(
            {
                "EARLY_CUM_IRRIGATION_CAP": base04026.EARLY_CUM_IRRIGATION_CAP,
                "MID_CUM_IRRIGATION_CAP": base04026.MID_CUM_IRRIGATION_CAP,
            }
        )
    return {"before": before, "after": after, "early_mid_caps_ablated": ABLATE_EARLY_MID_CAPS_TOO}


def build_env_config_for_station(config: dict[str, Any]) -> dict[str, Any]:
    """Same read-only selection -> env_config construction used in 040_47,
    scoped to STATION, without mutating base03222's shared module state."""
    split = pd.read_csv(base03222.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    split = split[split["station_code"].eq(STATION)].sort_values("year").reset_index(drop=True)
    pool = pd.read_csv(base03222.POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selection = pool.merge(split[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selection["selected_for_train"] = selection["split"].eq("train")
    selection["selected_for_eval"] = True
    selection["selection_reason"] = "040_48_dap90_gate_ablation_sy2014_only"
    return base03222.direct_ppo.build_env_config(config, selection)


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
        dap = int(round(float(named.get("dap", 0) or 0)))
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


def write_record(ablation_state: dict[str, Any], eval_result: dict[str, Any], model_path: Path, total_timesteps: int) -> None:
    same_sequence = eval_result["irrigation_sequence"].strip() == BASELINE_040_44_SY2014["irrigation_sequence"].strip()
    lines = [
        f"# {TASK} 记录",
        "",
        "## 任务目的",
        "",
        "检验040_44发现的“灌溉序列跨年固定”是否由040_36加的DAP<=90累计195mm硬约束",
        "（几何配额）导致，而不是policy学不会响应天气。只消融这一条约束，其余奖励/",
        "档位/超参数/seed全部与040_40保持一致，只在SY2014单年上重新训练一次。",
        "",
        "## 边界",
        "",
        "- 只改动作合法性（mask），不改reward、不改离散动作档位、不改PPO超参数、不改seed。",
        f"- 是否同时消融DAP<=30/DAP<=60分段配额：`{ablation_state['early_mid_caps_ablated']}`"
        + ("（默认关闭，未验证040_26源码前不建议打开）" if not ablation_state["early_mid_caps_ablated"] else ""),
        "- 这是对动作合法性mask的结构性修改，按项目预注册习惯，应视为新的预注册任务，",
        "  而不是常规checkpoint评估（参考027_04当时的重新授权流程）。",
        "",
        "## 消融前后的模块常量对比",
        "",
        f"- 消融前：`{ablation_state['before']}`",
        f"- 消融后：`{ablation_state['after']}`",
        "",
        "## 结论先说",
        "",
        f"- 训练步数：`{total_timesteps}`；模型：`{model_path.relative_to(ROOT).as_posix()}`。",
        f"- 消融后灌溉序列：`{eval_result['irrigation_sequence']}`",
        f"- 040_44基线（gate未消融）灌溉序列：`{BASELINE_040_44_SY2014['irrigation_sequence']}`",
        f"- 两者完全相同：`{same_sequence}`",
        f"- 消融后终产量：`{eval_result['final_grnwt_kg_ha']}` kg/ha"
        f"（基线：`{BASELINE_040_44_SY2014['final_grnwt_kg_ha']}` kg/ha）",
        f"- 消融后总灌溉/总施氮：`{eval_result['total_irrigation_mm']}` mm / `{eval_result['total_n_kg_ha']}` kg/ha"
        f"（基线：`{BASELINE_040_44_SY2014['total_irrigation_mm']}` mm / `{BASELINE_040_44_SY2014['total_n_kg_ha']}` kg/ha）",
        "",
        "## 解释边界",
        "",
        "- 若消融后序列仍与基线完全相同，说明固定时间表不是这一条mask几何造成的，",
        "  嫌疑应转回policy本身或reward对时机不敏感；若序列发生变化，说明该约束至少是",
        "  部分成因，但因为只训练了单年单seed，还不能断言在其它年份/站点上同样成立。",
        "- 本任务只是SY2014单年单seed的小规模检验，不能替代未来的多年多seed正式实验，",
        "  只用于决定该往哪个方向投入更大规模的预注册实验。",
        "- 若要据此得出正式结论，需要按预注册流程另开多年/多seed任务重复本检验。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(total_timesteps: int) -> None:
    ensure_dirs()
    ablation_state = apply_ablation()
    config = base04040.load_config()
    result = {
        "task": TASK,
        "mode": "dry_run",
        "station": STATION,
        "year": YEAR,
        "seed": SEED,
        "total_timesteps": total_timesteps,
        "ablation_state": ablation_state,
        "reward": config["reward"],
        "action_safety": config["action_safety"],
        "discrete_actions": config["discrete_actions"],
        "prompt_exists": PROMPT.exists(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(total_timesteps: int) -> None:
    ensure_dirs()
    from sb3_contrib import MaskablePPO

    ablation_state = apply_ablation()
    config = base04040.load_config()
    env_config = build_env_config_for_station(config)

    run_tag = f"{STATION}_{YEAR}_040_48_dap90_gate_ablation_train"
    train_env = base04040.make_env_with_yield_guardrail(config, env_config, STATION, YEAR, SEED, run_tag, evaluation=False)
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=SEED,
        tensorboard_log=str(OUT / "logs" / "tensorboard"),
        **base03222.base.ppo_kwargs(config),
    )
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True, progress_bar=False)
    if hasattr(train_env, "close"):
        train_env.close()

    model_path = OUT / "models" / f"{STATION}_{YEAR}_dap90_gate_ablation_seed{SEED}_ckpt{total_timesteps}.zip"
    model.save(str(model_path))

    eval_run_tag = f"{STATION}_{YEAR}_040_48_dap90_gate_ablation_eval"
    eval_env = base04040.make_env_with_yield_guardrail(config, env_config, STATION, YEAR, SEED, eval_run_tag, evaluation=True)
    eval_result = rollout(model, eval_env)
    if hasattr(eval_env, "close"):
        eval_env.close()

    pd.DataFrame([{**eval_result, "station_code": STATION, "year": YEAR, "checkpoint_step": total_timesteps}]).to_csv(
        OUT / "tables" / "040_48_ablated_sy2014_eval.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([{**BASELINE_040_44_SY2014, "station_code": STATION, "year": YEAR, "source": "040_44_record"}]).to_csv(
        OUT / "tables" / "040_48_baseline_reference_sy2014.csv", index=False, encoding="utf-8-sig"
    )

    write_record(ablation_state, eval_result, model_path, total_timesteps)

    result = {
        "task": TASK,
        "station": STATION,
        "year": YEAR,
        "seed": SEED,
        "total_timesteps": total_timesteps,
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "ablation_state": ablation_state,
        "eval_result": eval_result,
        "baseline_reference": BASELINE_040_44_SY2014,
        "sequences_identical_to_baseline": eval_result["irrigation_sequence"].strip() == BASELINE_040_44_SY2014["irrigation_sequence"].strip(),
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_48_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
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
