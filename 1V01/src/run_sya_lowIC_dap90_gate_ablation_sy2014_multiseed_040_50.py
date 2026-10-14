"""040_50: SY2014-only DAP90 gate ablation, controlled, multi-seed version.

Why this extends 040_49
------------------------
040_49 (single seed=0) found that turning the gate OFF made the policy
irrigate less (75mm vs 225mm), stress more (max_swfac 0.95 vs 0.25), and
yield much less (7636 vs 10895 kg/ha) -- the opposite direction from the
"gate is holding the policy back" hypothesis. But that is one training run
per arm; PPO training variance alone could produce this. This task repeats
the same control_gate_on / treatment_gate_off comparison across several
seeds, all still restricted to SY2014-only training, to see whether the
gate-off degradation is a stable, repeatable effect or a single-seed fluke.

Design (unchanged from 040_49 except for the seed loop)
---------------------------------------------------------
For each seed in SEEDS, train both arms (identical reward, discrete action
levels, PPO hyperparameters, training data = SY2014 only), differing ONLY in
whether LATE_RESERVE_DAP_END is 90 (gate on, control) or 0 (gate off,
treatment). Evaluate both, then aggregate across seeds.

Scope discipline
-----------------
  - len(SEEDS) x 2 new training runs, all single station-year (SYA 2014).
  - Does NOT change reward, discrete action levels, PPO hyperparameters.
  - Does NOT touch 040_26's DAP<=30/DAP<=60 caps (same caveat as 040_48/040_49).
  - Structural mask change -- same re-authorization discipline as 027_04
    applies before treating this as an official result.
  - Cost note: default SEEDS has 3 seeds x 2 arms x 100k timesteps = 600k
    timesteps total. Adjust --seeds / --timesteps if that's too expensive
    for a first pass; you can always widen it later once this shows signal.

Usage
-----
    python run_sya_lowIC_dap90_gate_ablation_sy2014_multiseed_040_50.py --dry-run
    python run_sya_lowIC_dap90_gate_ablation_sy2014_multiseed_040_50.py --seeds 0 1 2 --timesteps 100000
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
from ppo_evaluate import latest_observation_dict  # same helper used in 040_46/47/48/49

TASK = "040_50_sya_lowIC_dap90_gate_ablation_sy2014_multiseed"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = base04040.STATION
SITE = "SY"
YEAR = 2014
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_TOTAL_TIMESTEPS = 100_000

ARMS = ["control_gate_on", "treatment_gate_off"]
GATE_DAP_END_BY_ARM = {
    "control_gate_on": 90,
    "treatment_gate_off": 0,
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
    split = pd.read_csv(base03222.SPLIT_CSV, keep_default_na=False)
    split["year"] = pd.to_numeric(split["year"], errors="coerce").astype(int)
    split = split[split["station_code"].eq(STATION)].sort_values("year").reset_index(drop=True)
    pool = pd.read_csv(base03222.POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    selection = pool.merge(split[["station_code", "year", "split"]], on=["station_code", "year"], how="inner")
    selection["selected_for_train"] = selection["split"].eq("train")
    selection["selected_for_eval"] = True
    selection["selection_reason"] = "040_50_dap90_gate_ablation_sy2014_multiseed"
    return base03222.direct_ppo.build_env_config(config, selection)


def dap_for_logging(raw_dap: float) -> int:
    """Same convention as 040_49: reset's dap=0 logs as dap=1 (planting day)."""
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


def train_and_eval_arm(arm: str, seed: int, config: dict[str, Any], env_config: dict[str, Any], total_timesteps: int) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    gate_state = apply_gate(GATE_DAP_END_BY_ARM[arm])

    run_tag = f"{STATION}_{YEAR}_040_50_{arm}_seed{seed}_train"
    train_env = base04040.make_env_with_yield_guardrail(config, env_config, STATION, YEAR, seed, run_tag, evaluation=False)
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=seed,
        tensorboard_log=str(OUT / "logs" / "tensorboard" / arm / f"seed{seed}"),
        **base03222.base.ppo_kwargs(config),
    )
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True, progress_bar=False)
    if hasattr(train_env, "close"):
        train_env.close()

    model_path = OUT / "models" / f"{STATION}_{YEAR}_{arm}_seed{seed}_ckpt{total_timesteps}.zip"
    model.save(str(model_path))

    eval_run_tag = f"{STATION}_{YEAR}_040_50_{arm}_seed{seed}_eval"
    eval_env = base04040.make_env_with_yield_guardrail(config, env_config, STATION, YEAR, seed, eval_run_tag, evaluation=True)
    eval_result = rollout(model, eval_env)
    if hasattr(eval_env, "close"):
        eval_env.close()

    return {
        "arm": arm,
        "seed": seed,
        "gate_state": gate_state,
        "model_path": model_path.relative_to(ROOT).as_posix(),
        **eval_result,
    }


def aggregate(all_rows: pd.DataFrame) -> dict[str, Any]:
    pivot = all_rows.pivot(index="seed", columns="arm", values=[
        "irrigation_sequence", "total_irrigation_mm", "total_n_kg_ha", "final_grnwt_kg_ha", "max_swfac"
    ])
    per_seed = []
    for seed in all_rows["seed"].unique():
        c = all_rows[(all_rows["seed"].eq(seed)) & (all_rows["arm"].eq("control_gate_on"))].iloc[0]
        t = all_rows[(all_rows["seed"].eq(seed)) & (all_rows["arm"].eq("treatment_gate_off"))].iloc[0]
        per_seed.append(
            {
                "seed": int(seed),
                "sequences_identical": c["irrigation_sequence"].strip() == t["irrigation_sequence"].strip(),
                "control_irrigation_mm": c["total_irrigation_mm"],
                "treatment_irrigation_mm": t["total_irrigation_mm"],
                "irrigation_mm_delta_treatment_minus_control": t["total_irrigation_mm"] - c["total_irrigation_mm"],
                "control_yield_kg_ha": c["final_grnwt_kg_ha"],
                "treatment_yield_kg_ha": t["final_grnwt_kg_ha"],
                "yield_delta_treatment_minus_control": t["final_grnwt_kg_ha"] - c["final_grnwt_kg_ha"],
                "control_max_swfac": c["max_swfac"],
                "treatment_max_swfac": t["max_swfac"],
            }
        )
    per_seed_df = pd.DataFrame(per_seed)
    n_seeds = len(per_seed_df)
    n_identical = int(per_seed_df["sequences_identical"].sum())
    n_treatment_worse_yield = int((per_seed_df["yield_delta_treatment_minus_control"] < 0).sum())
    n_treatment_less_irrigation = int((per_seed_df["irrigation_mm_delta_treatment_minus_control"] < 0).sum())
    summary = {
        "n_seeds": n_seeds,
        "n_seeds_sequences_identical": n_identical,
        "n_seeds_treatment_yield_worse": n_treatment_worse_yield,
        "n_seeds_treatment_irrigation_less": n_treatment_less_irrigation,
        "mean_yield_delta_treatment_minus_control": float(per_seed_df["yield_delta_treatment_minus_control"].mean()),
        "std_yield_delta_treatment_minus_control": float(per_seed_df["yield_delta_treatment_minus_control"].std()) if n_seeds > 1 else None,
        "mean_irrigation_mm_delta_treatment_minus_control": float(per_seed_df["irrigation_mm_delta_treatment_minus_control"].mean()),
        "std_irrigation_mm_delta_treatment_minus_control": float(per_seed_df["irrigation_mm_delta_treatment_minus_control"].std()) if n_seeds > 1 else None,
    }
    return {"per_seed": per_seed_df, "summary": summary}


def write_record(all_rows: pd.DataFrame, agg: dict[str, Any], total_timesteps: int, seeds: list[int]) -> None:
    per_seed_df: pd.DataFrame = agg["per_seed"]
    summary = agg["summary"]
    lines = [
        f"# {TASK} 记录",
        "",
        "## 任务目的",
        "",
        "040_49只用了seed=0，一次训练看到treatment(gate关)比control(gate开)浇水更少、",
        "更缺水、产量更低，但单个seed无法排除PPO训练本身的随机波动。本任务在同样的",
        "SY2014单年、双臂设计下，重复多个seed，检验这个方向是否稳定出现。",
        "",
        "## 边界",
        "",
        f"- seeds：`{seeds}`；每个seed训练control/treatment各一次，均只用SY2014一年数据。",
        "- 不改reward、不改离散动作档位、不改PPO超参数，唯一变量是gate开关。",
        "- 不涉及DAP<=30/DAP<=60分段配额。",
        "- 结构性mask改动，按项目预注册习惯需要与027_04同等的重新授权流程。",
        "",
        "## 结论先说",
        "",
        f"- 训练步数：`{total_timesteps}`（所有seed/arm相同）。",
        f"- 有效seed数：`{summary['n_seeds']}`。",
        f"- 灌溉序列在control/treatment之间完全相同的seed数：`{summary['n_seeds_sequences_identical']}/{summary['n_seeds']}`。",
        f"- treatment（gate关）产量比control差的seed数：`{summary['n_seeds_treatment_yield_worse']}/{summary['n_seeds']}`。",
        f"- treatment（gate关）总灌溉量比control少的seed数：`{summary['n_seeds_treatment_irrigation_less']}/{summary['n_seeds']}`。",
        f"- 产量差值（treatment-control）均值：`{summary['mean_yield_delta_treatment_minus_control']:.2f}` kg/ha"
        + (f"，标准差：`{summary['std_yield_delta_treatment_minus_control']:.2f}`" if summary["std_yield_delta_treatment_minus_control"] is not None else "（仅1个seed，无标准差）"),
        f"- 灌溉量差值（treatment-control）均值：`{summary['mean_irrigation_mm_delta_treatment_minus_control']:.2f}` mm"
        + (f"，标准差：`{summary['std_irrigation_mm_delta_treatment_minus_control']:.2f}`" if summary["std_irrigation_mm_delta_treatment_minus_control"] is not None else "（仅1个seed，无标准差）"),
        "",
        "## 逐seed对比表",
        "",
        per_seed_df.to_markdown(index=False),
        "",
        "## 解释边界",
        "",
        "- 若多数/全部seed都显示treatment（gate关）灌溉更少、产量更差，说明这条gate在",
        "  当前奖励设计下实际起着“帮policy学会预留水量”的作用，关掉它不是解放了policy，",
        "  反而让它在单年训练下更容易学出偷懒策略；这是有价值的负面结果，应按项目惯例保留。",
        "- 若seed之间结果不稳定（有的相同有的不同、产量差值方向不一致），说明单年训练本身",
        "  方差就很大，gate的净效应暂时无法从这个规模的实验中分离出来，需要更多seed或",
        "  换成多年训练设置才能下结论。",
        "- 本任务仍只在SY2014单年上做，任何结论推广到其它年份/站点或多年训练设置前，",
        "  都需要另开预注册任务验证。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(seeds: list[int], total_timesteps: int) -> None:
    ensure_dirs()
    config = base04040.load_config()
    result = {
        "task": TASK,
        "mode": "dry_run",
        "station": STATION,
        "year": YEAR,
        "seeds": seeds,
        "total_timesteps": total_timesteps,
        "arms": ARMS,
        "gate_dap_end_by_arm": GATE_DAP_END_BY_ARM,
        "n_training_runs": len(seeds) * len(ARMS),
        "reward": config["reward"],
        "action_safety": config["action_safety"],
        "discrete_actions": config["discrete_actions"],
        "prompt_exists": PROMPT.exists(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_training(seeds: list[int], total_timesteps: int) -> None:
    ensure_dirs()
    config = base04040.load_config()
    env_config = build_env_config_for_station(config)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            rows.append(train_and_eval_arm(arm, seed, config, env_config, total_timesteps))

    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(OUT / "tables" / "040_50_multiseed_two_arm_eval.csv", index=False, encoding="utf-8-sig")

    agg = aggregate(all_rows)
    agg["per_seed"].to_csv(OUT / "tables" / "040_50_per_seed_comparison.csv", index=False, encoding="utf-8-sig")

    write_record(all_rows, agg, total_timesteps, seeds)

    result = {
        "task": TASK,
        "station": STATION,
        "year": YEAR,
        "seeds": seeds,
        "total_timesteps": total_timesteps,
        "summary": agg["summary"],
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_50_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.seeds, args.timesteps)
    else:
        run_training(args.seeds, args.timesteps)


if __name__ == "__main__":
    main()
