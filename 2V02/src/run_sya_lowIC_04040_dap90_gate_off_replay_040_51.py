"""040_51: frozen 040_40 checkpoint100000 policy, replayed with DAP90 gate on vs off.

Why this replaces the 040_49/040_50 retraining approach
----------------------------------------------------------
040_49/040_50 retrained from scratch on SY2014-only data, and found that
training-run-to-training-run variance (just from the random seed) was as
large as, or larger than, any effect attributable to the gate itself
(control_gate_on alone ranged from 75mm to 225mm total irrigation across 3
seeds). That noise makes single/few-seed retraining an expensive and
inconclusive way to isolate the gate's effect.

This task avoids retraining entirely. It takes the SAME frozen checkpoint
already used in 040_42/040_43/040_44/040_46/040_47 (040_40 checkpoint100000,
trained on many years of SYA data, gate ON throughout training) and simply
REPLAYS it at inference time under two conditions:
  1. gate_on  -- should exactly reproduce 040_44's recorded sequences per
     year (this is a built-in correctness check on this script's rollout
     logic itself, not just a research result)
  2. gate_off -- LATE_RESERVE_DAP_END monkeypatched to 0, so the DAP<=90 /
     195mm cap never blocks an action, while the policy network's weights
     are completely unchanged

Because MaskablePPO renormalizes its action probabilities over whatever the
current action mask says is legal, relaxing the mask CAN change which
action a frozen policy picks, even with no retraining -- so this is a
legitimate test of "does the already-trained policy want to deviate from
the fixed schedule but get blocked", without the retraining-variance
confound from 040_49/040_50.

Scope discipline
-----------------
  - Zero training. Pure inference/replay on a frozen checkpoint.
  - Does not change reward, discrete action levels, PPO hyperparameters,
    or the checkpoint itself.
  - Does NOT touch 040_26's DAP<=30/DAP<=60 caps (same caveat as
    040_48/040_49/040_50).
  - Still a structural mask change at evaluation time; treat conclusions
    here as diagnostic, not as grounds to alter 040_40's official recorded
    results.

Usage
-----
    python run_sya_lowIC_04040_dap90_gate_off_replay_040_51.py --dry-run
    python run_sya_lowIC_04040_dap90_gate_off_replay_040_51.py
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
from ppo_evaluate import latest_observation_dict  # same helper used in 040_46/47/48/49/50

TASK = "040_51_sya_lowIC_04040_dap90_gate_off_replay"
OUT = ROOT / "benchmark_results" / TASK
DOC = ROOT / "docs" / f"{TASK}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK}.md"

STATION = base04040.STATION
SITE = "SY"
SEED = base03222.SEED
CHECKPOINT_STEP = 100_000
VALIDATION_YEARS = list(range(2014, 2024))

GATE_CONDITIONS = {"gate_on": 90, "gate_off": 0}

# Sanity-check reference: 040_44's recorded sequence for SYA 2014 under the
# real (gate_on) checkpoint. If this script's gate_on replay does not match
# this, the rollout/logging logic has a bug and the gate_off numbers below
# should not be trusted until that is fixed.
SANITY_CHECK_2014 = {
    "irrigation_sequence": "DAP1:I45; DAP8:I30; DAP31:I45; DAP38:I30; DAP61:I45; DAP91:I45",
    "final_grnwt_kg_ha": 10210.0989,
    "source": "040_44_record",
}

MODEL_ZIP_OVERRIDE: Path | None = None


def ensure_dirs() -> None:
    for rel in ["tables", "logs", "configs"]:
        (OUT / rel).mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    if PROMPT.exists():
        shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)


def model_zip_path() -> Path:
    if MODEL_ZIP_OVERRIDE is not None:
        return MODEL_ZIP_OVERRIDE
    return (
        base04040.BASE_OUT
        / "models"
        / STATION
        / f"{STATION}_half_split_stress_aware_maskableppo_seed{SEED}_ckpt{CHECKPOINT_STEP}.zip"
    )


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
    selection["selection_reason"] = "040_51_dap90_gate_off_replay_readonly"
    return base03222.direct_ppo.build_env_config(config, selection)


def dap_for_logging(raw_dap: float) -> int:
    """Same convention as 040_49/040_50: reset's dap=0 logs as dap=1."""
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
    n_actions_that_would_be_illegal_under_gate_on = 0
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
            if dap <= 90 and total_irrigation > 195.0 + 1e-9:
                n_actions_that_would_be_illegal_under_gate_on += 1
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
        "n_actions_that_would_be_illegal_under_gate_on": n_actions_that_would_be_illegal_under_gate_on,
    }


def replay_all_years(model: Any, config: dict[str, Any], env_config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for condition, dap_end in GATE_CONDITIONS.items():
        apply_gate(dap_end)
        for year in VALIDATION_YEARS:
            run_tag = f"{STATION}_{year}_040_51_{condition}_replay"
            env = base04040.make_env_with_yield_guardrail(config, env_config, STATION, year, SEED, run_tag, evaluation=True)
            result = rollout(model, env)
            if hasattr(env, "close"):
                env.close()
            rows.append({"station_code": STATION, "year": year, "gate_condition": condition, **result})
    return pd.DataFrame(rows)


def summarize(all_rows: pd.DataFrame) -> dict[str, Any]:
    sanity = all_rows[(all_rows["year"].eq(2014)) & (all_rows["gate_condition"].eq("gate_on"))]
    sanity_ok = False
    sanity_detail = {}
    if not sanity.empty:
        row = sanity.iloc[0]
        sanity_ok = (
            row["irrigation_sequence"].strip() == SANITY_CHECK_2014["irrigation_sequence"].strip()
            and abs(float(row["final_grnwt_kg_ha"]) - SANITY_CHECK_2014["final_grnwt_kg_ha"]) < 1.0
        )
        sanity_detail = {
            "replayed_irrigation_sequence": row["irrigation_sequence"],
            "expected_irrigation_sequence": SANITY_CHECK_2014["irrigation_sequence"],
            "replayed_yield": float(row["final_grnwt_kg_ha"]),
            "expected_yield": SANITY_CHECK_2014["final_grnwt_kg_ha"],
        }

    per_year = []
    for year in VALIDATION_YEARS:
        on = all_rows[(all_rows["year"].eq(year)) & (all_rows["gate_condition"].eq("gate_on"))]
        off = all_rows[(all_rows["year"].eq(year)) & (all_rows["gate_condition"].eq("gate_off"))]
        if on.empty or off.empty:
            continue
        on_r, off_r = on.iloc[0], off.iloc[0]
        per_year.append(
            {
                "year": year,
                "sequences_identical": on_r["irrigation_sequence"].strip() == off_r["irrigation_sequence"].strip(),
                "gate_on_irrigation_mm": on_r["total_irrigation_mm"],
                "gate_off_irrigation_mm": off_r["total_irrigation_mm"],
                "irrigation_delta_off_minus_on": off_r["total_irrigation_mm"] - on_r["total_irrigation_mm"],
                "gate_on_yield": on_r["final_grnwt_kg_ha"],
                "gate_off_yield": off_r["final_grnwt_kg_ha"],
                "yield_delta_off_minus_on": off_r["final_grnwt_kg_ha"] - on_r["final_grnwt_kg_ha"],
                "n_actions_that_would_be_illegal_under_gate_on": off_r["n_actions_that_would_be_illegal_under_gate_on"],
            }
        )
    per_year_df = pd.DataFrame(per_year)
    n_years = len(per_year_df)
    n_identical = int(per_year_df["sequences_identical"].sum()) if n_years else 0
    n_wanted_to_deviate = int((per_year_df["n_actions_that_would_be_illegal_under_gate_on"] > 0).sum()) if n_years else 0
    return {
        "sanity_check_gate_on_matches_040_44": sanity_ok,
        "sanity_check_detail": sanity_detail,
        "per_year": per_year_df,
        "n_years": n_years,
        "n_years_sequences_identical": n_identical,
        "n_years_policy_would_exceed_195mm_by_dap90_if_allowed": n_wanted_to_deviate,
    }


def write_record(summary: dict[str, Any]) -> None:
    per_year_df: pd.DataFrame = summary["per_year"]
    lines = [
        f"# {TASK} 记录",
        "",
        "## 任务目的",
        "",
        "不重新训练，直接回放已经用多年数据训练好的040_40 checkpoint100000冻结策略，",
        "分别在gate开、gate关两种条件下跑全部10个验证年份，检验这个已经训练成熟的",
        "policy本身是否“想”偏离固定时间表，只是被这条约束挡住了，从而避开040_49/040_50",
        "重新训练带来的种子间方差问题。",
        "",
        "## 边界",
        "",
        "- 零训练，纯推理回放，checkpoint权重完全不变。",
        "- 不改reward、不改离散动作档位、不改PPO超参数。",
        "- 不涉及DAP<=30/DAP<=60分段配额。",
        "- 属于评估时的结构性mask改动，结论仅作诊断用，不改变040_40的正式记录结果。",
        "",
        "## 自检（重要，先看这个）",
        "",
        f"- gate开回放是否与040_44记录的SYA 2014结果完全一致：`{summary['sanity_check_gate_on_matches_040_44']}`",
        f"- 细节：`{summary['sanity_check_detail']}`",
        "- 若上面是False，说明本脚本的回放/记录逻辑有偏差，下面gate_off的结果暂不可信，",
        "  需要先排查这个自检不通过的原因。",
        "",
        "## 结论先说",
        "",
        f"- 验证年份数：`{summary['n_years']}`。",
        f"- gate开/关两种条件下灌溉序列完全相同的年份数：`{summary['n_years_sequences_identical']}/{summary['n_years']}`。",
        f"- 若gate开着，policy在gate关时的动作会有多少年触发“DAP<=90累计超195mm”："
        f"`{summary['n_years_policy_would_exceed_195mm_by_dap90_if_allowed']}/{summary['n_years']}`年"
        "（这个数只在gate_off回放里统计，反映policy在没有约束时是否主动想在90天内多用水）。",
        "",
        "## 逐年对比表",
        "",
        per_year_df.to_markdown(index=False) if not per_year_df.empty else "无记录。",
        "",
        "## 解释边界",
        "",
        "- 这是同一套冻结权重在不同mask下的推理结果，没有重新训练带来的方差问题，",
        "  比040_49/040_50更适合回答“policy本身是否被这条约束真正束缚住”这个问题。",
        "- 但MaskablePPO是在训练时的mask分布下学出来的概率分布；把mask在评估时换掉，",
        "  policy面对的“合法动作集合”和训练时不一样，其行为已经离开了它被训练优化的",
        "  分布范围，属于分布外（out-of-distribution）评估，结果能说明“网络内部的偏好”，",
        "  但不能直接当作“如果一开始就用这个mask训练，policy会学成什么样”的答案——",
        "  后者仍然需要长期多年训练+gate关的正式实验才能回答。",
        "- 若“若gate开着会想超195mm”的年份数很低，说明policy即使拿到自由也不倾向于在",
        "  90天内提前用完预算，那么040_44观察到的DAP91固定点更可能只是policy自己的",
        "  时机偏好，而不是被约束逼出来的假象。",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run() -> None:
    ensure_dirs()
    config = base04040.load_config()
    zip_path = model_zip_path()
    result = {
        "task": TASK,
        "mode": "dry_run",
        "station": STATION,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_exists": zip_path.exists(),
        "checkpoint_path": str(zip_path),
        "validation_years": VALIDATION_YEARS,
        "gate_conditions": GATE_CONDITIONS,
        "sanity_check_reference": SANITY_CHECK_2014,
        "prompt_exists": PROMPT.exists(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main_run() -> None:
    ensure_dirs()
    from sb3_contrib import MaskablePPO

    zip_path = model_zip_path()
    if not zip_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {zip_path}. Set MODEL_ZIP_OVERRIDE if the path differs.")
    model = MaskablePPO.load(str(zip_path))

    config = base04040.load_config()
    env_config = build_env_config_for_station(config)

    all_rows = replay_all_years(model, config, env_config)
    all_rows.to_csv(OUT / "tables" / "040_51_gate_on_off_replay_by_year.csv", index=False, encoding="utf-8-sig")

    summary = summarize(all_rows)
    summary["per_year"].to_csv(OUT / "tables" / "040_51_per_year_comparison.csv", index=False, encoding="utf-8-sig")

    write_record(summary)

    result = {
        "task": TASK,
        "checkpoint": zip_path.relative_to(ROOT).as_posix() if zip_path.is_relative_to(ROOT) else str(zip_path),
        "sanity_check_passed": summary["sanity_check_gate_on_matches_040_44"],
        "n_years": summary["n_years"],
        "n_years_sequences_identical": summary["n_years_sequences_identical"],
        "n_years_policy_would_exceed_195mm_by_dap90_if_allowed": summary["n_years_policy_would_exceed_195mm_by_dap90_if_allowed"],
        "record_md": DOC.relative_to(ROOT).as_posix(),
    }
    (OUT / "040_51_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        main_run()


if __name__ == "__main__":
    main()
