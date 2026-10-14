"""Read-only same-DAP base-observation swap audit for SY checkpoints."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_134_sya_within_site_weather_amplification as base_audit

CONFIG = ROOT / "configs/136_00_sya_base_state_swap_readonly.json"
BASE_CFG = ROOT / "configs/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k.json"
PROMPT = ROOT / "prompts/2026-08-16_sya_base_state_swap_readonly.md"
OUT = ROOT / "benchmark_results/136_00_sya_base_state_swap_readonly_rerun1"
MODEL_ROOTS = {
    "ckpt2000": ROOT / "benchmark_results/111_00_sya_originIC_rolling7d_perfect_forecast_smoke2k_smoke2k/models/SYA",
    "ckpt10000": ROOT / "benchmark_results/126_00_sya_originIC_rolling7d_perfect_forecast_5k10k/models/SYA",
}
MODEL_NAMES = {
    "ckpt2000": "SYA_half_split_stress_aware_maskableppo_seed0_ckpt2000.zip",
    "ckpt10000": "SYA_half_split_stress_aware_maskableppo_seed0_ckpt10000.zip",
}
YEARS = list(range(2014, 2024))
SEED = 0


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build_pairs(states: pd.DataFrame) -> dict[int, int | None]:
    pairs: dict[int, int | None] = {}
    for _dap, group in states.groupby("dap"):
        ids = list(group.index)
        for pos, idx in enumerate(ids):
            target_year = int(group.loc[idx, "year"])
            candidates = ids[pos + 1 :] + ids[:pos]
            source = next(
                (candidate for candidate in candidates if int(group.loc[candidate, "year"]) != target_year),
                None,
            )
            pairs[int(idx)] = int(source) if source is not None else None
    return pairs


def main() -> int:
    base_audit.rolling.patch_contract()
    base_cfg = base_audit.forecast.read_json(BASE_CFG)
    preflight = base_audit.forecast.preflight(base_cfg, ROOT / base_cfg["prompt"])
    if not preflight["next_step_allowed"]:
        raise RuntimeError(preflight["issues"])
    model_paths = {label: root / MODEL_NAMES[label] for label, root in MODEL_ROOTS.items()}
    for path in model_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if OUT.exists():
        raise FileExistsError(f"refuse to overwrite {rel(OUT)}")
    OUT.mkdir(parents=True)
    shutil.copy2(CONFIG, OUT / CONFIG.name)
    shutil.copy2(BASE_CFG, OUT / BASE_CFG.name)
    shutil.copy2(PROMPT, OUT / PROMPT.name)

    forecast = base_audit.forecast
    old_loader = forecast.engine.load_config
    old_make_env = forecast.engine.base03222.base.make_env
    old_class = forecast.engine.base03222.base.StressAwareDiscreteWrapper
    old_root = forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT
    old_i = list(forecast.engine.BINARY_IRRIGATION_LEVELS)
    old_n = list(forecast.engine.BINARY_NITROGEN_LEVELS)
    old_station, old_sites, old_input = forecast.engine.STATION, list(forecast.engine.SITES), forecast.engine.LOWIC_INPUT_ROOT
    summaries: list[dict] = []
    try:
        forecast.engine.load_config = lambda: base_audit.tuned_load_config(old_loader)
        forecast.engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        forecast.engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        forecast.engine.STATION, forecast.engine.SITES = "SYA", ["SYA"]
        forecast.engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = forecast.engine.base04036.LateIrrigationReserveMaskWrapper
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
        runtime_config = forecast.engine.load_config()
        selection = forecast.engine.base03222.build_selection(forecast.engine.base03222.load_split())
        env_config = forecast.direct_ppo.build_env_config(runtime_config, selection)
        make_env = forecast.make_forecast_env_factory(old_make_env, base_cfg)
        for label, model_path in model_paths.items():
            model = MaskablePPO.load(str(model_path), device="cpu")
            states = base_audit.collect_states(model, make_env, runtime_config, env_config)
            pairs = build_pairs(states)
            states.to_json(OUT / f"136_00_{label}_state_trace.json", orient="records", force_ascii=False)
            probe = make_env(runtime_config, env_config, "SYA", YEARS[0], SEED, f"136_00_{label}_grid_probe", evaluation=True)
            try:
                grid = base_audit.action_map(probe)
            finally:
                probe.close()

            rows: list[dict] = []
            for idx, state in states.iterrows():
                target_base = np.asarray(state["base_obs"], dtype=np.float32)
                real_fc = np.asarray(state["real_forecast"], dtype=np.float32)
                mask = np.asarray(state["mask"], dtype=bool)
                real_action, _ = model.predict(np.concatenate([target_base, real_fc]), action_masks=mask, deterministic=True)
                real_action = int(np.asarray(real_action).reshape(-1)[0])
                source_idx = pairs.get(int(idx))
                if source_idx is None:
                    rows.append(
                        {
                            "target_row": int(idx),
                            "target_year": int(state["year"]),
                            "source_year": None,
                            "dap": int(state["dap"]),
                            "paired": False,
                            "real_action": real_action,
                            "swap_action": None,
                            "real_i": grid[real_action][0],
                            "real_n": grid[real_action][1],
                            "swap_i": None,
                            "swap_n": None,
                            "swap_changed": None,
                            "real_legal": bool(mask[real_action]),
                            "swap_legal": None,
                            "base_max_feature_delta": None,
                        }
                    )
                    continue
                source = states.loc[source_idx]
                source_base = np.asarray(source["base_obs"], dtype=np.float32)
                swap_action, _ = model.predict(np.concatenate([source_base, real_fc]), action_masks=mask, deterministic=True)
                swap_action = int(np.asarray(swap_action).reshape(-1)[0])
                rows.append(
                    {
                        "target_row": int(idx),
                        "target_year": int(state["year"]),
                        "source_year": int(source["year"]),
                        "dap": int(state["dap"]),
                        "paired": True,
                        "real_action": real_action,
                        "swap_action": swap_action,
                        "real_i": grid[real_action][0],
                        "real_n": grid[real_action][1],
                        "swap_i": grid[swap_action][0],
                        "swap_n": grid[swap_action][1],
                        "swap_changed": bool(real_action != swap_action),
                        "real_legal": bool(mask[real_action]),
                        "swap_legal": bool(mask[swap_action]),
                        "base_max_feature_delta": float(np.max(np.abs(source_base - target_base))),
                    }
                )
            decisions = pd.DataFrame(rows)
            decisions.to_csv(OUT / f"136_00_{label}_base_state_swap_decisions.csv", index=False, encoding="utf-8-sig")
            paired = decisions[decisions["paired"]]
            after = paired["dap"] > 1
            summaries.append(
                {
                    "checkpoint": label,
                    "all_states": int(len(decisions)),
                    "paired_states": int(len(paired)),
                    "unpaired_states": int((~decisions["paired"]).sum()),
                    "swap_action_change_rate": float(paired["swap_changed"].mean()) if len(paired) else None,
                    "swap_action_change_rate_after_dap1": float(paired.loc[after, "swap_changed"].mean()) if after.any() else None,
                    "changed_rows": int(paired["swap_changed"].sum()) if len(paired) else 0,
                    "all_real_actions_legal": bool(decisions["real_legal"].all()),
                    "all_swap_actions_legal": bool(paired["swap_legal"].all()) if len(paired) else False,
                    "mean_base_max_feature_delta": float(paired["base_max_feature_delta"].mean()) if len(paired) else None,
                    "max_base_feature_delta": float(paired["base_max_feature_delta"].max()) if len(paired) else None,
                }
            )

        summary = {
            "status": "completed_read_only_same_dap_base_state_swap",
            "models": {label: rel(path) for label, path in model_paths.items()},
            "config": rel(CONFIG),
            "preflight": preflight,
            "years": YEARS,
            "summaries": summaries,
            "counterfactual_actions_executed_in_dssat": False,
            "training_executed": False,
            "physically_consistent_rollout": False,
            "diagnostic_only": True,
            "next_step_long_training_allowed": False,
        }
        (OUT / "136_00_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(OUT / "136_00_summary.csv", index=False, encoding="utf-8-sig")
        doc = ROOT / "docs/2026-08-16_sya_base_state_swap_execution_rerun1.md"
        doc.write_text(
            "\n".join(
                [
                    "# 136_00 SY 基础土壤/作物状态替换只读诊断",
                    "",
                    "- 使用既有 SY seed=0 的 2K 和 10K checkpoint；未重新训练。",
                    f"- 每个 checkpoint 采集 `{len(YEARS)}` 个验证年份的状态，并在同一 DAP 内跨年份循环替换基础 observation。",
                    "- 目标状态的 DAP、真实未来 7 天天气和 action mask 保持不变；没有执行 DSSAT 反事实动作。",
                    "- 同 DAP 的基础状态替换可能不是物理一致的完整轨迹，因此只能判断策略是否读取状态差异，不能解释为产量结果。",
                    "",
                    "## 汇总",
                    "",
                    summary_df.to_markdown(index=False),
                    "",
                    "## 输出",
                    "",
                    f"- JSON：`{rel(OUT / '136_00_summary.json')}`",
                    f"- CSV：`{rel(OUT / '136_00_summary.csv')}`",
                    "",
                    "本轮不批准 25K/50K/75K/100K 长训练。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        forecast.engine.load_config = old_loader
        forecast.engine.base03222.base.make_env = old_make_env
        forecast.engine.base03222.base.StressAwareDiscreteWrapper = old_class
        forecast.ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root
        forecast.engine.BINARY_IRRIGATION_LEVELS = old_i
        forecast.engine.BINARY_NITROGEN_LEVELS = old_n
        forecast.engine.STATION, forecast.engine.SITES, forecast.engine.LOWIC_INPUT_ROOT = old_station, old_sites, old_input
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
