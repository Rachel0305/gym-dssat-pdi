"""106_00 read-only FQ model-facing weather/action sensitivity audit.

This is deliberately isolated from all training and replay runners.  It replays
existing checkpoints through the same environment only to capture observations,
weather metadata, masks, and deterministic actions; it does not produce paper
performance claims or run five-scenario DSSAT comparisons.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as reserve

CONFIG = ROOT / "configs/106_00_fqa_weather_sensitivity_audit.json"
BASE_YAML = ROOT / "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml"
OUT = ROOT / "benchmark_results/106_00_fqa_weather_sensitivity_audit"
DOC = ROOT / "docs/106_00_fqa_weather_sensitivity_audit_record.md"
STATION = "FQA"
SITE = "FQ"
SEED1_LABEL = "seed1_ckpt2000"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def scalar(value: Any, default: float = np.nan) -> float:
    return direct_ppo.scalar(value, default=default)


def obs_names(env: Any, obs: Any) -> list[str]:
    flat = np.asarray(obs, dtype=float).reshape(-1)
    names = [str(x) for x in list(getattr(env.unwrapped, "observation_variables", []) or [])]
    if len(names) != len(flat):
        names = [f"obs_{i}" for i in range(len(flat))]
    return names


def masked_probs(model: Any, obs: Any, mask: np.ndarray) -> np.ndarray:
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor, action_masks=np.asarray(mask, dtype=bool).reshape(1, -1))
        raw = getattr(getattr(dist, "distribution", dist), "probs", None)
        if raw is None:
            return np.full(len(mask), np.nan)
        return raw.detach().cpu().numpy().reshape(-1).astype(float)
    except Exception:
        return np.full(len(mask), np.nan)


def make_env_config(config: dict[str, Any]) -> dict[str, Any]:
    split = batch.load_split()
    split = split[split["station_code"].eq(STATION)].copy()
    selection = batch.build_selection(split)
    selection = selection[selection["year"].astype(int).isin(config["validation_years"])].copy()
    cfg = direct_ppo.load_yaml(BASE_YAML)
    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = 1
    cfg["paths"]["output_root"] = rel(OUT)
    cfg["ppo"]["ent_coef"] = 0.02
    cfg["reward"]["nitrogen_cost"] = 2.5
    cfg["discrete_actions"]["irrigation_levels"] = [0.0, 15.0, 30.0, 45.0]
    cfg["discrete_actions"]["nitrogen_levels"] = [0.0, 40.0, 80.0, 120.0]
    return cfg, direct_ppo.build_env_config(cfg, selection)


def weather_row(weather: pd.DataFrame, year: int, date: pd.Timestamp) -> dict[str, Any]:
    rows = weather[(weather["station_code"].eq(STATION)) & (weather["date"].eq(date))]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def audit_model(model_cfg: dict[str, Any], config: dict[str, Any], env_config: dict[str, Any], weather: pd.DataFrame, grid: list[tuple[float, float]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from sb3_contrib.common.maskable.utils import get_action_masks
    from sb3_contrib import MaskablePPO

    model_path = ROOT / model_cfg["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = MaskablePPO.load(str(model_path), device="cpu")
    decisions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    for year in config["validation_years"]:
        env = batch.base.make_env(config, env_config, STATION, int(year), int(model_cfg["seed"]), f"FQA_{year}_106_weather_audit", evaluation=True)
        try:
            obs, info = env.reset()
            planting = pd.Timestamp(direct_ppo.find_year(env_config, STATION, int(year))["planting_date"])
            done = False
            step = 0
            signature: list[str] = []
            captured: set[int] = set()
            while not done and step < int(config["runtime"]["max_steps"]):
                latest = batch.base.latest_observation_dict(env, obs, info)
                dap_raw = scalar(latest.get("dap", step + 1), step + 1)
                dap = int(round(dap_raw)) if np.isfinite(dap_raw) and dap_raw > 0 else step + 1
                mask = np.asarray(get_action_masks(env), dtype=bool).reshape(-1)
                action_raw, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(np.asarray(action_raw).reshape(-1)[0])
                i_amt, n_amt = grid[action]
                date = planting + pd.Timedelta(days=max(dap - 1, 0))
                w = weather_row(weather, int(year), date)
                if i_amt > 1e-9 or n_amt > 1e-9:
                    signature.append(f"DAP{dap} I{i_amt:g}/N{n_amt:g}")
                if dap in config["target_daps"] and dap not in captured:
                    captured.add(dap)
                    names = obs_names(env, obs)
                    flat = np.asarray(obs, dtype=float).reshape(-1)
                    probs = masked_probs(model, obs, mask)
                    decisions.append({
                        "model_label": model_cfg["label"], "seed": int(model_cfg["seed"]), "checkpoint": int(model_cfg["checkpoint"]),
                        "year": int(year), "dap": int(dap), "date": date.strftime("%Y-%m-%d"),
                        "predicted_action": action, "irrigation_mm": i_amt, "nitrogen_kg_ha": n_amt,
                        "legal_action_count": int(mask.sum()), "top_probability": float(np.nanmax(probs)) if np.isfinite(probs).any() else np.nan,
                        "rain": scalar(w.get("rain")), "srad": scalar(w.get("srad")), "tmax": scalar(w.get("tmax")), "tmin": scalar(w.get("tmin")),
                        "swfac": scalar(latest.get("swfac")), "nstres": scalar(latest.get("nstres")), "topwt": scalar(latest.get("topwt")),
                        "grnwt": scalar(latest.get("grnwt")), "xlai": scalar(latest.get("xlai")), "obs_dim": int(len(flat)),
                    })
                    for idx, (name, value) in enumerate(zip(names, flat.tolist())):
                        observations.append({"model_label": model_cfg["label"], "seed": int(model_cfg["seed"]), "year": int(year), "dap": int(dap), "obs_index": idx, "obs_name": name, "obs_value": float(value)})
                obs, _reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                step += 1
            signatures.append({"model_label": model_cfg["label"], "seed": int(model_cfg["seed"]), "year": int(year), "action_signature": " | ".join(signature), "nonzero_action_count": len(signature)})
        finally:
            env.close()
    return decisions, observations, signatures


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "无记录。"
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(5)
    work = work.astype(object).where(pd.notna(work), "")
    return "\n".join([
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()],
    ])


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    config, env_config = make_env_config(cfg)
    # Carry audit-only selectors alongside the engine YAML so the replay loop
    # cannot accidentally fall back to a different year/DAP set.
    config["validation_years"] = list(cfg["validation_years"])
    config["target_daps"] = list(cfg["target_daps"])
    weather = direct_ppo.weather_for_daily(config)
    grid = [(float(i), float(n)) for i in config["discrete_actions"]["irrigation_levels"] for n in config["discrete_actions"]["nitrogen_levels"]]
    original_wrapper = batch.base.StressAwareDiscreteWrapper
    old_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / "DSSAT_auto_validation/multisite_new_cultivar_inputs_013"
    batch.base.StressAwareDiscreteWrapper = reserve.LateIrrigationReserveMaskWrapper
    decisions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    signatures: list[dict[str, Any]] = []
    try:
        for model_cfg in cfg["source_models"]:
            d, o, s = audit_model(model_cfg, config, env_config, weather, grid)
            decisions.extend(d); observations.extend(o); signatures.extend(s)
    finally:
        batch.base.StressAwareDiscreteWrapper = original_wrapper
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_root

    ddf = pd.DataFrame(decisions); odf = pd.DataFrame(observations); sdf = pd.DataFrame(signatures)
    ddf.to_csv(OUT / "tables/106_00_decision_weather_action.csv", index=False, encoding="utf-8-sig")
    odf.to_csv(OUT / "tables/106_00_model_observation_values.csv", index=False, encoding="utf-8-sig")
    sdf.to_csv(OUT / "tables/106_00_year_action_signatures.csv", index=False, encoding="utf-8-sig")

    weather_fields = cfg["weather_fields"]
    rows: list[dict[str, Any]] = []
    for (label, dap), g in ddf.groupby(["model_label", "dap"], sort=True):
        wranges = {f: float(pd.to_numeric(g[f], errors="coerce").max() - pd.to_numeric(g[f], errors="coerce").min()) for f in weather_fields}
        sranges = {f: float(pd.to_numeric(g[f], errors="coerce").max() - pd.to_numeric(g[f], errors="coerce").min()) for f in ["swfac", "nstres", "topwt", "grnwt", "xlai"]}
        rows.append({"model_label": label, "dap": int(dap), "n_years": int(len(g)), "weather_ranges": json.dumps(wranges, ensure_ascii=False), "state_ranges": json.dumps(sranges, ensure_ascii=False), "weather_varied_fields": int(sum(v > 1e-9 for v in wranges.values())), "action_unique": int(g[["irrigation_mm", "nitrogen_kg_ha"]].drop_duplicates().shape[0]), "legal_action_max": int(pd.to_numeric(g["legal_action_count"], errors="coerce").max()), "weather_varies": bool(any(v > 1e-9 for v in wranges.values())), "nonforced_action_invariant": bool(any(v > 1e-9 for v in wranges.values()) and int(g[["irrigation_mm", "nitrogen_kg_ha"]].drop_duplicates().shape[0]) == 1 and int(pd.to_numeric(g["legal_action_count"], errors="coerce").max()) > 1)})
    same = pd.DataFrame(rows)
    same.to_csv(OUT / "tables/106_00_same_dap_weather_action_sensitivity.csv", index=False, encoding="utf-8-sig")

    seed1_d = ddf[ddf["model_label"].eq(SEED1_LABEL)]
    varied_fields = {f: float(pd.to_numeric(seed1_d[f], errors="coerce").max() - pd.to_numeric(seed1_d[f], errors="coerce").min()) for f in weather_fields}
    seed1_same = same[same["model_label"].eq(SEED1_LABEL)]
    varied_groups = int(seed1_same["weather_varies"].sum()) if not seed1_same.empty else 0
    invariant_groups = int(seed1_same["nonforced_action_invariant"].sum()) if not seed1_same.empty else 0
    model_obs_names = sorted(set(odf[odf["model_label"].eq(SEED1_LABEL)]["obs_name"].astype(str)))
    semantic_weather_names = [x for x in model_obs_names if any(w in x.lower() for w in weather_fields)]
    gate = {
        "weather_fields_with_crossyear_variation": int(sum(v > 1e-9 for v in varied_fields.values())),
        "weather_field_ranges_seed1": varied_fields,
        "weather_varied_dap_groups_seed1": varied_groups,
        "nonforced_action_invariant_groups_seed1": invariant_groups,
        "semantic_weather_names_in_model_vector": semantic_weather_names,
        "observation_weather_names_resolved": bool(semantic_weather_names),
        "observation_variant_2k_candidate": bool(sum(v > 1e-9 for v in varied_fields.values()) >= int(cfg["gate"]["min_weather_fields_with_crossyear_variation"]) and varied_groups >= int(cfg["gate"]["min_weather_varied_dap_groups"]) and invariant_groups >= int(cfg["gate"]["min_nonforced_action_invariant_groups"])),
    }
    result = {"task_id": cfg["task_id"], "mode": cfg["mode"], "training_run": False, "five_scenario_replay": False, "source_models": cfg["source_models"], "gate": gate, "output_root": rel(OUT), "next_step": "run isolated current-weather append seed1 2K smoke" if gate["observation_variant_2k_candidate"] else "stop; do not change observation or train"}
    (OUT / "106_00_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    action_summary = ddf.groupby(["model_label", "dap"], as_index=False).agg(action_unique=("predicted_action", "nunique"), legal_action_max=("legal_action_count", "max"), rain_range=("rain", lambda x: float(pd.to_numeric(x, errors="coerce").max() - pd.to_numeric(x, errors="coerce").min())), tmax_range=("tmax", lambda x: float(pd.to_numeric(x, errors="coerce").max() - pd.to_numeric(x, errors="coerce").min())))
    names_note = "模型输入向量含有语义天气字段" if semantic_weather_names else "模型输入向量未暴露语义天气字段名；天气列来自环境当前天气元数据，不能据此声称模型已接收天气"
    lines = ["# 106_00 FQ weather-sensitivity audit", "", "## Scope", "", "- 只读加载 FQ seed=1 ckpt2000 与 seed=0 ckpt10000，验证 2014--2023；没有训练、五情景 replay 或论文性能重算。", "- 使用原 originIC、16 格动作、无 forecast、无 external-N、DSSAT 自动灌溉安全包装。", "", "## Gate", "", *[f"- `{k}`: `{v}`" for k, v in gate.items()], "", "## Interpretation", "", f"- {names_note}。", "- `nonforced_action_invariant_groups_seed1` 只表示相同 DAP 上天气有变化、合法动作不止一个但确定性动作仍相同；不是天气因果证明。", "- 只有 `observation_variant_2k_candidate=true` 才允许下一步追加当日 rain/srad/tmax/tmin 的隔离 2K smoke；不引入 future forecast。", "", "## Same-DAP action summary", "", md_table(action_summary), "", "## Signatures", "", md_table(sdf, max_rows=25), ""]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
