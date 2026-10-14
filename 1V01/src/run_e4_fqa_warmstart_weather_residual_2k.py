"""FQ E4 paired smoke: warm-start the same no-forecast checkpoint.

The source policy is the FQ seed-1 no-forecast checkpoint at 2K.  Each arm
continues for 2K adaptation steps with the same reward/action/safety code;
only the weather branch input differs (forecast features versus twelve zeros).
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import forecast_engineered_observation_056_057 as forecast
import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
import run_lc_multiyear_free_timing_ppo_smoke_032_10 as lc_multi
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_176_multisite_weather_framework_transfer as transfer
from run_141E1_sya_actionable_weather_encoding_smoke2k import patch_contract
from run_152E3_sya_dual_branch_noforecast_control import make_zero_weather_factory


STATION = "FQA"
SITE = "FQ"
SEED = 1
ADAPT_STEPS = 2_000
RESULT_BASENAME = "182_183_FQA_e4_warmstart_weather_residual_2k_smoke_result"
DOC_BASENAME = "182_183_FQA_e4_warmstart_weather_residual_2k_record"
SOURCE = ROOT / "benchmark_results/175FQA_fqa_originIC_dual_branch_noforecast_control_5k_seed1/models/FQA/FQA_half_split_stress_aware_maskableppo_seed1_ckpt2000.zip"
CFG_PATHS = {
    "forecast": ROOT / "configs/182FQA_fqa_originIC_e4_warmstart_weather_residual_2k_seed1.json",
    "noforecast": ROOT / "configs/183FQA_fqa_originIC_e4_warmstart_noforecast_control_2k_seed1.json",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_root(kind: str) -> Path:
    cfg = read_json(CFG_PATHS[kind])
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def validate_contracts() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    for kind, path in CFG_PATHS.items():
        cfg = read_json(path)
        if cfg["station_code"] != STATION or cfg["seed"] != SEED:
            raise ValueError(f"{kind}: station/seed mismatch")
        if cfg["actions"] != {"irrigation_levels_mm": [0, 15, 30, 45], "nitrogen_levels_kg_ha": [0, 40, 80, 120]}:
            raise ValueError(f"{kind}: action grid changed")
        if cfg["scope"]["reward_and_safety"] != "inherit_042_15_via_046_02_unchanged":
            raise ValueError(f"{kind}: reward contract changed")
        if cfg["training"]["adaptation_timesteps"] != ADAPT_STEPS:
            raise ValueError(f"{kind}: adaptation steps mismatch")


def configure_runtime() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, list[int], list[int]]:
    """Resolve the existing FQ environment exactly as the transfer runner does."""
    patch_contract()
    engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
    engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
    engine.LOWIC_INPUT_ROOT = forecast.INPUT_PROFILES["originIC"]
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.base03222.load_config = engine.load_config
    engine.base03222.base.StressAwareDiscreteWrapper = engine.base04036.LateIrrigationReserveMaskWrapper
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
    runtime_cfg = engine.load_config()
    runtime_cfg = json.loads(json.dumps(runtime_cfg))
    runtime_cfg["seed"] = SEED
    split = engine.base03222.load_split()
    split = split[split["station_code"].eq(STATION)].copy()
    selection = engine.base03222.build_selection(split)
    env_cfg = direct_ppo.build_env_config(runtime_cfg, selection)
    train_years = sorted(split.loc[split["split"].eq("train"), "year"].astype(int).tolist())
    valid_years = sorted(split.loc[split["split"].eq("validation"), "year"].astype(int).tolist())
    return runtime_cfg, env_cfg, selection, train_years, valid_years


def initialize_weather_residual(model: Any) -> dict[str, float]:
    """Zero weather sensitivity while preserving the source policy's zero-input output."""
    extractor = model.policy.features_extractor
    weather_encoder = extractor.weather_encoder
    linear = weather_encoder[0]
    norm = weather_encoder[1]
    with torch.no_grad():
        zero = torch.zeros((1, int(extractor.weather_observation_dim)), device=linear.weight.device)
        source_output = weather_encoder(zero).detach().reshape(-1)
        target_pre_tanh = torch.atanh(torch.clamp(source_output, -0.999999, 0.999999))
        linear.weight.zero_()
        linear.bias.zero_()
        # With zero linear input, LayerNorm output is its bias.  This keeps the
        # source policy's constant zero-weather branch output while opening a
        # trainable path for weather inputs during adaptation.
        norm.bias.copy_(target_pre_tanh)
        after = weather_encoder(zero).detach().reshape(-1)
    return {
        "source_zero_branch_l2": float(torch.linalg.vector_norm(source_output).cpu()),
        "initialized_zero_branch_l2": float(torch.linalg.vector_norm(after).cpu()),
        "zero_branch_max_abs_delta": float(torch.max(torch.abs(after - source_output)).cpu()),
        "weather_linear_weight_l2_after": float(torch.linalg.vector_norm(linear.weight).cpu()),
    }


def train_one(kind: str, runtime_cfg: dict[str, Any], env_cfg: dict[str, Any], train_years: list[int]) -> tuple[Path, dict[str, Any]]:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback

    cfg = read_json(CFG_PATHS[kind])
    out = output_root(kind)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing E4 output: {out.relative_to(ROOT)}")
    out.mkdir(parents=True, exist_ok=True)
    for rel in ["configs", "models/FQA", "evaluation", "daily_outputs/FQA", "logs", "audits"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG_PATHS[kind], out / "configs" / CFG_PATHS[kind].name)
    prompt = ROOT / cfg["prompt"]
    if prompt.exists():
        shutil.copy2(prompt, out / "configs" / prompt.name)
    (out / "configs" / "resolved_env_config.json").write_text(json.dumps(env_cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    old_make_env = engine.base03222.base.make_env
    exp_cfg = read_json(CFG_PATHS["forecast"])
    if kind == "forecast":
        make_env = forecast.make_forecast_env_factory(old_make_env, exp_cfg)
    else:
        make_env = make_zero_weather_factory(old_make_env)
    env = None
    model = None
    audit: dict[str, Any] = {}
    try:
        engine.base03222.base.make_env = make_env
        env = batch.RandomYearEnv(runtime_cfg, env_cfg, STATION, train_years, SEED)
        model = MaskablePPO.load(str(SOURCE), env=env, device="cpu")
        audit = initialize_weather_residual(model)

        class SaveAdaptationCheckpoint(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.saved = False

            def _on_step(self) -> bool:
                if not self.saved and int(self.model.num_timesteps) >= ADAPT_STEPS:
                    model_path = out / "models" / "FQA" / f"FQA_e4_{kind}_seed{SEED}_adaptation2k.zip"
                    self.model.save(str(model_path))
                    self.saved = True
                return True

        model.learn(total_timesteps=ADAPT_STEPS, reset_num_timesteps=True, progress_bar=False, callback=SaveAdaptationCheckpoint())
        model_path = out / "models" / "FQA" / f"FQA_e4_{kind}_seed{SEED}_adaptation2k.zip"
        if not model_path.exists():
            model.save(str(model_path))
        reset_df = pd.DataFrame([
            {"station_code": STATION, "year": int(y), "episode_count": int(env.reset_counts.get(y, 0))}
            for y in train_years
        ])
        reset_df.to_csv(out / "logs" / "training_year_reset_counts.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(env.switch_log).to_csv(out / "logs" / "training_year_switch_log.csv", index=False, encoding="utf-8-sig")
    finally:
        if env is not None:
            env.close()
        engine.base03222.base.make_env = old_make_env
    inventory = pd.DataFrame([{
        "station_code": STATION, "site": SITE, "seed": SEED, "source_checkpoint_step": 2000,
        "adaptation_timesteps": ADAPT_STEPS, "checkpoint_step": ADAPT_STEPS,
        "source_model_path": str(SOURCE.relative_to(ROOT).as_posix()),
        "model_path": str(model_path.relative_to(ROOT).as_posix()),
        "model_sha256": sha256_file(model_path), "run_status": "ok",
    }])
    inventory.to_csv(out / "evaluation" / "training_checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    (out / "audits" / "weather_residual_initialization.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return model_path, audit


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_one(kind: str, model_path: Path, runtime_cfg: dict[str, Any], env_cfg: dict[str, Any], valid_years: list[int]) -> pd.DataFrame:
    out = output_root(kind)
    batch.OUT = out
    batch.SITES = [STATION]
    batch.SEED = SEED
    batch.SITE_NAMES[STATION] = SITE
    row = pd.Series({"station_code": STATION, "site": SITE, "seed": SEED, "checkpoint_step": ADAPT_STEPS, "model_path": model_path.relative_to(ROOT).as_posix()})
    old_make_env = engine.base03222.base.make_env
    cfg = read_json(CFG_PATHS[kind])
    exp_cfg = read_json(CFG_PATHS["forecast"])
    make_env = forecast.make_forecast_env_factory(old_make_env, exp_cfg) if kind == "forecast" else make_zero_weather_factory(old_make_env)
    rows: list[dict[str, Any]] = []
    try:
        engine.base03222.base.make_env = make_env
        for year in valid_years:
            try:
                item = batch.evaluate_checkpoint(runtime_cfg, env_cfg, row, int(year))
            except Exception:
                item = {"station_code": STATION, "site": SITE, "year": int(year), "seed": SEED, "checkpoint_step": ADAPT_STEPS, "run_status": "failed", "notes": traceback.format_exc()[-4000:]}
            rows.append(item)
    finally:
        engine.base03222.base.make_env = old_make_env
    result = pd.DataFrame(rows)
    result.to_csv(out / "evaluation" / f"{cfg['task_id']}_checkpoint_validation_summary.csv", index=False, encoding="utf-8-sig")
    return result


def audit_actions(kind: str, eval_df: pd.DataFrame) -> dict[str, Any]:
    out = output_root(kind)
    rows: list[dict[str, Any]] = []
    for _, rec in eval_df.iterrows():
        daily_path = ROOT / str(rec.get("daily_csv_path", ""))
        if not daily_path.exists():
            continue
        daily = pd.read_csv(daily_path)
        irr = pd.to_numeric(daily.get("safe_action_amir", 0), errors="coerce").fillna(0.0)
        nit = pd.to_numeric(daily.get("safe_action_anfer", 0), errors="coerce").fillna(0.0)
        dap = pd.to_numeric(daily.get("dap", 0), errors="coerce").fillna(0.0)
        positive = (irr > 1e-9) | (nit > 1e-9)
        rows.append({
            "year": int(rec["year"]), "run_status": rec.get("run_status", ""), "row_count": len(daily),
            "off_grid_irrigation": int((~np.isclose(irr.to_numpy()[:, None], [0, 15, 30, 45], atol=1e-6).any(axis=1)).sum()),
            "off_grid_nitrogen": int((~np.isclose(nit.to_numpy()[:, None], [0, 40, 80, 120], atol=1e-6).any(axis=1)).sum()),
            "positive_action_rows": int(positive.sum()), "positive_after_dap1": int((positive & (dap > 1)).sum()),
            "action_sequence": rec.get("action_sequence", ""),
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "evaluation" / "action_audit.csv", index=False, encoding="utf-8-sig")
    return {
        "validation_rows": int(len(audit)),
        "all_daily_files_exist": bool(len(audit) == 10),
        "all_actions_on_grid": bool(len(audit) and audit[["off_grid_irrigation", "off_grid_nitrogen"]].to_numpy().sum() == 0),
        "positive_actions_present": bool(len(audit) and audit["positive_action_rows"].sum() > 0),
        "positive_actions_after_dap1": bool(len(audit) and audit["positive_after_dap1"].sum() > 0),
        "unique_action_sequences": int(audit["action_sequence"].nunique()) if not audit.empty else 0,
        "action_audit_path": str((out / "evaluation" / "action_audit.csv").relative_to(ROOT)),
    }


def write_record(results: dict[str, Any]) -> None:
    lines = [
        "# FQ E4 warm-start weather residual paired 2K smoke",
        "",
        "- Same FQ seed-1 no-forecast checkpoint 2000 used as source for both arms.",
        "- Each arm continued for 2000 adaptation timesteps; this is source exposure 2K + adaptation 2K, not a from-scratch 2K run.",
        "- Reward, action grid, action evaluation, safety masks, DSSAT inputs and year split were held fixed.",
        "",
        "## Results",
        "",
        "```json",
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        "```",
        "",
        "## Interpretation boundary",
        "",
        "- This is a paired FQ smoke result. It does not establish cross-seed or cross-site superiority.",
        "- WP_ET requires the separate DSSAT snapshot replay audit before final interpretation.",
        "- 5K adaptation is not authorized by this record unless both paired arms pass the smoke gates.",
    ]
    (ROOT / "docs" / f"{DOC_BASENAME}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    validate_contracts()
    runtime_cfg, env_cfg, selection, train_years, valid_years = configure_runtime()
    results: dict[str, Any] = {"source_checkpoint": str(SOURCE.relative_to(ROOT)), "train_years": train_years, "validation_years": valid_years, "arms": {}}
    for kind in ["forecast", "noforecast"]:
        model_path, init_audit = train_one(kind, runtime_cfg, env_cfg, train_years)
        eval_df = evaluate_one(kind, model_path, runtime_cfg, env_cfg, valid_years)
        action_gate = audit_actions(kind, eval_df)
        ok = eval_df[eval_df["run_status"].astype(str).str.startswith("ok")] if not eval_df.empty else pd.DataFrame()
        results["arms"][kind] = {
            "model_path": str(model_path.relative_to(ROOT)),
            "initialization": init_audit,
            "evaluation_rows": int(len(ok)),
            "mean_final_grnwt": float(pd.to_numeric(ok.get("final_grnwt"), errors="coerce").mean()) if not ok.empty else None,
            "mean_total_irrigation": float(pd.to_numeric(ok.get("total_irrigation"), errors="coerce").mean()) if not ok.empty else None,
            "mean_total_n": float(pd.to_numeric(ok.get("total_n"), errors="coerce").mean()) if not ok.empty else None,
            "mean_pfp_n": float(pd.to_numeric(ok.get("PFP_N"), errors="coerce").mean()) if not ok.empty else None,
            "action_gate": action_gate,
        }
    f = results["arms"].get("forecast", {})
    n = results["arms"].get("noforecast", {})
    results["paired_delta_forecast_minus_noforecast"] = {
        key: (f.get(key) - n.get(key)) if f.get(key) is not None and n.get(key) is not None else None
        for key in ["mean_final_grnwt", "mean_total_irrigation", "mean_total_n", "mean_pfp_n"]
    }
    results["next_step_allowed"] = bool(all(a.get("action_gate", {}).get("all_daily_files_exist") and a.get("action_gate", {}).get("all_actions_on_grid") and a.get("action_gate", {}).get("positive_actions_after_dap1") for a in results["arms"].values()))
    out_path = ROOT / "benchmark_results" / f"{RESULT_BASENAME}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_record(results)
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
