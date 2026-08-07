"""048_01: SYA originIC auto-0.05 teacher-guided MaskablePPO experiment.

This runner returns to the 046_10 action grid and adds only a lightweight
stress-triggered teacher-shaping reward derived from the auto-0.05 external
auto-N diagnostic baseline.  The single additional change is removal of the
inherited DAP90 fertilization cutoff; season-N cap, fertilizer interval and
the discrete action mask remain active.
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
import run_sya_ppo_configured_046_02 as base04602
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36 as base04036


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TASK_DIR / "config_048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo.json"
PROMPT = ROOT / "prompts" / "048_01_sya_originIC_auto005_teacher_guided_no_late_n_ban_maskableppo.md"
SMOKE_TIMESTEPS = 2_000
SMOKE_CHECKPOINTS = [1_000, 2_000]
EXPECTED_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
EXPECTED_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
ACTIVE_TEACHER_GUIDANCE: dict[str, Any] = {}
FERTILIZATION_ALLOWED_DAP_RANGE = [1, 150]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


class Auto005TeacherGuidedWrapper(base04036.LateIrrigationReserveMaskWrapper):
    """Add local teacher reward shaping on top of the inherited PPO safety wrapper."""

    def step(self, action):
        prev_obs = dict(getattr(self, "last_obs_dict", {}) or {})
        dap = int(self._dap())
        obs, reward, terminated, truncated, info = super().step(action)
        teacher = dict(ACTIVE_TEACHER_GUIDANCE or {})
        if not bool(teacher.get("enabled", False)):
            return obs, reward, terminated, truncated, info

        threshold = float(teacher.get("nitrogen_stress_threshold", 0.05))
        reward_scale = float(teacher.get("reward_scale", 0.001))
        safe_i = _finite_float(self.last_action_info.get("safe_action_amir", 0.0))
        safe_n = _finite_float(self.last_action_info.get("safe_action_anfer", 0.0))
        prev_nstres = _finite_float(prev_obs.get("nstres", 0.0))
        early_end = int(teacher.get("early_n_dap_end", 45))
        miss_start = int(teacher.get("missed_stress_dap_start", 45))
        miss_end = int(teacher.get("missed_stress_dap_end", 90))

        unscaled = 0.0
        components: dict[str, float] = {}
        if safe_n > 1e-9 and prev_nstres < threshold:
            penalty = float(teacher.get("low_stress_n_penalty_per_kg_unscaled", 0.0)) * safe_n
            unscaled -= penalty
            components["low_stress_n_penalty_unscaled"] = -penalty
        if safe_n > 1e-9 and dap <= early_end:
            penalty = float(teacher.get("early_n_penalty_per_kg_unscaled", 0.0)) * safe_n
            unscaled -= penalty
            components["early_n_penalty_unscaled"] = -penalty
        if prev_nstres >= threshold and safe_n > 1e-9:
            bonus = float(teacher.get("stress_n_bonus_unscaled", 0.0))
            unscaled += bonus
            components["stress_n_bonus_unscaled"] = bonus
        if prev_nstres >= threshold and safe_n <= 1e-9 and miss_start <= dap <= miss_end:
            penalty = float(teacher.get("missed_stress_n_penalty_unscaled", 0.0))
            unscaled -= penalty
            components["missed_stress_n_penalty_unscaled"] = -penalty
        if dap == 1 and (safe_i > 1e-9 or safe_n > 1e-9):
            penalty = float(teacher.get("dap1_positive_action_penalty_unscaled", 0.0))
            unscaled -= penalty
            components["dap1_positive_action_penalty_unscaled"] = -penalty

        shaped = float(unscaled) * reward_scale
        reward = float(reward) + shaped
        self.last_action_info.update(
            {
                "teacher_guidance_enabled": True,
                "teacher_nstres_threshold": threshold,
                "teacher_prev_nstres": prev_nstres,
                "teacher_reward_unscaled": float(unscaled),
                "teacher_reward_scaled": float(shaped),
                "reward_after_teacher_guidance": float(reward),
                **components,
            }
        )
        return obs, float(reward), terminated, truncated, info


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    cfg = base04602.read_config(path)
    actions = cfg.get("actions", {})
    irrigation = list(map(float, actions.get("irrigation_levels_mm", [])))
    nitrogen = list(map(float, actions.get("nitrogen_levels_kg_ha", [])))
    if irrigation != EXPECTED_IRRIGATION_LEVELS or nitrogen != EXPECTED_NITROGEN_LEVELS:
        raise ValueError(
            "048_01 action contract requires irrigation [0,15,30,45] and nitrogen [0,40,80,120]."
        )
    obs = cfg.get("observation_contract", {})
    if obs.get("base") != "046_02_raw_observation":
        raise ValueError("048_01 must inherit the 046_02 raw observation.")
    if bool(obs.get("normalization_enabled", True)) or bool(obs.get("weather_forecast_enabled", True)):
        raise ValueError("048_01 forbids observation normalization and weather forecast features.")
    if cfg.get("reference_run") != "046_10_sya_originIC_expanded_action_maskableppo":
        raise ValueError("048_01 must reference the 046_10 expanded-action PPO baseline.")
    teacher = cfg.get("teacher_guidance", {})
    if not bool(teacher.get("enabled", False)):
        raise ValueError("048_01 requires teacher guidance to be enabled.")
    if float(teacher.get("nitrogen_stress_threshold", -1.0)) != 0.05:
        raise ValueError("048_01 teacher must use nitrogen_stress_threshold=0.05.")
    safety = cfg.get("action_safety_override", {})
    if safety.get("fertilization_allowed_dap_range") != FERTILIZATION_ALLOWED_DAP_RANGE:
        raise ValueError("048_01 requires fertilization_allowed_dap_range=[1,150].")
    return cfg


def output_root(cfg: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def task_prefix(cfg: dict[str, Any]) -> str:
    return str(cfg["task_id"])


def prepare_config(cfg: dict[str, Any], smoke: bool) -> dict[str, Any]:
    planned = json.loads(json.dumps(cfg))
    if smoke:
        planned["task_name"] = f"{planned['task_name']}_smoke2k"
        planned["training"] = {"total_timesteps": SMOKE_TIMESTEPS, "checkpoint_steps": SMOKE_CHECKPOINTS}
    return planned


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    result = base04602.preflight(cfg)
    result.update(
        {
            "reference_run": cfg["reference_run"],
            "observation_contract": cfg["observation_contract"],
            "irrigation_levels_mm": cfg["actions"]["irrigation_levels_mm"],
            "nitrogen_levels_kg_ha": cfg["actions"]["nitrogen_levels_kg_ha"],
            "combined_action_count": len(cfg["actions"]["irrigation_levels_mm"])
            * len(cfg["actions"]["nitrogen_levels_kg_ha"]),
            "teacher_guidance": cfg.get("teacher_guidance", {}),
            "action_safety_override": cfg.get("action_safety_override", {}),
            "prompt_exists": PROMPT.exists(),
        }
    )
    if not PROMPT.exists():
        result["issues"].append(f"Prompt missing: {rel(PROMPT)}")
    result["next_step_allowed"] = not result["issues"]
    return result


def write_manifest(out: Path, cfg_path: Path, cfg: dict[str, Any], pf: dict[str, Any], status: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    config_dir = out / "configs"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(cfg_path, config_dir / cfg_path.name)
    shutil.copy2(PROMPT, config_dir / PROMPT.name)
    path = out / f"{task_prefix(cfg)}_run_manifest.json"
    path.write_text(json.dumps({"status": status, "config": cfg, "preflight": pf}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def copy_clean_names(out: Path, prefix: str) -> dict[str, str]:
    mapping = {
        "evaluation/042_10_training_checkpoint_inventory.csv": f"evaluation/{prefix}_training_checkpoint_inventory.csv",
        "evaluation/042_10_checkpoint_validation_summary.csv": f"evaluation/{prefix}_checkpoint_validation_summary.csv",
        "evaluation/042_10_validation_summary_by_station_checkpoint.csv": f"evaluation/{prefix}_validation_summary_by_station_checkpoint.csv",
        "042_10_result.json": f"{prefix}_engine_result.json",
    }
    copied: dict[str, str] = {}
    for source_rel, target_rel in mapping.items():
        source, target = out / source_rel, out / target_rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[target_rel] = source_rel
    return copied


def _on_grid(values: pd.Series, allowed: list[float]) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return values.map(lambda value: bool(np.isclose(float(value), allowed, atol=1e-6).any()))


def audit_actions(out: Path, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prefix = task_prefix(cfg)
    summary_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing validation summary: {summary_path}")
    summary = pd.read_csv(summary_path, keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for record in summary.itertuples(index=False):
        daily_path = ROOT / str(getattr(record, "daily_csv_path"))
        if not daily_path.exists():
            rows.append({"checkpoint_step": int(record.checkpoint_step), "year": int(record.year), "daily_exists": False})
            continue
        daily = pd.read_csv(daily_path)
        irrigation = pd.to_numeric(daily.get("safe_action_amir"), errors="coerce").fillna(0.0)
        nitrogen = pd.to_numeric(daily.get("safe_action_anfer"), errors="coerce").fillna(0.0)
        previous_i = pd.to_numeric(daily.get("previous_cumulative_irrigation"), errors="coerce")
        previous_n = pd.to_numeric(daily.get("previous_cumulative_n"), errors="coerce")
        cumulative_i = pd.to_numeric(daily.get("season_cumulative_irrigation"), errors="coerce")
        cumulative_n = pd.to_numeric(daily.get("season_cumulative_n"), errors="coerce")
        positive = (irrigation > 1e-9) | (nitrogen > 1e-9)
        delta_i = cumulative_i - previous_i
        delta_n = cumulative_n - previous_n
        transmitted = (~positive) | (np.isclose(delta_i, irrigation, atol=1e-6) & np.isclose(delta_n, nitrogen, atol=1e-6))
        nonzero_pairs = {
            (float(i), float(n))
            for i, n in zip(irrigation[positive], nitrogen[positive])
        }
        novel = irrigation.isin([15.0, 30.0]) | nitrogen.isin([40.0, 120.0])
        daps = pd.to_numeric(daily.get("dap"), errors="coerce")
        late_n = (nitrogen > 1e-9) & (daps > 90)
        rows.append(
            {
                "checkpoint_step": int(record.checkpoint_step),
                "year": int(record.year),
                "daily_exists": True,
                "row_count": int(len(daily)),
                "off_grid_irrigation_rows": int((~_on_grid(irrigation, EXPECTED_IRRIGATION_LEVELS)).sum()),
                "off_grid_nitrogen_rows": int((~_on_grid(nitrogen, EXPECTED_NITROGEN_LEVELS)).sum()),
                "positive_action_rows": int(positive.sum()),
                "positive_action_rows_after_dap1": int((positive & (daps > 1)).sum()),
                "positive_n_rows_after_dap90": int(late_n.sum()),
                "nitrogen_kg_ha_after_dap90": float(nitrogen[late_n].sum()),
                "transmission_mismatch_rows": int((positive & ~transmitted).sum()),
                "novel_level_event_rows": int((positive & novel).sum()),
                "unique_nonzero_action_pairs": len(nonzero_pairs),
                "nonzero_action_pairs": "; ".join(f"I{i:g}/N{n:g}" for i, n in sorted(nonzero_pairs)),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    final_checkpoint = int(max(cfg["training"]["checkpoint_steps"]))
    final = audit[audit["checkpoint_step"].eq(final_checkpoint)].copy()
    total_pairs = set()
    for text in final.get("nonzero_action_pairs", pd.Series(dtype=str)).fillna(""):
        total_pairs.update(piece for piece in str(text).split("; ") if piece)
    gate = {
        "final_checkpoint": final_checkpoint,
        "expected_validation_rows": len(cfg["scope"]["validation_years"]),
        "observed_validation_rows": int(len(final)),
        "all_daily_files_exist": bool(len(final) and final["daily_exists"].all()),
        "all_actions_on_declared_grid": bool(len(final) and final["off_grid_irrigation_rows"].sum() == 0 and final["off_grid_nitrogen_rows"].sum() == 0),
        "positive_actions_present": bool(len(final) and final["positive_action_rows"].sum() > 0),
        "positive_actions_transmitted": bool(len(final) and final["transmission_mismatch_rows"].sum() == 0),
        "not_all_positive_actions_at_dap1": bool(len(final) and final["positive_action_rows_after_dap1"].sum() > 0),
        "multiple_nonzero_action_pairs": len(total_pairs) >= 2,
        "new_action_levels_used": bool(len(final) and final["novel_level_event_rows"].sum() > 0),
    }
    gate["next_step_allowed"] = bool(
        gate["observed_validation_rows"] == gate["expected_validation_rows"]
        and all(value for key, value in gate.items() if key not in {"final_checkpoint", "expected_validation_rows", "observed_validation_rows", "next_step_allowed"})
    )
    return audit, gate


def verify_completed_smoke(base_cfg: dict[str, Any]) -> dict[str, Any]:
    """Verify the exact completed 2K smoke run before permitting 100K."""

    smoke_cfg = prepare_config(base_cfg, smoke=True)
    smoke_out = output_root(smoke_cfg)
    result_path = smoke_out / f"{task_prefix(smoke_cfg)}_smoke_result.json"
    manifest_path = smoke_out / f"{task_prefix(smoke_cfg)}_run_manifest.json"
    checks: dict[str, Any] = {
        "smoke_result_exists": result_path.exists(),
        "smoke_manifest_exists": manifest_path.exists(),
    }
    if not all(checks.values()):
        checks["next_step_allowed"] = False
        return checks
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke_config = manifest.get("config", {})
    gate = result.get("smoke_gate", {})
    checks.update(
        {
            "smoke_status_completed": manifest.get("status") == "completed_smoke",
            "smoke_gate_passed": gate.get("next_step_allowed") is True,
            "smoke_timesteps_match": smoke_config.get("training", {}).get("total_timesteps") == SMOKE_TIMESTEPS,
            "smoke_checkpoints_match": smoke_config.get("training", {}).get("checkpoint_steps") == SMOKE_CHECKPOINTS,
            "smoke_input_profile_match": smoke_config.get("input_profile") == base_cfg.get("input_profile"),
            "smoke_actions_match": smoke_config.get("actions") == base_cfg.get("actions"),
            "smoke_observation_contract_match": smoke_config.get("observation_contract") == base_cfg.get("observation_contract"),
            "smoke_teacher_guidance_enabled": bool(smoke_config.get("teacher_guidance", {}).get("enabled", False)),
            "smoke_action_safety_override_match": smoke_config.get("action_safety_override") == base_cfg.get("action_safety_override"),
            "smoke_output_root": rel(smoke_out),
        }
    )
    checks["next_step_allowed"] = all(
        value for key, value in checks.items() if key not in {"smoke_output_root", "next_step_allowed"}
    )
    return checks


def write_record(
    cfg: dict[str, Any],
    pf: dict[str, Any],
    copied: dict[str, str],
    audit: pd.DataFrame,
    gate: dict[str, Any],
    phase: str,
    smoke_verification: dict[str, Any] | None = None,
) -> Path:
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    lines = [
        f"# {cfg['task_id']} SYA originIC 扩展动作空间 MaskablePPO 记录",
        "",
        "## 设计",
        "",
        f"- 参考实验：`{cfg['reference_run']}`。",
        f"- 唯一实验变量：动作档位，灌溉 `{cfg['actions']['irrigation_levels_mm']}` mm，施氮 `{cfg['actions']['nitrogen_levels_kg_ha']}` kg/ha，共 {pf['combined_action_count']} 个组合。",
        "- 未改变：originIC 输入、原始 observation、无天气预报、无归一化、reward、安全约束、训练/验证年份和随机种子。",
        f"- 训练：`{cfg['training']['total_timesteps']}` steps；checkpoints：`{cfg['training']['checkpoint_steps']}`。",
        "",
        f"## {phase}动作审计",
        "",
    ]
    for key, value in gate.items():
        lines.append(f"- `{key}`: `{value}`")
    if smoke_verification is not None:
        lines.extend(["", "## 正式训练前 smoke 复核", ""])
        for key, value in smoke_verification.items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## 按年动作审计", "", audit.to_markdown(index=False), "", "## 规范化产物", ""])
    for target, source in copied.items():
        lines.append(f"- `{target}` <- `{source}`")
    lines.append("")
    doc.write_text("\n".join(lines), encoding="utf-8")
    return doc


def run(cfg_path: Path, dry_run: bool, smoke: bool, formal: bool) -> dict[str, Any]:
    base_cfg = read_config(cfg_path)
    cfg = prepare_config(base_cfg, smoke)
    pf = preflight(cfg)
    out = output_root(cfg)
    smoke_verification: dict[str, Any] | None = None
    if formal:
        smoke_verification = verify_completed_smoke(base_cfg)
        if not smoke_verification["next_step_allowed"]:
            pf["issues"].append("046_10 completed smoke verification did not pass")
            pf["next_step_allowed"] = False
    if dry_run:
        result = {**pf, "mode": "dry_run", "output_root": rel(out)}
        if smoke_verification is not None:
            result["smoke_verification"] = smoke_verification
        return result
    if not smoke and not formal:
        raise ValueError("Use --smoke for 2K validation or --formal for explicitly authorized 100K training.")
    if not pf["next_step_allowed"]:
        raise RuntimeError("Preflight failed: " + "; ".join(pf["issues"]))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {rel(out)}")

    total = int(cfg["training"]["total_timesteps"])
    checkpoints = list(map(int, cfg["training"]["checkpoint_steps"]))
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    old = {key: getattr(engine, key) for key in ["TASK_ID", "TASK_NAME", "BASE_OUT", "BASE_DOC", "PROMPT", "LOWIC_INPUT_ROOT", "STATION", "SITES", "BINARY_IRRIGATION_LEVELS", "BINARY_NITROGEN_LEVELS", "patch_base_module"]}
    old_base_wrapper = base03222.base.StressAwareDiscreteWrapper
    old_base_load_config = base03222.load_config
    old_teacher = dict(ACTIVE_TEACHER_GUIDANCE)
    try:
        ACTIVE_TEACHER_GUIDANCE.clear()
        ACTIVE_TEACHER_GUIDANCE.update(dict(cfg.get("teacher_guidance", {})))

        original_patch_base_module = engine.patch_base_module

        def patch_base_module_with_teacher(out_path: Path, doc_path: Path, total_timesteps: int, checkpoint_steps: list[int]) -> None:
            original_patch_base_module(out_path, doc_path, total_timesteps, checkpoint_steps)
            inherited_load_config = base03222.load_config

            def load_config_without_late_n_ban() -> dict[str, Any]:
                runtime_config = inherited_load_config()
                safety = runtime_config.setdefault("action_safety", {})
                safety["fertilization_allowed_dap_range"] = list(FERTILIZATION_ALLOWED_DAP_RANGE)
                safety["late_n_ban_removed"] = True
                safety["late_n_ban_removal_note"] = (
                    "048_01 permits nitrogen through DAP150; season-N cap and minimum fertilizer interval remain active."
                )
                return runtime_config

            base03222.load_config = load_config_without_late_n_ban
            base03222.base.StressAwareDiscreteWrapper = Auto005TeacherGuidedWrapper

        engine.patch_base_module = patch_base_module_with_teacher
        engine.TASK_ID = str(cfg["task_id"])
        engine.TASK_NAME = str(cfg["task_name"])
        engine.BASE_OUT = out
        engine.BASE_DOC = doc
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = base04602.INPUT_PROFILES[str(cfg["input_profile"])]
        engine.STATION = str(cfg["station_code"])
        engine.SITES = [engine.STATION]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        engine.run_training(total, checkpoints, suffix="")
    finally:
        ACTIVE_TEACHER_GUIDANCE.clear()
        ACTIVE_TEACHER_GUIDANCE.update(old_teacher)
        base03222.base.StressAwareDiscreteWrapper = old_base_wrapper
        base03222.load_config = old_base_load_config
        for key, value in old.items():
            setattr(engine, key, value)

    copied = copy_clean_names(out, task_prefix(cfg))
    audit, gate = audit_actions(out, cfg)
    phase = "smoke" if smoke else "formal"
    audit_path = out / "audits" / f"{task_prefix(cfg)}_{phase}_action_audit.csv"
    audit_path.parent.mkdir(exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    manifest = write_manifest(out, cfg_path, cfg, pf, f"completed_{phase}")
    doc = write_record(cfg, pf, copied, audit, gate, "2K Smoke" if smoke else "100K 正式训练后", smoke_verification)
    result = {
        **pf,
        "mode": f"{phase}_train_and_validate",
        "output_root": rel(out),
        "manifest": rel(manifest),
        "record_md": rel(doc),
        "action_audit": rel(audit_path),
        "action_audit_gate": gate,
    }
    if smoke:
        result["smoke_gate"] = gate
        result["next_step_allowed"] = gate["next_step_allowed"]
        result_path = out / f"{task_prefix(cfg)}_smoke_result.json"
    else:
        result["smoke_verification"] = smoke_verification
        result["next_step_allowed"] = gate["next_step_allowed"]
        result_path = out / f"{task_prefix(cfg)}_formal_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--smoke", action="store_true", help="Run the isolated 2K smoke test.")
    phase.add_argument("--formal", action="store_true", help="Run the explicitly authorized 100K training after smoke verification.")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(run(cfg_path, args.dry_run, args.smoke, args.formal), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
