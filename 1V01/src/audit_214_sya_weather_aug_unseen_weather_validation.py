"""214: read-only unseen-weather validation for the frozen task-168 PPO.

The training policy is not updated here.  Task 168 trained online over six
RAIN-only weather variants derived from SYA 2005.  This audit replays fixed
checkpoints on original SYA 2014-2023 weather, which is an unseen weather-year
holdout for that training run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ppo_safe_rendering
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


TASK_ID = "214"
TASK_NAME = "sya_weather_aug_unseen_weather_validation"
CONFIG_PATH = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
PROMPT_PATH = ROOT / "prompts" / f"2026-09-02_{TASK_NAME}.md"
SOURCE_RUN = ROOT / "benchmark_results" / "168_sya_weather_aug_pilot_smoke2k_v2"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
SOURCE_INPUT_ROOT = ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013"
STATION = "SYA"
SITE = "SY"
SEED = 0
CHECKPOINTS = [1000, 2000]
PRIMARY_CHECKPOINT = 2000
VALIDATION_YEARS = list(range(2014, 2024))
IRRIGATION_GRID = np.asarray([0.0, 15.0, 30.0, 45.0])
NITROGEN_GRID = np.asarray([0.0, 40.0, 80.0, 120.0])


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def weather_name(year: int) -> str:
    return f"CNSY{int(year) % 100:02d}01.WTH"


def load_contract() -> dict[str, Any]:
    cfg = read_json(CONFIG_PATH)
    if cfg.get("task_id") != TASK_ID or cfg.get("task_name") != TASK_NAME:
        raise ValueError("214 config identity mismatch")
    if cfg.get("source_training_task") != "168_sya_weather_aug_pilot_smoke2k_v2":
        raise ValueError("214 must use frozen task 168 as its source")
    if int(cfg.get("primary_checkpoint")) != PRIMARY_CHECKPOINT:
        raise ValueError("primary checkpoint must remain fixed at 2000")
    if list(map(int, cfg.get("unseen_weather", {}).get("years", []))) != VALIDATION_YEARS:
        raise ValueError("unexpected unseen-weather year set")
    return cfg


def source_inventory() -> pd.DataFrame:
    path = SOURCE_RUN / "evaluation" / "032_22_training_checkpoint_inventory.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    inventory = pd.read_csv(path, keep_default_na=False)
    inventory["checkpoint_step"] = pd.to_numeric(inventory["checkpoint_step"], errors="raise").astype(int)
    selected = inventory[inventory["checkpoint_step"].isin(CHECKPOINTS)].copy()
    selected = selected[selected["run_status"].astype(str).str.startswith("ok")].copy()
    if sorted(selected["checkpoint_step"].unique().tolist()) != CHECKPOINTS:
        raise RuntimeError(f"task 168 inventory lacks required checkpoints: {selected.to_dict(orient='records')}")
    if selected["checkpoint_step"].duplicated().any():
        raise RuntimeError("expected one SYA model row per checkpoint")
    for path_value in selected["model_path"].astype(str):
        model_path = ROOT / path_value
        if not model_path.exists():
            raise FileNotFoundError(model_path)
    return selected.sort_values("checkpoint_step").reset_index(drop=True)


def validation_split() -> pd.DataFrame:
    rows = []
    for year in VALIDATION_YEARS:
        rows.append(
            {
                "station_code": STATION,
                "site": SITE,
                "station_name": SITE,
                "source_year": year,
                "year": year,
                "split": "validation_unseen_weather",
                "weather_variant": "validation_original",
                "scenario_type": f"{TASK_ID}_unseen_original_{year}",
                "weather_file": weather_name(year),
                "target_weather": rel(SOURCE_INPUT_ROOT / SITE / weather_name(year)),
                "selected_for_train": False,
                "selected_for_eval": True,
                "selection_reason": "original_year_not_used_in_task_168_training",
            }
        )
    return pd.DataFrame(rows)


def validate_inputs(split: pd.DataFrame) -> dict[str, Any]:
    missing = []
    for year in VALIDATION_YEARS:
        path = SOURCE_INPUT_ROOT / SITE / weather_name(year)
        if not path.exists():
            missing.append(rel(path))
    if missing:
        raise FileNotFoundError(f"missing original unseen weather: {missing}")
    static_names = ["MZCER048.CUL"]
    static_hashes = {}
    for name in static_names:
        path = SOURCE_INPUT_ROOT / SITE / name
        if path.exists():
            static_hashes[name] = sha256(path)
    return {
        "validation_years": VALIDATION_YEARS,
        "weather_files_exist": True,
        "weather_modified": False,
        "static_input_hashes": static_hashes,
        "train_rows": int((split["selected_for_train"] == True).sum()),
        "validation_rows": int(len(split)),
    }


def patch_runtime() -> dict[str, Any]:
    old_engine = {
        key: copy.deepcopy(getattr(engine, key))
        for key in [
            "TASK_ID",
            "TASK_NAME",
            "BASE_OUT",
            "BASE_DOC",
            "PROMPT",
            "LOWIC_INPUT_ROOT",
            "STATION",
            "SITES",
            "BINARY_IRRIGATION_LEVELS",
            "BINARY_NITROGEN_LEVELS",
        ]
    }
    old_base = {
        "SITES": list(base03222.SITES),
        "SITE_NAMES": dict(base03222.SITE_NAMES),
        "OUT": base03222.OUT,
        "DOC": base03222.DOC,
        "PROMPT": base03222.PROMPT,
    }
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT

    engine.TASK_ID = TASK_ID
    engine.TASK_NAME = TASK_NAME
    engine.BASE_OUT = OUT
    engine.BASE_DOC = DOC
    engine.PROMPT = PROMPT_PATH
    engine.LOWIC_INPUT_ROOT = SOURCE_INPUT_ROOT
    engine.STATION = STATION
    engine.SITES = [STATION]
    engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
    engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
    base03222.SITES = [STATION]
    base03222.SITE_NAMES.clear()
    base03222.SITE_NAMES.update({STATION: SITE})
    base03222.OUT = OUT
    base03222.DOC = DOC
    base03222.PROMPT = PROMPT_PATH
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = SOURCE_INPUT_ROOT
    return {"engine": old_engine, "base": old_base, "input_root": old_input_root}


def restore_runtime(old: dict[str, Any]) -> None:
    ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["input_root"]
    for key, value in old["base"].items():
        if key == "SITE_NAMES":
            base03222.SITE_NAMES.clear()
            base03222.SITE_NAMES.update(value)
        else:
            setattr(base03222, key, value)
    for key, value in old["engine"].items():
        setattr(engine, key, value)


def action_signature(daily: pd.DataFrame) -> str:
    irrigation = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nitrogen = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    dap = pd.to_numeric(daily.get("dap", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    rows = []
    for d, irr, fert in zip(dap, irrigation, nitrogen):
        if abs(float(irr)) + abs(float(fert)) > 1e-9:
            rows.append(f"DAP{int(round(d))} I{float(irr):g}/N{float(fert):g}")
    return "; ".join(rows)


def audit_daily(daily_path: Path, model_path: Path, checkpoint: int, year: int) -> dict[str, Any]:
    daily = pd.read_csv(daily_path, keep_default_na=False)
    irrigation = pd.to_numeric(daily.get("safe_action_amir", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    nitrogen = pd.to_numeric(daily.get("safe_action_anfer", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    positive = (irrigation.abs() + nitrogen.abs()) > 1e-9
    on_grid = bool(
        irrigation.map(lambda x: bool(np.isclose(float(x), IRRIGATION_GRID).any())).all()
        and nitrogen.map(lambda x: bool(np.isclose(float(x), NITROGEN_GRID).any())).all()
    )
    final_grnwt = pd.to_numeric(daily.get("grnwt", pd.Series(dtype=float)), errors="coerce")
    swfac = pd.to_numeric(daily.get("swfac", pd.Series(dtype=float)), errors="coerce")
    nstres = pd.to_numeric(daily.get("nstres", pd.Series(dtype=float)), errors="coerce")
    done = bool(len(daily) and str(daily.get("done", pd.Series([False])).iloc[-1]).lower() in {"true", "1"})
    return {
        "station_code": STATION,
        "year": int(year),
        "checkpoint_step": int(checkpoint),
        "run_status": "ok" if done else "incomplete",
        "episode_length": int(len(daily)),
        "episode_completed": done,
        "final_grnwt": float(final_grnwt.iloc[-1]) if len(final_grnwt) else np.nan,
        "total_irrigation": float(irrigation.sum()),
        "total_n": float(nitrogen.sum()),
        "PFP_N": float(final_grnwt.iloc[-1] / nitrogen.sum()) if len(final_grnwt) and nitrogen.sum() > 1e-9 else np.nan,
        "positive_action_rows": int(positive.sum()),
        "unique_nonzero_action_pairs": int(pd.DataFrame({"i": irrigation[positive], "n": nitrogen[positive]}).drop_duplicates().shape[0]),
        "action_signature": action_signature(daily),
        "all_actions_on_grid": on_grid,
        "max_swfac": float(swfac.max()) if len(swfac) else np.nan,
        "swfac_days_gt_0p05": int((swfac > 0.05).sum()) if len(swfac) else 0,
        "max_nstres": float(nstres.max()) if len(nstres) else np.nan,
        "nstres_days_gt_0p05": int((nstres > 0.05).sum()) if len(nstres) else 0,
        "model_path": rel(model_path),
        "daily_csv_path": rel(daily_path),
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for checkpoint, group in detail.groupby("checkpoint_step", sort=True):
        rows.append(
            {
                "checkpoint_step": int(checkpoint),
                "validation_years": int(group["year"].nunique()),
                "completed_episodes": int(group["episode_completed"].sum()),
                "all_actions_on_grid": bool(group["all_actions_on_grid"].all()),
                "all_zero_management_episodes": int(((group["total_irrigation"].abs() + group["total_n"].abs()) <= 1e-9).sum()),
                "mean_final_grnwt": float(group["final_grnwt"].mean()),
                "std_final_grnwt": float(group["final_grnwt"].std(ddof=0)),
                "min_final_grnwt": float(group["final_grnwt"].min()),
                "mean_PFP_N": float(group["PFP_N"].mean()),
                "mean_total_irrigation": float(group["total_irrigation"].mean()),
                "mean_total_n": float(group["total_n"].mean()),
                "unique_action_signatures": int(group["action_signature"].nunique()),
                "mean_positive_action_rows": float(group["positive_action_rows"].mean()),
                "max_swfac": float(group["max_swfac"].max()),
                "max_nstres": float(group["max_nstres"].max()),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    cfg = load_contract()
    if OUT.exists():
        existing_files = {path.relative_to(OUT).as_posix() for path in OUT.rglob("*") if path.is_file()}
        allowed_partial = {f"configs/{CONFIG_PATH.name}"}
        if existing_files != allowed_partial:
            raise FileExistsError(f"Refuse to overwrite existing output: {rel(OUT)}; files={sorted(existing_files)}")
    inventory = source_inventory()
    split = validation_split()
    input_audit = validate_inputs(split)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(parents=True, exist_ok=True)
    (OUT / "daily_outputs" / STATION).mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, OUT / "configs" / CONFIG_PATH.name)
    shutil.copy2(PROMPT_PATH, OUT / "configs" / PROMPT_PATH.name)
    shutil.copy2(SOURCE_RUN / "168_result.json", OUT / "configs" / "168_result_source.json")
    split.to_csv(OUT / "configs" / "214_unseen_weather_split.csv", index=False, encoding="utf-8-sig")
    (OUT / "configs" / "214_input_audit.json").write_text(json.dumps(input_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    old = patch_runtime()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        runtime_config = engine.load_config()
        env_config = base03222.direct_ppo.build_env_config(runtime_config, split)
        for model_row in inventory.to_dict(orient="records"):
            checkpoint = int(model_row["checkpoint_step"])
            model_path = ROOT / str(model_row["model_path"])
            for year in VALIDATION_YEARS:
                row = pd.Series(model_row)
                row["station_code"] = STATION
                try:
                    result = base03222.evaluate_checkpoint(runtime_config, env_config, row, year)
                    daily_path = ROOT / str(result["daily_csv_path"])
                    rows.append(audit_daily(daily_path, model_path, checkpoint, year))
                except Exception as exc:
                    errors.append(f"checkpoint={checkpoint} year={year}: {exc!r}")
    finally:
        restore_runtime(old)

    detail = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    summary = summarize(detail) if not detail.empty else pd.DataFrame()
    detail.to_csv(OUT / "evaluation" / "214_unseen_weather_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "evaluation" / "214_unseen_weather_checkpoint_summary.csv", index=False, encoding="utf-8-sig")

    primary = summary[summary["checkpoint_step"].eq(PRIMARY_CHECKPOINT)] if not summary.empty else pd.DataFrame()
    stability_passed = bool(
        len(primary)
        and int(primary.iloc[0]["completed_episodes"]) == len(VALIDATION_YEARS)
        and bool(primary.iloc[0]["all_actions_on_grid"])
        and int(primary.iloc[0]["all_zero_management_episodes"]) == 0
        and not errors
    )
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed" if not errors else "completed_with_errors",
        "training_executed": False,
        "source_training_task": "168_sya_weather_aug_pilot_smoke2k_v2",
        "primary_checkpoint": PRIMARY_CHECKPOINT,
        "diagnostic_checkpoints": CHECKPOINTS,
        "unseen_weather_years": VALIDATION_YEARS,
        "input_audit": input_audit,
        "observed_detail_rows": int(len(detail)),
        "expected_detail_rows": int(len(CHECKPOINTS) * len(VALIDATION_YEARS)),
        "stability_passed": stability_passed,
        "errors": errors,
        "outputs": {
            "detail": rel(OUT / "evaluation" / "214_unseen_weather_detail.csv"),
            "checkpoint_summary": rel(OUT / "evaluation" / "214_unseen_weather_checkpoint_summary.csv"),
        },
        "WP_ET_status": "pending exact Summary.OUT/ETCP replay; never inferred from daily CSV",
    }
    (OUT / "214_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {TASK_ID} SYA weather-only unseen-weather validation",
        "",
        "- 只读验证 task 168 的冻结 PPO checkpoint；本任务不训练、不修改 168 输出。",
        "- 训练天气：task 168 的 2005 六情景在线训练；验证天气：原始 SYA 2014-2023，均未用于 task 168 训练。",
        f"- 主 checkpoint：{PRIMARY_CHECKPOINT}；诊断 checkpoint：{', '.join(map(str, CHECKPOINTS))}。",
        f"- 稳定性门：`{stability_passed}`。",
        "",
        "## Checkpoint 汇总",
        "",
        summary.to_markdown(index=False) if not summary.empty else "无成功验证记录。",
        "",
        "## 结果边界",
        "",
        "- 本审计判断未见天气上的运行稳定性、动作合法性和是否出现全零管理坍缩。",
        "- 不把未见天气结果解释为天气因果或 PPO 普遍优越性。",
        "- WP_ET 仍需 Summary.OUT/ETCP 精确回放后才能报告。",
        "",
        "## Errors",
        "",
        "```text",
        "\n".join(errors) if errors else "none",
        "```",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
