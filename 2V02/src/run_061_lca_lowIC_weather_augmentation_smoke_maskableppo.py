"""061_00: LC lowIC weather-augmentation MaskablePPO smoke.

This runner expands the LC training weather set using rainfall-only WTH
perturbations, trains a short MaskablePPO smoke, and evaluates only on the
original LC validation years. It intentionally does not run a second synthetic
validation set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import ppo_safe_rendering
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine
from weather_augmentation_061_00 import prepare_augmented_inputs, read_config


TASK_ID = "061_00"
TASK_NAME = "lca_lowIC_weather_augmentation_smoke_maskableppo"
DEFAULT_CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
DOC = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
PROMPT = ROOT / "prompts" / f"{TASK_ID}_{TASK_NAME}.md"
STATION = "LCA"
SITE = "LC"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ensure_prompt() -> None:
    if not PROMPT.exists():
        PROMPT.write_text(
            "\n".join(
                [
                    "# 061_00 LC weather-augmentation smoke",
                    "",
                    "Goal: test whether rainfall perturbation during training can break fixed DAP/resource templates.",
                    "",
                    "- Train on LC original train years plus early-dry, mid-dry, and early-wet pseudo-year variants.",
                    "- Validate only on original LC 2014-2023 years.",
                    "- First smoke changes rainfall only; temperature is unchanged.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def make_split(cfg: dict[str, Any]) -> pd.DataFrame:
    aug = cfg["weather_augmentation"]
    rows = []
    for year in map(int, aug["train_years"]):
        for variant in aug["variants"]:
            pseudo_year = int(year + int(variant["pseudo_year_offset"]))
            rows.append(
                {
                    "station_code": STATION,
                    "site": SITE,
                    "source_year": year,
                    "year": pseudo_year,
                    "variant": str(variant["name"]),
                    "split": "train",
                    "rain_multiplier": float(variant["rain_multiplier"]),
                    "window": str(variant["window"]),
                }
            )
    for year in map(int, aug["validation_years_original_only"]):
        rows.append(
            {
                "station_code": STATION,
                "site": SITE,
                "source_year": year,
                "year": year,
                "variant": "validation_original",
                "split": "validation",
                "rain_multiplier": 1.0,
                "window": "none",
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "source_year", "variant"]).reset_index(drop=True)


def make_selection(cfg: dict[str, Any], split: pd.DataFrame) -> pd.DataFrame:
    pool = pd.read_csv(engine.base03222.POOL, keep_default_na=False)
    pool["year"] = pd.to_numeric(pool["year"], errors="coerce").astype(int)
    pool_lc = pool[pool["station_code"].astype(str).eq(STATION)].copy()
    rows: list[dict[str, Any]] = []
    for row in split.itertuples(index=False):
        src = pool_lc[pool_lc["year"].eq(int(row.source_year))]
        base = src.iloc[0].to_dict() if len(src) else {}
        base.update(
            {
                "station_code": STATION,
                "station_name": SITE,
                "site": SITE,
                "year": int(row.year),
                "source_year": int(row.source_year),
                "weather_variant": str(row.variant),
                "scenario_type": f"061_00_{row.variant}",
                "weather_file": f"CNLC{int(row.year) % 100:02d}01.WTH",
                "split": str(row.split),
                "selected_for_train": str(row.split) == "train",
                "selected_for_eval": str(row.split) == "validation",
                "selection_reason": "061_00_weather_augmentation_train" if str(row.split) == "train" else "061_00_original_validation",
                "rain_multiplier": float(row.rain_multiplier),
                "perturbation_window": str(row.window),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["station_code", "year"]).reset_index(drop=True)


def write_clean_names(out: Path, prefix: str) -> dict[str, str]:
    mapping = {
        "evaluation/032_22_training_checkpoint_inventory.csv": f"evaluation/{prefix}_training_checkpoint_inventory.csv",
        "evaluation/032_22_checkpoint_validation_summary.csv": f"evaluation/{prefix}_checkpoint_validation_summary.csv",
        "evaluation/032_22_validation_summary_by_station_checkpoint.csv": f"evaluation/{prefix}_validation_summary_by_station_checkpoint.csv",
        "logs/032_22_training_year_reset_counts.csv": f"logs/{prefix}_training_year_reset_counts.csv",
        "032_22_result.json": f"{prefix}_engine_result.json",
    }
    copied = {}
    for src_rel, dst_rel in mapping.items():
        src = out / src_rel
        dst = out / dst_rel
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied[dst_rel] = src_rel
    return copied


def action_signature(daily: pd.DataFrame) -> str:
    events = []
    for r in daily.sort_values("dap").itertuples(index=False):
        i = float(getattr(r, "safe_action_amir", 0.0))
        n = float(getattr(r, "safe_action_anfer", 0.0))
        if i > 0 or n > 0:
            events.append(f"DAP{int(round(float(r.dap)))}:I{i:.0f}/N{n:.0f}")
    return "; ".join(events)


def audit_validation_actions(out: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_path = out / "evaluation" / f"{prefix}_checkpoint_validation_summary.csv"
    if not eval_path.exists():
        eval_path = out / "evaluation" / "032_22_checkpoint_validation_summary.csv"
    eval_df = pd.read_csv(eval_path, keep_default_na=False)
    rows = []
    for row in eval_df.itertuples(index=False):
        daily_path = ROOT / str(getattr(row, "daily_csv_path", ""))
        rec = {
            "station_code": getattr(row, "station_code", STATION),
            "year": int(getattr(row, "year")),
            "checkpoint_step": int(getattr(row, "checkpoint_step")),
            "daily_exists": daily_path.exists(),
            "action_signature": "",
        }
        if daily_path.exists():
            daily = pd.read_csv(daily_path, keep_default_na=False)
            for col in ["dap", "rain", "swfac", "nstres", "safe_action_amir", "safe_action_anfer", "grnwt"]:
                if col in daily:
                    daily[col] = pd.to_numeric(daily[col], errors="coerce")
            i = daily["safe_action_amir"].fillna(0.0)
            n = daily["safe_action_anfer"].fillna(0.0)
            rec.update(
                {
                    "row_count": int(len(daily)),
                    "total_irrigation_mm": float(i.sum()),
                    "total_nitrogen_kg_ha": float(n.sum()),
                    "irrigation_event_count": int(i.gt(0).sum()),
                    "nitrogen_event_count": int(n.gt(0).sum()),
                    "max_swfac": float(daily["swfac"].max()) if "swfac" in daily else np.nan,
                    "max_nstres": float(daily["nstres"].max()) if "nstres" in daily else np.nan,
                    "season_rain_mm": float(daily["rain"].fillna(0).sum()) if "rain" in daily else np.nan,
                    "final_grain_kg_ha": float(daily["grnwt"].iloc[-1]) if "grnwt" in daily and len(daily) else np.nan,
                    "irrigation_event_daps": ";".join(str(int(round(x))) for x in daily.loc[i.gt(0), "dap"].tolist()),
                    "nitrogen_event_daps": ";".join(str(int(round(x))) for x in daily.loc[n.gt(0), "dap"].tolist()),
                    "action_signature": action_signature(daily),
                }
            )
        rows.append(rec)
    audit = pd.DataFrame(rows).sort_values(["checkpoint_step", "year"]).reset_index(drop=True)
    summary = (
        audit.groupby("checkpoint_step", as_index=False)
        .agg(
            years=("year", "nunique"),
            unique_action_signatures=("action_signature", "nunique"),
            mean_total_irrigation_mm=("total_irrigation_mm", "mean"),
            sd_total_irrigation_mm=("total_irrigation_mm", lambda x: pd.to_numeric(x, errors="coerce").std(ddof=0)),
            mean_total_nitrogen_kg_ha=("total_nitrogen_kg_ha", "mean"),
            sd_total_nitrogen_kg_ha=("total_nitrogen_kg_ha", lambda x: pd.to_numeric(x, errors="coerce").std(ddof=0)),
            corr_i_vs_rain=("total_irrigation_mm", lambda x: np.nan),
        )
    )
    corr_rows = []
    for ck, sub in audit.groupby("checkpoint_step"):
        corr_rows.append(
            {
                "checkpoint_step": int(ck),
                "corr_total_i_vs_rain": pd.to_numeric(sub["total_irrigation_mm"], errors="coerce").corr(pd.to_numeric(sub["season_rain_mm"], errors="coerce")),
                "corr_total_i_vs_max_swfac": pd.to_numeric(sub["total_irrigation_mm"], errors="coerce").corr(pd.to_numeric(sub["max_swfac"], errors="coerce")),
                "corr_total_n_vs_max_nstres": pd.to_numeric(sub["total_nitrogen_kg_ha"], errors="coerce").corr(pd.to_numeric(sub["max_nstres"], errors="coerce")),
            }
        )
    summary = summary.drop(columns=["corr_i_vs_rain"]).merge(pd.DataFrame(corr_rows), on="checkpoint_step", how="left")
    audit_dir = out / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_dir / f"{prefix}_validation_action_weather_audit.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(audit_dir / f"{prefix}_validation_action_weather_audit_summary.csv", index=False, encoding="utf-8-sig")
    return audit, summary


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "No rows."
    work = df.head(max_rows).copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    lines = [
        "| " + " | ".join(map(str, work.columns)) + " |",
        "| " + " | ".join(["---"] * len(work.columns)) + " |",
    ]
    for row in work.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def write_record(cfg: dict[str, Any], prep: dict[str, Any], split: pd.DataFrame, copied: dict[str, str], audit: pd.DataFrame, summary: pd.DataFrame, status: str) -> None:
    lines = [
        "# 061_00 LC weather-augmentation smoke record",
        "",
        f"- Status: `{status}`.",
        f"- Augmented input root: `{prep.get('augmented_input_root', '')}`.",
        f"- Weather mapping CSV: `{prep.get('mapping_csv', '')}`.",
        f"- Training timesteps: `{cfg['training']['total_timesteps']}`.",
        f"- Checkpoints: `{cfg['training']['checkpoint_steps']}`.",
        "- Train variants: original, early_dry, mid_dry, early_wet.",
        "- Validation: original 2014-2023 only.",
        "- First smoke perturbation: rainfall only; no temperature perturbation.",
        "",
        "## Perturbation design",
        "",
        "- `early_dry`: DAP 1-30 rainfall x 0.5.",
        "- `mid_dry`: DAP 31-60 rainfall x 0.5.",
        "- `early_wet`: DAP 1-30 rainfall x 1.5.",
        "- Rationale: AgMIP/C3MP-style climate sensitivity perturbations modify daily climate series directly; 0.5/1.5 rainfall multipliers are an appropriate smoke-level stress range. RL rationale is domain randomization: broaden environment distribution so fixed templates are no longer always safe.",
        "",
        "## Split / pseudo-year mapping",
        "",
        md_table(split, 80),
        "",
        "## Validation action-weather audit summary",
        "",
        md_table(summary, 40),
        "",
        "## Validation action details",
        "",
        md_table(audit, 120),
        "",
        "## Normalized outputs",
        "",
    ]
    for dst, src in copied.items():
        lines.append(f"- `{dst}` <- `{src}`")
    lines.append("")
    DOC.write_text("\n".join(lines), encoding="utf-8")


def run(cfg_path: Path, dry_run: bool = False) -> dict[str, Any]:
    cfg = read_config(cfg_path)
    ensure_prompt()
    prep = prepare_augmented_inputs(cfg_path, reuse_existing=True)
    split = make_split(cfg)
    selection = make_selection(cfg, split)
    if dry_run:
        return {
            "task": f"{TASK_ID}_{TASK_NAME}",
            "mode": "dry_run",
            "output_root": rel(OUT),
            "augmented_input_root": prep["augmented_input_root"],
            "train_rows": int((split["split"] == "train").sum()),
            "validation_rows": int((split["split"] == "validation").sum()),
            "train_year_min_max": [int(split.loc[split["split"].eq("train"), "year"].min()), int(split.loc[split["split"].eq("train"), "year"].max())],
            "validation_years": split.loc[split["split"].eq("validation"), "year"].astype(int).tolist(),
            "next_step_allowed": True,
        }
    if OUT.exists() and (OUT / "061_00_smoke_result.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed 061_00 smoke: {rel(OUT)}")
    for sub in ["configs", "evaluation", "logs", "models", "daily_outputs", "audits"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, OUT / "configs" / cfg_path.name)
    shutil.copy2(PROMPT, OUT / "configs" / PROMPT.name)
    split.to_csv(OUT / "configs" / "061_00_augmented_split.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUT / "configs" / "061_00_augmented_selection.csv", index=False, encoding="utf-8-sig")

    old_values = {
        "TASK_ID": engine.TASK_ID,
        "TASK_NAME": engine.TASK_NAME,
        "BASE_OUT": engine.BASE_OUT,
        "BASE_DOC": engine.BASE_DOC,
        "PROMPT": engine.PROMPT,
        "LOWIC_INPUT_ROOT": engine.LOWIC_INPUT_ROOT,
        "STATION": engine.STATION,
        "SITES": list(engine.SITES),
        "BINARY_IRRIGATION_LEVELS": list(engine.BINARY_IRRIGATION_LEVELS),
        "BINARY_NITROGEN_LEVELS": list(engine.BINARY_NITROGEN_LEVELS),
    }
    base03222 = engine.base03222
    old_base = {
        "SITES": list(base03222.SITES),
        "SITE_NAMES": dict(base03222.SITE_NAMES),
        "OUT": base03222.OUT,
        "DOC": base03222.DOC,
        "PROMPT": base03222.PROMPT,
        "TOTAL_TIMESTEPS": base03222.TOTAL_TIMESTEPS,
        "CHECKPOINT_STEPS": list(base03222.CHECKPOINT_STEPS),
        "load_split": base03222.load_split,
        "build_selection": base03222.build_selection,
    }
    old_input_root = ppo_safe_rendering.MULTISITE_INPUT_ROOT
    status = "completed"
    error = ""
    try:
        engine.TASK_ID = TASK_ID
        engine.TASK_NAME = TASK_NAME
        engine.BASE_OUT = OUT
        engine.BASE_DOC = DOC
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = ROOT / cfg["weather_augmentation"]["augmented_input_root"]
        engine.STATION = STATION
        engine.SITES = [STATION]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        engine.patch_base_module(OUT, DOC, int(cfg["training"]["total_timesteps"]), list(map(int, cfg["training"]["checkpoint_steps"])))
        base03222.SITES = [STATION]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update({STATION: SITE})
        base03222.load_split = lambda: split.copy()
        base03222.build_selection = lambda _split: selection.copy()
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = engine.LOWIC_INPUT_ROOT
        base03222.main()
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old_input_root
        for key, value in old_values.items():
            setattr(engine, key, value)
        for key, value in old_base.items():
            if key == "SITE_NAMES":
                base03222.SITE_NAMES.clear()
                base03222.SITE_NAMES.update(value)
            else:
                setattr(base03222, key, value)
    copied = write_clean_names(OUT, TASK_ID)
    if status == "completed":
        audit, summary = audit_validation_actions(OUT, TASK_ID)
    else:
        audit, summary = pd.DataFrame(), pd.DataFrame()
    write_record(cfg, prep, split, copied, audit, summary, status)
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": status,
        "error": error[-4000:] if error else "",
        "output_root": rel(OUT),
        "record_md": rel(DOC),
        "augmented_input_root": prep["augmented_input_root"],
        "mapping_csv": prep["mapping_csv"],
        "train_variant_count": prep["train_variant_count"],
        "validation_original_count": prep["validation_original_count"],
        "validation_action_audit": rel(OUT / "audits" / f"{TASK_ID}_validation_action_weather_audit.csv") if status == "completed" else "",
        "validation_action_audit_summary": rel(OUT / "audits" / f"{TASK_ID}_validation_action_weather_audit_summary.csv") if status == "completed" else "",
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if status != "completed":
        raise RuntimeError(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(run(cfg_path, dry_run=args.dry_run), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
