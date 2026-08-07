"""046_02: one-config runner for the SYA binary-timing MaskablePPO comparison.

Only ``configs/046_02_sya_originIC_binary_timing_ppo.json`` is intended to be
edited.  The established 042_15 implementation remains untouched; this file
sets its input profile and output folder in one audited place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_multisite_input_ic1_four_baseline_rebuild_034_00 as baseline034
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


DEFAULT_CONFIG = ROOT / "configs" / "046_02_sya_originIC_binary_timing_ppo.json"
PROMPT = ROOT / "prompts" / "046_02_sya_originIC_configured_rebuild.md"
INPUT_PROFILES = {
    "originIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013",
    "lowIC": ROOT / "DSSAT_auto_validation" / "multisite_new_cultivar_inputs_013_lowIC_manual",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = ["task_id", "task_name", "station_code", "input_profile", "training", "actions", "scope"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"配置缺少字段: {missing}")
    if str(cfg["input_profile"]) not in INPUT_PROFILES:
        raise ValueError(f"input_profile 只能是 {list(INPUT_PROFILES)}，当前为 {cfg['input_profile']!r}")
    if str(cfg["station_code"]) != "SYA":
        raise ValueError("046_02 当前只允许 SYA；站点扩展应新开任务，不要静默复用本轮。")
    if not cfg["training"].get("checkpoint_steps"):
        raise ValueError("training.checkpoint_steps 不能为空")
    return cfg


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recorded_schedule_summary() -> dict[str, Any]:
    schedules = baseline034.recorded_template_schedules()
    schedule = schedules.get("SY", {})
    return {
        "recorded_schedule_source": rel(baseline034.RECORDED_TEMPLATE_DAILY),
        "recorded_schedule_source_exists": baseline034.RECORDED_TEMPLATE_DAILY.exists(),
        "recorded_template_event_days": len(schedule),
        "recorded_template_irrigation_mm": float(sum(float(v.get("amir", 0.0)) for v in schedule.values())),
        "recorded_template_nitrogen_kg_ha": float(sum(float(v.get("anfer", 0.0)) for v in schedule.values())),
        "recorded_template_daps": sorted(map(int, schedule.keys())),
    }


def preflight(cfg: dict[str, Any]) -> dict[str, Any]:
    root = INPUT_PROFILES[str(cfg["input_profile"])]
    source_mzx = root / "SY" / "CNSY1201.MZX"
    expected_train = list(map(int, cfg["scope"]["train_years"]))
    expected_validation = list(map(int, cfg["scope"]["validation_years"]))
    split = engine.base03222.load_split()
    split = split[split["station_code"].astype(str).eq("SYA")].copy()
    train_actual = sorted(pd.to_numeric(split.loc[split["split"].eq("train"), "year"], errors="coerce").dropna().astype(int).tolist())
    validation_actual = sorted(pd.to_numeric(split.loc[split["split"].eq("validation"), "year"], errors="coerce").dropna().astype(int).tolist())
    issues: list[str] = []
    if not root.exists():
        issues.append(f"输入目录不存在: {rel(root)}")
    if not source_mzx.exists():
        issues.append(f"SY 模板不存在: {rel(source_mzx)}")
    if train_actual != expected_train:
        issues.append(f"训练年份与既有半分割不一致: config={expected_train}, engine={train_actual}")
    if validation_actual != expected_validation:
        issues.append(f"验证年份与既有半分割不一致: config={expected_validation}, engine={validation_actual}")
    mzx_hash = sha256(source_mzx) if source_mzx.exists() else ""
    return {
        "task": f"{cfg['task_id']}_{cfg['task_name']}",
        "station_code": cfg["station_code"],
        "input_profile": cfg["input_profile"],
        "resolved_input_root": rel(root),
        "source_mzx": rel(source_mzx),
        "source_mzx_sha256": mzx_hash,
        "config_train_years": expected_train,
        "config_validation_years": expected_validation,
        "engine_train_years": train_actual,
        "engine_validation_years": validation_actual,
        "recorded_template": recorded_schedule_summary(),
        "issues": issues,
        "next_step_allowed": not issues,
    }


def output_root(cfg: dict[str, Any]) -> Path:
    return ROOT / "benchmark_results" / f"{cfg['task_id']}_{cfg['task_name']}"


def write_manifest(out: Path, cfg_path: Path, cfg: dict[str, Any], pf: dict[str, Any], status: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(exist_ok=True)
    shutil.copy2(cfg_path, out / "configs" / cfg_path.name)
    if PROMPT.exists():
        shutil.copy2(PROMPT, out / "configs" / PROMPT.name)
    payload = {"status": status, "config": cfg, "preflight": pf}
    path = out / "046_02_run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def copy_clean_names(out: Path) -> dict[str, str]:
    mapping = {
        "evaluation/042_10_training_checkpoint_inventory.csv": "evaluation/046_02_training_checkpoint_inventory.csv",
        "evaluation/042_10_checkpoint_validation_summary.csv": "evaluation/046_02_checkpoint_validation_summary.csv",
        "evaluation/042_10_validation_summary_by_station_checkpoint.csv": "evaluation/046_02_validation_summary_by_station_checkpoint.csv",
        "042_10_result.json": "046_02_engine_result.json",
    }
    done: dict[str, str] = {}
    for source_rel, target_rel in mapping.items():
        source, target = out / source_rel, out / target_rel
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            done[target_rel] = source_rel
    return done


def write_record(cfg: dict[str, Any], pf: dict[str, Any], copied: dict[str, str], status: str) -> Path:
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    lines = [
        f"# {cfg['task_id']} SYA originIC 配置化 MaskablePPO 重建记录",
        "",
        "## 状态",
        "",
        f"- `{status}`",
        f"- 输入 profile：`{pf['input_profile']}`",
        f"- 解析后输入目录：`{pf['resolved_input_root']}`",
        f"- SY 模板 SHA256：`{pf['source_mzx_sha256']}`",
        "",
        "## 框架保持不变",
        "",
        "- 沿用 042_15 的 binary-timing MaskablePPO、奖励和 safety constraints。",
        f"- 灌溉动作：`{cfg['actions']['irrigation_levels_mm']}` mm；施氮动作：`{cfg['actions']['nitrogen_levels_kg_ha']}` kg/ha。",
        f"- 训练步数：`{cfg['training']['total_timesteps']}`；checkpoint：`{cfg['training']['checkpoint_steps']}`。",
        "- 本轮唯一实验变量是 input profile：originIC。",
        "",
        "## recorded 模板预审计",
        "",
        f"- 模板来源：`{pf['recorded_template']['recorded_schedule_source']}`",
        f"- 模板总灌溉：`{pf['recorded_template']['recorded_template_irrigation_mm']}` mm；总施氮：`{pf['recorded_template']['recorded_template_nitrogen_kg_ha']}` kg/ha。",
        "- 该项是静态站点模板复用，不是逐年真实 recorded farmer；后续表图须保留此边界。",
        "",
        "## 规范化输出",
        "",
    ]
    for target, source in copied.items():
        lines.append(f"- `{target}`（复制自引擎兼容输出 `{source}`）")
    if pf["issues"]:
        lines.extend(["", "## 阻塞", "", *[f"- {x}" for x in pf["issues"]]])
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return doc


def run(cfg_path: Path, dry_run: bool, smoke: bool = False) -> dict[str, Any]:
    cfg = read_config(cfg_path)
    if smoke:
        # A separate folder prevents a source-input smoke from being confused
        # with or overwriting the formal 25K comparison.
        cfg = json.loads(json.dumps(cfg))
        cfg["task_name"] = f"{cfg['task_name']}_smoke2k"
        cfg["training"] = {"total_timesteps": 2000, "checkpoint_steps": [1000, 2000]}
    pf = preflight(cfg)
    out = output_root(cfg)
    if dry_run:
        manifest = write_manifest(out, cfg_path, cfg, pf, "dry_run")
        return {**pf, "mode": "dry_run", "manifest": rel(manifest)}
    if not pf["next_step_allowed"]:
        raise RuntimeError("preflight 未通过：" + "; ".join(pf["issues"]))

    input_root = INPUT_PROFILES[str(cfg["input_profile"])]
    doc = ROOT / "docs" / f"{cfg['task_id']}_{cfg['task_name']}_record.md"
    checkpoints = list(map(int, cfg["training"]["checkpoint_steps"]))
    total = int(cfg["training"]["total_timesteps"])
    old = {
        "TASK_ID": engine.TASK_ID, "TASK_NAME": engine.TASK_NAME, "BASE_OUT": engine.BASE_OUT,
        "BASE_DOC": engine.BASE_DOC, "PROMPT": engine.PROMPT, "LOWIC_INPUT_ROOT": engine.LOWIC_INPUT_ROOT,
        "STATION": engine.STATION, "SITES": list(engine.SITES),
        "BINARY_IRRIGATION_LEVELS": list(engine.BINARY_IRRIGATION_LEVELS),
        "BINARY_NITROGEN_LEVELS": list(engine.BINARY_NITROGEN_LEVELS),
    }
    try:
        engine.TASK_ID = str(cfg["task_id"])
        engine.TASK_NAME = str(cfg["task_name"])
        engine.BASE_OUT = out
        engine.BASE_DOC = doc
        engine.PROMPT = PROMPT
        engine.LOWIC_INPUT_ROOT = input_root
        engine.STATION = str(cfg["station_code"])
        engine.SITES = [str(cfg["station_code"])]
        engine.BINARY_IRRIGATION_LEVELS = list(map(float, cfg["actions"]["irrigation_levels_mm"]))
        engine.BINARY_NITROGEN_LEVELS = list(map(float, cfg["actions"]["nitrogen_levels_kg_ha"]))
        engine.run_training(total, checkpoints, suffix="")
    finally:
        for key, value in old.items():
            setattr(engine, key, value)
    copied = copy_clean_names(out)
    manifest = write_manifest(out, cfg_path, cfg, pf, "completed")
    record = write_record(cfg, pf, copied, "completed")
    return {
        **pf,
        "mode": "train_and_validate",
        "manifest": rel(manifest),
        "record_md": rel(record),
        "output_root": rel(out),
        "clean_outputs": copied,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="originIC 2K smoke，输出到独立 *_smoke2k 目录")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config).resolve()
    print(json.dumps(run(cfg_path, args.dry_run, args.smoke), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
