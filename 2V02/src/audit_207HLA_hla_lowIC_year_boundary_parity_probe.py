"""207HLA: read-only HLA DSSAT two-digit-year parity probe.

The probe clones the exact 2005 WTH meteorology to 2029 and 2030, evaluates
the already-frozen 204HLA 10K checkpoint, and compares physical outcomes.
No training, reward, action grid, mask, cultivar, soil, or source file changes
are allowed.  A new small input root is created and never overwritten.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for module_path in [ROOT, ROOT / "src"]:
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import ppo_safe_rendering
import run_all_year_direct_action_safe_ppo as direct_ppo
import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as base03222
import run_sya_lowIC_binary_timing_maskableppo_042_10 as engine


TASK_ID = "207HLA"
TASK_NAME = "hla_lowIC_year_boundary_parity_probe"
CONFIG = ROOT / "configs" / f"{TASK_ID}_{TASK_NAME}.json"
RUN_204 = ROOT / "benchmark_results" / "204HLA_hla_lowIC_10y_six_weather_balanced_aug_maskableppo"
INVENTORY = RUN_204 / "evaluation" / "204HLA_training_checkpoint_inventory.csv"
OUT = ROOT / "benchmark_results" / f"{TASK_ID}_{TASK_NAME}"
RECORD = ROOT / "docs" / f"{TASK_ID}_{TASK_NAME}_record.md"
DATA_RE = re.compile(r"^(\s*)(\d{4})(\d{3})(\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s+[-+]?\d*\.?\d+\s*)$")
LOG_ROW_RE = re.compile(
    r"^\s*1\s+MZ\s+1\s+(?P<flo>-?\d+)\s+(?P<mat>-?\d+)\s+"
    r"(?P<topwt>-?\d+)\s+(?P<harwt>-?\d+)\s+(?P<rain>-?\d+)\s+"
    r"(?P<tirr>-?\d+)\s+(?P<cet>-?\d+)\s+(?P<pesw>-?\d+)\s+"
    r"(?P<tnup>-?\d+)\s+(?P<tnlf>-?\d+)"
)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meteorology_signature(path: Path) -> str:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DATA_RE.match(line)
        if match:
            values.append(match.group(3) + match.group(4).strip())
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def clone_weather(source: Path, target: Path, year: int) -> None:
    output: list[str] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        match = DATA_RE.match(line)
        if match:
            output.append(f"{match.group(1)}{int(year):04d}{match.group(3)}{match.group(4)}")
        else:
            output.append(line)
    target.write_text("\n".join(output) + "\n", encoding="utf-8")


def parse_last_log_row(path: Path) -> dict[str, int]:
    matches = [LOG_ROW_RE.match(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    rows = [match for match in matches if match]
    if not rows:
        raise RuntimeError(f"No DSSAT summary row found in {rel(path)}")
    return {key: int(value) for key, value in rows[-1].groupdict().items()}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refuse to overwrite existing diagnostic: {rel(OUT)}")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_root = ROOT / cfg["source_input_root"]
    probe_root = ROOT / cfg["probe_input_root"]
    if probe_root.exists():
        raise FileExistsError(f"Refuse to overwrite existing probe input root: {rel(probe_root)}")
    required = [source_root / "HL" / "CNHL0501.WTH", INVENTORY, RUN_204 / "204HLA_formal_result.json"]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing 207HLA inputs: " + "; ".join(missing))

    for sub in ["configs", "evaluation", "logs", "daily_outputs", "rendered_inputs"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG, OUT / "configs" / CONFIG.name)
    shutil.copytree(source_root, probe_root)
    source_weather = probe_root / "HL" / "CNHL0501.WTH"
    paths = {2005: source_weather}
    for year in map(int, cfg["clone_years"]):
        target = probe_root / "HL" / f"CNHL{year % 100:02d}01.WTH"
        clone_weather(source_weather, target, year)
        paths[year] = target

    signatures = {year: meteorology_signature(path) for year, path in paths.items()}
    if len(set(signatures.values())) != 1:
        raise RuntimeError("207HLA clone meteorology is not identical")
    provenance = pd.DataFrame(
        [
            {
                "year": year,
                "weather_file": rel(path),
                "sha256": sha256(path),
                "meteorology_signature_ignoring_year": signatures[year],
            }
            for year, path in paths.items()
        ]
    )
    provenance.to_csv(OUT / "configs" / f"{TASK_ID}_weather_clone_provenance.csv", index=False, encoding="utf-8-sig")

    selection = pd.DataFrame(
        [
            {
                "station_code": "HLA",
                "year": year,
                "scenario_type": f"207HLA_exact_2005_weather_year_{year}",
                "weather_file": path.name,
            }
            for year, path in paths.items()
        ]
    )
    selection.to_csv(OUT / "configs" / f"{TASK_ID}_probe_selection.csv", index=False, encoding="utf-8-sig")

    inventory = pd.read_csv(INVENTORY, keep_default_na=False)
    inventory["checkpoint_step"] = pd.to_numeric(inventory["checkpoint_step"], errors="raise").astype(int)
    model_rows = inventory[
        inventory["checkpoint_step"].eq(int(cfg["checkpoint_step"]))
        & inventory["run_status"].astype(str).str.startswith("ok")
    ]
    if len(model_rows) != 1:
        raise RuntimeError("Expected exactly one 204HLA 10K checkpoint")
    model_row = model_rows.iloc[0].copy()

    old: dict[str, Any] = {
        "engine_lowic": engine.LOWIC_INPUT_ROOT,
        "engine_i": list(engine.BINARY_IRRIGATION_LEVELS),
        "engine_n": list(engine.BINARY_NITROGEN_LEVELS),
        "base_out": base03222.OUT,
        "base_seed": base03222.SEED,
        "base_sites": list(base03222.SITES),
        "base_site_names": dict(base03222.SITE_NAMES),
        "renderer_root": ppo_safe_rendering.MULTISITE_INPUT_ROOT,
    }
    rows: list[dict[str, Any]] = []
    try:
        engine.LOWIC_INPUT_ROOT = probe_root
        engine.BINARY_IRRIGATION_LEVELS = [0.0, 15.0, 30.0, 45.0]
        engine.BINARY_NITROGEN_LEVELS = [0.0, 40.0, 80.0, 120.0]
        base03222.OUT = OUT
        base03222.SEED = 0
        base03222.SITES = ["HLA"]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update({"HLA": "HLA"})
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = probe_root
        run_config = engine.load_config()
        run_config["paths"]["output_root"] = rel(OUT)
        env_config = direct_ppo.build_env_config(run_config, selection)
        for year in paths:
            result = base03222.evaluate_checkpoint(run_config, env_config, model_row, year)
            log = OUT / "logs" / "HLA" / f"HLA_{year}_HLA_{year}_032_10_ckpt10000_eval.log"
            result.update(parse_last_log_row(log))
            result["weather_sha256"] = sha256(paths[year])
            result["meteorology_signature_ignoring_year"] = signatures[year]
            rows.append(result)
    finally:
        engine.LOWIC_INPUT_ROOT = old["engine_lowic"]
        engine.BINARY_IRRIGATION_LEVELS = old["engine_i"]
        engine.BINARY_NITROGEN_LEVELS = old["engine_n"]
        base03222.OUT = old["base_out"]
        base03222.SEED = old["base_seed"]
        base03222.SITES = old["base_sites"]
        base03222.SITE_NAMES.clear()
        base03222.SITE_NAMES.update(old["base_site_names"]
        )
        ppo_safe_rendering.MULTISITE_INPUT_ROOT = old["renderer_root"]

    results = pd.DataFrame(rows).sort_values("year")
    reference = results[results["year"].eq(int(cfg["source_year"]))].iloc[0]
    results["yield_delta_vs_2005_kg_ha"] = pd.to_numeric(results["final_grnwt"], errors="coerce") - float(reference["final_grnwt"])
    results["action_sequence_equal_2005"] = results["action_sequence"].astype(str).eq(str(reference["action_sequence"]))
    results["parity_yield_within_1kg_ha"] = results["yield_delta_vs_2005_kg_ha"].abs().le(1.0)
    results.to_csv(OUT / "evaluation" / f"{TASK_ID}_year_boundary_parity_results.csv", index=False, encoding="utf-8-sig")

    year_2029 = results[results["year"].eq(2029)].iloc[0]
    year_2030 = results[results["year"].eq(2030)].iloc[0]
    boundary_supported = bool(
        year_2029["parity_yield_within_1kg_ha"]
        and not year_2030["parity_yield_within_1kg_ha"]
        and int(year_2029["mat"]) > 0
        and int(year_2030["mat"]) < 0
    )
    result = {
        "task": f"{TASK_ID}_{TASK_NAME}",
        "status": "completed_diagnostic_only",
        "training_run": False,
        "meteorology_identical_ignoring_year": len(set(signatures.values())) == 1,
        "year_boundary_failure_supported": boundary_supported,
        "interpretation": "A sharp 2029/2030 parity break supports a DSSAT YYDDD century-boundary failure in the pseudo-year augmentation path." if boundary_supported else "The probe does not isolate a sharp 2029/2030 boundary; keep the cause at the broader pseudo-year rendering level.",
        "results_csv": rel(OUT / "evaluation" / f"{TASK_ID}_year_boundary_parity_results.csv"),
        "record": rel(RECORD),
    }
    (OUT / f"{TASK_ID}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    display = results[["year", "final_grnwt", "total_irrigation", "total_n", "flo", "mat", "rain", "yield_delta_vs_2005_kg_ha", "action_sequence_equal_2005", "parity_yield_within_1kg_ha"]].copy()
    record = [
        "# 207HLA HLA 两位年份边界同天气克隆探针",
        "",
        "## 结论",
        "",
        f"- `year_boundary_failure_supported={boundary_supported}`。",
        f"- {result['interpretation']}",
        "- 该探针不训练；复用 204HLA 10K checkpoint、原 reward、16 格动作和 mask。",
        "- 2005、2029、2030 的逐日 SRAD/TMAX/TMIN/RAIN 完全一致，只有 WTH DATE 的四位年份不同。",
        "",
        "## 结果",
        "",
        display.to_markdown(index=False),
        "",
        "## 文件",
        "",
        f"- 配置：`{rel(CONFIG)}`。",
        f"- 代码：`{rel(Path(__file__))}`。",
        f"- 天气克隆来源：`{rel(source_root / 'HL' / 'CNHL0501.WTH')}`；原文件未修改。",
        f"- 克隆审计：`{rel(OUT / 'configs' / f'{TASK_ID}_weather_clone_provenance.csv')}`。",
        f"- 逐年结果：`{result['results_csv']}`。",
    ]
    RECORD.write_text("\n".join(record) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
