"""004_23A: read-only audit of YC external WTH injection and augmentation design.

This runner reads source/provenance files only. It never imports DSSAT/Gym,
launches a simulator, or generates/rewrites weather data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK = "004_23A"
OUT = ROOT / "results/yc_random_weather_ppo/004_23A_yc_external_weather_injection_audit_and_augmentation_design"
REPORT = ROOT / "docs/yc_random_weather_004_23A_external_weather_injection_audit_and_augmentation_design.md"

SOURCES = {
    "004_03_runner": ROOT / "results/yc_random_weather_ppo/004_03/run_controlled_pilot.py",
    "yc_renderer": ROOT / "src/ppo_safe_rendering.py",
    "gym_dssat_source_snapshot": ROOT / "results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/dssat_pdi.py",
    "yc_filex_template_snapshot": ROOT / "results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/gym_dssat_pdi/envs/configs/maize/UFGA8201.jinja2",
    "weather_archive_contract": ROOT / "docs/random_weather_archival_contract.md",
    "004_17_decision": ROOT / "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/decision.json",
    "004_17_verified_manifest": ROOT / "results/yc_random_weather_ppo/004_17_yc_runtime_weather_recovery/weather_archive_manifest_verified.csv",
    "004_21_decision": ROOT / "results/yc_random_weather_ppo/004_21_yc_wgen_srad_generation_mechanism_and_correction_design/decision.json",
    "004_22_decision": ROOT / "results/yc_random_weather_ppo/004_22_yc_isolated_srad_floor_weather_only_pilot/decision.json",
    "004_22_report": ROOT / "docs/yc_random_weather_004_22_isolated_srad_floor_weather_only_pilot.md",
    "fitting_weather": ROOT / "results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv",
    "frozen_cli": ROOT / "results/yc_wgen_cli_pilot/003_06_05_02/final/CNYC.CLI",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def get_evidence(source_key: str, terms: list[str]) -> list[dict[str, Any]]:
    path = SOURCES[source_key]
    if not path.exists():
        return [{"source_key": source_key, "source_path": rel(path), "line": "", "evidence": "SOURCE_MISSING", "status": "UNKNOWN"}]
    lines = read_text(path).splitlines()
    found: list[dict[str, Any]] = []
    for term in terms:
        for number, line in enumerate(lines, 1):
            if term in line:
                found.append({
                    "source_key": source_key,
                    "source_path": rel(path),
                    "line": number,
                    "evidence": line.strip()[:350],
                    "status": "VERIFIED_IN_PROJECT_SOURCE",
                })
                break
        else:
            found.append({"source_key": source_key, "source_path": rel(path), "line": "", "evidence": f"NOT_FOUND: {term}", "status": "UNKNOWN"})
    return found


def inspect_manifest(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"exists": path.exists(), "rows": 0, "sets": {}, "canonical_files_present": 0}
    if not path.exists():
        return result
    set_counts: Counter[str] = Counter()
    present = 0
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            result["rows"] += 1
            set_counts[row.get("set", "UNKNOWN")] += 1
            canonical = row.get("canonical_csv_path", "").strip()
            if canonical and (path.parent / canonical).is_file():
                present += 1
    result["sets"] = dict(set_counts)
    result["canonical_files_present"] = present
    return result


def build_report(decision: dict[str, Any], evidence: list[dict[str, Any]], hashes: list[dict[str, Any]], designs: list[dict[str, Any]]) -> str:
    evidence_table = "\n".join(
        f"| `{item['source_key']}` | `{item['source_path']}:{item['line']}` | {item['evidence'].replace('|', '\\|')} | {item['status']} |"
        for item in evidence
    )
    hash_table = "\n".join(
        f"| `{item['source_key']}` | `{item['path']}` | `{item['sha256']}` | {item['status']} |"
        for item in hashes
    )
    design_table = "\n".join(
        f"| {item['method']} | {item['preserves']} | {item['support_gain']} | {item['main_risk']} | {item['role']} |"
        for item in designs
    )
    archive = decision["archive_inventory"]
    return f"""# YC 外部天气接入路径审计与增强设计（004_23A）

## 执行结论

```text
external_weather_injection_status = {decision['external_weather_injection_status']}
augmentation_design_status = {decision['augmentation_design_status']}
weather_generation_started = false
dssat_started = false
ppo_started = false
next_step = {decision['next_step']}
```

本轮是源码/provenance 静态审计和设计，不是天气生成或 crop experiment。没有导入 Gym/DSSAT，没有加载模型，没有读写旧天气文件；唯一写入是 004_23A 新产物。

## 接入链审计

| 来源 | 位置 | 观察到的证据 | 证据状态 |
|---|---|---|---|
{evidence_table}

项目代码显示 YC renderer 会按年构造站点 weather stem，并将 `.WTH` 加入 `auxiliary_file_paths`；环境 source snapshot 显示 auxiliary 文件复制到 DSSAT 临时目录，DSSAT 子进程以该目录为 working directory。004_03 的实验代码区分 `WTHER=W` 的 WGEN 方案与 measured-weather 路径；该项目历史也记录了 `WTHER=M` 未启动 WGEN、改用任务级 `WTHER=W` 后才得到随机 WGEN 天气。

**静态可行路径**：在任务隔离输入目录中，把 fitting-only candidate WTH 放到 FileX `WSTA` 所指向的 basename，并将该文件路径作为 weather auxiliary input；FileX 使用 measured-weather 模式。这样复用现有文件输入链，不需要改 canonical runtime 或重编 DSSAT。

**尚未证明**：本轮没有启动 runtime，因而没有证明特定候选文件被安装版 DSSAT 正确解析并逐日消费；项目 source snapshot 与当前安装二进制的完整 build identity 也没有在本任务重新绑定。故结果只达到 `STATICALLY_SUPPORTED_UNSMOKED`，不能声称 injection 已通过运行验证。

## 输入与归档状态

| 资产 | 路径 | SHA256 | 状态 |
|---|---|---|---|
{hash_table}

004_17 verified manifest 共登记 {archive['rows']} 行；按 set 统计为 `{json.dumps(archive['sets'], ensure_ascii=False)}`，manifest 指向且当前可见的 canonical CSV 为 {archive['canonical_files_present']} 个。该核对仅确认归档清单及文件存在，不在本任务重算气候覆盖结论。004_17 历史 raw WTH 状态是 `{decision['historical_raw_wth_status']}`；canonical state-derived CSV 与 raw WTH 必须继续区分。

## 增强候选比较

| 候选方法 | 保留结构 | 可增加的支持 | 主要风险 | 本轮定位 |
|---|---|---|---|---|
{design_table}

### 推荐的最小起步设计

先核验 2004–2013 fitting weather 日期与 YC planting/season context 的映射；只有起点/日期对齐可追溯时才按 DAP 分段，否则按固定月份匹配。随后做多变量连续 block resampling，整体搬运每日 `RAIN/SRAD/TMAX/TMIN` 向量，不做逐变量独立扰动。block 长度和分段规则须仅用 fitting 数据的连续性诊断预先冻结；随后做 fitting-only leave-one-year-out 检查，评估分布保持、时间结构与块边界伪影。整季 donor resampling作为低复杂度基线；条件多变量 residual synthesis暂列备选，只有前者无法满足已预先定义的天气 QA 时才考虑。

这类基于历史供体/块的方案可以把 fitting期已有的低辐射、降雨和温度组合保留到候选 weather 中，但不会凭空保证超出 2004–2013 支持；也不能预先宣称一定覆盖 2014–2023 中已暴露的年份。Observed 2014–2023仅可在方法冻结后作描述性外部 QA，不能参与参数拟合或方法优选。该约束与用户已明确的 YC 目标一致：关注各验证年的产量、水氮投入、效率与 reward 的尽可能好的权衡，而非要求每个单项指标都成为五情景冠军或改写成单一跨年均值目标。 

## 下一阶段 gate

建议先做 004_23B 的单情景 weather injection loop smoke：选一份现有、已知 provenance 的 YC WTH，在任务独立目录中按 WSTA basename 接入，确认 FileX mode、传入文件 hash、实际 runtime daily-state 的日期和 `SRAD/TMAX/TMIN/RAIN` 逐日一致，并确认临时目录/输入隔离。此 smoke 只验证路径闭环，不调 PPO、不更改配置、不形成作物性能结果。通过后再冻结 fitting-only block construction 和小批候选 weather-only QA；初始 20–40 套只是建议规模，非本轮执行。

### QA gate

- date/DAP 对齐、日序连续、目标 season 长度及 leap-year 规则均可追溯；
- 缺测/哨兵、单位、`RAIN >= 0`、`SRAD >= 0`、`TMAX >= TMIN` 检查；
- SRAD 与太阳几何 plausibility、降雨 occurrence/amount、热/干事件持续性和多变量联合结构；
- 与 fitting donors 的来源映射、block seam诊断、候选间重复率；
- raw WTH 与 canonical daily series 分开归档、各自 SHA256、FileX/WSTA/CLI/runtime provenance、episode到 archive ID映射；
- 2014–2023只作已暴露的外部描述性 QA，不用于生成参数或调参；held-out WGEN维持隔离。

## 当前边界与决策

004_22 的 source/build gate 阻止的是修改 DSSAT 内部 WGEN，不阻止未来复用外部 measured-weather 输入路径；也不自动证明外部接入成功。当前不建议直接做 `80→160` 同分布 WGEN扩容或 PPO retraining。下一步先通过一份既有天气文件的注入回环，之后才考虑 fitting-only weather candidate pilot。

```text
external_weather_injection_status = {decision['external_weather_injection_status']}
augmentation_design_status = {decision['augmentation_design_status']}
weather_generation_started = false
dssat_started = false
ppo_started = false
next_step = {decision['next_step']}
```

## 产物

- Prompt：`prompt_02/004_23A_yc_external_weather_injection_audit_and_augmentation_design.md`
- Runner：`src/run_004_23A_yc_external_weather_injection_audit_and_augmentation_design.py`
- 机器审计：`results/yc_random_weather_ppo/004_23A_yc_external_weather_injection_audit_and_augmentation_design/`
"""


def main() -> None:
    allowed_partial = {
        "source_hashes.csv",
        "external_weather_injection_evidence.csv",
        "augmentation_candidate_design.csv",
        "injection_path_audit.json",
        "decision.json",
        "execution_record.json",
    }
    if REPORT.exists() and not read_text(REPORT).startswith("# YC 外部天气接入路径审计与增强设计（004_23A）"):
        raise FileExistsError("004_23A report exists but is not this task's generated report; refusing to overwrite")
    if OUT.exists():
        unexpected = {path.name for path in OUT.iterdir()} - allowed_partial
        if unexpected:
            raise FileExistsError(f"004_23A output contains unrecognized files; refusing to overwrite: {sorted(unexpected)}")
    else:
        OUT.mkdir(parents=True, exist_ok=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    source_hash_rows: list[dict[str, Any]] = []
    for key, path in SOURCES.items():
        exists = path.is_file()
        source_hash_rows.append({
            "source_key": key,
            "path": rel(path),
            "exists": exists,
            "bytes": path.stat().st_size if exists else "",
            "sha256": sha256(path) if exists else "",
            "status": "READ_ONLY_INPUT" if exists else "MISSING",
        })
    write_csv(OUT / "source_hashes.csv", source_hash_rows, ["source_key", "path", "exists", "bytes", "sha256", "status"])

    terms = {
        "004_03_runner": [
            'args["random_weather"] = regime == "RANDOM_WEATHER_WGEN"',
            'args["auxiliary_file_paths"] = auxiliary',
            'instance._rseed1 = int(bootstrap_weather_seed)',
            'wgen_text, replacements = re.subn(',
            'r"\\g<1>W",',
            'any(mode != "W" for mode in modes)',
            'args["fileX_template_path"] = str(target)',
        ],
        "yc_renderer": [
            'standard_name = f"{info[\'weather_prefix\']}{int(year) % 100:02d}01.WTH"',
            'return f"{info[\'weather_prefix\']}{int(year) % 100:02d}01"',
            'shutil.copyfile(source_weather, rendered_weather)',
            '"random_weather": False',
            '"auxiliary_file_paths": [str(cultivar), str(weather), str(soil)]',
        ],
        "gym_dssat_source_snapshot": [
            "self.wther = 'W' if random_weather else 'M'",
            "def _copy_auxiliary_files(self):",
            "shutil.copyfile(path, f'{self._tmp_folder}/{file_name}')",
            "cwd=self._tmp_folder,",
        ],
        "yc_filex_template_snapshot": [
            "@L ID_FIELD WSTA....",
            "@N METHODS     WTHER",
        ],
        "weather_archive_contract": [
            "Preserve the actual runtime WTH file byte-for-byte",
            "Preserve the canonical daily series",
            "Call `src/archive_runtime_weather.py::archive_runtime_weather`",
        ],
        "004_22_report": [
            "source_build_gate = FAIL",
            "weather_generation_started = false",
            "当前容器没有可核对的安装版 Fortran 源码",
        ],
    }
    evidence: list[dict[str, Any]] = []
    for key, wanted in terms.items():
        evidence.extend(get_evidence(key, wanted))
    write_csv(OUT / "external_weather_injection_evidence.csv", evidence, ["source_key", "source_path", "line", "evidence", "status"])

    designs = [
        {
            "method": "Whole-season donor resampling",
            "preserves": "整季RAIN/SRAD/TMAX/TMIN联合轨迹、全部原始日序",
            "support_gain": "重抽样不会超出2004-2013供体支持；唯一季节数仍有限",
            "main_risk": "年份级样本仅10个，重复率高；标签年份与实际气候供体必须分开记录",
            "role": "低复杂度基线，不作为最终扩增首选",
        },
        {
            "method": "Season/date-context-aligned multivariate moving-block bootstrap",
            "preserves": "块内四变量联合关系、短期自相关与局部wet/dry和热/辐射事件；DAP仅在fitting context可核验时使用",
            "support_gain": "可组合fitting供体块形成新季节序列，但不保证超出单变量或联合供体支持",
            "main_risk": "拼接处可能产生不连续；block长度与DAP/月分层须fitting-only预先冻结",
            "role": "推荐的最小 weather-only pilot 候选",
        },
        {
            "method": "Fitting-only conditional multivariate residual synthesis",
            "preserves": "取决于模型；可显式保留阶段/湿干条件及变量协方差",
            "support_gain": "可能探索供体未出现的变量组合；必须证明物理与统计可行",
            "main_risk": "10个年份支撑参数有限；模型错设、尾部放大、RAIN离散分布与物理约束伪影",
            "role": "较复杂备选；只有块重采样不足且QA失败机制明确后再设计",
        },
    ]
    write_csv(OUT / "augmentation_candidate_design.csv", designs, ["method", "preserves", "support_gain", "main_risk", "role"])

    archive = inspect_manifest(SOURCES["004_17_verified_manifest"])
    d21 = json.loads(read_text(SOURCES["004_21_decision"])) if SOURCES["004_21_decision"].is_file() else {}
    d22 = json.loads(read_text(SOURCES["004_22_decision"])) if SOURCES["004_22_decision"].is_file() else {}
    historical = json.loads(read_text(SOURCES["004_17_decision"])) if SOURCES["004_17_decision"].is_file() else {}

    required_evidence = [
        "args[\"random_weather\"] = regime == \"RANDOM_WEATHER_WGEN\"",
        "\"auxiliary_file_paths\": [str(cultivar), str(weather), str(soil)]",
        "self.wther = 'W' if random_weather else 'M'",
        "shutil.copyfile(path, f'{self._tmp_folder}/{file_name}')",
        "cwd=self._tmp_folder,",
    ]
    combined = "\n".join(item["evidence"] for item in evidence)
    static_supported = all(any(token in line for line in combined.splitlines()) for token in required_evidence)
    status = "STATICALLY_SUPPORTED_UNSMOKED" if static_supported else "PARTIALLY_SUPPORTED"

    decision = {
        "task_id": TASK,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_weather_injection_status": status,
        "augmentation_design_status": "DESIGN_ONLY",
        "weather_generation_started": False,
        "dssat_started": False,
        "ppo_started": False,
        "next_step": "WEATHER_ONLY_INJECTION_SMOKE" if static_supported else "INSUFFICIENT",
        "static_path_components": {
            "year_station_weather_stem": "VERIFIED_IN_PROJECT_RENDERER",
            "WTH_in_auxiliary_file_paths": "VERIFIED_IN_PROJECT_RENDERER",
            "auxiliary_copy_to_runtime_temp_directory": "VERIFIED_IN_ARCHIVED_GYM_DSSAT_SOURCE_SNAPSHOT",
            "runtime_uses_candidate_values_exactly": "UNSMOKED",
            "snapshot_to_installed_binary_build_identity": "NOT_REVALIDATED_IN_004_23A",
        },
        "archive_inventory": archive,
        "historical_raw_wth_status": historical.get("raw_wth_emitted", "UNKNOWN"),
        "004_21_mechanism_status": d21.get("runtime_srad_mechanism_status", "UNKNOWN"),
        "004_22_source_build_gate": d22.get("source_build_gate", "UNKNOWN"),
        "augmentation_recommendation": "Start with fitting-only season/date-context-aligned multivariate block resampling; use DAP only when its calendar mapping is verified, otherwise month-match; keep whole-season donor as baseline and defer residual synthesis unless simpler candidates fail predefined weather QA.",
        "observed_2014_2023_role": "EXPOSED_EXTERNAL_DESCRIPTIVE_QA_ONLY; NOT_FOR_FITTING_OR_METHOD_SELECTION",
        "no_overwrite": True,
        "source_hash_manifest": "source_hashes.csv",
    }
    (OUT / "injection_path_audit.json").write_text(json.dumps({"decision": decision, "evidence": evidence}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    execution = {
        "task_id": TASK,
        "executed_at_utc": decision["executed_at_utc"],
        "mode": "READ_ONLY_SOURCE_AND_PROVENANCE_AUDIT",
        "commands_or_tools_started": ["python runner (stdlib only)"],
        "dssat_or_wgen_invoked": False,
        "gym_or_ppo_imported": False,
        "weather_generated_or_copied": False,
        "existing_experiment_outputs_modified": False,
        "input_files_hashed": sum(bool(row["exists"]) for row in source_hash_rows),
        "source_evidence_rows": len(evidence),
        "archive_inventory": archive,
    }
    (OUT / "execution_record.json").write_text(json.dumps(execution, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(build_report(decision, evidence, source_hash_rows, designs), encoding="utf-8")
    print(json.dumps({"task_id": TASK, "decision": decision, "outputs": [rel(path) for path in [OUT, REPORT]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
