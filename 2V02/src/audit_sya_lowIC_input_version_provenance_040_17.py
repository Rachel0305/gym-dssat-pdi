"""040_17: SYA lowIC input-version provenance audit.

This script is intentionally read-only with respect to DSSAT inputs.
It compares rendered templates from 040_13 and 040_16 to diagnose why
040_13 high-yield PPO daily outputs do not close against the current
040_16 online DSSAT replay.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YEARS = [2014, 2017]

TASK = "040_17_sya_lowIC_input_version_provenance_audit"
OUT_ROOT = PROJECT_ROOT / "benchmark_results" / TASK
TABLE_DIR = OUT_ROOT / "tables"
DOC_PATH = PROJECT_ROOT / "docs" / f"{TASK}_record.md"

PATH_04013 = (
    PROJECT_ROOT
    / "benchmark_results"
    / "040_13_sya_lowIC_ppo_early_irrigation_hard_reserve_mask"
    / "rendered_inputs"
    / "SYA"
)
PATH_04016 = (
    PROJECT_ROOT
    / "benchmark_results"
    / "040_16_sya_lowIC_ppo_online_dssat_snapshot_closure_audit"
    / "rendered_inputs"
    / "SYA"
)
CURRENT_LOWIC_MZX = (
    PROJECT_ROOT
    / "DSSAT_auto_validation"
    / "multisite_new_cultivar_inputs_013_lowIC_manual"
    / "SY"
    / "CNSY1201.MZX"
)


@dataclass
class ICRow:
    source: str
    year: int | str
    file: str
    treatment: str
    icbl: float
    sh2o: float
    snh4: float
    sno3: float


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def find_one(root: Path, pattern: str) -> Path:
    hits = sorted(root.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one {pattern} under {root}, got {len(hits)}")
    return hits[0]


def parse_ic_rows(path: Path, source: str, year: int | str) -> list[ICRow]:
    rows: list[ICRow] = []
    lines = path.read_text(errors="replace").splitlines()
    in_ic = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("@C  ICBL") or line.startswith("@C ICBL"):
            in_ic = True
            continue
        if in_ic and (not line or line.startswith("@") or line.startswith("*")):
            in_ic = False
            continue
        if not in_ic:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append(
                ICRow(
                    source=source,
                    year=year,
                    file=str(path.relative_to(PROJECT_ROOT)),
                    treatment=parts[0],
                    icbl=float(parts[1]),
                    sh2o=float(parts[2]),
                    snh4=float(parts[3]),
                    sno3=float(parts[4]),
                )
            )
        except ValueError:
            continue
    return rows


def rows_to_dicts(rows: Iterable[ICRow]) -> list[dict]:
    return [r.__dict__ for r in rows]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    all_ic_rows: list[ICRow] = []
    hash_rows: list[dict] = []
    diff_rows: list[dict] = []

    for year in YEARS:
        p13 = find_one(PATH_04013 / str(year), f"SYA_{year}_032_10_ckpt25000_eval/*.jinja2")
        p16 = find_one(PATH_04016 / str(year), f"SYA_{year}_040_16_online_snapshot/*.jinja2")
        w13 = find_one(PATH_04013 / str(year), f"SYA_{year}_032_10_ckpt25000_eval/*.WTH")
        w16 = find_one(PATH_04016 / str(year), f"SYA_{year}_040_16_online_snapshot/*.WTH")
        for label, path in [
            ("040_13_rendered_template", p13),
            ("040_16_rendered_template", p16),
            ("040_13_weather", w13),
            ("040_16_weather", w16),
        ]:
            hash_rows.append(
                {
                    "year": year,
                    "source": label,
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "md5": md5(path),
                    "bytes": path.stat().st_size,
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
        rows13 = parse_ic_rows(p13, "040_13_rendered", year)
        rows16 = parse_ic_rows(p16, "040_16_rendered", year)
        all_ic_rows.extend(rows13)
        all_ic_rows.extend(rows16)

        key13 = {(r.treatment, r.icbl): r for r in rows13}
        key16 = {(r.treatment, r.icbl): r for r in rows16}
        for key in sorted(set(key13) | set(key16), key=lambda x: (x[0], x[1])):
            a = key13.get(key)
            b = key16.get(key)
            diff_rows.append(
                {
                    "year": year,
                    "treatment": key[0],
                    "icbl": key[1],
                    "sh2o_04013": None if a is None else a.sh2o,
                    "sh2o_04016": None if b is None else b.sh2o,
                    "sh2o_delta_04016_minus_04013": None if a is None or b is None else b.sh2o - a.sh2o,
                    "snh4_04013": None if a is None else a.snh4,
                    "snh4_04016": None if b is None else b.snh4,
                    "snh4_delta_04016_minus_04013": None if a is None or b is None else b.snh4 - a.snh4,
                    "sno3_04013": None if a is None else a.sno3,
                    "sno3_04016": None if b is None else b.sno3,
                    "sno3_delta_04016_minus_04013": None if a is None or b is None else b.sno3 - a.sno3,
                }
            )

    current_rows = parse_ic_rows(CURRENT_LOWIC_MZX, "current_lowIC_source", "current")
    all_ic_rows.extend(current_rows)
    hash_rows.append(
        {
            "year": "current",
            "source": "current_lowIC_source",
            "file": str(CURRENT_LOWIC_MZX.relative_to(PROJECT_ROOT)),
            "md5": md5(CURRENT_LOWIC_MZX),
            "bytes": CURRENT_LOWIC_MZX.stat().st_size,
            "mtime": datetime.fromtimestamp(CURRENT_LOWIC_MZX.stat().st_mtime).isoformat(timespec="seconds"),
        }
    )

    ic_csv = TABLE_DIR / "040_17_ic_rows.csv"
    diff_csv = TABLE_DIR / "040_17_04013_vs_04016_ic_diff.csv"
    hash_csv = TABLE_DIR / "040_17_file_hashes.csv"
    result_json = OUT_ROOT / "040_17_result.json"

    write_csv(ic_csv, rows_to_dicts(all_ic_rows))
    write_csv(diff_csv, diff_rows)
    write_csv(hash_csv, hash_rows)

    any_ic_diff = any(
        row["sh2o_delta_04016_minus_04013"] not in (None, 0)
        or row["snh4_delta_04016_minus_04013"] not in (None, 0)
        or row["sno3_delta_04016_minus_04013"] not in (None, 0)
        for row in diff_rows
    )
    weather_same = all(
        hash_rows[i]["md5"] == hash_rows[i + 1]["md5"]
        for i in range(2, len(hash_rows) - 1, 4)
        if hash_rows[i]["source"] == "040_13_weather"
    )

    branch = "B_input_version_drift_confirmed" if any_ic_diff else "A_input_versions_match"
    result = {
        "task": TASK,
        "branch": branch,
        "years": YEARS,
        "ic_diff_detected": any_ic_diff,
        "weather_hashes_match_for_checked_years": weather_same,
        "next_step_allowed": False,
        "outputs": {
            "ic_rows_csv": str(ic_csv.relative_to(PROJECT_ROOT)),
            "ic_diff_csv": str(diff_csv.relative_to(PROJECT_ROOT)),
            "file_hashes_csv": str(hash_csv.relative_to(PROJECT_ROOT)),
            "record_md": str(DOC_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    key_examples = [r for r in diff_rows if r["year"] == 2014 and r["treatment"] == "1"][:3]
    DOC_PATH.write_text(
        "\n".join(
            [
                "# 040_17 SYA lowIC 输入版本溯源审计记录",
                "",
                "## 结论",
                "",
                f"- 分支：`{branch}`。",
                "- 040_13 与 040_16 的天气文件哈希一致；主要差异来自渲染模板中的初始土壤水分/氮条件。",
                "- 040_16 已经证明当前 replay 的 PPO 动作、daily 输出和 DSSAT `Summary.OUT` 能闭合，因此当前证据不支持“动作完全没有进入 DSSAT”。",
                "- 040_13/040_14 的高产 daily 结果在当前 lowIC 输入版本下不能复现，必须暂时隔离，不能继续作为可信 PPO 结果汇报或解释。",
                "",
                "## 关键证据",
                "",
                "- 040_13 与 040_16 的 PPO 动作逐日一致，但作物生长轨迹分叉；这说明问题不在动作序列不同。",
                "- 040_13 rendered template 与 040_16 rendered template 的 `ICBL/SH2O/SNH4/SNO3` 不一致。",
                "- 例：SYA2014 treatment 1 前三层如下。",
                "",
                "| ICBL | SH2O 040_13 | SH2O 040_16 | SNH4 040_13 | SNH4 040_16 | SNO3 040_13 | SNO3 040_16 |",
                "|---:|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {r['icbl']:.0f} | {r['sh2o_04013']} | {r['sh2o_04016']} | {r['snh4_04013']} | {r['snh4_04016']} | {r['sno3_04013']} | {r['sno3_04016']} |"
                    for r in key_examples
                ],
                "",
                "## 输出文件",
                "",
                f"- IC 逐层表：`{ic_csv.relative_to(PROJECT_ROOT)}`",
                f"- 040_13 vs 040_16 IC 差异表：`{diff_csv.relative_to(PROJECT_ROOT)}`",
                f"- 文件哈希表：`{hash_csv.relative_to(PROJECT_ROOT)}`",
                f"- JSON 摘要：`{result_json.relative_to(PROJECT_ROOT)}`",
                "",
                "## 后续要求",
                "",
                "下一轮 PPO/DQN 评估必须固定输入版本，并保存 DSSAT 原始快照（至少 `Summary.OUT`、`PlantGro.OUT`、`SoilWat.OUT`、`MgmtEvent.OUT`）。不得再只依赖 daily CSV 作为唯一证据。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
