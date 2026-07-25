from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_five_site_half_split_stress_aware_maskableppo_batch_032_22 as batch
from ppo_safe_rendering import (
    build_env_args,
    source_cultivar_path,
    source_soil_path,
    source_template_path,
    source_weather_path,
    target_weather_stem,
)


OUT = ROOT / "benchmark_results" / "033_05_multisite_input_ic1_render_audit"
DOC = ROOT / "docs" / "033_05_multisite_input_ic1_render_audit_record.md"
FACTORS = ["CU", "FL", "SA", "IC", "MP", "MI", "MF", "MR", "MC", "MT", "ME", "MH", "SM"]


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无记录。"
    work = df.copy()
    for col in work.select_dtypes(include=["number"]).columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").round(4)
    work = work.astype(object).where(pd.notna(work), "")
    header = "| " + " | ".join(map(str, work.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(work.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in work.to_numpy().tolist()]
    return "\n".join([header, sep, *rows])


def parse_treatment_one(text: str) -> dict[str, str]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("@N R O C TNAME"):
            for row in lines[idx + 1 :]:
                stripped = row.strip()
                if not stripped or stripped.startswith("@") or stripped.startswith("*"):
                    continue
                parts = stripped.split()
                if parts and parts[0] == "1":
                    return {"treatment_row": stripped, **dict(zip(FACTORS, parts[5 : 5 + len(FACTORS)]))}
    return {"treatment_row": ""}


def parse_ic_ids(text: str) -> set[str]:
    ids: set[str] = set()
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("*INITIAL CONDITIONS"):
            in_section = True
            continue
        if in_section and stripped.startswith("*"):
            break
        if not in_section or not stripped or stripped.startswith("@"):
            continue
        parts = stripped.split()
        if parts and re.fullmatch(r"\d+", parts[0]):
            ids.add(parts[0])
    return ids


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)

    config = batch.load_config()
    config["paths"]["output_root"] = str(OUT.relative_to(ROOT)).replace("\\", "/")
    split = batch.load_split()
    selection = batch.build_selection(split)
    env_config = batch.direct_ppo.build_env_config(config, selection)

    rows: list[dict] = []
    skipped: list[dict] = []
    for _, item in split.sort_values(["station_code", "year"]).iterrows():
        station = str(item["station_code"])
        year = int(item["year"])
        try:
            expected_source_weather = source_weather_path(station, year)
        except FileNotFoundError as exc:
            skipped.append(
                {
                    "station_code": station,
                    "site": item.get("site", ""),
                    "year": year,
                    "split": item["split"],
                    "reason": str(exc),
                }
            )
            continue
        year_info = batch.direct_ppo.find_year(env_config, station, year)
        args = build_env_args(
            station=station,
            year=year,
            planting_date=year_info["planting_date"],
            seed=0,
            config=config,
            run_tag=f"{station}_{year}_033_05_ic1_audit",
            evaluation=True,
            mode=config.get("runtime", {}).get("mode", "all"),
        )
        template = Path(args["fileX_template_path"])
        text = template.read_text(encoding="utf-8", errors="replace")
        tr = parse_treatment_one(text)
        ic_ids = parse_ic_ids(text)
        expected_wsta = target_weather_stem(station, year)
        wsta_tokens = sorted(set(re.findall(r"\bCN[A-Z]{2}(?:\d{2}01|20\d{2})\b", text)))
        rendered_wth = sorted(p.name for p in template.parent.glob("*.WTH"))
        aux_paths = [Path(p) for p in args["auxiliary_file_paths"]]
        row = {
            "station_code": station,
            "site": item.get("site", ""),
            "year": year,
            "split": item["split"],
            "IC": tr.get("IC", ""),
            "MI": tr.get("MI", ""),
            "MF": tr.get("MF", ""),
            "initial_condition_ids": ";".join(sorted(ic_ids, key=lambda x: int(x))),
            "ic_id_valid": tr.get("IC", "") in ic_ids,
            "expected_wsta": expected_wsta,
            "wsta_tokens": ";".join(wsta_tokens),
            "wsta_all_expected": wsta_tokens == [expected_wsta],
            "rendered_wth": ";".join(rendered_wth),
            "rendered_wth_ok": rendered_wth == [f"{expected_wsta}.WTH"],
            "source_template": str(source_template_path(station).relative_to(ROOT)).replace("\\", "/"),
            "source_weather": str(expected_source_weather.relative_to(ROOT)).replace("\\", "/"),
            "source_soil": str(source_soil_path(station).relative_to(ROOT)).replace("\\", "/"),
            "source_cultivar": str(source_cultivar_path(station).relative_to(ROOT)).replace("\\", "/"),
            "rendered_template": str(template.relative_to(ROOT)).replace("\\", "/"),
            "aux_paths": " ; ".join(str(p.relative_to(ROOT)).replace("\\", "/") for p in aux_paths),
            "treatment_row": tr.get("treatment_row", ""),
        }
        row["pass"] = bool(
            row["IC"] == "1"
            and row["MI"] == "1"
            and row["MF"] == "1"
            and row["ic_id_valid"]
            and row["wsta_all_expected"]
            and row["rendered_wth_ok"]
            and "DSSAT_auto_validation/multisite_new_cultivar_inputs_013" in row["source_template"]
            and "DSSAT_auto_validation/multisite_new_cultivar_inputs_013" in row["source_soil"]
            and "DSSAT_auto_validation/multisite_new_cultivar_inputs_013" in row["source_cultivar"]
            and "DSSAT_auto_validation/multisite_new_cultivar_inputs_013" in row["source_weather"]
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    csv_path = OUT / "033_05_multisite_input_ic1_render_audit.csv"
    fail_path = OUT / "033_05_multisite_input_ic1_render_audit_failures.csv"
    skipped_path = OUT / "033_05_multisite_input_ic1_render_audit_skipped_missing_wth.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df[~df["pass"]].to_csv(fail_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")

    passed = int(df["pass"].sum())
    total = len(df)
    by_station = df.groupby("station_code", as_index=False).agg(rows=("year", "count"), passed=("pass", "sum"))
    skipped_summary = (
        skipped_df.groupby(["station_code", "site", "split"], as_index=False)
        .agg(n_skipped=("year", "nunique"), skipped_years=("year", lambda s: ",".join(map(str, sorted(pd.to_numeric(s).astype(int).tolist())))))
        if not skipped_df.empty
        else pd.DataFrame()
    )

    DOC.write_text(
        "\n".join(
            [
                "# 033_05 multisite 输入源 IC=1 渲染审计记录",
                "",
                "## 任务性质",
                "",
                "本任务只渲染并解析即将进入 033_04 的五站点 half-split 年份输入，不跑 DSSAT、不训练。",
                "",
                "## 结果",
                "",
                f"- 总通过：{passed}/{total}",
                f"- 因 multisite 包缺少目标年 WTH 而跳过：{len(skipped_df)}",
                "",
                "## 按站点通过情况",
                "",
                md_table(by_station),
                "",
                "## 缺天气年份汇总",
                "",
                md_table(skipped_summary),
                "",
                "## 输出",
                "",
                f"- 全量表：`{csv_path.relative_to(ROOT)}`",
                f"- 失败表：`{fail_path.relative_to(ROOT)}`",
                f"- 缺天气跳过表：`{skipped_path.relative_to(ROOT)}`",
                "",
                "## 判定",
                "",
                "- 只有有天气数据的年份总通过率为 100% 时，才允许继续进入 033_04 正式训练。",
                "- 缺少 multisite WTH 的年份不得静默训练，必须记录在跳过表或先补齐天气。",
                "- 审计条件包括：`IC=1, MI=1, MF=1`、IC id 有效、WSTA 与目标年 WTH 一致、源模板/天气/土壤/品种均来自 multisite 输入包。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"passed {passed}/{total}")
    print(csv_path)
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
