from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "DSSAT_auto_validation" / "five_site_wue_nue_019_10"
BASELINE = (
    ROOT
    / "DSSAT_auto_validation"
    / "extension_expert_baseline_018_03"
    / "018_03_clean_multisite_comparison_with_extension_expert.csv"
)


def p(relative: str) -> Path:
    return ROOT / relative


RAW_SOURCE_MAP: dict[tuple[str, str], Path] = {
    ("HLA", "Null"): p("DSSAT_auto_validation/HLA_2004/hla_2010_2015_four_scenario_with_ppo/null/2010/pdi_tmp_snapshot/Summary.OUT"),
    ("HLA", "DSSAT auto"): p("DSSAT_auto_validation/HLA_2004/hla_2010_2015_four_scenario_with_ppo/dssat_auto/2010/pdi_tmp_snapshot/Summary.OUT"),
    ("HLA", "DQN best checkpoint"): p("DSSAT_auto_validation/HLA_2004/hla_baseline_relative_dqn_checkpoint_015_12/2010/baseline_relative_seed0_50000steps/checkpoint_35000/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("YC", "Null"): p("DSSAT_auto_validation/multisite_new_cultivar_yc2014_four_scenario_smoke_013_03/null/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("YC", "DSSAT auto"): p("DSSAT_auto_validation/multisite_new_cultivar_yc2014_four_scenario_smoke_013_03/dssat_auto/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("YC", "DQN best checkpoint"): p("DSSAT_auto_validation/yc2014_unified_dqn_checkpoint_diagnostic_015_04/seed0/checkpoint_5000/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("FQ", "Null"): p("DSSAT_auto_validation/fq2016_four_scenario_process_017_02/runs/null/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("FQ", "DSSAT auto"): p("DSSAT_auto_validation/fq2016_four_scenario_process_017_02/runs/dssat_auto/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("FQ", "DQN best checkpoint"): p("DSSAT_auto_validation/fq2016_baseline_relative_dqn_checkpoint_015_14/seed1_50000steps/pdi_tmp_snapshot_eval_30000/Summary.OUT"),
    ("SY", "Null"): p("DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08/runs/2014/seed0/null/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("SY", "DSSAT auto"): p("DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08/runs/2014/seed0/dssat_auto/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("SY", "DQN ckpt15000"): p("DSSAT_auto_validation/sy_local_dqn_train_cross_year_transfer_017_08/eval_runs/2014/seed0/dqn_ckpt15000/pdi_tmp_snapshot_eval/Summary.OUT"),
    ("LC", "Null"): p("DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2010/null/pdi_tmp_snapshot/Summary.OUT"),
    ("LC", "DSSAT auto"): p("DSSAT_auto_validation/lc_fixed_input_year_screening_017_11/runs/2010/dssat_auto/pdi_tmp_snapshot/Summary.OUT"),
    ("LC", "DQN best checkpoint"): p("DSSAT_auto_validation/lc2010_baseline_relative_dqn_smoke_017_12/seed0_5000steps/pdi_tmp_snapshot_eval_5000/Summary.OUT"),
}


for site, year in {"HLA": 2010, "YC": 2014, "FQ": 2016, "SY": 2014, "LC": 2010}.items():
    RAW_SOURCE_MAP[(site, "Official extension expert fixed DAP")] = p(
        f"DSSAT_auto_validation/extension_expert_baseline_018_03/{site}{year}/extension_expert_fixed_dap/pdi_tmp_snapshot_eval/Summary.OUT"
    )


def clean_name(name: str) -> str:
    return name.strip().rstrip(".")


def parse_value(text: str):
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return text
    if value == -99:
        return None
    return value


def parse_summary_out(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("@"))
    header = lines[header_index]
    matches = list(re.finditer(r"\S+", header))
    names = [clean_name(match.group()) for match in matches[1:]]
    token_ends = [match.end() for match in matches[1:]]
    field_starts = [0] + token_ends[:-1]

    rows: list[dict[str, object]] = []
    for line in lines[header_index + 1 :]:
        if line.startswith("@") or line.startswith("!") or line.startswith("*") or not line.strip():
            continue
        if not re.match(r"^\s*\d", line):
            continue
        values = [
            parse_value(line[start:stop])
            for start, stop in zip(field_starts, token_ends)
        ]
        rows.append(dict(zip(names, values)))
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    return rows


def num(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def select_matching_row(
    rows: list[dict[str, object]], expected_yield: float, expected_i: float, expected_n: float
) -> tuple[dict[str, object], float, int]:
    scored = []
    for index, row in enumerate(rows):
        hwam, ircm, nicm = num(row, "HWAM"), num(row, "IRCM"), num(row, "NICM")
        if hwam is None or ircm is None or nicm is None:
            continue
        score = abs(hwam - expected_yield) + abs(ircm - expected_i) + abs(nicm - expected_n)
        scored.append((score, -index, row, index))
    if not scored:
        raise ValueError("No Summary.OUT row contains numeric HWAM/IRCM/NICM")
    score, _, row, index = min(scored, key=lambda item: (item[0], item[1]))
    if abs(num(row, "HWAM") - expected_yield) > 2.0:
        raise ValueError(f"Yield mismatch: expected {expected_yield}, got {num(row, 'HWAM')}")
    if abs(num(row, "IRCM") - expected_i) > 2.0:
        raise ValueError(f"Irrigation mismatch: expected {expected_i}, got {num(row, 'IRCM')}")
    if abs(num(row, "NICM") - expected_n) > 2.0:
        raise ValueError(f"Nitrogen mismatch: expected {expected_n}, got {num(row, 'NICM')}")
    return row, score, index


def valid_positive(value: float | None) -> bool:
    return value is not None and value > 0


def native_or_none(value: float | None) -> float | None:
    return value if value is not None and value >= 0 else None


def main() -> None:
    baselines = pd.read_csv(BASELINE)
    selected_labels = {"Null", "DSSAT auto", "Official extension expert fixed DAP", "DQN best checkpoint", "DQN ckpt15000"}
    baselines = baselines[baselines["scenario_label"].isin(selected_labels)].copy()

    missing = [str(path.relative_to(ROOT)) for path in RAW_SOURCE_MAP.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Summary.OUT files:\n" + "\n".join(missing))

    metric_rows = []
    audit_rows = []
    for _, target in baselines.iterrows():
        site = str(target["site"])
        label = str(target["scenario_label"])
        source = RAW_SOURCE_MAP[(site, label)]
        parsed = parse_summary_out(source)
        row, score, row_index = select_matching_row(
            parsed,
            float(target["grain_yield_kg_ha"]),
            float(target["irrigation_mm"]),
            float(target["nitrogen_kg_ha"]),
        )

        hwam = num(row, "HWAM")
        ircm = num(row, "IRCM")
        nicm = num(row, "NICM")
        nucm = num(row, "NUCM")
        nlcm = num(row, "NLCM")
        etcp = num(row, "ETCP")
        ypem = native_or_none(num(row, "YPEM"))
        ypim = native_or_none(num(row, "YPIM"))
        ypnam = native_or_none(num(row, "YPNAM"))
        ypnum = native_or_none(num(row, "YPNUM"))

        wp_et_recalc = hwam / etcp / 10.0 if valid_positive(etcp) else None
        wp_et_native = ypem * 0.1 if ypem is not None else None
        iwp_native = ypim * 0.1 if valid_positive(ircm) and ypim is not None else None
        pfp_n_native = ypnam if valid_positive(nicm) and ypnam is not None else None
        nute_native = ypnum if valid_positive(nucm) and ypnum is not None else None
        pnb_n = nucm / nicm if valid_positive(nicm) and nucm is not None else None

        metric_rows.append(
            {
                "site": site,
                "station": target["station"],
                "year": int(target["year"]),
                "scenario": label,
                "HWAM_kg_ha": hwam,
                "IRCM_mm_summary": ircm,
                "irrigation_mm_event_table": float(target["irrigation_mm"]),
                "NICM_kg_ha": nicm,
                "NUCM_kg_ha": nucm,
                "NLCM_kg_ha": nlcm,
                "ETCM_mm": num(row, "ETCM"),
                "ETCP_mm_planting_to_harvest": etcp,
                "EPCM_mm": num(row, "EPCM"),
                "EPCP_mm_planting_to_harvest": num(row, "EPCP"),
                "WP_ET_kg_m3_DSSAT_YPEM": wp_et_native,
                "WP_ET_kg_m3_recalculated": wp_et_recalc,
                "WP_ET_abs_check_error": abs(wp_et_native - wp_et_recalc) if wp_et_native is not None and wp_et_recalc is not None else None,
                "IWP_gross_kg_m3_DSSAT_YPIM": iwp_native,
                "PFP_N_kg_grain_per_kg_applied_N_DSSAT_YPNAM": pfp_n_native,
                "NUtE_kg_grain_per_kg_uptake_N_DSSAT_YPNUM": nute_native,
                "PNB_N_uptake_per_applied_N": pnb_n,
                "source_summary_out": source.relative_to(ROOT).as_posix(),
            }
        )
        audit_rows.append(
            {
                "site": site,
                "year": int(target["year"]),
                "scenario": label,
                "parsed_rows": len(parsed),
                "selected_row_index": row_index,
                "match_score_H_W_N": score,
                "expected_yield": float(target["grain_yield_kg_ha"]),
                "selected_HWAM": hwam,
                "expected_irrigation": float(target["irrigation_mm"]),
                "selected_IRCM": ircm,
                "expected_nitrogen": float(target["nitrogen_kg_ha"]),
                "selected_NICM": nicm,
                "source_summary_out": source.relative_to(ROOT).as_posix(),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values(["site", "scenario"])
    audit = pd.DataFrame(audit_rows).sort_values(["site", "scenario"])

    comparisons = []
    for site, group in metrics.groupby("site"):
        dqn = group[group["scenario"].str.startswith("DQN")].iloc[0]
        for comparator_label in ["DSSAT auto", "Official extension expert fixed DAP"]:
            comparator = group[group["scenario"] == comparator_label].iloc[0]
            comparisons.append(
                {
                    "site": site,
                    "year": int(dqn["year"]),
                    "comparator": comparator_label,
                    "yield_diff_kg_ha": dqn["HWAM_kg_ha"] - comparator["HWAM_kg_ha"],
                    "WP_ET_diff_kg_m3": dqn["WP_ET_kg_m3_DSSAT_YPEM"] - comparator["WP_ET_kg_m3_DSSAT_YPEM"],
                    "WP_ET_ratio": dqn["WP_ET_kg_m3_DSSAT_YPEM"] / comparator["WP_ET_kg_m3_DSSAT_YPEM"],
                    "NUtE_diff_kg_kg": dqn["NUtE_kg_grain_per_kg_uptake_N_DSSAT_YPNUM"] - comparator["NUtE_kg_grain_per_kg_uptake_N_DSSAT_YPNUM"],
                    "NUtE_ratio": dqn["NUtE_kg_grain_per_kg_uptake_N_DSSAT_YPNUM"] / comparator["NUtE_kg_grain_per_kg_uptake_N_DSSAT_YPNUM"],
                    "DQN_PFP_N": dqn["PFP_N_kg_grain_per_kg_applied_N_DSSAT_YPNAM"],
                    "comparator_PFP_N": comparator["PFP_N_kg_grain_per_kg_applied_N_DSSAT_YPNAM"],
                    "PFP_N_comparable": pd.notna(dqn["PFP_N_kg_grain_per_kg_applied_N_DSSAT_YPNAM"]) and pd.notna(comparator["PFP_N_kg_grain_per_kg_applied_N_DSSAT_YPNAM"]),
                    "N_leaching_diff_kg_ha": dqn["NLCM_kg_ha"] - comparator["NLCM_kg_ha"],
                    "input_pareto_with_1kg_yield_tolerance": (
                        dqn["HWAM_kg_ha"] >= comparator["HWAM_kg_ha"] - 1.0
                        and dqn["irrigation_mm_event_table"] <= comparator["irrigation_mm_event_table"]
                        and dqn["NICM_kg_ha"] <= comparator["NICM_kg_ha"]
                    ),
                }
            )
    comparison = pd.DataFrame(comparisons).sort_values(["site", "comparator"])

    dictionary = pd.DataFrame(
        [
            {
                "metric": "WP_ET",
                "full_name": "基于生育季实际蒸散的籽粒水分生产率",
                "formula": "HWAM / ETCP / 10",
                "DSSAT_native_code": "YPEM * 0.1",
                "unit": "kg m-3",
                "formal_role": "主水分效率指标；所有ETCP>0情景可比",
                "zero_input_rule": "ETCP<=0时NA",
                "limitation": "反映降雨、灌溉和初始土壤水共同形成的总蒸散生产率",
            },
            {
                "metric": "IWP_gross",
                "full_name": "单位灌溉水籽粒生产率（总产量口径）",
                "formula": "HWAM / irrigation / 10",
                "DSSAT_native_code": "YPIM * 0.1",
                "unit": "kg m-3",
                "formal_role": "辅助灌溉效率指标",
                "zero_input_rule": "灌溉量=0时NA",
                "limitation": "分子包含降雨和土壤水贡献，不是灌溉的因果边际效应",
            },
            {
                "metric": "PFP_N",
                "full_name": "氮肥偏生产力",
                "formula": "HWAM / NICM",
                "DSSAT_native_code": "YPNAM",
                "unit": "kg grain kg-1 applied N",
                "formal_role": "主施肥投入效率指标；仅施氮情景可比",
                "zero_input_rule": "NICM=0时NA，不填0或无穷大",
                "limitation": "未扣除土壤本底氮贡献，跨站点解释需谨慎",
            },
            {
                "metric": "NUtE",
                "full_name": "植株吸收氮的内部利用效率",
                "formula": "HWAM / NUCM",
                "DSSAT_native_code": "YPNUM",
                "unit": "kg grain kg-1 crop N uptake",
                "formal_role": "跨零施氮与施氮情景的辅助氮效率指标",
                "zero_input_rule": "NUCM<=0时NA",
                "limitation": "高值可能来自氮胁迫和低吸氮，必须与产量、胁迫和NUCM一起解释",
            },
            {
                "metric": "PNB_N",
                "full_name": "植株吸氮量与施氮量之比",
                "formula": "NUCM / NICM",
                "DSSAT_native_code": "由NUCM和NICM计算",
                "unit": "kg crop N uptake kg-1 applied N",
                "formal_role": "辅助氮平衡指标",
                "zero_input_rule": "NICM=0时NA",
                "limitation": "包含土壤供氮，不等同于肥料氮回收率；大于1可能表示土壤供氮/土壤氮消耗",
            },
            {
                "metric": "N_leaching",
                "full_name": "季节氮淋洗量",
                "formula": "NLCM",
                "DSSAT_native_code": "NLCM",
                "unit": "kg N ha-1",
                "formal_role": "环境效益/约束指标",
                "zero_input_rule": "允许为0",
                "limitation": "不能替代完整氮平衡，需与排水和土壤残留氮结合解释",
            },
            {
                "metric": "AE_N_future",
                "full_name": "氮肥农学效率",
                "formula": "(Y_N - Y_0_matched) / NICM",
                "DSSAT_native_code": "需配对反事实",
                "unit": "kg grain increase kg-1 applied N",
                "formal_role": "未来严格因果补充指标",
                "zero_input_rule": "NICM=0时NA",
                "limitation": "Y_0必须保持同一灌溉时机、IC和其余管理，仅把施氮设为0",
            },
            {
                "metric": "RE_N_future",
                "full_name": "肥料氮表观回收效率",
                "formula": "(NUCM_N - NUCM_0_matched) / NICM",
                "DSSAT_native_code": "需配对反事实",
                "unit": "kg uptake increase kg-1 applied N",
                "formal_role": "未来严格因果补充指标",
                "zero_input_rule": "NICM=0时NA",
                "limitation": "NUCM_0必须来自相同灌溉和初始条件的无氮对照",
            },
            {
                "metric": "IWP_incremental_future",
                "full_name": "灌溉增量水分生产率",
                "formula": "(Y_I - Y_noI_matched) / irrigation / 10",
                "DSSAT_native_code": "需配对反事实",
                "unit": "kg m-3",
                "formal_role": "未来严格因果补充指标",
                "zero_input_rule": "灌溉量=0时NA",
                "limitation": "无灌溉对照必须保持相同施氮时机、IC和其余管理",
            },
        ]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUT / "019_10_native_wue_nue_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT / "019_10_dqn_vs_baseline_efficiency_comparison.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "019_10_source_match_audit.csv", index=False, encoding="utf-8-sig")
    dictionary.to_csv(OUT / "019_10_metric_dictionary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
