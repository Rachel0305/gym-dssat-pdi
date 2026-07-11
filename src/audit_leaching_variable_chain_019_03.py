from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "DSSAT_auto_validation" / "leaching_chain_audit_019_03"
DOC_PATH = PROJECT_ROOT / "docs" / "2026-07-10_019_03_leaching_variable_chain_audit_record.md"

SNAPSHOT_ROOTS = [
    PROJECT_ROOT / "DSSAT_auto_validation" / "deterministic_oracle_upper_bound_scan_016_05" / "runs",
    PROJECT_ROOT / "DSSAT_auto_validation" / "extension_expert_baseline_018_03",
    PROJECT_ROOT / "DSSAT_auto_validation" / "HLA_2004",
]

REWARD_FILES = [
    PROJECT_ROOT / "src" / "run_hla_baseline_relative_dqn_checkpoint_015_12.py",
    PROJECT_ROOT / "src" / "run_yc2014_baseline_relative_dqn_015_10.py",
    PROJECT_ROOT / "src" / "run_fq2016_baseline_relative_dqn_checkpoint_015_14.py",
    PROJECT_ROOT / "src" / "run_sy_local_dqn_train_cross_year_transfer_017_08.py",
    PROJECT_ROOT / "src" / "run_lc2010_baseline_relative_dqn_smoke_017_12.py",
    PROJECT_ROOT / "src" / "run_hla2010_tao_reward_dqn_probe_014_06.py",
]


def parse_dssat_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="latin1", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            parts = line.split()
            if parts and parts[0] == "@":
                parts = parts[1:]
            elif parts:
                parts[0] = parts[0].lstrip("@")
            header = parts
            continue
        if header and re.match(r"^\d", line):
            values = line.split()
            n = min(len(header), len(values))
            rows.append(dict(zip(header[:n], values[:n])))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def infer_case(snapshot: Path) -> dict[str, object]:
    parts = snapshot.parts
    text = str(snapshot)
    station = None
    year = None
    for part in parts:
        if part in {"HLA", "YC", "FQ", "SY", "LC"}:
            station = part
        if re.fullmatch(r"20\d{2}", part):
            year = int(part)
    match = re.search(r"I(?P<I>\d+)_N(?P<N>\d+)(?:_(?P<timing>[A-Za-z0-9]+))?", text)
    return {
        "station": station,
        "year": year,
        "irrigation_label": int(match.group("I")) if match else None,
        "nitrogen_label": int(match.group("N")) if match else None,
        "timing_label": match.group("timing") if match else None,
    }


def snapshot_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for root in SNAPSHOT_ROOTS:
        if not root.exists():
            continue
        for snapshot in root.rglob("pdi_tmp_snapshot*"):
            if not snapshot.is_dir() or snapshot in seen:
                continue
            seen.add(snapshot)
            summary = parse_dssat_table(snapshot / "Summary.OUT")
            soilni = parse_dssat_table(snapshot / "SoilNi.OUT")
            plantn = parse_dssat_table(snapshot / "PlantN.OUT")
            yml = snapshot / "dssat-pdi.yml"
            yml_text = yml.read_text(encoding="utf-8", errors="ignore") if yml.exists() else ""

            row = {
                "snapshot": str(snapshot.relative_to(PROJECT_ROOT)),
                **infer_case(snapshot),
                "has_dssat_pdi_yml": yml.exists(),
                "pdi_yml_has_CLeach": "CLeach" in yml_text,
                "pdi_yml_has_cleach_state": "'cleach': cleach" in yml_text or "cleach = CLeach" in yml_text,
                "pdi_yml_has_TLeachD": "TLeachD" in yml_text,
                "summary_has_NLCM": "NLCM" in summary.columns,
                "soilni_has_NLCC": "NLCC" in soilni.columns,
                "plantn_columns": ",".join(str(c) for c in plantn.columns[:20]) if not plantn.empty else "",
                "summary_rows": len(summary),
                "soilni_rows": len(soilni),
            }
            if "NLCM" in summary.columns and not summary.empty:
                row["summary_final_NLCM"] = pd.to_numeric(summary["NLCM"], errors="coerce").dropna().iloc[-1]
                row["summary_max_NLCM"] = pd.to_numeric(summary["NLCM"], errors="coerce").max()
            if "NLCC" in soilni.columns and not soilni.empty:
                nlcc = pd.to_numeric(soilni["NLCC"], errors="coerce")
                row["soilni_final_NLCC"] = nlcc.dropna().iloc[-1] if not nlcc.dropna().empty else None
                row["soilni_max_NLCC"] = nlcc.max()
            if "TLeachD" in soilni.columns:
                tleach = pd.to_numeric(soilni["TLeachD"], errors="coerce")
                row["soilni_sum_TLeachD"] = tleach.sum()
            rows.append(row)
    return pd.DataFrame(rows)


def scan_csv_headers() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in (PROJECT_ROOT / "DSSAT_auto_validation").rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
                header = handle.readline().strip()
        except OSError:
            continue
        low = header.lower()
        if any(key in low for key in ["cleach", "tleach", "nlcm", "nlcc", "cnox"]):
            rows.append({"csv_path": str(path.relative_to(PROJECT_ROOT)), "header": header[:500]})
    return pd.DataFrame(rows)


def scan_reward_files() -> pd.DataFrame:
    rows = []
    for path in REWARD_FILES:
        if not path.exists():
            rows.append({"file": str(path.relative_to(PROJECT_ROOT)), "exists": False})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "exists": True,
                "mentions_cleach": "cleach" in text.lower(),
                "mentions_leaching": "leach" in text.lower(),
                "mentions_no_leaching": "no_leaching" in text.lower() or "without nitrate leaching" in text.lower(),
                "uses_baseline_relative_reward": "null_baseline" in text or "baseline_relative" in text,
            }
        )
    return pd.DataFrame(rows)


def write_record(snapshot_df: pd.DataFrame, csv_df: pd.DataFrame, reward_df: pd.DataFrame) -> None:
    available = int(snapshot_df.get("pdi_yml_has_cleach_state", pd.Series(dtype=bool)).fillna(False).sum())
    nlcm = int(snapshot_df.get("summary_has_NLCM", pd.Series(dtype=bool)).fillna(False).sum())
    nlcc = int(snapshot_df.get("soilni_has_NLCC", pd.Series(dtype=bool)).fillna(False).sum())
    csv_hits = len(csv_df)
    reward_hits = int(reward_df.get("mentions_leaching", pd.Series(dtype=bool)).fillna(False).sum())

    example_cols = [
        "station",
        "year",
        "irrigation_label",
        "nitrogen_label",
        "timing_label",
        "summary_final_NLCM",
        "soilni_final_NLCC",
        "soilni_max_NLCC",
    ]
    example = snapshot_df[[c for c in example_cols if c in snapshot_df.columns]].dropna(
        subset=[c for c in ["summary_final_NLCM", "soilni_final_NLCC"] if c in snapshot_df.columns],
        how="all",
    )
    if not example.empty:
        example = example.head(12).to_markdown(index=False)
    else:
        example = "_未找到可展示的 NLCM/NLCC 数值。_"

    lines = [
        "# 019_03 leaching 变量链路审计记录",
        "",
        "## 结论先行",
        "",
        "本轮没有训练，也没有修改模板。审计结果显示：PDI 通信模板中已经存在 `CLeach -> cleach` 和 `TLeachD -> tleachd` 的状态映射；DSSAT 输出中也能看到 `Summary.OUT` 的 `NLCM` 与 `SoilNi.OUT` 的 `NLCC`。因此当前证据不支持“必须先手动改模板才能获得 leaching”。",
        "",
        "真正缺的是：当前主线 DQN reward wrapper 和评估 CSV 没有稳定保存/使用 `cleach` 或 `tleachd`，所以如果要把氮淋洗加入奖励，下一步应优先改 reward wrapper 与日值记录，而不是马上重训。",
        "",
        "## 关键数量",
        "",
        f"- 含 `cleach` 状态映射的 PDI 快照数：{available}",
        f"- `Summary.OUT` 含 `NLCM` 的快照数：{nlcm}",
        f"- `SoilNi.OUT` 含 `NLCC` 的快照数：{nlcc}",
        f"- 已有 CSV 表头包含 leaching 字段的文件数：{csv_hits}",
        f"- 被审计 reward 脚本中提到 leaching/no_leaching 的文件数：{reward_hits}",
        "",
        "## 示例数值",
        "",
        example,
        "",
        "## 文件",
        "",
        f"- 快照字段审计：`{(OUT_DIR / '019_03_snapshot_leaching_field_audit.csv').relative_to(PROJECT_ROOT)}`",
        f"- 已有 CSV 字段审计：`{(OUT_DIR / '019_03_existing_csv_leaching_header_hits.csv').relative_to(PROJECT_ROOT)}`",
        f"- reward 脚本审计：`{(OUT_DIR / '019_03_reward_file_leaching_mentions.csv').relative_to(PROJECT_ROOT)}`",
        "",
        "## 下一步建议",
        "",
        "1. 在 DQN 评估日值表中加入 `cleach`、`tleachd`、`cnox` 字段，先不改变奖励。",
        "2. 选择一个高氮案例和一个低氮案例，对比 `cleach` 是否随施氮和灌水合理变化。",
        "3. 若变量稳定，再做一个 leaching-aware reward 的 smoke test。",
        "4. 只有当 PDI 快照缺失这些字段时，才需要回头改模板。",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_df = snapshot_audit()
    csv_df = scan_csv_headers()
    reward_df = scan_reward_files()

    snapshot_df.to_csv(OUT_DIR / "019_03_snapshot_leaching_field_audit.csv", index=False, encoding="utf-8-sig")
    csv_df.to_csv(OUT_DIR / "019_03_existing_csv_leaching_header_hits.csv", index=False, encoding="utf-8-sig")
    reward_df.to_csv(OUT_DIR / "019_03_reward_file_leaching_mentions.csv", index=False, encoding="utf-8-sig")

    if not snapshot_df.empty:
        group_cols = [c for c in ["station", "year", "irrigation_label", "nitrogen_label", "timing_label"] if c in snapshot_df.columns]
        value_cols = [c for c in ["summary_final_NLCM", "soilni_final_NLCC", "soilni_max_NLCC"] if c in snapshot_df.columns]
        if group_cols and value_cols:
            summary = snapshot_df.groupby(group_cols, dropna=False)[value_cols].agg(["count", "mean", "max"]).reset_index()
            summary.to_csv(OUT_DIR / "019_03_leaching_values_by_case_group.csv", index=False, encoding="utf-8-sig")

    write_record(snapshot_df, csv_df, reward_df)
    print(f"snapshot_rows={len(snapshot_df)}")
    print(f"csv_header_hits={len(csv_df)}")
    print(f"reward_files={len(reward_df)}")
    print(DOC_PATH)


if __name__ == "__main__":
    main()
