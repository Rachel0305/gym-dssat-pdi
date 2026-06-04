from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


STATIONS = ["HLA", "SYA", "LCA", "YCA", "FQA"]
COMMON_STATIONS = ["HLA", "LCA", "YCA", "FQA"]
WEATHER_COLUMNS = ["SRAD", "TMAX", "TMIN", "RAIN"]
SCAN_EXTENSIONS = {".jinja2", ".sol", ".wth", ".cul", ".xls", ".xlsx"}


@dataclass(frozen=True)
class SourceSpec:
    variable: str
    path: Path
    stations: list[str]
    source_column: str | None = None


def normalize_text(value: object) -> str:
    return str(value).strip().replace("\ufeff", "")


def find_file(my_data: Path, *, exact: str | None = None, includes: list[str] | None = None) -> Path:
    if exact:
        path = my_data / exact
        if path.exists():
            return path
    includes = includes or []
    matches = []
    for path in my_data.iterdir():
        name = path.name
        if all(token in name for token in includes):
            matches.append(path)
    if not matches:
        criteria = exact or " + ".join(includes)
        raise FileNotFoundError(f"Could not find my_data file matching {criteria}")
    return sorted(matches, key=lambda p: len(p.name))[0]


def scan_my_data(my_data: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(my_data.iterdir(), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in SCAN_EXTENSIONS:
            rows.append(
                {
                    "name": path.name,
                    "extension": path.suffix,
                    "size_bytes": path.stat().st_size,
                    "last_modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "my_data_scan.csv", index=False, encoding="utf-8-sig")
    return df


def read_excel_all_sheets(path: Path) -> pd.DataFrame:
    frames = []
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet, header=1)
        frame["source_file"] = path.name
        frame["source_sheet"] = sheet
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def pick_column(columns: list[object], keywords: list[str], *, all_keywords: bool = False) -> str:
    clean_columns = [normalize_text(c) for c in columns]
    scored: list[tuple[int, str]] = []
    for col in clean_columns:
        haystack = col.upper()
        hits = [kw for kw in keywords if kw.upper() in haystack]
        if all_keywords and len(hits) != len(keywords):
            continue
        if hits:
            scored.append((len(hits), col))
    if not scored:
        raise ValueError(f"No column matched keywords {keywords}. Columns: {clean_columns}")
    return sorted(scored, key=lambda item: (-item[0], len(item[1])))[0][1]


def coerce_numeric(series: pd.Series) -> pd.Series:
    text = (
        series.astype(str)
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "－": pd.NA,
                "-": pd.NA,
                "--": pd.NA,
                "—": pd.NA,
                "/": pd.NA,
                "\\": pd.NA,
                "缺测": pd.NA,
                "缺": pd.NA,
                "999999": pd.NA,
                "99999": pd.NA,
                "9999": pd.NA,
                "32766": pd.NA,
            }
        )
    )
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("℃", "", regex=False).str.replace("mm", "", regex=False)
    text = text.str.replace("MJ/m^2", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def normalize_source(spec: SourceSpec) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = read_excel_all_sheets(spec.path)
    raw.columns = [normalize_text(c) for c in raw.columns]

    station_col = pick_column(raw.columns, ["生态站代码"])
    year_col = pick_column(raw.columns, ["年"])
    month_col = pick_column(raw.columns, ["月"])
    day_col = pick_column(raw.columns, ["日"])

    if spec.source_column:
        value_col = spec.source_column
    elif spec.variable == "SRAD":
        value_col = pick_column(raw.columns, ["总辐射总量"])
    elif spec.variable == "TMAX":
        value_col = pick_column(raw.columns, ["日最大值"])
    elif spec.variable == "TMIN":
        value_col = pick_column(raw.columns, ["日最小值"])
    elif spec.variable == "RAIN":
        try:
            value_col = pick_column(raw.columns, ["20-20", "合计"], all_keywords=True)
        except ValueError:
            value_col = pick_column(raw.columns, ["RAIN", "降雨", "降水", "合计"])
    else:
        raise ValueError(f"Unsupported variable: {spec.variable}")

    values = coerce_numeric(raw[value_col])
    rain_blank_to_zero = 0
    if spec.variable == "RAIN":
        rain_blank_to_zero = int(values.isna().sum())
        values = values.fillna(0)

    data = pd.DataFrame(
        {
            "station": raw[station_col].astype(str).str.strip(),
            "year": coerce_numeric(raw[year_col]),
            "month": coerce_numeric(raw[month_col]),
            "day": coerce_numeric(raw[day_col]),
            spec.variable: values,
        }
    )
    data = data[data["station"].isin(spec.stations)].copy()
    data = data[data["year"] >= 2000].copy()
    for col in ["year", "month", "day"]:
        data[col] = data[col].astype("Int64")
    data["date"] = pd.to_datetime(
        {
            "year": data["year"].astype(float),
            "month": data["month"].astype(float),
            "day": data["day"].astype(float),
        },
        errors="coerce",
    )
    invalid_dates = int(data["date"].isna().sum())
    data = data.dropna(subset=["date"]).copy()
    data = data[["station", "date", spec.variable]].drop_duplicates(["station", "date"], keep="last")
    metadata = {
        "variable": spec.variable,
        "source_file": spec.path.name,
        "source_column": value_col,
        "rows_after_filter": int(len(data)),
        "invalid_dates": invalid_dates,
        "numeric_missing_before_fill": int(data[spec.variable].isna().sum()),
        "rain_blank_to_zero": rain_blank_to_zero if spec.variable == "RAIN" else 0,
        "stations_found": ",".join(sorted(data["station"].dropna().unique())),
        "year_min": int(data["date"].dt.year.min()) if len(data) else None,
        "year_max": int(data["date"].dt.year.max()) if len(data) else None,
    }
    return data, metadata


def build_specs(my_data: Path) -> list[SourceSpec]:
    common_srad = find_file(my_data, exact="D32.xls")
    common_temp = find_file(my_data, exact="T2.xls")
    common_rain = find_file(my_data, includes=["HLLCYCFQ", "降雨"])
    sy_srad = find_file(my_data, includes=["SYCS", "太阳辐射"])
    sy_temp = find_file(my_data, includes=["SYCS", "气温"])
    sy_rain = find_file(my_data, includes=["SYCS", "降雨"])
    return [
        SourceSpec("SRAD", common_srad, COMMON_STATIONS),
        SourceSpec("TMAX", common_temp, COMMON_STATIONS),
        SourceSpec("TMIN", common_temp, COMMON_STATIONS),
        SourceSpec("RAIN", common_rain, COMMON_STATIONS),
        SourceSpec("SRAD", sy_srad, ["SYA"]),
        SourceSpec("TMAX", sy_temp, ["SYA"]),
        SourceSpec("TMIN", sy_temp, ["SYA"]),
        SourceSpec("RAIN", sy_rain, ["SYA"]),
    ]


def merge_weather(sources: list[pd.DataFrame]) -> pd.DataFrame:
    by_variable: dict[str, list[pd.DataFrame]] = {variable: [] for variable in WEATHER_COLUMNS}
    for frame in sources:
        variables = [col for col in WEATHER_COLUMNS if col in frame.columns]
        if len(variables) != 1:
            raise ValueError(f"Each source frame must contain exactly one weather variable, got {variables}")
        by_variable[variables[0]].append(frame)

    combined_sources = []
    for variable, frames in by_variable.items():
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(["station", "date"], keep="last")
        combined_sources.append(combined[["station", "date", variable]])

    merged: pd.DataFrame | None = None
    for frame in combined_sources:
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["station", "date"], how="outer")
    assert merged is not None
    merged = merged.sort_values(["station", "date"]).reset_index(drop=True)
    merged["year"] = merged["date"].dt.year
    merged["month"] = merged["date"].dt.month
    merged["day"] = merged["date"].dt.day
    return merged[["station", "date", "year", "month", "day", *WEATHER_COLUMNS]]


def reindex_full_years(weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    inserted_rows = []
    for station, station_df in weather.groupby("station"):
        years = sorted(station_df["year"].dropna().astype(int).unique())
        station_df = station_df.set_index("date").sort_index()
        for year in years:
            expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            year_df = station_df[station_df["year"] == year].reindex(expected)
            missing_dates = int(year_df["station"].isna().sum())
            year_df["station"] = station
            year_df["date"] = year_df.index
            year_df["year"] = year
            year_df["month"] = year_df.index.month
            year_df["day"] = year_df.index.day
            frames.append(year_df.reset_index(drop=True))
            if missing_dates:
                inserted_rows.append({"station": station, "year": year, "inserted_missing_dates": missing_dates})
    out = pd.concat(frames, ignore_index=True)
    return out[["station", "date", "year", "month", "day", *WEATHER_COLUMNS]], pd.DataFrame(inserted_rows)


def fill_missing(weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather = weather.copy()
    logs = []
    for station in STATIONS:
        station_mask = weather["station"] == station
        for variable in WEATHER_COLUMNS:
            before = int(weather.loc[station_mask, variable].isna().sum())
            month_means = weather.loc[station_mask].groupby("month")[variable].transform("mean")
            fill_mask = station_mask & weather[variable].isna() & month_means.notna()
            weather.loc[fill_mask, variable] = month_means[fill_mask]
            filled_month = int(fill_mask.sum())

            annual_mean = weather.loc[station_mask, variable].mean()
            annual_mask = station_mask & weather[variable].isna()
            if not math.isnan(annual_mean):
                weather.loc[annual_mask, variable] = annual_mean
            filled_annual = int(annual_mask.sum())
            after = int(weather.loc[station_mask, variable].isna().sum())
            logs.append(
                {
                    "station": station,
                    "variable": variable,
                    "missing_before_fill": before,
                    "filled_by_station_month_mean": filled_month,
                    "filled_by_station_annual_mean": filled_annual,
                    "missing_after_fill": after,
                    "station_annual_mean_used": annual_mean if not math.isnan(annual_mean) else pd.NA,
                }
            )
    return weather, pd.DataFrame(logs)


def add_calendar_columns(weather: pd.DataFrame) -> pd.DataFrame:
    out = weather.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    dt = pd.to_datetime(out["date"])
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["doy"] = dt.dt.dayofyear
    out["year_doy"] = out["year"].astype(str) + out["doy"].astype(str).str.zfill(3)
    return out[["station", "date", "year", "month", "day", "doy", "year_doy", *WEATHER_COLUMNS]]


def summarize_years(weather: pd.DataFrame, inserted_dates: pd.DataFrame) -> pd.DataFrame:
    missing_by_year = (
        weather.groupby(["station", "year"])[WEATHER_COLUMNS]
        .apply(lambda df: df.isna().sum())
        .reset_index()
    )
    counts = weather.groupby(["station", "year"]).size().reset_index(name="row_count")
    summary = counts.merge(missing_by_year, on=["station", "year"], how="left")
    if inserted_dates.empty:
        summary["inserted_missing_dates"] = 0
    else:
        summary = summary.merge(inserted_dates, on=["station", "year"], how="left")
        summary["inserted_missing_dates"] = summary["inserted_missing_dates"].fillna(0).astype(int)
    return summary


def write_report(
    out_dir: Path,
    scan: pd.DataFrame,
    source_meta: pd.DataFrame,
    fill_log: pd.DataFrame,
    year_summary_before: pd.DataFrame,
    year_summary_after: pd.DataFrame,
    weather: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# 气象数据检查报告")
    lines.append("")
    lines.append("范围：仅完成 TASK_LEAVE_ONE_YEAR_STRATEGY.md 的前 6 步；未训练 PPO。")
    lines.append("")
    lines.append("## 1. my_data 文件扫描")
    lines.append(f"- 已扫描指定扩展名文件数：{len(scan)}")
    for ext, count in scan["extension"].value_counts().sort_index().items():
        lines.append(f"- {ext}: {count}")
    lines.append("")
    lines.append("## 2. 成功识别的气象字段")
    for row in source_meta.to_dict("records"):
        lines.append(
            f"- {row['variable']}：来源文件 {row['source_file']}，来源列 `{row['source_column']}`；"
            f"站点={row['stations_found']}；年份={row['year_min']}-{row['year_max']}；"
            f"填补前缺失={row['numeric_missing_before_fill']}；无效日期={row['invalid_dates']}"
        )
        if row.get("variable") == "RAIN":
            lines.append(f"  - RAIN 空白按无降雨处理为 0：{row.get('rain_blank_to_zero', 0)} 个单元格")
    lines.append("")
    lines.append("## 3. 整理后的数据文件")
    for station in STATIONS:
        station_df = weather[weather["station"] == station]
        years = sorted(station_df["year"].unique())
        year_text = f"{min(years)}-{max(years)}" if years else "none"
        lines.append(f"- {station}_weather_cleaned.csv：{len(station_df)} 行，年份 {year_text}")
    lines.append(f"- all_sites_weather_cleaned.csv：{len(weather)} 行")
    lines.append("")
    lines.append("## 4. 缺失值填补记录")
    for row in fill_log.to_dict("records"):
        lines.append(
            f"- {row['station']} {row['variable']}：填补前={row['missing_before_fill']}，"
            f"站点-月份均值填补={row['filled_by_station_month_mean']}，"
            f"站点全年均值兜底填补={row['filled_by_station_annual_mean']}，"
            f"填补后={row['missing_after_fill']}"
        )
    lines.append("")
    lines.append("## 5. 发现的问题和注意事项")
    after_problem = year_summary_after[WEATHER_COLUMNS].sum()
    remaining_missing = int(after_problem.sum())
    if remaining_missing:
        lines.append(f"- 填补后仍存在缺失单元格：{remaining_missing}。详见 data_check_by_year_after_fill.csv。")
    else:
        lines.append("- 按站点-月份均值及全年均值兜底填补后，SRAD/TMAX/TMIN/RAIN 均无剩余 NaN。")
    inserted_total = int(year_summary_before["inserted_missing_dates"].sum())
    if inserted_total:
        lines.append(
            f"- 清洗前补齐了 {inserted_total} 个缺失日历日期。"
            "具体站点-年份见 data_check_by_year_before_fill.csv。"
        )
    rain_zero_total = int(source_meta.get("rain_blank_to_zero", pd.Series(dtype=int)).sum())
    if rain_zero_total:
        lines.append(f"- 已根据用户确认，将 RAIN 原表空白解释为无降雨，并转换为 0；共转换 {rain_zero_total} 个单元格。")
    if "SYA" in set(weather["station"]):
        sy_years = sorted(weather.loc[weather["station"] == "SYA", "year"].unique())
        if sy_years and min(sy_years) > 2000:
            lines.append(f"- SYA 清洗后可用年份从 {min(sy_years)} 年开始，没有 2000 年记录。")
    lines.append("- 本阶段只整理气象数据；WTH 生成和年份分类尚未执行。")
    lines.append("")
    (out_dir / "data_check_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean five-site weather Excel files for leave-one-year experiments.")
    parser.add_argument("--my-data", default="my_data")
    parser.add_argument("--out-dir", default="weather_clean")
    args = parser.parse_args()

    my_data = Path(args.my_data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = scan_my_data(my_data, out_dir)
    specs = build_specs(my_data)

    source_frames = []
    metadata_rows = []
    for spec in specs:
        frame, metadata = normalize_source(spec)
        source_frames.append(frame)
        metadata_rows.append(metadata)

    source_meta = pd.DataFrame(metadata_rows)
    source_meta.to_csv(out_dir / "recognized_weather_fields.csv", index=False, encoding="utf-8-sig")

    weather = merge_weather(source_frames)
    weather, inserted_dates = reindex_full_years(weather)
    year_summary_before = summarize_years(weather, inserted_dates)
    weather, fill_log = fill_missing(weather)
    year_summary_after = summarize_years(weather, inserted_dates)
    weather = add_calendar_columns(weather)

    for station in STATIONS:
        station_df = weather[weather["station"] == station].copy()
        station_df.to_csv(out_dir / f"{station}_weather_cleaned.csv", index=False, encoding="utf-8-sig")
    weather.to_csv(out_dir / "all_sites_weather_cleaned.csv", index=False, encoding="utf-8-sig")
    fill_log.to_csv(out_dir / "missing_value_fill_log.csv", index=False, encoding="utf-8-sig")
    year_summary_before.to_csv(out_dir / "data_check_by_year_before_fill.csv", index=False, encoding="utf-8-sig")
    year_summary_after.to_csv(out_dir / "data_check_by_year_after_fill.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, scan, source_meta, fill_log, year_summary_before, year_summary_after, weather)
    print(f"Wrote cleaned weather data and reports to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
