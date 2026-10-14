#!/usr/bin/env python3
"""Create and structurally validate the Chinese 003_05_01 YC weather deck."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "yc_weather_gapfill_finalize"
OUTPUT = ROOT / "docs" / "yc_weather_gapfill_finalize.pptx"
SUMMARY = RESULTS / "audit_summary.json"
GATES = RESULTS / "variable_gate_status.json"
RAIN = RESULTS / "rain_station_validation.json"
ANNUAL = RESULTS / "candidate_annual_summary.csv"

INK = RGBColor(34, 50, 55)
MUTED = RGBColor(101, 117, 119)
TEAL = RGBColor(26, 116, 112)
PALE_TEAL = RGBColor(227, 241, 238)
ORANGE = RGBColor(211, 111, 67)
PALE_ORANGE = RGBColor(250, 238, 229)
GOLD = RGBColor(208, 164, 72)
PALE_GOLD = RGBColor(249, 245, 226)
LINE = RGBColor(220, 229, 226)
PAPER = RGBColor(250, 252, 251)
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(81, 127, 99)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_text(slide, x, y, w, h, value, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.01)
    frame.margin_bottom = Inches(0.01)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = str(value)
    run.font.name = "Microsoft YaHei"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def add_rect(slide, x, y, w, h, fill, border=None):
    shape = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border or fill
    return shape


def base_slide(prs, title, section, page):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = PAPER
    add_rect(slide, 0, 0, 13.333, 0.10, TEAL)
    add_text(slide, 0.62, 0.34, 5.7, 0.28, section, 9, TEAL, True)
    add_text(slide, 0.62, 0.72, 12.1, 0.58, title, 24, INK, True)
    add_rect(slide, 0.62, 7.12, 12.08, 0.012, LINE)
    add_text(slide, 0.64, 7.20, 9.7, 0.22, "YC TRAIN WEATHER  |  003_05_01", 8, MUTED, True)
    add_text(slide, 12.0, 7.18, 0.7, 0.24, f"{page:02d} / 06", 8, MUTED, False, PP_ALIGN.RIGHT)
    return slide


def add_bullet(slide, x, y, w, value, size=14, marker=TEAL):
    add_rect(slide, x, y + 0.10, 0.08, 0.08, marker)
    add_text(slide, x + 0.20, y, w - 0.20, 0.48, value, size, INK)


def add_metric(slide, x, y, w, label, value, detail, accent=TEAL):
    add_rect(slide, x, y, w, 1.30, WHITE, LINE)
    add_rect(slide, x, y, 0.07, 1.30, accent)
    add_text(slide, x + 0.18, y + 0.12, w - 0.32, 0.25, label, 9, MUTED, True)
    add_text(slide, x + 0.18, y + 0.38, w - 0.32, 0.45, value, 22, accent, True)
    add_text(slide, x + 0.18, y + 0.91, w - 0.32, 0.25, detail, 8, MUTED)


def set_cell(cell, value, size=10, color=INK, bold=False, fill=None):
    cell.text = str(value)
    cell.margin_left = Inches(0.05)
    cell.margin_right = Inches(0.05)
    cell.margin_top = Inches(0.025)
    cell.margin_bottom = Inches(0.025)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    for paragraph in cell.text_frame.paragraphs:
        paragraph.space_after = Pt(0)
        for run in paragraph.runs:
            run.font.name = "Microsoft YaHei"
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color


def add_table(slide, x, y, w, h, headers, rows, widths=None, size=9):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    table = shape.table
    if widths:
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
    for col, header in enumerate(headers):
        set_cell(table.cell(0, col), header, size, WHITE, True, INK)
    for ri, row in enumerate(rows, start=1):
        for ci, value in enumerate(row):
            set_cell(table.cell(ri, ci), value, size, INK, False, WHITE if ri % 2 else PALE_TEAL)
    return shape


def backup_output():
    if not OUTPUT.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = RESULTS / "backups" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(OUTPUT, backup / OUTPUT.name)
    return str(backup.relative_to(ROOT))


def build() -> dict:
    summary = read_json(SUMMARY)
    gates = read_json(GATES)["variables"]
    station_audit = read_json(RAIN)
    selected_id = station_audit["selected_station"]["station_id"]
    selected = next(row for row in station_audit["stations"] if row["station_id"] == selected_id)
    selected_2013 = next(row for row in selected["annual_total_ratios"] if row["period"] == "2013")
    annual_rows = []
    with ANNUAL.open("r", encoding="utf-8-sig", newline="") as stream:
        import csv
        annual_rows = list(csv.DictReader(stream))
    annual_2004 = next(row for row in annual_rows if row.get("year") == "2004")
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = base_slide(prs, "YC 训练天气候选已通过变量级补齐", "决策摘要", 1)
    add_rect(slide, 0.62, 1.55, 7.15, 4.95, PALE_TEAL)
    add_text(slide, 0.96, 1.84, 6.35, 0.30, "最终状态", 11, TEAL, True)
    add_text(slide, 0.96, 2.20, 6.35, 0.60, summary["final_status"], 21, INK, True)
    add_text(slide, 0.96, 3.00, 6.25, 0.82,
             "温度与辐射缺口按独立 Gate 接受；降雨缺口仅接受通过重叠验证的近邻地面站值。", 16, INK)
    add_bullet(slide, 0.98, 4.12, 6.20, "NASA POWER：TMAX 107 天、TMIN 107 天、SRAD 151 天；RAIN 0 天。", 13)
    add_bullet(slide, 0.98, 4.82, 6.20, f"五个雨日：{selected_id} JINAN，{selected['distance_km']:.1f} km，事件一致率 {selected['event_agreement_rate']:.2%}。", 13)
    add_bullet(slide, 0.98, 5.52, 6.20, "本任务未生成 CLI，也未启动 WeatherMan、WGEN、DSSAT 或 PPO。", 13, ORANGE)
    add_metric(slide, 8.08, 1.65, 2.10, "天气日数", "3,653 / 3,653", "2004–2013", GREEN)
    add_metric(slide, 10.40, 1.65, 2.10, "候选 QC", "PASS", "完整性 / 物理 / provenance", GREEN)
    add_metric(slide, 8.08, 3.22, 2.10, "邻站候选", "8 / 12", "通过雨量 Gate", TEAL)
    add_metric(slide, 10.40, 3.22, 2.10, "泄漏上限", "2013", "2014+ 天气值未用", GREEN)
    add_text(slide, 8.15, 5.08, 4.25, 0.90, "训练天气 candidate 已形成，可交给独立的 train-only `.CLI` / WGEN pilot 任务审阅。", 14, INK, True)

    slide = base_slide(prs, "变量 Gate：只放行已通过的列", "验证结果", 2)
    rows = []
    for variable in ("TMAX", "TMIN", "SRAD"):
        gate = gates[variable]
        rows.append([variable, gate["overlap_n"], f"{gate['corrected_mae']:.3f}", f"{gate['corrected_rmse']:.3f}", f"{gate['pearson_r']:.4f}", summary["accepted_fill_counts"][variable], "PASS"])
    rain_gate = gates["RAIN"]
    rows.append(["RAIN (NASA)", rain_gate["overlap_n"], "-", "-", "-", 0, f"BLOCKED {rain_gate['event_agreement_rate']:.2%}"])
    add_table(slide, 0.72, 1.62, 11.88, 2.42,
             ["变量", "Overlap n", "校正 MAE", "校正 RMSE", "Pearson r", "NASA 填补日", "Gate"], rows,
             [1.55, 1.35, 1.45, 1.45, 1.35, 1.65, 3.08], 10)
    add_rect(slide, 0.72, 4.35, 11.88, 1.62, PALE_ORANGE)
    add_text(slide, 1.02, 4.62, 2.1, 0.28, "RAIN overlap", 10, ORANGE, True)
    add_text(slide, 1.02, 4.98, 3.1, 0.50, f"{rain_gate['event_agreement_days']} / {rain_gate['overlap_n']} 日", 20, ORANGE, True)
    add_text(slide, 4.40, 4.58, 7.80, 0.95,
             "71.98% 未达到本项目预先设定的 80% 门槛；阈值未下调。NASA RAIN 不进入 candidate，原官方 2005–2013 日值和正式 0 mm 原样保留。", 14, INK)

    slide = base_slide(prs, "近邻日雨量站比较", "地面站筛选", 3)
    station_rows = []
    for row in station_audit["stations"]:
        event = f"{row['event_agreement_rate']:.1%}" if row["event_agreement_rate"] is not None else "NA"
        station_rows.append([row["station_id"], row["station_name"], f"{row['distance_km']:.1f}", row["paired_days"], event, f"{row['prcp_valid_days_2004_target']}/5", row["gate_status"]])
    add_table(slide, 0.66, 1.47, 12.00, 5.25,
             ["站号", "站名", "km", "配对日", "事件一致率", "2004 五日", "Gate"], station_rows,
             [2.15, 1.75, 0.85, 1.20, 1.65, 1.35, 3.05], 8)
    add_text(slide, 0.78, 6.78, 11.8, 0.20,
             f"配对期 2005–2013；2013 年仅 {selected_2013['paired_days']}/{selected_2013['expected_days']} 日有效配对，未计入完整年总量比摘要。原始子集、flags 与文件 SHA 已记录。", 8, MUTED)

    slide = base_slide(prs, "五天降雨值与来源", "gap-fill 决议", 4)
    resolution = summary["five_day_resolution"]
    rain_rows = [[row["date"], f"{row['selected_rain_mm']:.1f} mm", selected_id, "s", "空白", "PASS"] for row in resolution]
    add_table(slide, 0.82, 1.65, 11.65, 3.30,
             ["日期", "PRCP", "Station", "Source flag", "Q flag", "决议"], rain_rows,
             [2.55, 1.55, 3.15, 1.35, 1.20, 1.85], 12)
    add_rect(slide, 0.82, 5.28, 11.65, 0.95, PALE_GOLD)
    add_text(slide, 1.10, 5.50, 11.05, 0.42,
             "五天均采用 Jinan 的 GHCN-Daily PRCP 0.0 mm。NASA POWER 同日也为 0.0 mm，但只作辅助比较，没有参与选择或填补。", 14, INK)
    add_text(slide, 0.88, 6.36, 11.45, 0.48,
             f"Jinan overlap：{selected['paired_days']} 日；事件一致 {selected['event_agreement_rate']:.2%}，wet-day precision {selected['wet_day_precision']:.2%}，recall {selected['wet_day_recall']:.2%}。湿日 precision 偏低，来源仅支持本次五天限定填补。", 9, MUTED)

    slide = base_slide(prs, "完整性、气候统计与填补比例", "Candidate QA", 5)
    add_metric(slide, 0.72, 1.55, 2.55, "完整天气日", "3,653", "无重复 / 无缺日", GREEN)
    add_metric(slide, 3.48, 1.55, 2.55, "物理异常", "0", "TMAX≥TMIN / SRAD≥0 / RAIN≥0", GREEN)
    add_metric(slide, 6.24, 1.55, 2.55, "2004 年降水", f"{float(annual_2004['annual_precipitation_mm']):.1f} mm", "ChinaFLUX 年产品同为 846.2", TEAL)
    add_metric(slide, 9.00, 1.55, 2.55, "2004 雨日", annual_2004["rainy_days"], f"最长干旱期 {annual_2004['longest_dry_spell_days']} 日", GOLD)
    gap_rows = []
    for variable in ("TMAX", "TMIN", "SRAD", "RAIN"):
        count = summary["candidate_gapfill_counts"][variable]
        gap_rows.append([variable, count, f"{count / 3653:.2%}"])
    add_table(slide, 0.82, 3.36, 5.45, 2.10, ["变量", "填补日", "占 3653 日"], gap_rows, [2.25, 1.45, 1.75], 11)
    add_rect(slide, 6.60, 3.36, 5.85, 2.10, WHITE, LINE)
    add_text(slide, 6.92, 3.62, 5.20, 0.28, "2004 气候摘要", 11, TEAL, True)
    add_text(slide, 6.92, 4.02, 5.15, 1.08,
             f"TMAX 均值 {float(annual_2004['TMAX_mean']):.2f}°C（范围 {float(annual_2004['TMAX_min']):.1f}–{float(annual_2004['TMAX_max']):.1f}）；"
             f"TMIN 均值 {float(annual_2004['TMIN_mean']):.2f}°C；SRAD 均值 {float(annual_2004['SRAD_mean']):.2f} MJ/m²/d。", 12, INK)
    add_text(slide, 0.86, 5.88, 11.40, 0.65,
             "ChinaFLUX 年值/月值只用于交叉对照；匹配结果不是填补规则，也没有按聚合总量反推逐日降水。", 12, MUTED)

    slide = base_slide(prs, "泄漏审计与本轮边界", "复现与下一步", 6)
    add_rect(slide, 0.72, 1.55, 5.52, 4.98, PALE_TEAL)
    add_text(slide, 1.02, 1.84, 4.88, 0.30, "泄漏审计：通过", 14, GREEN, True)
    add_bullet(slide, 1.02, 2.42, 4.88, "用于值或校准的最高天气年份：2013。", 13, GREEN)
    add_bullet(slide, 1.02, 3.20, 4.88, "GHCN 2014+ 行只纳入文件 SHA，不解析天气值。", 13, GREEN)
    add_bullet(slide, 1.02, 3.98, 4.88, "NASA 校正和 overlap 只用 2005–2013。", 13, GREEN)
    add_bullet(slide, 1.02, 4.76, 4.88, "Candidate 的逐日 source / QC 字段完整。", 13, GREEN)
    add_rect(slide, 6.60, 1.55, 5.85, 3.12, PALE_ORANGE)
    add_text(slide, 6.92, 1.84, 5.20, 0.30, "本轮停止边界", 14, ORANGE, True)
    add_bullet(slide, 6.92, 2.38, 5.05, "未生成 `.CLI`，未运行 WeatherMan 或 WGEN。", 13, ORANGE)
    add_bullet(slide, 6.92, 3.10, 5.05, "未运行 DSSAT smoke 或 PPO。", 13, ORANGE)
    add_rect(slide, 6.60, 4.93, 5.85, 1.60, PALE_GOLD)
    add_text(slide, 6.92, 5.18, 5.18, 0.30, "下一项最小任务", 12, GOLD, True)
    add_text(slide, 6.92, 5.55, 5.10, 0.72, "003_06：在冻结本 candidate 后，独立准备 train-only `.CLI` 与 WGEN pilot。", 13, INK, True)
    add_text(slide, 0.82, 6.74, 11.8, 0.18, "NOAA/NCEI GHCN-Daily README：ncei.noaa.gov/pub/data/ghcn/daily/readme.txt", 8, MUTED)

    backup = backup_output()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    reopened = Presentation(OUTPUT)
    with zipfile.ZipFile(OUTPUT, "r") as package:
        zip_error = package.testzip()
    slide_w, slide_h = reopened.slide_width, reopened.slide_height
    bounds = []
    for slide_i, slide in enumerate(reopened.slides, start=1):
        for shape_i, shape in enumerate(slide.shapes, start=1):
            if shape.left < 0 or shape.top < 0 or shape.left + shape.width > slide_w or shape.top + shape.height > slide_h:
                bounds.append({"slide": slide_i, "shape": shape_i, "name": shape.name})
    renderer = shutil.which("soffice") or shutil.which("libreoffice")
    result = {
        "path": str(OUTPUT.relative_to(ROOT)),
        "slide_count": len(reopened.slides),
        "expected_slide_count": 6,
        "zip_package_integrity": zip_error is None,
        "all_slides_have_shapes": all(len(slide.shapes) > 0 for slide in reopened.slides),
        "shapes_within_slide_bounds": not bounds,
        "out_of_bounds_shapes": bounds,
        "visual_renderer_available": bool(renderer),
        "renderer_path": renderer,
        "visual_rendered": False,
        "final_status": summary["final_status"],
        "sha256": sha256_file(OUTPUT),
        "backup_path": backup,
        "validation_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RESULTS / "pptx_validation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["slide_count"] == 6 and result["zip_package_integrity"] and result["shapes_within_slide_bounds"] else 1


if __name__ == "__main__":
    raise SystemExit(build())
