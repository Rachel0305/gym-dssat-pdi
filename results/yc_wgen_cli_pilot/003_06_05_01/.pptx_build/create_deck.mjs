import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const { SKILL_DIR, TMP_DIR, WORKSPACE_DIR, FINAL_PPTX, RUNTIME_PYTHON } = process.env;
const RUNTIME_NODE_MODULES = process.env.RUNTIME_NODE_MODULES;
for (const value of [SKILL_DIR, TMP_DIR, WORKSPACE_DIR, FINAL_PPTX, RUNTIME_PYTHON, RUNTIME_NODE_MODULES]) {
  if (!path.isAbsolute(value ?? "")) throw new Error("Presentation runtime paths must be absolute");
}
if (await fs.stat(FINAL_PPTX).then(() => true).catch(() => false)) {
  throw new Error(`Refusing to overwrite existing deck: ${FINAL_PPTX}`);
}

const {
  finalizePresentation,
  resolvePresentationFont,
} = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

const FONT = resolvePresentationFont({ fontFamily: "Arial" });
const C = {
  ink: "#202B2E",
  muted: "#5C6B6E",
  teal: "#167A78",
  green: "#287452",
  coral: "#B64B3D",
  line: "#D8E0DE",
  pale: "#F2F6F5",
  white: "#FFFFFF",
};
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slideRefs = [];

function addText(slide, name, text, x, y, w, h, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: 25,
    color: C.ink,
    alignment: "left",
    verticalAlignment: "top",
    wrap: "square",
    autoFit: "shrinkText",
    insets: { top: 2, right: 4, bottom: 2, left: 4 },
    ...style,
  };
  return shape;
}

function newSlide(index, title) {
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  slideRefs.push(slide);
  addText(slide, `slide-${index}-title`, title, 76, 62, 1120, 68, {
    fontSize: 41,
    bold: true,
    color: C.ink,
  });
  addText(slide, `slide-${index}-index`, `YC WEATHER PIPELINE    /    003_06_05_01    /    ${String(index).padStart(2, "0")}`, 80, 18, 900, 28, {
    fontSize: 16,
    color: C.teal,
  });
  addText(slide, `slide-${index}-page`, String(index).padStart(2, "0"), 1160, 670, 50, 24, {
    fontSize: 16,
    color: C.muted,
    alignment: "right",
  });
  return slide;
}

function notes(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}

// Cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.white;
  slideRefs.push(slide);
  addText(slide, "cover-kicker", "YC WEATHER PIPELINE    /    003_06_05_01", 88, 88, 900, 34, {
    fontSize: 19,
    color: C.teal,
  });
  addText(slide, "cover-title", "YC WGENIN 5010\n格式修复", 84, 190, 1060, 180, {
    fontSize: 61,
    bold: true,
    lineSpacing: 0.94,
  });
  addText(slide, "cover-subtitle", "定位固定列错位，只重排序列化格式，保留全部月参数值", 91, 400, 1050, 70, {
    fontSize: 29,
    color: C.muted,
  });
  addText(slide, "cover-result", "PARSE_FIX_RUNTIME_PASS", 91, 526, 960, 48, {
    fontSize: 32,
    bold: true,
    color: C.green,
  });
  addText(slide, "cover-scope", "DSSAT 4.8.0.024    |    单次 weather_seed = 101", 93, 582, 1050, 42, {
    fontSize: 23,
    color: C.ink,
  });
  notes(slide, "实验记录基于 results/yc_wgen_cli_pilot/003_06_05_01 下的逐列诊断、静态检查和单次 runtime smoke。未运行其他 seed 或 PPO。");
}

// Read contract
{
  const slide = newSlide(2, "WGENIN 月记录读取合同");
  addText(slide, "s2-error", "原始错误    STOP 99  |  5010  |  CNYC.CLI 第 27 行", 88, 166, 1100, 54, {
    fontSize: 27,
    color: C.coral,
    bold: true,
  });
  addText(slide, "s2-read-label", "WGENIN READ FORMAT", 90, 260, 620, 34, {
    fontSize: 19,
    color: C.muted,
  });
  addText(slide, "s2-format", "(I6,14(1X,F5.0))", 88, 308, 960, 66, {
    fontSize: 47,
    bold: true,
    color: C.teal,
  });
  addText(slide, "s2-count", "6 列整数 MTH  +  14 × 6 列实数槽位  =  90 列", 91, 406, 1090, 52, {
    fontSize: 30,
  });
  addText(slide, "s2-clarifier", "14 个 WGEN 统计值；含月份字段共 15 列。", 92, 478, 1080, 42, {
    fontSize: 25,
    color: C.muted,
  });
  addText(slide, "s2-version", "v4.8.0.24 与 v4.8.5.0 公共源码读取格式一致", 92, 548, 1080, 42, {
    fontSize: 23,
    color: C.green,
  });
  notes(slide, "读取合同来源：DSSAT dssat-csm-os v4.8.0.24 Weather/WGEN.for, https://github.com/DSSAT/dssat-csm-os/blob/v4.8.0.24/Weather/WGEN.for#L408-L414；DSSAT v4.8.5.0 同模块 https://github.com/DSSAT/dssat-csm-os/blob/v4.8.5.0/Weather/WGEN.for#L412-L418。运行时为 DSSAT 4.8.0.024。本次记录 14 个统计字段另加 MTH 共 15 个输入列。");
}

// Root cause with editable evidence table
{
  const slide = newSlide(3, "根因：三个统计字段各多一列");
  const table = slide.tables.add({
    rows: 4,
    columns: 4,
    left: 88,
    top: 165,
    width: 1100,
    height: 286,
    columnWidths: [260, 275, 275, 290],
    values: [
      ["字段", "原生成器", "WGENIN 要求", "每行影响"],
      ["XDMN", "7 列", "6 列", "+1 列"],
      ["XWMN", "7 列", "6 列", "+1 列"],
      ["NAMN", "7 列", "6 列", "+1 列"],
    ],
  });
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.styleOptions = { headerRow: false, bandedRows: false };
  for (let row = 0; row < 4; row += 1) {
    for (let col = 0; col < 4; col += 1) {
      const cell = table.getCell(row, col);
      cell.fill = row === 0 ? C.ink : (row % 2 === 0 ? C.pale : C.white);
      cell.text.style = {
        typeface: FONT,
        fontSize: row === 0 ? 21 : 24,
        bold: row === 0 || col === 0,
        color: row === 0 ? C.white : (col === 3 ? C.coral : C.ink),
        verticalAlignment: "middle",
        alignment: col === 0 ? "left" : "center",
        autoFit: "shrinkText",
      };
    }
  }
  table.rows[0].height = 58;
  addText(slide, "s3-length", "12 条月行均由 90 列变成 93 列。固定槽位从 XDMN 溢出后，XDSD 起出现错位。", 91, 496, 1090, 75, {
    fontSize: 26,
  });
  addText(slide, "s3-reference", "仓库 CNSY.CLI 结构样本：相同 header，月行 90 列；仅比较格式，未复制参数值。", 92, 592, 1080, 50, {
    fontSize: 19,
    color: C.muted,
  });
  notes(slide, "源字段宽度来自 scripts/build_dssat_cli.py 原格式化表达式；读取格式来源同上一页。逐字符切片、token 和样本比对见 results/yc_wgen_cli_pilot/003_06_05_01/diagnostics/original_line27_layout.txt 与 wgen_row_schema_comparison.txt。根因是静态列布局诊断，具体运行时行为由修复后的单次 DSSAT smoke 验证。");
}

// Fix and static validation
{
  const slide = newSlide(4, "修复只改变序列化，不改参数");
  addText(slide, "s4-fix", "I6 + 14 × (1X,F5.0)", 90, 164, 1080, 60, {
    fontSize: 40,
    bold: true,
    color: C.teal,
  });
  addText(slide, "s4-checks", "12 / 12 月行符合 90 列\n15 列可解析，数值有限\n原始参数 token 全部保留\n23 项回归测试通过", 92, 264, 730, 255, {
    fontSize: 27,
    lineSpacing: 1.25,
  });
  addText(slide, "s4-hash-label", "SHA256 provenance", 872, 270, 330, 34, {
    fontSize: 18,
    color: C.muted,
  });
  addText(slide, "s4-source-hash", "冻结原件\n5ABF5D7BB97EFAAE5E8361ABB4C773E1554B58116CBCA2213E75DCFF838285F0", 870, 314, 340, 110, {
    fontSize: 18,
    color: C.ink,
  });
  addText(slide, "s4-fixed-hash", "隔离修复件\n65CF134600A5881706A5D435E1A09B276ED92A21FA5ABE2E18AAF63AF1E3A929", 870, 450, 340, 110, {
    fontSize: 18,
    color: C.teal,
  });
  addText(slide, "s4-note", "原 CLI 和冻结训练天气运行前后 hash 均未改变。", 92, 584, 1080, 44, {
    fontSize: 23,
    color: C.green,
  });
  notes(slide, "静态校验来源：results/yc_wgen_cli_pilot/003_06_05_01/validation/static_parse_check.json。回归测试输出：tests/test_build_dssat_cli.py 与 tests/test_yc_wgen_seed_pilot.py，23 passed。冻结 train weather hash 为 4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34。");
}

// Runtime outcome
{
  const slide = newSlide(5, "唯一有效 seed 101 runtime smoke");
  addText(slide, "s5-status", "PARSE_FIX_RUNTIME_PASS", 91, 158, 1090, 54, {
    fontSize: 34,
    bold: true,
    color: C.green,
  });
  addText(slide, "s5-steps", "120 步", 94, 252, 480, 70, {
    fontSize: 52,
    bold: true,
    color: C.ink,
  });
  addText(slide, "s5-weather", "120 行天气状态", 610, 252, 550, 70, {
    fontSize: 42,
    bold: true,
    color: C.teal,
  });
  addText(slide, "s5-setup", "YC 单季   |   random_weather=True   |   WTHER=W\nWSTA=CNYC0801   |   PPO 未运行", 95, 365, 1070, 96, {
    fontSize: 26,
    lineSpacing: 1.2,
  });
  addText(slide, "s5-date", "天气状态窗口    2008-06-01 至 2008-09-28", 96, 490, 1050, 42, {
    fontSize: 24,
    color: C.muted,
  });
  addText(slide, "s5-qc", "天气 screening QC 通过；WGENIN 5010 未再出现。", 96, 552, 1050, 48, {
    fontSize: 26,
    color: C.green,
  });
  notes(slide, "单次运行结果见 results/yc_wgen_cli_pilot/003_06_05_01/runtime_smoke_summary.json、runtime/seed_101_parsefix_01_basename_ok/wgen_status.json 和生成的 120 行 CSV。DSSAT 4.8.0.024；runtime 使用的 CNYC.CLI SHA256 与隔离修复件相同。QC 是输出范围的 screening checks，不代表多 seed 随机性或气候学验证。");
}

// Scope and next gate
{
  const slide = newSlide(6, "结论与停止边界");
  addText(slide, "s6-conclusion", "格式根因已定位，单次 seed 101 smoke 越过 WGENIN 并进入模拟。", 90, 159, 1090, 84, {
    fontSize: 30,
    bold: true,
  });
  addText(slide, "s6-first-attempt", "首次包装尝试在 MAKEFW 找不到 CNYC.CLI，未进入 WGENIN，已保留并标记 NOT_TESTED。", 92, 282, 1090, 75, {
    fontSize: 24,
    color: C.coral,
  });
  addText(slide, "s6-next", "本任务停止于此。下一步恢复 003_06_05 seed pilot，再检查 101a / 101b 与 102–105。", 92, 405, 1080, 92, {
    fontSize: 27,
  });
  addText(slide, "s6-warning", "FileX 经纬度与海拔 warning 仍存在，本轮未改动输入。", 93, 548, 1080, 48, {
    fontSize: 22,
    color: C.muted,
  });
  notes(slide, "任务边界来自 prompt_02/003_06_05_01_fix_wgenin_5010.md。首次 wrapper lookup 失败证据在 runtime/attempt_01_basename_lookup_failure.json；没有把该次误判为 WGENIN 解析成功，也没有重复运行 seed 101 以外的其他 seed。");
}

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
const draftPath = path.join(TMP_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(draftPath);

for (let index = 0; index < slideRefs.length; index += 1) {
  const slide = slideRefs[index];
  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`),
    new Uint8Array(await preview.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(TMP_DIR, `slide-${String(index + 1).padStart(2, "0")}.layout.json`), await layout.text());
}
const snapshot = await presentation.inspect({ kind: "slide,textbox,table,notes,layout", maxChars: 30000 });
await fs.writeFile(path.join(TMP_DIR, "deck-inspect.ndjson"), snapshot.ndjson, "utf8");

const stagingDir = path.join(WORKSPACE_DIR, "results/yc_wgen_cli_pilot/003_06_05_01/.pptx_finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const result = await finalizePresentation({
  explicitTotalSlideCount: 6,
  requiredNativeTableOwnerSlides: [3],
  workspaceDir: WORKSPACE_DIR,
  candidatePath: draftPath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
    "--require-native-table-slide", "3",
  ],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "yc_wgen_5010_parse_fix.validation.json"),
});
await fs.writeFile(path.join(TMP_DIR, "finalizer-result.json"), JSON.stringify(result, null, 2), "utf8");
console.log(JSON.stringify({ final: FINAL_PPTX, slides: slideRefs.length, font: FONT, finalizer: result }, null, 2));
