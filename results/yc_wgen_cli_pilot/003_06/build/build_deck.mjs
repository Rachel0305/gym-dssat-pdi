import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const { SKILL_DIR, BUILD_DIR, WORKSPACE_DIR, FINAL_PPTX, RUNTIME_PYTHON } = process.env;
for (const value of [SKILL_DIR, BUILD_DIR, WORKSPACE_DIR, FINAL_PPTX, RUNTIME_PYTHON]) {
  if (!path.isAbsolute(value ?? "")) throw new Error("Presentation paths must be absolute.");
}

const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);
const font = resolvePresentationFont({ fontFamily: "Arial" });
const C = {
  ink: "#182B35",
  muted: "#52656E",
  teal: "#087E78",
  tealPale: "#E5F3F0",
  amber: "#A56A0A",
  amberPale: "#FBF2DE",
  red: "#A8403A",
  redPale: "#F8E9E7",
  line: "#D7E0E2",
  paper: "#FFFFFF",
};

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const addText = (slide, text, x, y, w, h, size = 24, color = C.ink, bold = false, align = "left") => {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: font,
    fontSize: size,
    color,
    bold,
    alignment: align,
    autoFit: "none",
  };
  return shape;
};
const rule = (slide, x, y, w, color = C.line, h = 2) => {
  slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: color,
    line: { fill: "none", width: 0 },
  });
};
const note = (slide, text) => slide.speakerNotes.textFrame.setText(text);
const addSlide = (section, title, number) => {
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  rule(slide, 0, 0, 1280, C.teal, 8);
  addText(slide, section, 72, 38, 800, 26, 18, C.teal, true);
  addText(slide, title, 72, 82, 1136, 58, 39, C.ink, true);
  rule(slide, 72, 652, 1136, C.line, 1);
  addText(slide, "YC / TRAIN-ONLY CLI / WGEN PILOT", 72, 666, 780, 22, 15, C.muted);
  addText(slide, `${number} / 4`, 1080, 666, 128, 22, 15, C.muted, false, "right");
  return slide;
};

{
  const s = deck.slides.add();
  s.background.fill = C.paper;
  rule(s, 0, 0, 1280, C.teal, 8);
  addText(s, "003_06  /  YC-YCA  /  WGEN PILOT", 76, 58, 1080, 30, 18, C.teal, true);
  addText(s, "天气候选已冻结\nWGEN pilot 停在 CLI Gate", 76, 128, 1070, 142, 48, C.ink, true);
  addText(s, "Candidate 通过复核；没有可验证的 2004-2013 参数估计工具链。", 80, 302, 1080, 42, 24, C.muted);
  const metrics = [
    ["2004-2013", "冻结输入窗口", C.teal],
    ["3,653", "天，四变量完整", C.ink],
    ["0", "eligible train-only CLI", C.red],
    ["0", "WGEN / DSSAT smoke", C.red],
  ];
  metrics.forEach(([value, label, color], i) => {
    const x = 80 + i * 285;
    rule(s, x, 430, 235, i > 1 ? C.red : C.teal, 3);
    addText(s, value, x, 452, 250, 51, 31, color, true);
    addText(s, label, x, 510, 250, 42, 19, C.muted, true);
  });
  addText(s, "Final status  BLOCKED_CLI_GENERATION", 80, 590, 1070, 45, 24, C.red, true);
  note(s, "依据：docs/yc_train_only_cli_and_wgen_pilot.md；results/yc_wgen_cli_pilot/003_06/experiment_summary.json。Candidate SHA256 与上一轮 004 upstream_gate.json 一致。未启动 WGEN、DSSAT 或 PPO。");
}

{
  const s = addSlide("01  /  Candidate freeze", "候选数据完整；物理检查无失败", 2);
  addText(s, "文件", 80, 172, 180, 30, 19, C.muted, true);
  addText(s, "yc_wgen_fitting_weather_2004_2013.csv", 275, 166, 900, 38, 24, C.ink, true);
  rule(s, 80, 218, 1100, C.line, 1);
  addText(s, "SHA256", 80, 250, 180, 30, 19, C.muted, true);
  addText(s, "4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34", 275, 245, 920, 52, 20, C.ink, true);
  rule(s, 80, 309, 1100, C.line, 1);
  const facts = [
    ["日期与日数", "2004-01-01 至 2013-12-31  |  3,653 天"],
    ["数据完整性", "RAIN / TMAX / TMIN / SRAD 均无空值；日期无重复"],
    ["物理 QC", "负雨量 0  |  负 SRAD 0  |  TMAX < TMIN 0"],
    ["前轮冻结核对", "与 004 upstream_gate.json 的 candidate SHA256 一致"],
  ];
  facts.forEach(([label, value], i) => {
    const y = 346 + i * 63;
    addText(s, label, 80, y, 190, 34, 19, C.muted, true);
    addText(s, value, 275, y, 900, 42, 21, C.ink, i === 0);
  });
  addText(s, "验证期 2014-2023 排除；原 candidate 文件未改动。", 80, 610, 1100, 31, 19, C.teal, true);
  note(s, "来源：results/yc_wgen_cli_pilot/003_06/weather_candidate_freeze.json；results/yc_weather_gapfill_finalize/candidate_qc.json；results/yc_ppo_weather_augmentation/upstream_gate.json。当前 CSV 复核 3653 行、首末日期、唯一日期、空值和基础物理约束。");
}

{
  const s = addSlide("02  /  Toolchain and provenance", "当前参数估计工具链无法核实", 3);
  addText(s, "仓库证据", 80, 166, 200, 31, 19, C.teal, true);
  addText(s, "记录过 gym_dssat_pdi 0.0.5；random_weather=True 会请求 W 模式，并从 Gym RNG 派生 rseed1。", 275, 162, 900, 62, 21, C.ink);
  rule(s, 80, 241, 1100, C.line, 1);
  addText(s, "未核实", 80, 267, 200, 31, 19, C.red, true);
  addText(s, "当前 DSSAT / WeatherMan 版本与路径、WeatherMan CLI/GUI、WGEN 可执行文件、WSTA 到 CLI 的映射。", 275, 262, 900, 63, 21, C.ink);
  rule(s, 80, 340, 1100, C.line, 1);
  addText(s, "旧 CLI", 80, 367, 200, 31, 19, C.amber, true);
  addText(s, "SHA 58dbe11f…；记录窗口 2008-01-01 至 2014-12-31。", 275, 362, 900, 40, 21, C.ink, true);
  addText(s, "估计年份 provenance 不完整；关联运行 random_weather=false。排除，不复制、不改名。", 275, 405, 900, 55, 20, C.red);
  addText(s, "项目目录未发现官方参数估计自动化；项目外安装位置未检查。", 80, 512, 1100, 42, 22, C.ink, true);
  addText(s, "因此不能声称 GUI-only，也不能伪造 CLI。", 80, 572, 1100, 35, 21, C.red, true);
  note(s, "来源：references/dssat_pdi.py；results/yc_weather_audit/yc_weather_reset_diagnostic.json；results/yc_wgen_cli_pilot/yc_cli_provenance.json；docs/yc_wgen_cli_pilot.md。DSSAT 4.8 仅见历史 DSSAT480 目录标签，不作为当前版本证据。");
}

{
  const s = addSlide("03  /  Restart conditions", "满足四项证据后再启动随机天气 pilot", 4);
  const rows = [
    ["01", "官方估计路径", "参数唯一来自冻结 candidate；年份严格为 2004-2013。"],
    ["02", "CLI provenance / QC", "工具与版本、日期窗、输入 SHA、输出 SHA、CNYC 结构均可复核。"],
    ["03", "WGEN runtime contract", "确认 FileX WSTA 映射；weather seed 与 PPO seed 独立；能导出逐日天气。"],
    ["04", "先小规模验证", "101 重复哈希一致；101 与 102 天气不同；再生成 3-5 套并 smoke 至少 3 季。"],
  ];
  rows.forEach(([n, label, detail], i) => {
    const y = 165 + i * 95;
    addText(s, n, 82, y, 64, 43, 25, C.teal, true);
    addText(s, label, 160, y, 280, 38, 21, C.ink, true);
    addText(s, detail, 450, y, 720, 60, 20, C.muted);
    if (i < rows.length - 1) rule(s, 82, y + 69, 1088, C.line, 1);
  });
  addText(s, "本轮 PPO seed = NOT_USED；不训练 PPO，不作策略比较。", 82, 574, 1090, 33, 20, C.red, true);
  addText(s, "下一最小任务：提供项目约束内可审计的 WeatherMan/转换路径与 train-only CLI provenance。", 82, 612, 1090, 31, 18, C.ink, true);
  note(s, "来源：prompt_02/003_06_yc_train_only_cli_and_wgen_pilot.md；docs/yc_train_only_cli_and_wgen_pilot.md。各 Gate 依赖顺序严格遵循任务定义。本任务未训练 PPO。");
}

await fs.mkdir(BUILD_DIR, { recursive: true });
const candidatePath = path.join(BUILD_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);

for (let i = 0; i < deck.slides.items.length; i++) {
  const slide = deck.slides.items[i];
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(BUILD_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(BUILD_DIR, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
}
const montage = await deck.export({ format: "webp", montage: true, scale: 0.5 });
await fs.writeFile(path.join(BUILD_DIR, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const receiptPath = path.join(WORKSPACE_DIR, "results/yc_wgen_cli_pilot/003_06/presentation_validation.json");
const result = await finalizePresentation({
  explicitTotalSlideCount: 4,
  requiredNativeChartOwnerSlides: [],
  requiredNativeTableOwnerSlides: [],
  workspaceDir: WORKSPACE_DIR,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit"],
  fontPolicy: { basis: "design", families: ["Arial"] },
  verifyArtifactToolImport: true,
  receiptPath,
});
console.log(JSON.stringify({ candidatePath, finalPath: FINAL_PPTX, result }, null, 2));
