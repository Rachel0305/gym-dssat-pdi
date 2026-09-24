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
  red: "#A8403A",
  redPale: "#F8E9E7",
  line: "#D7E0E2",
  paper: "#FFFFFF",
};

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const addText = (slide, text, x, y, w, h, size = 24, color = C.ink, bold = false) => {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { typeface: font, fontSize: size, color, bold, autoFit: "none" };
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
const page = (section, title, n) => {
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  rule(slide, 0, 0, 1280, C.teal, 8);
  addText(slide, section, 72, 38, 900, 26, 18, C.teal, true);
  addText(slide, title, 72, 82, 1136, 62, 38, C.ink, true);
  rule(slide, 72, 652, 1136, C.line, 1);
  addText(slide, "YC / WEATHERMAN CLI PATH AUDIT", 72, 666, 850, 22, 15, C.muted);
  addText(slide, `${n} / 3`, 1080, 666, 128, 22, 15, C.muted);
  return slide;
};

{
  const s = deck.slides.add();
  s.background.fill = C.paper;
  rule(s, 0, 0, 1280, C.teal, 8);
  addText(s, "003_06_01  /  YC-YCA  /  CLI GENERATION", 76, 58, 1100, 30, 18, C.teal, true);
  addText(s, "Candidate 已复冻\nWeatherMan 路径探测受限", 76, 130, 1080, 140, 46, C.ink, true);
  addText(s, "项目边界禁止读取项目外安装位置；WeatherMan 是否安装仍未知。", 80, 302, 1080, 45, 23, C.muted);
  rule(s, 80, 421, 1080, C.red, 3);
  addText(s, "BLOCKED_TOOL_PATH_ACCESS", 80, 452, 830, 50, 32, C.red, true);
  addText(s, "SHA256 与任务要求一致", 80, 526, 480, 34, 21, C.teal, true);
  addText(s, "3653 天  |  2004-01-01 至 2013-12-31", 80, 566, 700, 38, 21, C.ink, true);
  addText(s, "CLI 0  |  WGEN 0  |  PPO 0", 80, 612, 650, 34, 20, C.muted, true);
  note(s, "来源：results/yc_wgen_cli_pilot/003_06_01/input_freeze.json、weatherman_inventory.json、experiment_summary.json。状态表达访问边界导致未检查，不表示 WeatherMan 未安装。");
}

{
  const s = page("01  /  Gate B", "当前工具链版本与路径均未核实", 2);
  addText(s, "项目约束", 82, 170, 190, 32, 20, C.teal, true);
  addText(s, "AGENTS.md 仅允许读取和修改当前项目目录内文件。", 290, 166, 865, 42, 22, C.ink);
  rule(s, 82, 224, 1090, C.line, 1);
  addText(s, "未探测位置", 82, 252, 190, 32, 20, C.red, true);
  addText(s, "C:\\DSSAT*  |  Program Files  |  PATH / shortcuts\n/opt  |  /usr/local  |  /usr/bin  |  项目外 Python / Gym runtime", 290, 247, 865, 76, 21, C.ink);
  rule(s, 82, 344, 1090, C.line, 1);
  addText(s, "未知项", 82, 372, 190, 32, 20, C.muted, true);
  addText(s, "WeatherMan 是否存在、executable/version、GUI/CLI 能力；当前 DSSAT 路径与版本。", 290, 367, 865, 64, 21, C.ink);
  addText(s, "“未知”不等于“未安装”", 82, 496, 1090, 48, 28, C.red, true);
  addText(s, "未查询导入格式或站点元数据，未转换 candidate，也未生成 CNYC.CLI。", 82, 566, 1090, 52, 21, C.ink);
  note(s, "安全边界来源：项目根目录 AGENTS.md。官方 WeatherMan 工具说明：https://dssat.net/tools/ 。官方 DSSAT 下载系统：https://get.dssat.net/ 。网页说明只用于引用官方工具入口，不代表本机安装状态。");
}

{
  const s = page("02  /  Minimal user input", "提供三项本机证据后再继续 Gate C", 3);
  const rows = [
    ["01", "WeatherMan", "executable 完整路径与文件名；若未找到，请说明人工检查入口。"],
    ["02", "DSSAT", "安装根目录与实际运行版本证据，例如输出头或 About 信息。"],
    ["03", "版本资料", "WeatherMan About/version 文本或截图；若有，附原始 Help 输出。"],
  ];
  rows.forEach(([n, label, text], i) => {
    const y = 174 + i * 100;
    addText(s, n, 84, y, 65, 40, 25, C.teal, true);
    addText(s, label, 164, y, 210, 40, 22, C.ink, true);
    addText(s, text, 390, y, 770, 58, 20, C.muted);
    if (i < rows.length - 1) rule(s, 84, y + 71, 1080, C.line, 1);
  });
  addText(s, "把文本或截图复制进项目目录，再核对该版本的导入格式和 YC 元数据。", 84, 500, 1080, 60, 21, C.ink, true);
  addText(s, "先验证与当前 DSSAT runtime 的兼容性，不要盲目切换版本。", 84, 578, 1080, 38, 20, C.red, true);
  note(s, "官方参考：DSSAT Tools 页面介绍 WeatherMan 的导入、分析、导出和站点功能：https://dssat.net/tools/ 。官方 DSSAT Download System：https://get.dssat.net/ 。当前项目 DSSAT 版本尚未查明，因此不推荐具体安装版本。candidate SHA256 见本 deck 第 1 页及 results/yc_wgen_cli_pilot/003_06_01/input_freeze.json。");
}

await fs.mkdir(BUILD_DIR, { recursive: true });
const candidatePath = path.join(BUILD_DIR, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
for (let i = 0; i < deck.slides.items.length; i++) {
  const slide = deck.slides.items[i];
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(BUILD_DIR, `slide-${i + 1}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(BUILD_DIR, `slide-${i + 1}.layout.json`), await layout.text());
}
const receiptPath = path.join(WORKSPACE_DIR, "results/yc_wgen_cli_pilot/003_06_01/presentation_validation.json");
await finalizePresentation({
  explicitTotalSlideCount: 3,
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
console.log(JSON.stringify({ candidatePath, finalPath: FINAL_PPTX, receiptPath }, null, 2));
