import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const { SKILL_DIR, TMP_DIR, WORKSPACE_DIR, RUNTIME_PYTHON } = process.env;
for (const [name, value] of Object.entries({ SKILL_DIR, TMP_DIR, WORKSPACE_DIR, RUNTIME_PYTHON })) {
  if (!path.isAbsolute(value ?? "")) throw new Error(`${name} must be an absolute path`);
}

const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);
const FONT = resolvePresentationFont();
const W = 1280;
const H = 720;
const C = {
  ink: "#1C302D",
  muted: "#596B66",
  teal: "#0D6E68",
  tealDark: "#174D49",
  tealPale: "#E5F1EE",
  amber: "#E8A54A",
  amberPale: "#FFF2DE",
  coral: "#C65D4D",
  coralPale: "#F9E9E5",
  green: "#477B5B",
  greenPale: "#E9F2E9",
  line: "#D6E0DC",
  paper: "#F5F8F6",
  white: "#FFFFFF",
};

await fs.mkdir(TMP_DIR, { recursive: true });
const deck = Presentation.create({ slideSize: { width: W, height: H } });
const slides = [];

function addText(slide, text, x, y, w, h, size, color = C.ink, bold = false, align = "left") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: size,
    bold,
    color,
    alignment: align,
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    wrap: "square",
    insets: { left: 2, right: 2, top: 2, bottom: 2 },
  };
  return shape;
}

function addRect(slide, x, y, w, h, fill, line = "none", radius = 0) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function addLine(slide, x1, y1, x2, y2, color = C.line, width = 2) {
  slide.shapes.add({
    geometry: "line",
    position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function base(title, kicker, page) {
  const slide = deck.slides.add();
  slides.push(slide);
  slide.background.fill = C.white;
  addRect(slide, 0, 0, W, 8, C.teal);
  addText(slide, kicker.toUpperCase(), 64, 28, 800, 26, 15, C.teal, true);
  addText(slide, title, 64, 58, 1144, 56, 38, C.ink, true);
  addLine(slide, 64, 652, 1216, 652, C.line, 1);
  addText(slide, "YC 天气增强 · 003_06_03", 64, 664, 650, 26, 14, C.muted, false);
  addText(slide, `${String(page).padStart(2, "0")} / 07`, 1110, 664, 106, 26, 14, C.muted, false, "right");
  return slide;
}

function addTable(slide, values, x, y, w, h, columnWidths, fontSize = 18) {
  const rows = values.length;
  const columns = values[0].length;
  const table = slide.tables.add({ rows, columns, left: x, top: y, width: w, height: h, columnWidths, values });
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.styleOptions = { headerRow: false, bandedRows: false };
  for (let r = 0; r < rows; r += 1) {
    for (let col = 0; col < columns; col += 1) {
      const cell = table.getCell(r, col);
      cell.fill = r === 0 ? C.tealDark : (r % 2 === 1 ? C.paper : C.white);
      cell.text.style = {
        typeface: FONT,
        fontSize: r === 0 ? fontSize : fontSize - 1,
        bold: r === 0 || col === 0,
        color: r === 0 ? C.white : (col === 0 ? C.tealDark : C.ink),
        verticalAlignment: "middle",
        autoFit: "shrinkText",
        wrap: "square",
        insets: { left: 10, right: 10, top: 7, bottom: 7 },
      };
    }
  }
  return table;
}

function note(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
}

{
  const slide = base("YC CNYC.CLI 生成路径确认", "Weather generation · decision record", 1);
  addText(slide, "当前结论", 68, 154, 260, 32, 19, C.coral, true);
  addText(slide, "本轮不生成 CLI", 68, 194, 620, 74, 46, C.ink, true);
  addText(slide, "BLOCKED_BY_WEATHERMAN_ACCESS", 70, 276, 660, 36, 21, C.coral, true);
  addText(slide, "并列阻塞：导入格式、4.8.x 严格字段、完整站点元数据尚未核实", 70, 318, 686, 52, 21, C.muted);

  addRect(slide, 790, 150, 420, 250, C.paper, C.line, 6);
  addText(slide, "冻结拟合输入", 822, 176, 340, 34, 20, C.teal, true);
  addText(slide, "2004-01-01  →  2013-12-31", 822, 224, 350, 38, 23, C.ink, true);
  addText(slide, "3653 日 · 仅训练天气 · SHA256 已核", 822, 270, 350, 32, 18, C.muted);
  addText(slide, "4B8FFE9E…088ED7B34", 822, 316, 350, 32, 17, C.tealDark, true);

  addRect(slide, 68, 438, 1142, 140, C.tealPale, "none", 6);
  addText(slide, "路径已确认到“官方工具计算参数”这一步；运行入口和输入合同仍需补证。", 96, 464, 1080, 42, 25, C.tealDark, true);
  addText(slide, "不拟合参数 · 不拼接 CLI · 不启动 WGEN / DSSAT / PPO", 96, 518, 1080, 34, 19, C.ink);
  note(slide, "本页依据本轮冻结候选审计和项目内证据。冻结文件 SHA256：4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34。此报告不声称已验证当前 WeatherMan 或 DSSAT runtime。任务说明要求本轮默认不得生成 CNYC.CLI，除非官方流程已无不确定参数可直接运行。来源：仓库文件 results/yc_weather_gapfill_finalize/yc_wgen_fitting_weather_2004_2013.csv；项目任务 prompt_02/003_06_03_confirm_cli_generation_path.md。");
}

{
  const slide = base("这轮只解决拟合入口，不启动随机天气", "Scope and data boundary", 2);
  const steps = [
    { x: 70, title: "冻结候选", body: "YC 2004–2013\n3653 daily rows", fill: C.tealPale, accent: C.teal },
    { x: 357, title: "官方导入", body: "WeatherMan\n格式合同待核", fill: C.amberPale, accent: C.amber },
    { x: 644, title: "计算并导出", body: "WGEN statistics\n写入 CNYC.CLI", fill: C.paper, accent: C.tealDark },
    { x: 931, title: "后续验证", body: "CLI QC → seed pilot\n→ gated DSSAT smoke", fill: C.coralPale, accent: C.coral },
  ];
  for (const [i, step] of steps.entries()) {
    addRect(slide, step.x, 205, 218, 194, step.fill, C.line, 6);
    addRect(slide, step.x, 205, 218, 9, step.accent);
    addText(slide, `0${i + 1}`, step.x + 20, 232, 48, 32, 18, step.accent, true);
    addText(slide, step.title, step.x + 20, 272, 180, 36, 23, C.ink, true);
    addText(slide, step.body, step.x + 20, 323, 180, 60, 18, C.muted);
    if (i < steps.length - 1) {
      addLine(slide, step.x + 228, 302, step.x + 276, 302, C.teal, 3);
      addText(slide, ">", step.x + 241, 278, 28, 44, 28, C.teal, true, "center");
    }
  }
  addRect(slide, 70, 450, 1140, 118, C.coralPale, "none", 6);
  addText(slide, "隔离线", 94, 470, 110, 34, 20, C.coral, true);
  addText(slide, "验证天气 2014–2023 不参与拟合、偏差订正或站点气候统计。", 215, 466, 944, 38, 22, C.ink, true);
  addText(slide, "当前 YC safe-render 仍为 random_weather=False；正式 PPO 与其他站点保持不动。", 215, 511, 944, 34, 18, C.muted);
  note(slide, "路线顺序取自任务阶段门槛：先确认官方生成路径，再生成 CLI、QC、WGEN pilot、DSSAT smoke，最后才讨论正式 PPO。禁止使用 2014–2023 validation weather 进行任何参数估计。当前 YC safe-render 参数和输入由 src/ppo_safe_rendering.py 静态审计得出。没有运行模拟。");
}

{
  const slide = base("项目证据显示：CLI 是分区文本，精确 schema 仍未锁定", "Observed format versus active parser", 3);
  addTable(slide, [
    ["区段", "旧官方指南列出的字段组", "本轮可确认到什么"],
    ["Climate / station", "LAT · LONG · ELEV · TAV · AMP · SRAY · TMXY / TMNY · START / DURN 等", "结构有旧指南依据；active 4.8.x 必填集合未知"],
    ["Monthly averages", "MTH · SAMN / XAMN / NAMN · RTOT / RNUM · SHMN · AMTH / BMTH", "当前 CNLC 样本只有简化字段；不可当完整 WGEN 样本"],
    ["WGEN parameters", "MTH · SDMN / SDSD · SWMN / SWSD · XDMN / XDSD · XWMN / XWSD · ALPHA · PDW 等", "旧指南说明 WGEN 需要该区段；未做 4.8.x parser 测试"],
    ["QC / counts", "MIN / MAX / RATE；TOTAL / VALID / MISSING / ERROR / ABOVE / BELOW", "CNSY 样本含对应区段；样本来源版本未核实"],
  ], 68, 160, 1144, 352, [190, 525, 429], 18);
  addText(slide, "只借鉴结构，不借用任何其他站点的气候参数；字段单位、月尺度定义逐项见 JSON 证据表。", 72, 548, 1136, 48, 19, C.muted);
  note(slide, "样本：benchmark_results/027_07_site_specific_stage_maskable_ppo/LC/readiness/baseline_runs/dssat_auto/input/CNLC.CLI（简化，9 行）；benchmark_results/028_05_sy_crossyear_frozen_ppo_daily/2012/seed0/snapshot/CNSY.CLI（含 WGEN 与 QC 区段）。均仅用于格式结构，不复制参数。字段细节、单位、时间尺度和证据限制记录于 results/yc_wgen_cli_pilot/003_06_03/cli_field_structure.json。官方旧版指南：DSSAT User's Guide Vol. 3, https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol3.pdf 。");
}

{
  const slide = base("YC 核心位置一致，但 WeatherMan 建站元数据不完整", "Station identity and metadata", 4);
  addTable(slide, [
    ["核对项", "仓库证据 / 当前状态"],
    ["站点标签 / DSSAT ID", "YC / CNYC（2004–2013 十个 WTH 头部一致）"],
    ["位置", "36.830°N · 116.570°E · 22 m（十年 WTH 一致）"],
    ["FileX WSTA", "示例 CNYC0801；与年度 WTH 命名中的 CNYC 前缀一致"],
    ["CLI basename 规则", "公开 4.8.5 SECLI.for 静态推导：W 模式以 WSTA 前 4 字符导出 / 查找 CNYC.CLI"],
    ["未知 / 不可直接沿用", "Angstrom A/B、growing-season start/duration 缺失；REFHT/WNDHT=-99；TAV/AMP 参考期未知"],
  ], 68, 156, 1144, 350, [250, 894], 18);
  addRect(slide, 68, 532, 1144, 72, C.amberPale, "none", 5);
  addText(slide, "CN / YC 的字符语义未定义；当前静态命名推导不等于 active runtime lookup 已通过。", 90, 550, 1098, 34, 19, C.ink, true);
  note(slide, "位置和站点 ID 来自训练期 CNYC0401.WTH 至 CNYC1301.WTH 的头部。示例 FileX CNYC0801.MZX 的 WSTA=CNYC0801。公开 4.8.5 SECLI.for 依据 FILEW(5:12) 推导 CLI basename；该静态证据未覆盖本机 DSSAT 4.8.0.024 lookup。官方旧 WeatherMan guide 的新站点表列出 latitude、longitude、elevation、Angstrom A/B、REFHT、WNDHT、TAV、AMP、START、DURN；当前 active版本要求未知。参考：https://github.com/DSSAT/dssat-csm-os/blob/develop/InputModule/SECLI.for 。");
}

{
  const slide = base("优先路线是 WeatherMan，但 CSV 与版本合同未验证", "Official generation route", 5);
  const route = [
    ["1", "导入", "daily weather\n用户定义列 / 日期 / 单位"],
    ["2", "归档", "WeatherMan station\narchive (.WTD)"],
    ["3", "限定窗口", "仅 2004/001–\n2013/365"],
    ["4", "计算", "Calculate WGEN\nparameters"],
    ["5", "导出", "新建 CNYC.CLI\n记录版本与 hash"],
  ];
  for (const [i, item] of route.entries()) {
    const x = 70 + i * 229;
    addRect(slide, x, 184, 196, 176, i === 0 ? C.tealPale : C.paper, C.line, 6);
    addText(slide, item[0], x + 16, 200, 38, 30, 18, C.teal, true);
    addText(slide, item[1], x + 16, 238, 164, 32, 22, C.ink, true);
    addText(slide, item[2], x + 16, 282, 164, 62, 17, C.muted);
    if (i < route.length - 1) addText(slide, "→", x + 199, 248, 30, 34, 23, C.amber, true, "center");
  }
  addText(slide, "已由官方旧指南支持", 74, 397, 340, 28, 17, C.green, true);
  addText(slide, "WeatherMan：导入 / 分析 / 计算 WGEN / 保存 CLI 的通用菜单流程。", 74, 430, 1090, 34, 20, C.ink);
  addText(slide, "尚待本机版本证据", 74, 488, 340, 28, 17, C.coral, true);
  addText(slide, "About 版本、CSV 逗号与 ISO 日期支持、缺失值合同、batch/CLI 接口、4.8.0↔4.8.5 格式差异。", 74, 521, 1098, 54, 19, C.ink);
  note(slide, "WeatherMan 官方 Tools 页面说明其天气导入/分析/导出能力：https://dssat.net/tools/ 。DSSAT User's Guide Vol. 3 描述以用户定义格式导入 daily weather、存入 station archive、选期计算 WGEN 参数并写入 CLI；此指南版本较旧，不能视为 active 4.8.x 导入规则。Vol. 1 对用户定义的导入格式另有说明：https://dssat.net/wp-content/uploads/2011/10/DSSAT-vol1.pdf 。未找到本仓库内官方替代 CLI estimator；本轮不实现 Python fitting。");
}

{
  const slide = base("Gym-DSSAT 可以消费调用方提供的 CLI；当前 YC 尚未启用", "Runtime hand-off · source audit", 6);
  const chain = [
    { x: 72, w: 236, title: "调用方", body: "auxiliary_file_paths\n显式加入 CNYC.CLI" },
    { x: 375, w: 236, title: "Wrapper", body: "按 basename 复制\n到临时运行目录" },
    { x: 678, w: 236, title: "DSSAT", body: "以临时目录为 cwd\nFileX weather mode W" },
    { x: 981, w: 226, title: "WGEN", body: "WSTA 前四位\n静态导出 CNYC.CLI" },
  ];
  for (const [i, step] of chain.entries()) {
    addRect(slide, step.x, 188, step.w, 144, i === 2 ? C.amberPale : C.tealPale, C.line, 6);
    addText(slide, step.title, step.x + 16, 208, step.w - 32, 32, 22, C.tealDark, true);
    addText(slide, step.body, step.x + 16, 251, step.w - 32, 62, 18, C.ink);
    if (i < chain.length - 1) addText(slide, "→", step.x + step.w + 17, 244, 36, 34, 24, C.teal, true, "center");
  }
  addLine(slide, 72, 374, 1208, 374, C.line, 1);
  addText(slide, "当前 YC 配置", 74, 402, 210, 32, 19, C.coral, true);
  addText(slide, "random_weather=False · 只传历史 WTH / soil / cultivar · 没有自动附加 CNYC.CLI", 286, 397, 900, 46, 20, C.ink, true);
  addRect(slide, 72, 477, 1136, 106, C.paper, "none", 5);
  addText(slide, "确认边界", 94, 497, 156, 28, 18, C.teal, true);
  addText(slide, "这只是 wrapper + 公开 DSSAT 4.8.5 源码的静态机制；未验证当前 runtime lookup、路径 fallback 或跨平台读取。", 250, 488, 930, 66, 18, C.muted);
  note(slide, "references/dssat_pdi.py 显示 auxiliary_file_paths 复制机制、临时工作目录、random_weather 到 W 与 rseed1 的参数映射。src/ppo_safe_rendering.py 当前 YC 路径用 random_weather=False，且没有 CLI 参数。公开 DSSAT 4.8.5 SECLI.for 静态推导 basename；当前 active runtime 未确认。来源：https://github.com/DSSAT/dssat-csm-os/blob/develop/InputModule/SECLI.for 。");
}

{
  const slide = base("Readiness：先补工具证据，再决定是否转换和生成", "Decision and next action", 7);
  addRect(slide, 68, 150, 1144, 102, C.coralPale, "none", 6);
  addText(slide, "can_generate_cnyc_cli_now", 92, 168, 392, 28, 17, C.coral, true);
  addText(slide, "BLOCKED_BY_WEATHERMAN_ACCESS", 92, 198, 900, 40, 27, C.ink, true);
  addTable(slide, [
    ["阻塞项", "需补齐的证据"],
    ["WeatherMan access", "本机 About/version 与生成入口（不探测项目外安装路径）"],
    ["Unknown import format", "CSV delimiter / ISO DATE / 列映射 / units / missing-value 规则"],
    ["Unknown CLI + station schema", "active 4.8.x 字段要求；Angstrom 与生长季等元数据来源"],
  ], 68, 278, 1144, 204, [340, 804], 18);
  addText(slide, "下一步：把版本与 Import/Export Help 截图或文本放入项目证据目录；先核合同，再决定无损转换；仅用冻结的 2004–2013 生成并做 hash/QC。", 74, 512, 1128, 88, 20, C.tealDark, true);
  note(slide, "阻塞项对应机器可读状态 results/yc_wgen_cli_pilot/003_06_03/cli_generation_readiness.json。本轮决策为 BLOCKED_BY_WEATHERMAN_ACCESS，附加 BLOCKED_BY_UNKNOWN_IMPORT_FORMAT、BLOCKED_BY_UNKNOWN_CLI_FORMAT、BLOCKED_BY_UNKNOWN_STATION_METADATA。用户下一步提供项目内 WeatherMan About/version、站点编辑字段要求、Import/Export format/help 证据后，重新核导入合同；本轮无 CLI、WGEN、DSSAT 或 PPO 结果。");
}

const candidatePath = path.join(TMP_DIR, "yc_cli_generation_path_candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);

for (const [index, slide] of slides.entries()) {
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(TMP_DIR, `slide-${String(index + 1).padStart(2, "0")}.layout.json`), await layout.text());
}

const finalPath = path.join(WORKSPACE_DIR, "docs", "yc_cli_generation_path.pptx");
const receiptPath = path.join(WORKSPACE_DIR, "results", "yc_wgen_cli_pilot", "003_06_03", "pptx_validation.json");
const finalization = await finalizePresentation({
  explicitTotalSlideCount: 7,
  workspaceDir: WORKSPACE_DIR,
  candidatePath,
  finalPath,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit",
    ...[3, 4, 7].flatMap((number) => ["--require-native-table-slide", String(number)]),
  ],
  requiredNativeTableOwnerSlides: [3, 4, 7],
  fontPolicy: { basis: "design", families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath,
});
console.log(JSON.stringify({ finalPath: finalization.finalPath, receiptPath: finalization.receiptPath, slides: 7, font: FONT, validationPassed: finalization.presentationLayout?.passed ?? null }, null, 2));
