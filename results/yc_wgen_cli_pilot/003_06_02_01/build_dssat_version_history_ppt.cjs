const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Codex';
pptx.subject = 'DSSAT 4.8.0 and 4.8.5 repository/Git history trace';
pptx.title = 'DSSAT Version History Trace';
pptx.lang = 'zh-CN';
pptx.theme = { headFontFace: 'Microsoft YaHei', bodyFontFace: 'Microsoft YaHei', lang: 'zh-CN' };

const C = {
  ink: '203039', sub: '586A70', muted: '849397', paper: 'F7F9F8', white: 'FFFFFF',
  teal: '168C88', tealPale: 'DCEFED', amber: 'D79A26', amberPale: 'FBF0D7',
  coral: 'C96661', coralPale: 'F7E4E2', line: 'D8E1DF', slate: '35474D',
};
const FONT = 'Microsoft YaHei';
const W = 13.333;
const H = 7.5;

function shape(slide, x, y, w, h, color, line = color, round = false) {
  slide.addShape(round ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color },
    line: { color: line, transparency: color === line ? 100 : 0, width: 0.8 },
  });
}

function text(slide, value, x, y, w, h, options = {}) {
  slide.addText(value, {
    x, y, w, h, margin: 0, fontFace: FONT, fontSize: 13, color: C.ink,
    valign: 'mid', fit: 'shrink', breakLine: false, paraSpaceAfterPt: 0,
    ...options,
  });
}

function base(slide, label, page) {
  slide.background = { color: C.paper };
  shape(slide, 0, 0, W, 0.12, C.teal);
  text(slide, 'DSSAT VERSION HISTORY TRACE', 0.55, 0.28, 4.5, 0.22, {
    fontSize: 9, bold: true, color: C.teal, charSpacing: 0.4,
  });
  text(slide, label, 9.6, 0.28, 3.18, 0.22, { fontSize: 9, color: C.muted, align: 'right' });
  shape(slide, 0.55, 7.08, 12.23, 0.012, C.line);
  text(slide, '003_06_02_01  ·  2026-09-24  ·  仓库与本地 Git 只读追溯', 0.55, 7.16, 9.5, 0.18, {
    fontSize: 8, color: C.muted,
  });
  text(slide, String(page).padStart(2, '0'), 12.1, 7.14, 0.68, 0.2, {
    fontSize: 9, bold: true, color: C.teal, align: 'right',
  });
}

function heading(slide, title, subtitle) {
  text(slide, title, 0.58, 0.75, 12.15, 0.44, { fontSize: 24, bold: true });
  text(slide, subtitle, 0.6, 1.25, 12.05, 0.33, { fontSize: 11, color: C.sub });
}

function tag(slide, label, x, y, w, fill, color) {
  shape(slide, x, y, w, 0.31, fill, fill, true);
  text(slide, label, x + 0.07, y + 0.02, w - 0.14, 0.25, {
    fontSize: 9, bold: true, color, align: 'center',
  });
}

// 1. Cover and findings
{
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  shape(slide, 0, 0, 0.18, H, C.teal);
  text(slide, '003_06_02_01   /   REPOSITORY + GIT HISTORY', 0.82, 0.78, 10.8, 0.25, {
    fontSize: 10, bold: true, color: C.teal, charSpacing: 0.4,
  });
  text(slide, 'DSSAT 版本历史\n有据可循，当前仍未知', 0.82, 1.42, 11.8, 1.5, {
    fontSize: 35, bold: true, breakLine: false,
  });
  text(slide, '历史输出头确认 DSSAT 4.8.0.024；档案记载 Windows DSSAT 4.8.5 实验。两者都不能自动代表 2026-09 当前 runtime。',
    0.86, 3.28, 10.9, 0.72, { fontSize: 15, color: C.sub, valign: 'top' });
  const cards = [
    ['历史 4.8.0', 'CONFIRMED', C.tealPale, C.teal],
    ['历史 4.8.5', 'SUPPORTED', C.amberPale, C.amber],
    ['当前 runtime', 'UNKNOWN', C.coralPale, C.coral],
  ];
  cards.forEach((item, i) => {
    const x = 0.85 + i * 3.92;
    shape(slide, x, 4.65, 3.55, 1.02, C.white, C.line);
    shape(slide, x, 4.65, 3.55, 0.07, item[2]);
    text(slide, item[0], x + 0.18, 4.86, 3.18, 0.22, { fontSize: 10, color: C.sub });
    text(slide, item[1], x + 0.18, 5.17, 3.18, 0.28, { fontSize: 16, bold: true, color: item[3] });
  });
  text(slide, '仅追溯仓库与本地 Git；未访问容器、未运行 DSSAT/WGEN、未改动实验输入。',
    0.87, 6.35, 11.4, 0.25, { fontSize: 10, color: C.muted });
}

// 2. Search method and evidence layers
{
  const slide = pptx.addSlide();
  base(slide, '01  /  检索方法', 2);
  heading(slide, '把版本头、实验记录和命名线索分开', '当前树覆盖 docs/results/src/prompts/references/scripts/backups 与相关输出目录；Git history 只读检查所有本地 refs。');
  const rows = [
    ['DIRECT_VERSION_EVIDENCE', '保存的 `Summary.OUT` 版本头：DSSAT 4.8.0.024', C.tealPale, C.teal],
    ['DIRECT_VERSION_EVIDENCE', '2026-06-27 实验总结明确记载 Windows DSSAT 4.8.5', C.amberPale, C.amber],
    ['STRONG_INDIRECT_EVIDENCE', '`C:\\DSSAT48\\DSCSM048.EXE`：仅支持 4.8 家族', C.amberPale, C.amber],
    ['USER_OR_DOCUMENT_RECOLLECTION', '“另一台电脑”只见于本任务引用的用户回忆', C.coralPale, C.coral],
  ];
  rows.forEach((r, i) => {
    const y = 1.95 + i * 0.82;
    shape(slide, 0.62, y, 12.05, 0.64, C.white, C.line);
    tag(slide, r[0], 0.82, y + 0.16, i === 3 ? 2.7 : 2.5, r[2], r[3]);
    text(slide, r[1], 3.65, y + 0.12, 8.7, 0.38, { fontSize: 11, color: C.ink });
  });
  shape(slide, 0.62, 5.55, 12.05, 0.88, C.slate, C.slate);
  text(slide, '排除混淆', 0.89, 5.79, 1.15, 0.25, { fontSize: 11, bold: true, color: 'C8D4D5' });
  text(slide, '`gym_dssat_pdi 0.0.5 / 0.0.9` 是 Python package 版本；不能替代 DSSAT 模型版本。',
    2.08, 5.75, 9.95, 0.34, { fontSize: 12, bold: true, color: C.white });
  text(slide, '任务提示里的线索也纳入搜索，但不作为独立实验事实。', 0.64, 6.63, 11.8, 0.24, { fontSize: 10, color: C.sub });
}

// 3. Direct 4.8.0 evidence
{
  const slide = pptx.addSlide();
  base(slide, '02  /  DSSAT 4.8.0', 3);
  heading(slide, 'YC 历史 PDI 运行头直接报出 4.8.0.024', '这是已保存的历史模拟输出，不是本轮启动的新模拟，也不是当前 runtime 的实时查询。');
  shape(slide, 0.62, 1.92, 12.05, 1.17, C.slate, C.slate);
  text(slide, 'DSSAT Cropping System Model Ver. 4.8.0.024 -stable', 0.9, 2.17, 11.5, 0.33, {
    fontFace: 'Consolas', fontSize: 17, bold: true, color: C.white,
  });
  text(slide, 'Summary.OUT  ·  2026-07-02 08:37:47', 0.91, 2.64, 8.7, 0.22, {
    fontSize: 9, color: 'C8D4D5',
  });
  const facts = [
    ['实验', 'YC / CNYC · 2014 · seed0 · null'],
    ['输出位置', '`pdi_tmp_snapshot_eval/Summary.OUT`'],
    ['同次配置', '`env_args.json` → `/opt/dssat_pdi/run_dssat`'],
    ['证据等级', 'DIRECT_VERSION_EVIDENCE'],
  ];
  facts.forEach((f, i) => {
    const y = 3.48 + i * 0.55;
    text(slide, f[0], 0.8, y, 1.3, 0.25, { fontSize: 10, bold: true, color: C.teal });
    text(slide, f[1], 2.1, y - 0.01, 9.9, 0.29, { fontSize: 11, color: C.ink });
  });
  shape(slide, 0.62, 5.93, 12.05, 0.64, C.tealPale, C.tealPale);
  text(slide, '结论：至少这次历史 Gym-DSSAT/PDI run 确实调用了报告 4.8.0.024 的 DSSAT。',
    0.88, 6.12, 11.55, 0.26, { fontSize: 12, bold: true, color: C.teal });
}

// 4. 4.8.5 and historical comparison
{
  const slide = pptx.addSlide();
  base(slide, '03  /  DSSAT 4.8.5', 4);
  heading(slide, '4.8.5 有实验记录，但不是纯版本因果试验', '2026-06-27 HLA 2004 CNHL0404 记录：Windows 4.8.5 与 PDI/Gym 4.8.0 的结果出现差异。');
  const cols = [
    { x: 0.62, color: C.amber, pale: C.amberPale, title: 'Windows DSSAT 4.8.5', body: 'HLA 2004 · CNHL0404\n记录为成熟，HWAM 约 2038 kg/ha\nWindows standalone 环境' },
    { x: 6.8, color: C.teal, pale: C.tealPale, title: 'PDI / Gym DSSAT 4.8.0', body: '同名输入诊断\n记录为早熟，HWAM = 0\n后续报告提醒版本不可混比' },
  ];
  cols.forEach((c) => {
    shape(slide, c.x, 2.0, 5.84, 2.05, C.white, C.line);
    shape(slide, c.x, 2.0, 5.84, 0.08, c.pale);
    text(slide, c.title, c.x + 0.24, 2.26, 5.36, 0.34, { fontSize: 16, bold: true, color: c.color });
    text(slide, c.body, c.x + 0.24, 2.82, 5.28, 0.9, { fontSize: 11, color: C.sub, valign: 'top' });
  });
  text(slide, '记录层面的版本对照：有', 0.82, 4.48, 3.25, 0.28, { fontSize: 13, bold: true });
  text(slide, '版本差异的独立因果结论：没有', 6.86, 4.48, 4.95, 0.28, { fontSize: 13, bold: true, color: C.coral });
  shape(slide, 0.62, 5.03, 12.05, 1.0, C.amberPale, C.amberPale);
  text(slide, '证据边界', 0.88, 5.31, 1.35, 0.25, { fontSize: 11, bold: true, color: C.amber });
  text(slide, '4.8.5 原始输出版本头未在本次检查的 CNHL0404 文件夹找到；“另一台电脑”也未获项目文件独立佐证。',
    2.25, 5.23, 9.95, 0.48, { fontSize: 11, color: C.ink, valign: 'top' });
  text(slide, 'Git commit 7bc4024（2026-06-27）新增该总结与对照脚本；后续 4.8.0 standalone 输出替换了旧 4.8.5 对照侧。',
    0.65, 6.48, 12.0, 0.28, { fontSize: 9, color: C.sub });
}

// 5. Historical/current distinction and next step
{
  const slide = pptx.addSlide();
  base(slide, '04  /  连续性与下一步', 5);
  heading(slide, '历史已证实，不代表今天仍相同', '本轮没有访问容器或外部安装路径；current active runtime 保持 UNKNOWN。');
  const blocks = [
    ['已确认', '2026-07-02 保存的 YC PDI 输出为 DSSAT 4.8.0.024；同次 env args 指向 `/opt/dssat_pdi/run_dssat`。', C.tealPale, C.teal],
    ['未确认', '当前 container/launcher 是否延续、当前 DSSAT 版本、该次 gym_dssat_pdi 安装版本、4.8.5 的原始 header。', C.coralPale, C.coral],
    ['建议', '先取得当前 runtime 的只读版本证据；如需核实 4.8.5，补齐既有 Windows 输出头和机器/date provenance。', C.amberPale, C.amber],
  ];
  blocks.forEach((b, i) => {
    const y = 1.95 + i * 1.04;
    shape(slide, 0.63, y, 12.05, 0.83, C.white, C.line);
    shape(slide, 0.63, y, 1.45, 0.83, b[2], b[2]);
    text(slide, b[0], 0.77, y + 0.26, 1.16, 0.25, { fontSize: 12, bold: true, color: b[3], align: 'center' });
    text(slide, b[1], 2.32, y + 0.13, 9.95, 0.54, { fontSize: 11, color: C.ink, valign: 'mid' });
  });
  shape(slide, 0.63, 5.45, 12.05, 0.89, C.slate, C.slate);
  text(slide, '本任务未改动', 0.9, 5.75, 1.5, 0.23, { fontSize: 11, bold: true, color: 'C8D4D5' });
  text(slide, 'weather candidate · CLI · DSSAT inputs · PPO · other sites',
    2.52, 5.71, 8.9, 0.3, { fontSize: 12, bold: true, color: C.white });
  text(slide, '无 DSSAT simulation / WGEN / WeatherMan / PPO；未 git push。',
    0.66, 6.58, 11.9, 0.24, { fontSize: 10, color: C.sub });
}

pptx.writeFile({ fileName: 'docs/yc_dssat_version_history_trace.pptx' })
  .then(() => console.log('Wrote docs/yc_dssat_version_history_trace.pptx'))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
