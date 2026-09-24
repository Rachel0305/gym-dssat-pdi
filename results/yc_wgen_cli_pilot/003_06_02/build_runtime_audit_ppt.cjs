const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Codex';
pptx.subject = 'YC DSSAT runtime/version/WGEN capability audit';
pptx.title = 'YC DSSAT Runtime Audit';
pptx.company = 'Research audit record';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Microsoft YaHei',
  bodyFontFace: 'Microsoft YaHei',
  lang: 'zh-CN',
};
pptx.margin = 0;

const C = {
  ink: '203039',
  sub: '586A70',
  muted: '849397',
  paper: 'F7F9F8',
  white: 'FFFFFF',
  teal: '168C88',
  tealPale: 'DCEFED',
  amber: 'D79A26',
  amberPale: 'FBF0D7',
  coral: 'C96661',
  coralPale: 'F7E4E2',
  line: 'D8E1DF',
  slate: '35474D',
};
const FONT = 'Microsoft YaHei';
const W = 13.333;
const H = 7.5;

function rect(slide, x, y, w, h, fill, line = fill, radius = false) {
  slide.addShape(radius ? pptx.ShapeType.roundRect : pptx.ShapeType.rect, {
    x, y, w, h,
    rectRadius: 0.06,
    fill: { color: fill },
    line: { color: line, transparency: line === fill ? 100 : 0, width: 0.8 },
  });
}

function txt(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    margin: 0,
    fontFace: FONT,
    fontSize: 14,
    color: C.ink,
    breakLine: false,
    valign: 'mid',
    fit: 'shrink',
    paraSpaceAfterPt: 0,
    ...opts,
  });
}

function base(slide, section, page) {
  slide.background = { color: C.paper };
  rect(slide, 0, 0, W, 0.12, C.teal);
  txt(slide, 'YC  /  DSSAT RUNTIME AUDIT', 0.55, 0.28, 4.8, 0.22, {
    fontSize: 9, bold: true, color: C.teal, charSpacing: 0.4,
  });
  txt(slide, section, 10.1, 0.28, 2.68, 0.22, {
    fontSize: 9, color: C.muted, align: 'right',
  });
  rect(slide, 0.55, 7.08, 12.23, 0.012, C.line);
  txt(slide, '003_06_02  ·  2026-09-24  ·  仅项目内证据', 0.55, 7.16, 8.8, 0.18, {
    fontSize: 8, color: C.muted,
  });
  txt(slide, String(page).padStart(2, '0'), 12.1, 7.14, 0.68, 0.2, {
    fontSize: 9, color: C.teal, align: 'right', bold: true,
  });
}

function heading(slide, title, subtitle) {
  txt(slide, title, 0.55, 0.74, 12.2, 0.48, { fontSize: 25, bold: true });
  txt(slide, subtitle, 0.57, 1.27, 12.1, 0.32, { fontSize: 11, color: C.sub });
}

function pill(slide, label, x, y, w, fill, color) {
  rect(slide, x, y, w, 0.31, fill, fill, true);
  txt(slide, label, x + 0.08, y + 0.02, w - 0.16, 0.26, {
    fontSize: 9, bold: true, color, align: 'center',
  });
}

// Slide 1
{
  const slide = pptx.addSlide();
  slide.background = { color: C.paper };
  rect(slide, 0, 0, 0.18, H, C.teal);
  txt(slide, '003_06_02   /   RUNTIME · VERSION · CAPABILITY', 0.82, 0.75, 10.8, 0.27, {
    fontSize: 10, bold: true, color: C.teal, charSpacing: 0.5,
  });
  txt(slide, 'Gym-DSSAT runtime\n仍待实时核验', 0.82, 1.45, 11.7, 1.55, {
    fontSize: 36, bold: true, breakLine: false, valign: 'mid',
  });
  txt(slide, '仓库记录支持 Docker/Linux 历史执行路径，但当前安装版本、DSSAT 本体与 WeatherMan 均无实时证据。',
    0.85, 3.32, 10.9, 0.7, { fontSize: 16, color: C.sub, breakLine: false, valign: 'top' });
  rect(slide, 0.85, 4.55, 11.6, 0.95, C.white, C.line);
  rect(slide, 0.85, 4.55, 0.09, 0.95, C.amber);
  txt(slide, '审计结论', 1.15, 4.76, 1.25, 0.25, { fontSize: 11, bold: true, color: C.amber });
  txt(slide, '不要把依赖声明、旧运行记录或接口代码当成当前 runtime 已验证。',
    2.45, 4.69, 9.5, 0.42, { fontSize: 15, bold: true });
  txt(slide, '范围：仅项目内静态配置、源码和历史记录；未启动 DSSAT / WGEN / WeatherMan / PPO。',
    0.87, 6.2, 11.5, 0.27, { fontSize: 10, color: C.muted });
  txt(slide, 'YC  /  2026-09-24', 10.55, 6.73, 1.9, 0.2, { fontSize: 9, color: C.muted, align: 'right' });
}

// Slide 2
{
  const slide = pptx.addSlide();
  base(slide, '01  /  架构与版本', 2);
  heading(slide, '历史调用链可追溯，当前实例不可确认', '容器路径来自仓库记录；runtime 与包版本没有在本轮实时探测。');

  const items = [
    ['Windows', '项目工作区', C.tealPale, C.teal],
    ['Docker / Linux', '`nifty_taussig`', C.tealPale, C.teal],
    ['Python', '`/opt/gym_dssat_pdi/bin/python`', C.tealPale, C.teal],
    ['Gym-DSSAT', '安装版本未知', C.amberPale, C.amber],
    ['DSSAT', '`run_dssat` 后端未知', C.coralPale, C.coral],
  ];
  const x0 = 0.58, gap = 0.21, bw = 2.28, by = 2.03, bh = 1.08;
  items.forEach((item, i) => {
    const x = x0 + i * (bw + gap);
    rect(slide, x, by, bw, bh, item[2], item[2]);
    txt(slide, item[0], x + 0.13, by + 0.18, bw - 0.26, 0.25, {
      fontSize: 14, bold: true, color: item[3], align: 'center',
    });
    txt(slide, item[1], x + 0.12, by + 0.55, bw - 0.24, 0.3, {
      fontSize: 9, color: C.ink, align: 'center',
    });
    if (i < items.length - 1) {
      txt(slide, '›', x + bw + 0.035, by + 0.34, 0.14, 0.3, { fontSize: 21, bold: true, color: C.muted, align: 'center' });
    }
  });
  txt(slide, '证据中的版本差异', 0.6, 3.55, 4.0, 0.28, { fontSize: 14, bold: true });
  const versions = [
    ['依赖声明', '0.0.9', 'requirements.txt', C.tealPale, C.teal],
    ['旧运行诊断', '0.0.5', '历史 package_info', C.amberPale, C.amber],
    ['当前安装', 'UNKNOWN', '需 runtime 只读证据', C.coralPale, C.coral],
  ];
  versions.forEach((v, i) => {
    const x = 0.58 + i * 4.13;
    rect(slide, x, 4.0, 3.84, 1.54, C.white, C.line);
    rect(slide, x, 4.0, 3.84, 0.08, v[3]);
    txt(slide, v[0], x + 0.19, 4.22, 3.4, 0.22, { fontSize: 10, color: C.sub });
    txt(slide, v[1], x + 0.19, 4.56, 3.4, 0.42, { fontSize: 25, bold: true, color: v[4] });
    txt(slide, v[2], x + 0.19, 5.12, 3.4, 0.22, { fontSize: 9, color: C.muted });
  });
  txt(slide, 'runtime_type：Docker/Linux（项目历史记录支持，当前状态未核验）', 0.6, 6.0, 11.9, 0.3, {
    fontSize: 11, color: C.sub,
  });
}

// Slide 3
{
  const slide = pptx.addSlide();
  base(slide, '02  /  executable 与能力', 3);
  heading(slide, '“读取已有 CLI”不等于“拟合新 CLI”', '接口、配置与活动 runtime 三个证据层级分别呈现。');

  rect(slide, 0.58, 1.9, 5.92, 3.72, C.white, C.line);
  rect(slide, 0.58, 1.9, 5.92, 0.08, C.teal);
  txt(slide, '已有 .CLI  →  WGEN 随机天气', 0.86, 2.18, 5.25, 0.36, { fontSize: 17, bold: true });
  pill(slide, '接口可请求', 0.86, 2.76, 1.3, C.tealPale, C.teal);
  txt(slide, '`random_weather=True` → FileX `W` + `rseed1`', 2.32, 2.79, 3.85, 0.28, { fontSize: 10 });
  pill(slide, '当前 YC 配置关闭', 0.86, 3.36, 1.75, C.amberPale, C.amber);
  txt(slide, '`random_weather=False`；未传入 `.CLI`', 2.76, 3.39, 3.42, 0.28, { fontSize: 10 });
  pill(slide, '当前 runtime 未验证', 0.86, 3.96, 1.82, C.coralPale, C.coral);
  txt(slide, 'WGEN 是否可用、是否解析目标站点 `.CLI`：未知', 2.82, 3.99, 3.35, 0.42, { fontSize: 10, valign: 'top' });
  txt(slide, '配置 launcher：`/opt/dssat_pdi/run_dssat`\n底层 DSSAT executable / path：未知\nDSSAT version：UNKNOWN',
    0.88, 4.68, 5.15, 0.75, { fontSize: 10, color: C.sub, breakLine: false, valign: 'top' });

  rect(slide, 6.82, 1.9, 5.92, 3.72, C.white, C.line);
  rect(slide, 6.82, 1.9, 5.92, 0.08, C.amber);
  txt(slide, '历史日天气  →  新建 `.CLI`', 7.1, 2.18, 5.25, 0.36, { fontSize: 17, bold: true });
  pill(slide, 'wrapper 不拟合', 7.1, 2.76, 1.6, C.tealPale, C.teal);
  txt(slide, '只复制调用者显式提供的 auxiliary files', 8.89, 2.79, 3.54, 0.28, { fontSize: 10 });
  pill(slide, 'WeatherMan', 7.1, 3.36, 1.48, C.coralPale, C.coral);
  txt(slide, '当前容器是否安装：UNKNOWN_DUE_TO_ACCESS_RESTRICTION',
    8.79, 3.33, 3.66, 0.5, { fontSize: 9, valign: 'top' });
  pill(slide, '其他拟合工具', 7.1, 4.02, 1.48, C.coralPale, C.coral);
  txt(slide, 'runtime 未枚举；没有创建或猜测 `CNYC.CLI`', 8.79, 4.03, 3.66, 0.44, { fontSize: 9, valign: 'top' });
  txt(slide, 'CNLC.CLI 只作为仓库格式样例；未复制参数、未用于 YC。',
    7.12, 4.8, 5.18, 0.45, { fontSize: 10, color: C.sub, valign: 'top' });

  rect(slide, 0.58, 5.9, 12.16, 0.63, C.slate, C.slate);
  txt(slide, 'WeatherMan status', 0.83, 6.07, 1.72, 0.22, { fontSize: 10, color: 'C8D4D5', bold: true });
  txt(slide, 'UNKNOWN_DUE_TO_ACCESS_RESTRICTION', 2.6, 6.05, 4.0, 0.25, { fontSize: 11, color: C.white, bold: true });
  txt(slide, '不等于 NOT_INSTALLED', 8.52, 6.07, 3.65, 0.22, { fontSize: 10, color: 'F1C976', align: 'right', bold: true });
}

// Slide 4
{
  const slide = pptx.addSlide();
  base(slide, '03  /  blocker 与下一步', 4);
  heading(slide, '先拿到只读 runtime 证据，再决定 CLI 路线', '本轮 blocker 是访问边界；不能把“没有检查”写成“没有安装”。');

  const steps = [
    ['01', '包与容器', '容器标识/镜像摘要；`pip show gym-dssat-pdi`；import 文件路径'],
    ['02', '启动链', '`command -v run_dssat`；launcher 归属；最终 DSSAT executable 路径'],
    ['03', '版本证据', '版本输出、安装元数据或既有模拟输出头；不为取证启动模拟'],
    ['04', '天气工具', '只查已知 runtime 应用目录；分别确认 WeatherMan、WGEN 与 CLI 拟合器'],
  ];
  steps.forEach((s, i) => {
    const y = 1.95 + i * 0.82;
    rect(slide, 0.6, y, 12.1, 0.65, C.white, C.line);
    rect(slide, 0.6, y, 0.72, 0.65, i === 0 ? C.tealPale : C.paper, i === 0 ? C.tealPale : C.paper);
    txt(slide, s[0], 0.74, y + 0.17, 0.42, 0.25, { fontSize: 11, bold: true, color: i === 0 ? C.teal : C.sub, align: 'center' });
    txt(slide, s[1], 1.55, y + 0.1, 1.55, 0.42, { fontSize: 12, bold: true });
    txt(slide, s[2], 3.05, y + 0.11, 9.3, 0.4, { fontSize: 10, color: C.sub });
  });
  rect(slide, 0.6, 5.47, 12.1, 0.94, C.tealPale, C.tealPale);
  txt(slide, '本轮不变更', 0.86, 5.7, 1.35, 0.26, { fontSize: 11, bold: true, color: C.teal });
  txt(slide, 'frozen weather candidate · CNYC.CLI · PPO · reward/action/observation · 其他站点',
    2.25, 5.68, 9.95, 0.3, { fontSize: 11, color: C.ink });
  txt(slide, '下一阶段仅在获得授权/项目内证据后做 runtime 只读复核；然后才评估 train-only CLI 参数来源。',
    0.63, 6.62, 11.95, 0.26, { fontSize: 10, color: C.sub });
}

pptx.writeFile({ fileName: 'docs/yc_dssat_runtime_version_audit.pptx' })
  .then(() => console.log('Wrote docs/yc_dssat_runtime_version_audit.pptx'))
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
