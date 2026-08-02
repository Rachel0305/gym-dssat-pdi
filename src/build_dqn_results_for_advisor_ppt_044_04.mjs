import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const ROOT = process.env.PROJECT_ROOT
  ? path.resolve(process.env.PROJECT_ROOT)
  : path.resolve(path.dirname(__filename), "..");
const RESULT_DIR = path.join(ROOT, "benchmark_results", "044_03_dqn_results_for_advisor_summary");
const OUT_DIR = path.join(RESULT_DIR, "ppt");
const RENDER_DIR = path.join(OUT_DIR, "rendered");
const FINAL_PPTX = path.join(OUT_DIR, "044_03_dqn_results_for_advisor_summary_minimal.pptx");

const checkpointCsv = path.join(RESULT_DIR, "tables", "044_03_dqn_checkpoint_mean_metrics.csv");
const qCsv = path.join(RESULT_DIR, "tables", "044_03_demo_dqn_q_ranking_summary.csv");
const figCheckpoint = path.join(RESULT_DIR, "figures", "044_03_dqn_checkpoint_metric_overview.png");
const figYearly = path.join(RESULT_DIR, "figures", "044_03_dqn_best_checkpoint_yearly_yield.png");
const figQ = path.join(RESULT_DIR, "figures", "044_03_demo_dqn_q_ranking_diagnostics.png");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"' && quoted && next === '"') {
      cell += '"';
      i++;
    } else if (ch === '"') {
      quoted = !quoted;
    } else if (ch === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && next === "\n") i++;
      row.push(cell);
      if (row.some((x) => x.length > 0)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    if (row.some((x) => x.length > 0)) rows.push(row);
  }
  const header = rows[0].map((h, i) => (i === 0 ? h.replace(/^\uFEFF/, "") : h));
  return rows.slice(1).map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

async function readCsv(file) {
  return parseCsv(await fs.readFile(file, "utf8"));
}

function num(v, digits = 0) {
  if (v === undefined || v === null || String(v).trim() === "") return "NA";
  const x = Number(v);
  if (!Number.isFinite(x)) return "NA";
  return x.toFixed(digits);
}

async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}

function addTitle(slide, text) {
  const title = slide.shapes.add({
    geometry: "textbox",
    position: { left: 48, top: 28, width: 1184, height: 54 },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  title.text = text;
  title.text.style = { fontSize: 34, bold: true, color: "black" };
}

function addText(slide, text, position, fontSize = 22, bold = false) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = { fontSize, bold, color: "black" };
  return box;
}

function addPlainTable(slide, values, position, fontSize = 14) {
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: position.left,
    top: position.top,
    width: position.width,
    height: position.height,
    values,
  });
  table.borders.assign({ style: "solid", fill: "808080", width: 1 });
  for (let r = 0; r < values.length; r++) {
    for (let c = 0; c < values[0].length; c++) {
      const cell = table.getCell(r, c);
      cell.text.style = { fontSize, color: "black", bold: r === 0 };
      if (r === 0) cell.fill = "F2F2F2";
    }
  }
}

async function addImage(slide, file, position, alt) {
  const bytes = await fs.readFile(file);
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
  });
}

function bestByMethod(rows) {
  const best = new Map();
  for (const r of rows) {
    const m = r.method;
    const y = Number(r.mean_final_grnwt);
    if (!best.has(m) || y > Number(best.get(m).mean_final_grnwt)) best.set(m, r);
  }
  return Array.from(best.values());
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(RENDER_DIR, { recursive: true });

  const checkpointRows = await readCsv(checkpointCsv);
  const qRows = await readCsv(qCsv);
  const bestRows = bestByMethod(checkpointRows);

  const pres = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // Slide 1
  {
    const slide = pres.slides.add();
    slide.background.fill = "white";
    addText(slide, "DQN / DQfD 对照结果整理", { left: 70, top: 140, width: 1140, height: 70 }, 44, true);
    addText(slide, "SYA lowIC 自由时序框架；只整理已有结果，不重新训练", { left: 70, top: 230, width: 1140, height: 40 }, 24);
    addText(slide, "纳入：普通 DQN 040_01、严格 MaskableDQN 040_02、Demo-DQN/DQfD 044_00–044_02", { left: 70, top: 310, width: 1140, height: 80 }, 22);
    addText(slide, "结论：当前 DQN 系列没有形成可作为主线正向结果的稳定策略，适合作为 PPO 主线的对照证据。", { left: 70, top: 430, width: 1100, height: 90 }, 24, true);
  }

  // Slide 2
  {
    const slide = pres.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "最佳 checkpoint 汇总（按验证年平均产量）");
    const vals = [
      ["方法", "checkpoint", "平均产量", "灌溉", "施氮", "PFP_N", "水胁迫天数", "氮胁迫天数"],
      ...bestRows.map((r) => [
        r.method,
        r.checkpoint_step,
        num(r.mean_final_grnwt, 0),
        num(r.mean_total_irrigation, 1),
        num(r.mean_total_n, 1),
        num(r.mean_PFP_N, 1),
        num(r.mean_swfac_stress_days_gt_0p05, 1),
        num(r.mean_nstres_days_gt_0p05, 1),
      ]),
    ];
    addPlainTable(slide, vals, { left: 45, top: 110, width: 1190, height: 230 }, 14);
    addText(slide, "解读：普通 DQN 的平均产量最高但仍偏低；严格 MaskableDQN 训练加深后退化；Demo-DQN/DQfD 在 smoke 中退化到 no-op。", { left: 60, top: 395, width: 1160, height: 100 }, 22);
  }

  // Slide 3
  {
    const slide = pres.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "训练 checkpoint 指标变化");
    await addImage(slide, figCheckpoint, { left: 40, top: 90, width: 1200, height: 590 }, "DQN checkpoint metric overview");
  }

  // Slide 4
  {
    const slide = pres.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "最佳 checkpoint 的逐年产量");
    await addImage(slide, figYearly, { left: 40, top: 90, width: 1200, height: 590 }, "DQN yearly yield");
  }

  // Slide 5
  {
    const slide = pres.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "Demo-DQN / DQfD 没有把非零 teacher 动作学成高 Q 值");
    await addImage(slide, figQ, { left: 40, top: 90, width: 1200, height: 460 }, "Demo-DQN Q ranking diagnostics");
    const vals = [
      ["方法", "step/epoch", "非零样本数", "非零teacher为argmax", "Qteacher-Qnoop"],
      ...qRows.map((r) => [
        r.method,
        r.checkpoint_step,
        r.nonzero_count,
        num(r.nonzero_teacher_argmax_rate, 3),
        num(r.mean_q_teacher_minus_noop, 3),
      ]),
    ];
    addPlainTable(slide, vals, { left: 55, top: 560, width: 1170, height: 115 }, 11);
  }

  // Slide 6
  {
    const slide = pres.slides.add();
    slide.background.fill = "white";
    addTitle(slide, "汇报口径");
    const text = [
      "1. 已按导师要求补充 DQN/DQfD 对照：普通 DQN、严格 MaskableDQN、Demo-DQN/DQfD。",
      "2. 三条 DQN 路线在当前 SYA lowIC 自由时序设置下均未优于当前 PPO 主线。",
      "3. Demo 经验确实进入了训练/诊断链，但非零 teacher 动作没有稳定成为 Q 值最高动作。",
      "4. 目前建议：DQN 作为对照和阴性证据保留；主线继续优化 PPO 的天气响应性、动作合理性和指标表现。",
    ].join("\\n");
    addText(slide, text, { left: 70, top: 120, width: 1120, height: 360 }, 26);
    addText(slide, "注意：本 PPT 只整理已有 DQN 结果，不代表数学上证明 DQN 不可能成功。", { left: 70, top: 560, width: 1120, height: 50 }, 22, true);
  }

  for (const [i, slide] of pres.slides.items.entries()) {
    const png = await pres.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(RENDER_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), png);
  }
  const montage = await pres.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(OUT_DIR, "044_03_dqn_results_for_advisor_summary_minimal_montage.webp"), montage);

  const pptx = await PresentationFile.exportPptx(pres);
  await pptx.save(FINAL_PPTX);

  const result = {
    task: "044_04_build_dqn_results_for_advisor_minimal_ppt",
    pptx: path.relative(ROOT, FINAL_PPTX),
    rendered_dir: path.relative(ROOT, RENDER_DIR),
    montage: path.relative(ROOT, path.join(OUT_DIR, "044_03_dqn_results_for_advisor_summary_minimal_montage.webp")),
    slides: pres.slides.items.length,
  };
  await fs.writeFile(path.join(OUT_DIR, "044_04_ppt_result.json"), JSON.stringify(result, null, 2), "utf8");
  console.log(JSON.stringify(result, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
