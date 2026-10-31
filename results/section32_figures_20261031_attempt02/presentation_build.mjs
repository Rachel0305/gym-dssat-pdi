import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const root=process.cwd(),build=path.join(root,'.codex-section32-136'),out=path.join(root,'results/section32_figures_20261031_attempt02');
const skill='C:/Users/DELL/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const p=Presentation.create({slideSize:{width:1280,height:960}});
function text(s,v,x,y,w,h,size=32,bold=false,color='#25364A'){const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});t.text=v;t.text.style={typeface:'Microsoft YaHei',fontSize:size,color,bold};return t;}
function slide(title,notes){const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,64,34,1150,75,42,true,'#12334E');text(s,`Results 3.2 图表记录   2026-10-31   ${p.slides.items.length}/4`,64,916,1150,28,18,false,'#68798C');s.speakerNotes.text=notes;return s;}
let s=slide('图表目标与数据合同','来源：136任务prompt；135审计；076、077、079冻结统计输出。');
text(s,'FIGURES_READY_WITH_CAVEATS',64,165,1150,80,45,true,'#166466');
text(s,'只比较历史训练H与天气增强A\n五站，每组8seed，共同评价2014—2023年\n保持已审计Bootstrap区间与Welch敏感性结果',64,290,1150,220,33);
text(s,'主图：平均产量、年际SD、资源管理权衡\n补充图：全部十年，统一纵轴，不平滑\n补充表：五站 × 六指标，保留单位与有效口径',64,560,1150,210,31);
text(s,'不训练、不模拟、不重算区间、不改变结论',64,815,1150,55,29);
s=slide('Figure X：三类证据的互补展示','嵌入最终figure_section32_main.png。PDF和SVG保留文字矢量；400dpi PNG用于本页。A均产，B年际SD及Welch，C描述性相对变化。');
s.images.add({blob:new Uint8Array(await fs.readFile(path.join(out,'figure_section32_main.png'))),contentType:'image/png',alt:'Three-panel management performance figure',fit:'contain',position:{left:180,top:110,width:920,height:790}});
s=slide('Figure Sx：十年年度差异全部保留','嵌入最终figure_section32_annual.png。每年组内8seed均值之差，纵轴相同，无逐年检验。正差年数SY1、HL5、LC3、YC2、FQ7。');
s.images.add({blob:new Uint8Array(await fs.readFile(path.join(out,'figure_section32_annual.png'))),contentType:'image/png',alt:'Year-specific yield differences across five sites',fit:'contain',position:{left:64,top:125,width:1150,height:735}});
s=slide('图表检查与论文引用','来源：figure_quality_check.md；section_3_2_with_figures_zh.md；table_section32_supplementary.csv；figure_captions.md。');
text(s,'正文第一段：Figure X A\n第二段：Figure X C及Table Sx\n第三段：Figure Sx\n第四段：Figure X B及Table Sx',64,155,1150,260,33);
text(s,'检查通过：30条统计表、50条年度差值\nPDF/SVG矢量、400dpi PNG，英文图注与源数据\n数值可追溯，H/A未混入预报组，无星号',64,455,1150,190,31);
text(s,'保留限制：固定十年、8seed、多重比较未校正\nHL区间排零对方法敏感；天气来源限制仍适用\n目标期刊未指定，最终版式需按其要求调整',64,700,1150,170,29,false,'#9C562B');
for(let i=0;i<p.slides.items.length;i++){const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'docs/2026-10-31_section32_figure_design.pptx'),pythonExecutable:'C:/Users/DELL/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,9144000','--validate-heading-fit'],fontPolicy:{basis:'design',families:['Microsoft YaHei']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log('PPTX finalized');
