import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const root=process.cwd(),dir=path.join(root,'.codex-section32-135');
const SKILL='C:/Users/DELL/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const data=JSON.parse(await fs.readFile(path.join(dir,'data.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Microsoft YaHei';
function text(s,v,x,y,w,h,size=28,color='#25364A',bold=false){const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});t.text=v;t.text.style={typeface:font,fontSize:size,color,bold};return t;}
function slide(title,source){let s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,64,48,1150,80,42,'#12334E',true);text(s,`Results 3.2 统计审计   2026-10-31   ${p.slides.items.length}/5`,64,664,1150,30,18,'#68798C');s.speakerNotes.text=`来源：${source}\n固定2014—2023评价期，未经多重比较校正，探索性统计。`;return s;}
let s=slide('统计计算可复现，HL区间判断对方法敏感','bootstrap_audit_report.md; verification.json');
text(s,'READY_WITH_CAVEATS',64,160,1150,75,48,'#166466',true);
text(s,'五站 × 两组 × 8个训练seed × 固定10年\n80个策略，800条年度产量，模型哈希全部核对\n本轮复现原区间，仅补LOSO与Welch两项检查',64,280,1150,220,32);
text(s,'不训练、不运行DSSAT、不生成天气',64,550,1150,55,26);
s=slide('Bootstrap以训练seed为单位','scripts/analyze_weather_increment_076.py; scripts/section32_year_stability_080.py; methods_statistical_analysis.md');
text(s,'每seed：十年均产，或十年样本SD（ddof=1）\n每组：汇总8个策略统计量，计算增强组减历史组\n组内独立有放回抽样，不按相同seed编号配对',64,160,1150,200,30);
text(s,'均产：4,000次，随机种子20261030\n年际SD：10,000次，随机种子20261031\n经验Percentile区间，完整复现原随机流与索引',64,390,1150,170,29);
text(s,'固定年份下的训练随机性，不代表未来任意年份',64,590,1150,45,25,'#9C562B');
s=slide('五站年际SD差异的稳健性检查','bootstrap_robustness_summary.csv; leave_one_seed_out_80.csv');
const f=x=>(+x).toFixed(1),range=(l,h)=>`[${f(l)}, ${f(h)}]`;
let vals=[['站点','ΔSD','Bootstrap 95%','Welch 95%','LOSO范围'],...data.map(r=>[r.site,f(r.delta_sd),range(r.bootstrap_low,r.bootstrap_high),range(r.welch_low,r.welch_high),range(r.loso_min,r.loso_max)])];
let tb=s.tables.add({rows:6,columns:5,left:64,top:170,width:1150,height:350,columnWidths:[100,130,310,310,300],values:vals});
for(let r=0;r<6;r++)for(let c=0;c<5;c++){let cell=tb.getCell(r,c);cell.fill=r===0?'#12334E':r%2?'#F1F5F8':'#FFFFFF';cell.text.style={typeface:font,fontSize:23,color:r===0?'#FFFFFF':'#25364A',bold:r===0};}
text(s,'单位 kg/ha。SY删除单seed后可反向，原本已无明确方向证据。\n除HL外四站的Bootstrap与Welch区间均跨零。',64,550,1150,90,25);
s=slide('HL增加方向保持，区间排零不稳健','bootstrap_robustness_summary.csv; leave_one_seed_out_80.csv');
text(s,'年际SD均值：884.0 → 1029.2 kg/ha\n差值：+145.2 kg/ha',64,160,1150,125,34);
text(s,'Bootstrap 95%：[7.6, 304.0]\nWelch 95%：[−39.0, 329.4]',64,320,1150,110,32);
text(s,'逐次删一个seed：差值 +97.7 至 +170.4，方向未反转\n最小值：删除增强组seed5；最大值：删除历史组seed1',64,470,1150,100,27);
text(s,'保留“探索性增加”，不作正式显著性声明',64,595,1150,45,28,'#9C562B',true);
s=slide('3.2可按限定范围收尾','bootstrap_audit_report.md; section_3_2_final_zh.md; existing weather audit sources');
text(s,'分类B：计算正确，HL区间判断对统计方法敏感\n原均产差异及区间全部复现，四站结论保持\nResults仅调整HL确定性，Methods补齐抽样与限制',64,160,1150,180,31);
text(s,'8seed的小样本，所有95%区间未校正多重比较\n固定十年与天气调度，不能推广任意未来年份\n天气来源限制保留，统计审计不等于统一天气QC认证',64,385,1150,170,28);
text(s,'READY_WITH_CAVEATS',64,590,1150,45,31,'#166466',true);
for(let i=0;i<p.slides.items.length;i++){let png=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(dir,`slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));}
const candidate=path.join(dir,'candidate_v3.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'docs/2026-10-31_section32_bootstrap_audit_v2.pptx'),pythonExecutable:'C:/Users/DELL/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3'],requiredNativeTableOwnerSlides:[3],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(dir,'validation_v2.json')});
console.log('PPTX finalized');
