"""Read completed offline diagnostics; report nested descriptive rates."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/forecast_policy_response_127_20261027'
r=json.loads((OUT/'result.json').read_text());assert r['status']=='completed'
assert json.loads((OUT/'protected_files_check.json').read_text())['unchanged']
s=pd.read_csv(OUT/'site_variant_20.csv');p=pd.read_csv(OUT/'policy_variant_160.csv');y=pd.read_csv(OUT/'model_year_variant_1600.csv')
assert len(s)==20 and len(p)==160 and len(y)==1600
rows=[]
for site,g in s.groupby('site'):
    sh=g[g.variant.str.startswith('shuffle')];z=g[g.variant.eq('zero_block')].iloc[0]
    rows.append(dict(site=site,eligible_unique_states=int(z.eligible),shuffle_trials=int(sh.eligible.sum()),shuffle_changed=int(sh.action_changed.sum()),shuffle_pct=100*sh.action_changed.sum()/sh.eligible.sum(),shuffle_input_changed_trials=int(sh.eligible_changed_input.sum()),shuffle_TV_mean=sh.TV_sum_eligible.sum()/sh.eligible.sum(),zero_changed=int(z.action_changed),zero_pct=100*z.action_change_rate))
t=pd.DataFrame(rows);t.to_csv(OUT/'site_readout_5.csv',index=False)
sh=s[s.variant.str.startswith('shuffle')];z=s[s.variant.eq('zero_block')];changed=int(sh.action_changed.sum());n=int(sh.eligible.sum());zerochanged=int(z.action_changed.sum());eligible=int(z.eligible.sum())
affected=p[p.variant.str.startswith('shuffle')].groupby(['site','seed']).action_changed.sum().gt(0)
unique_sets={}
for chunk in pd.read_csv(OUT/'state_response.csv',chunksize=20000):
    g=chunk[chunk.variant.str.startswith('shuffle')&chunk.action_changed&chunk.legal_count.gt(1)]
    for row in g.itertuples():unique_sets[(row.site,row.seed,row.year,row.step)]=True
stats=dict(models=40,model_years=400,original_states=int(z.states.sum()),eligible_unique_states=eligible,shuffle_action_changes=changed,shuffle_trials=n,shuffle_pct=100*changed/n,unique_states_changed_by_any_shuffle=len(unique_sets),models_with_any_shuffle_action_change=int(affected.sum()),models_without_shuffle_action_change=int((~affected).sum()),zero_changes=zerochanged,zero_pct=100*zerochanged/eligible)
(OUT/'readout_summary.json').write_text(json.dumps(stats,indent=2))
lines=['# 预报通道响应：40模型冻结状态诊断','', '采用全部40个100000步模型与2014—2023归档状态；FQ2018使用125质量控制版本。先SY seed0 2014技术smoke，再串行所有模型；没有模拟或训练。','',f"## 主要结果\n\n原始决策状态{stats['original_states']}个，其中{eligible}个具有至少两个合法动作。3次整块预报置换产生{n}个嵌套状态—置换组合，确定性动作改变{changed}次（{stats['shuffle_pct']:.4f}%）。任一置换引起动作改变的原状态共{len(unique_sets)}个；40模型中{int(affected.sum())}个至少出现一次动作改变。置换组合不是独立样本，不据此计算总体成功率或显著性。",'', '|站点|多合法动作状态数|三次置换动作变化/组合|变化比例|全零屏蔽变化比例|','|---|---:|---:|---:|---:|']
for row in t.itertuples():lines.append(f'|{row.site}|{row.eligible_unique_states}|{row.shuffle_changed}/{row.shuffle_trials}|{row.shuffle_pct:.4f}%|{row.zero_pct:.4f}%|')
lines+=['',f"全零屏蔽下动作改变{zerochanged}/{eligible}（{stats['zero_pct']:.4f}%）。屏蔽可能分布外，只用于通道敏感性诊断，不代表现实无预报或真实误差预报。置换下多合法状态的平均TV={sh.TV_sum_eligible.sum()/sh.eligible.sum():.8f}，最大TV={sh.TV_max_eligible.max():.6f}；概率会变化但大多数argmax不变。TV/JS为数值描述，不设置事后有效性阈值。",'', '## 可以写入论文的解释','', '在归档测试轨迹及预先固定的置换下，当前训练得到的确定性策略对追加预报块呈弱动作响应，概率分布总体变化也较小。这为“增加预报输入未自动转化为稳定收益”提供了策略层面的解释证据。不能说完全没有使用预报；不能将历史预报路线与原PPO的收益差直接归因于预报信息的有效利用。','', '基础观测仍含历史天气和作物/土壤状态；冻结状态不测量长期状态分布或价值函数变化，不能证明策略不使用任何天气信息。动作频率低也不证明少数关键动作没有产量作用。此处未做闭环产量或奖励比较，不能声称打乱预报不影响收获。','', '## 对论文和后续工作的意义','', '将天气结果组织为“收益与投入权衡—种子重复性—操作特征—天气扩展及通道响应”。这套诊断可以作为第四条的机制补充。当前不优先再训练增强+预报：仅增加输入或训练数据的研究路线并未展示稳定优势，先以完整现有证据组织论文。若需要闭环诊断，应优先覆盖发生动作改变的模型—年份并包含预先固定的无变化对照，明确其为机制样本而非总体效果估计，不按产量择优。','', '天气增强已实际进入训练与是否改善泛化仍是两个问题，本诊断只针对预报通道，不替代增强机制检验。','', '## 质量与资源记录','', '400个年度原动作全部逐状态复现，smoke重复推理完全一致；所有概率有限、和为1、非法动作概率近0。基础观测和mask保持不变，置换索引逐年保存，逐状态概率距离和动作见state_response.csv；原始模型及轨迹440个文件保护校验通过。宿主机采样线程发生一次Windows原子替换占用错误，替换监测器后继续同一推理进程；没有重启模型任务、放宽门槛或新增模拟。该问题记录于telemetry_repair.json。', '', f"完成耗时约{r['elapsed_seconds']:.1f}秒，模型释放后采样RSS最大约{r['post_model_sample_peak_rss_MiB']:.1f} MiB；该采样不是全过程峰值。"]
(OUT/'report_zh.md').write_text('\n'.join(lines),encoding='utf-8')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'frozen_plan.json',OUT/'readout_summary.json',ROOT/'scripts/diagnose_forecast_response_127.py',ROOT/'scripts/summarize_forecast_response_127.py',ROOT/'prompts/2026-10-27_127_forecast_policy_response.md']}
(OUT/'execution_sources.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(stats,ensure_ascii=False,indent=2));print(t.to_string(index=False))
