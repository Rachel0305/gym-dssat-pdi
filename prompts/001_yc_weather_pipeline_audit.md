# YC 站点天气增强训练前环境审计

## 任务背景

当前项目基于 `gym-dssat` + PPO 进行玉米水氮联合管理优化。

本阶段研究路线已经调整：

- 暂不改进 PPO 算法结构；
- 优先研究“合理天气数据增强是否能够提升 PPO 训练表现”；
- 当前只处理 `YC` 站点；
- `LC`、`SY` 已冻结且已有成功结果，**禁止修改其数据、配置、脚本和结果**；
- `HL`、`FQ` 后续继续调整，但本任务暂不处理；
- 最终目标仍是五站点中 PPO 在产量、水分利用效率、氮素利用效率、投入量、管理措施合理性等方面取得综合较优表现；
- 当前不要求每一个 PPO 随机种子都成功，但后续必须报告全部随机种子的结果、成功比例与评价标准，不能只挑最好的一颗种子。

本任务不是立即大规模训练，而是先把 `YC` 的天气生成与训练入口彻底查清楚，为下一步构造增强天气数据集做准备。

---

## 核心目标

请对当前项目进行一次**只读优先、非破坏性**的环境审计，回答：

1. `YC` 当前 PPO 训练到底使用什么天气数据？
2. 是否已经启用了 `random_weather` / WGEN / 其他随机天气机制？
3. 每个 episode 的天气是否真的发生变化？
4. 当前天气随机种子与 PPO 随机种子是否彼此独立？
5. `YC` 当前关联的 DSSAT 试验文件、天气文件、气候参数文件、土壤文件分别是什么？
6. 当前 `gym-dssat` 版本是否支持给 `YC` 接入自定义 `.CLI` / WGEN 参数？
7. 如果现有环境已经在随机天气下训练，当前随机天气来自哪里、基于什么气候参数？
8. 下一步最小改动方案应如何实现“YC 历史天气 vs YC 增强天气”的公平对照？

---

## 关键执行要求

1. **只处理 `YC`。**
   - 不修改 `LC`、`SY` 的任何文件。
   - 不修改 `HL`、`FQ` 的训练逻辑。
   - 如果发现公共代码改动会影响其他站点，先停止修改，只记录影响范围与建议方案。

2. **禁止改 PPO 算法。**
   - 不更换 PPO。
   - 不引入新的策略网络结构。
   - 不修改 PPO loss、clip、GAE 等算法核心。
   - 本任务只围绕天气输入、环境初始化、随机种子、数据记录与可复现性。

3. **先审计，再决定是否改代码。**
   - 第一阶段只读检查项目结构、配置和调用链。
   - 只有当某个检查需要极小、可逆的诊断代码时才允许新增诊断脚本。
   - 不直接启动大规模 PPO 训练。

4. **查清天气调用链。**
   对 `YC` 从训练入口一路追踪到 DSSAT：
   - PPO training script
   - environment creation
   - `env_args`
   - `random_weather`
   - `seed`
   - `set_seed()`
   - `fileX_template_path`
   - `auxiliary_file_paths`
   - `.WTH`
   - `.CLI`
   - `WSTA`
   - DSSAT experiment / soil / weather references

5. **验证“随机天气”是否真的随机。**
   不要只看配置值。
   至少做一个轻量诊断：
   - 固定 PPO seed；
   - 连续 reset 若干次环境；
   - 记录每个 episode 实际使用的天气标识、天气文件或关键逐日天气摘要；
   - 判断 episode 间天气是否不同；
   - 再固定 weather seed 验证是否可以复现同一序列。

6. **区分两类随机种子。**
   必须明确：
   - `ppo_seed`
   - `weather_seed`

   如果当前代码只有一个统一 `seed`，记录其影响范围，不要直接重构公共逻辑。提出最小侵入的分离方案。

7. **检查 YC 的气候参数来源。**
   如果启用了 WGEN / `random_weather=True`：
   - 找到真正被读取的 `.CLI` 或等价气候参数；
   - 检查是否属于 `YC`；
   - 如果来自默认 Florida / UF 环境或其他站点，明确标记为高优先级问题；
   - 不允许把默认气候参数误当作 YC 天气增强。

8. **不要擅自生成 `.CLI`。**
   如果 YC 缺少 `.CLI`：
   - 先寻找项目中已有的 YC 历史逐日天气；
   - 记录可用于拟合 WGEN 的字段、年份范围、缺失值情况；
   - 查明当前 DSSAT / WeatherMan / gym-dssat 版本支持的生成方式；
   - 不根据猜测手写气候统计参数。

9. **设计下一阶段公平对照，但本任务不大规模执行。**
   最终给出一个建议方案：

   `baseline_weather`：当前历史天气训练

   `augmented_weather`：基于 YC 训练期历史气候生成的合理随机天气训练

   两组保持一致：
   - PPO algorithm
   - reward function
   - action space
   - observation space
   - training steps
   - hyperparameters
   - evaluation years
   - evaluation seeds

10. **测试集必须独立。**
    在方案中明确：
    - 用完整生长季划分 train / validation / test；
    - 生成天气参数只能由 train 部分估计；
    - validation / test 真实天气不得用于天气生成器参数拟合。

11. **成功标准暂不凭单个最好 seed 定义。**
    本阶段只设计记录结构。后续至少要报告：
    - 每颗随机种子的产量；
    - irrigation；
    - fertilizer；
    - WUE；
    - NUE；
    - constraint violations；
    - reward；
    - 管理时序摘要；
    - 成功 seed 数 / 总 seed 数。

12. **保护现有成果。**
    - 不覆盖现有结果目录；
    - 新诊断输出放入独立目录，例如 `results/yc_weather_audit/`；
    - 修改前备份关键配置；
    - 所有新增脚本、配置与文档应可追踪。

---

## 建议搜索关键词

请在项目内系统搜索：

```text
random_weather
WGEN
WeatherMan
.CLI
.WTH
WSTA
fileX_template_path
auxiliary_file_paths
env_args
set_seed
seed
reset
YC
```

如果项目安装的 `gym-dssat` 位于 site-packages、Docker image 或子模块，也要追踪实际安装版本的源码，不要只看仓库中的旧副本。

---

## 需要输出的结果

### 1. 审计报告

创建：

```text
docs/yc_weather_pipeline_audit.md
```

必须包含：

- 当前训练入口；
- YC 环境配置；
- 实际天气来源；
- `random_weather` 当前值；
- WGEN 是否启用；
- `.CLI` / `.WTH` / experiment / soil 文件路径；
- PPO seed 与 weather seed 的关系；
- episode 间天气是否变化；
- 当前天气是否可复现；
- 是否存在默认 Florida 气候参数污染；
- 当前发现的问题；
- 下一阶段最小改动建议；
- 明确哪些结论已通过运行验证，哪些只是代码静态判断。

### 2. 机器可读配置快照

创建：

```text
results/yc_weather_audit/yc_weather_config_snapshot.json
```

至少记录：

```json
{
  "site": "YC",
  "gym_dssat_version": "",
  "training_entry": "",
  "random_weather": null,
  "ppo_seed": null,
  "weather_seed": null,
  "experiment_file": "",
  "weather_file": "",
  "climate_file": "",
  "soil_file": "",
  "wsta": "",
  "episode_weather_changes": null,
  "weather_reproducible": null
}
```

未知字段保持 `null` 或空字符串，不得猜测。

### 3. 轻量诊断脚本

如确有必要，可创建：

```text
scripts/diagnose_yc_weather.py
```

要求：

- 不训练 PPO；
- 只做环境 reset / 少量 step；
- 输出天气是否变化与 seed 复现结果；
- 不影响 LC/SY/HL/FQ。

### 4. 阶段记录

同步更新/创建：

```text
docs/yc_weather_audit_plan.md
docs/yc_weather_audit_plan.pptx
```

PPT 用于记录：研究路线调整、为什么先做 YC、审计问题、天气数据增强的下一步实验结构。

---

## Git / GitHub 要求

1. 在开始修改前记录：

```bash
git status
git branch --show-current
git remote -v
```

2. 不提交已有的无关改动。

3. 关键新增文件应加入版本控制：

```text
prompts/001_yc_weather_pipeline_audit.md
docs/yc_weather_pipeline_audit.md
docs/yc_weather_audit_plan.md
docs/yc_weather_audit_plan.pptx
scripts/diagnose_yc_weather.py          # 若创建
results/yc_weather_audit/yc_weather_config_snapshot.json
```

4. 如果仓库已配置可用 GitHub remote，并且当前权限允许 push，则将本任务关键文件备份到 GitHub。

5. 如果没有 GitHub remote、没有认证或 push 失败：
   - 不要伪造“已备份成功”；
   - 在审计报告中记录具体阻塞原因和终端错误信息；
   - 本地 commit 仍可完成（前提是不会混入无关改动）。

建议 commit message：

```text
chore: audit YC weather pipeline before PPO augmentation
```

---

## 本任务禁止事项

- 不修改 LC、SY 冻结成果；
- 不启动五站点大规模训练；
- 不修改 PPO 算法；
- 不为了得到“天气增强有效”的结论而调整测试集；
- 不只展示最优随机种子；
- 不覆盖旧结果；
- 不把代码中“看起来启用”当作运行时已验证；
- 不根据猜测填写 DSSAT / WGEN 参数。

---

## 最终回答格式

执行完成后，请按以下结构汇报：

1. **YC 当前天气机制一句话结论**
2. **已验证事实**
3. **发现的问题**
4. **是否已经在使用 WGEN / random weather**
5. **YC 当前使用的天气与气候参数来源**
6. **PPO seed 与 weather seed 的关系**
7. **下一阶段天气数据增强的最小实现方案**
8. **本次新增/修改文件列表**
9. **Git commit / GitHub backup 状态**
10. **明确说明 LC、SY 是否保持完全未动**
