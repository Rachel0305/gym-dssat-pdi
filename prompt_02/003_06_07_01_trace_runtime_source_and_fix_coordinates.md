# 003_06_07_01：导出当前 Runtime 源码快照并定位/修复 YC 坐标映射

## 一、任务背景

上一轮 `003_06_07` 已将坐标丢失定位到：

```text
rendered runtime FileX
LAT  = 36.830
LONG = 116.570
ELEV = 22
        ↓
PDI parser / renderer
        ↓
DSSAT48.INP / DSSAT48.INH
LAT  = -99
LONG = -999
ELEV = -99
        ↓
DSSAT runtime
coordinates -> 0
```

已知：

```text
source FileX placeholders: -99/-99/-99
isolated rendered FileX: 36.830 / 116.570 / 22
parsed internal coordinates: NOT_OBSERVED
DSSAT48.INP: placeholders remain
runtime coordinates: 0
```

因此问题已经不在 WGEN、`CNYC.CLI` 或天气数据，而在：

```text
Gym-DSSAT / PDI parser
or
DSSAT48.INP renderer / variable mapping
```

上一轮未继续，是因为相关源码位于当前 Docker/runtime 安装环境，不在项目 repo。

---

# 二、本轮授权边界

本任务明确授权：

> **只读访问当前实际运行容器 `nifty_taussig` 中安装的 Gym-DSSAT/PDI Python 源码及相关模板，并将诊断所需的最小源码快照复制到项目目录用于检查。**

允许读取：

```text
/opt/gym_dssat_pdi/
/opt/dssat_pdi/
```

以及当前 Python environment 中实际 import 到的：

```text
gym_dssat_pdi
gym_dssat
PDI-related modules
```

具体路径必须通过当前 runtime 自动定位，不能只根据上述路径猜测。

禁止：

```text
修改容器内已安装 package
修改 /opt/dssat_pdi
修改 DSSAT executable
pip install / uninstall
重建 container
```

所有修复必须优先发生在：

```text
repository-side code
isolated wrapper
isolated template
repo-side override
```

---

# 三、第一步：自动定位当前真正使用的安装源码

在 `nifty_taussig` 内使用当前实际 Python：

```text
/opt/gym_dssat_pdi/bin/python
```

通过：

```python
import inspect
import importlib
import importlib.metadata
```

以及必要的：

```text
pip show
python -c
find / grep
```

确认：

```text
distribution name
distribution version
module import path
package root
relevant source files
```

不要再混淆：

```text
gym-dssat-pdi package version
vs
DSSAT model version 4.8.0.024
```

生成：

```text
results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_inventory.json
```

至少记录：

```text
distribution
version
module
module_file
package_root
sha256
```

---

# 四、第二步：只读搜索坐标与 INP 渲染路径

在实际安装源码中搜索：

```text
DSSAT48.INP
DSSAT48.INH
*FIELDS
XCRD
YCRD
ELEV
CXCRDin
CYCRDin
CELEVin
LAT
LONG
MAKEFW
FileX
```

同时搜索：

```text
jinja
template
render
parse
parser
input
```

目标是定位：

```text
rendered FileX
→ parser
→ internal variables
→ DSSAT48.INP renderer
```

的实际调用链。

输出：

```text
results/yc_wgen_cli_pilot/003_06_07_01/coordinate_runtime_code_trace.md
```

必须具体到：

```text
module/file
class/function
relevant variable names
input
output
mapping logic
```

---

# 五、第三步：复制最小源码快照到 repo

为了遵守 repo 内诊断规则，只复制**与坐标传播直接相关的最小文件集合**。

保存到：

```text
results/yc_wgen_cli_pilot/003_06_07_01/runtime_source_snapshot/
```

例如可能包含：

```text
parser .py
renderer .py
template file
config/schema file
```

但实际文件必须由第二步证据决定。

每个文件记录：

```text
original runtime path
copied path
SHA256
reason needed
```

保存：

```text
runtime_source_snapshot_manifest.json
```

### 重要

这些第三方 runtime 源码快照：

```text
只用于本地诊断
不要 git add
不要提交 Git
不要复制整个 package
```

Git 只提交：

```text
我们自己的修复代码
测试
实验报告
机器可读诊断结果
```

---

# 六、第四步：精确找出坐标在哪一层丢失

必须建立实际变量链，例如：

```text
FileX XCRD/YCRD/ELEV
→ parser field A/B/C
→ instance variable ...
→ PDI config ...
→ INP renderer key ...
→ DSSAT48.INP
```

针对：

```text
36.830
116.570
22
```

逐步打印/记录实际值。

生成：

```text
coordinate_value_trace.json
```

至少包括：

```text
rendered_filex
parser_input
parser_output
environment/object_state
renderer_input
dssat48_inp
```

不能只通过静态源码猜测。

允许添加：

```text
repo-side debug instrumentation
temporary isolated probe
```

禁止修改 installed package。

---

# 七、根因分类

最终根因必须具体归入一类或多类：

```text
PARSER_FIELD_NOT_READ
PARSER_COLUMN_INDEX_ERROR
FIELD_NAME_MISMATCH
X_Y_MAPPING_ERROR
VALUE_DROPPED_AFTER_PARSE
RENDER_CONTEXT_MISSING
RENDER_KEY_MISMATCH
DEFAULT_OVERWRITE
TEMPLATE_PLACEHOLDER_BUG
PACKAGE_BEHAVIOR_BY_DESIGN
OTHER_SPECIFIC_ROOT_CAUSE
```

不得停在：

```text
PDI source unavailable
```

因为本轮已经授权只读导出当前 runtime source。

---

# 八、第五步：选择最小修复层

修复优先级：

```text
1. repo-side existing adapter / wrapper
2. repo-side runtime config propagation
3. repo-side template/context override
4. isolated repo-side subclass/patch adapter
```

最后才考虑：

```text
installed package modification
```

而本轮禁止直接修改 installed package。

如果必须修改第三方 package 才能修：

不要改 `/opt/...`。

应输出：

```text
exact upstream patch proposal
file/function
before behavior
after behavior
```

并评估能否通过 repo-side override 实现。

---

# 九、禁止 YC 坐标硬编码进通用第三方逻辑

禁止：

```python
lat = 36.830
lon = 116.570
elev = 22
```

直接写死进通用 parser/renderer。

正确目标是：

```text
parser/renderer correctly propagates whatever station coordinates are supplied
```

YC 值只能存在于：

```text
YC config
YC isolated input
test fixture
```

未来 HL/SY/LC/FQ 应可复用同一传播逻辑。

---

# 十、修复后第一门禁：不跑 DSSAT，先检查 INP

修复后先运行最小 probe：

```text
render YC isolated FileX
→ parse
→ render DSSAT48.INP
```

必须看到：

```text
LAT / Y coordinate = 36.830
LONG / X coordinate = 116.570
ELEV = 22
```

具体字段名按 DSSAT 实际 schema。

生成：

```text
results/yc_wgen_cli_pilot/003_06_07_01/dssat48_coordinate_gate.json
```

状态：

```text
PASS
FAIL
```

只有 PASS 才允许进入 runtime smoke。

---

# 十一、第二门禁：Runtime 单次历史 control

先只运行：

```text
historical-weather control
```

不先跑 WGEN seed。

必须检查：

```text
WARNING.OUT
INFO.OUT
Summary.OUT
DSSAT48.INP
DSSAT48.INH
```

确认：

```text
CYCRDin/CXCRDin/CELEVin 不再最终置零
runtime 坐标可验证
```

如果 historical control 坐标门禁仍失败：

停止。

不要继续 seed 101/104。

---

# 十二、第三门禁：重跑 seed 101 和 seed 104

只有 historical control 坐标 PASS 后才运行：

```text
weather_seed=101
weather_seed=104
```

保持：

```text
same corrected CNYC.CLI
same management
same soil
same cultivar
same DSSAT runtime 4.8.0.024
```

确认三组坐标一致正确。

---

# 十三、比较修复前后作物输出

对：

```text
historical control
seed 101
seed 104
```

比较 coordinate fix 前后：

```text
anthesis
maturity
season length
yield
biomass
irrigation
fertilizer
N uptake
water stress
N stress
```

生成：

```text
coordinate_fix_crop_output_comparison.csv
```

必须明确：

```text
absolute difference
relative difference where meaningful
```

不要预设修复后一定变化或一定不变化。

---

# 十四、坐标 warning 最终判定

如果修复成功：

```text
coordinate_warning_resolved = YES
```

并把上一轮：

```text
36 class-C coordinate events
```

重新核验。

如果 PHOTO、STONES、ADCOEF 仍存在：

继续按既有 baseline limitation 记录，不在本轮擅自修改。

---

# 十五、PPO 准入标准

必须全部满足：

```text
1. parser/renderer root cause明确
2. repo-side or isolated fix implemented
3. DSSAT48.INP coordinate gate PASS
4. historical control runtime coordinate PASS
5. seed 101 coordinate PASS
6. seed 104 coordinate PASS
7. three crop simulations end normally
8. crop outputs finite / interpretable
9. no new fatal runtime issue
```

成功：

```text
PPO_COORDINATE_GATE_PASS
YC_RANDOM_WEATHER_PPO_READY = YES
```

否则：

```text
BLOCKED_BEFORE_PPO
```

---

# 十六、测试

新增针对性 regression tests，至少覆盖：

```text
FileX coordinate parse
coordinate mapping
renderer context
DSSAT48.INP output
LAT/LONG/ELEV preservation
non-YC fixture to prove no hard-coding
```

如果修改 repo-side shared wrapper：

额外确保：

```text
historical random_weather=False path remains valid
random_weather=True path remains valid
```

运行相关 pytest。

---

# 十七、实验记录

只生成：

```text
docs/yc_dssat_coordinate_runtime_source_fix.md
```

中文 Markdown 实验报告。

另保留：

```text
experiment_log.md
JSON / CSV / TXT evidence
```

### 不制作

```text
PPT
PPTX
slide preview
font/layout checks
```

---

# 十八、结果目录

保存到：

```text
results/yc_wgen_cli_pilot/003_06_07_01/
```

建议：

```text
runtime_source_inventory.json
runtime_source_snapshot_manifest.json
runtime_source_snapshot/   # local diagnostic only, do not stage
coordinate_runtime_code_trace.md
coordinate_value_trace.json
dssat48_coordinate_gate.json
runtime/
├── historical_control/
├── seed_101/
└── seed_104/
coordinate_fix_crop_output_comparison.csv
warning_comparison.csv
experiment_log.md
```

---

# 十九、Git

完成后：

```text
git diff
```

只 stage：

```text
repo-side fix
tests
docs
machine-readable results
prompt
```

不要 stage：

```text
runtime_source_snapshot/
```

建议 commit message：

```text
fix: propagate DSSAT station coordinates through PDI
```

未经明确批准：

```text
git push = NO
```

---

# 二十、最终摘要

必须返回：

```text
=== YC PDI COORDINATE SOURCE TRACE + FIX SUMMARY ===

runtime_distribution:
runtime_distribution_version:
runtime_package_root:

source_snapshot_created:
source_snapshot_staged_to_git: NO

root_cause:
root_cause_category:

expected_lat:
expected_long:
expected_elev:

rendered_filex_coordinates:
parser_coordinates:
renderer_context_coordinates:
dssat48_inp_coordinates:
runtime_coordinates:

repo_side_fix:
installed_package_modified: NO

dssat48_coordinate_gate:
historical_control_coordinate_gate:
seed_101_coordinate_gate:
seed_104_coordinate_gate:

coordinate_warning_resolved:

historical_control_status:
seed_101_status:
seed_104_status:

crop_output_comparison_path:

tests_status:

ppo_coordinate_gate:
yc_random_weather_ppo_ready:

wgen_modified: NO
cli_modified: NO
weather_fitting_modified: NO
wet_day_definition_changed: NO
ppo_training_run: NO
ppt_created: NO

recommended_next_step:

report_md:
results_directory:

git_commit:
git_push: NO
github_backup_status:
```

成功时：

```text
ppo_coordinate_gate: PPO_COORDINATE_GATE_PASS
yc_random_weather_ppo_ready: YES
recommended_next_step: design YC random-weather PPO pilot
```

如果第三方 package 必须修改且 repo-side override 无法安全完成：

```text
ppo_coordinate_gate: BLOCKED_BEFORE_PPO
yc_random_weather_ppo_ready: NO
recommended_next_step: apply reviewed upstream-compatible patch in controlled environment
```

但必须已经给出精确到文件/函数/变量的 patch proposal，不能再以“源码不可访问”为理由停止。
