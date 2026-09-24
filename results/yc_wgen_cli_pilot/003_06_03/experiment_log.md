# 003_06_03 实验记录

## 执行边界

- 日期：2026-09-24（Asia/Shanghai）
- 范围：YC train-only CLI 生成路径确认；冻结输入仅限 2004–2013。
- 当前 DSSAT active runtime 版本未知；本轮未探测项目外路径、容器、PATH 或安装目录。
- 没有生成 CSV 转换文件、`.CLI`、随机天气；没有启动 WeatherMan、WGEN、DSSAT 或 PPO。

## 证据与结果

- 冻结候选的 SHA256 与任务记录一致：`4B8FFE9E881D0A0743921B78B9C0E0EBFB1D2D645C5AA9737948B2B088ED7B34`。只读核对 3653 行数据、表头和起止日期，不重新 QC 或重建候选。
- 仅读取 2004–2013 `.WTH` 头部的站点 ID、位置及 WeatherMan 站点字段；十年均为 `CNYC`, `36.830`, `116.570`, `22`，`TAV/AMP` 均为 `14.0/28.9` 但参考期未知，`REFHT/WNDHT=-99.0`，未见 Angstrom A/B 或生长季起止字段。未读取 validation 年逐日内容，也不把未知期 TAV/AMP 作为 train-only 统计。
- 检查仓库的 `CNLC.CLI` / `CNSY.CLI` 结构，没有复制其他站点参数。旧 YC `.CLI` 的源文件路径当前不存在；仅保留前序报告中记录的 SHA、2008–2014 统计窗口和排除理由作为历史线索。
- 阅读 `references/dssat_pdi.py`、`src/ppo_safe_rendering.py` 和前序 YC 审计，确认 `auxiliary_file_paths` 的 basename 复制机制、临时运行目录以及当前 `random_weather=false` 设置。
- 查阅 DSSAT 官方 Tools 页、User's Guide Vol. 1/3 与公开 `SECLI.for`；WeatherMan 通用 GUI 工作流和老版 CLI 文件字段有依据，但官方手册版本旧，不能代替当前 4.8.x Help/About 或 active runtime smoke。

## 阻塞与决策

- `can_generate_cnyc_cli_now=BLOCKED_BY_WEATHERMAN_ACCESS`。
- 附加阻塞：`BLOCKED_BY_UNKNOWN_IMPORT_FORMAT`, `BLOCKED_BY_UNKNOWN_CLI_FORMAT`（指 active 4.8.x 严格字段/导入合同尚未核验，不是说完全没有历史结构信息）。
- YC 核心站点 ID/位置已确认；设备是否安装 WeatherMan、其版本、GUI/batch 能力及 Windows 输出到 Linux runtime 互操作未确认。
- CSV 是否可用目标 WeatherMan 版本直接导入未知；本轮不创建 conversion file。手写气候统计或 Python 拟合未启动。
- 下一步：把 WeatherMan About/version 和 Import/Export format/help 证据放入项目；再决定是否需要只读转换并单独审阅哈希与映射。

## 失败尝试与修正

- 一次文件读取命令最初使用了错误的工作目录，导致相对路径查找失败；改回项目根目录后读取成功。未写入或改变任何文件。
- 对前序报告记录的旧 `CNYC.CLI` 路径进行只读存在性检查，当前 checkout 中不存在；没有将缺失误判为系统未安装 WeatherMan。
- PPT 构建首次启动时，从 `.build` 工作目录解析相对 `TMP_DIR`，导致环境变量无法设为绝对路径；Node 在导出前退出，未产生候选或最终 PPTX。修正为绝对路径后重跑。
- 第二次 PPT 构建已生成 7 页候选及预览，但 finalizer 拒绝未与 `--require-native-table-slide` 参数对应的必需原生表格声明；补齐表格所有页的显式布局校验参数后重跑。
- 完成视觉检查后尝试清理本轮私有 `.build` 暂存目录，删除命令被安全策略拒绝；保留该目录，不改用其他删除方式，并在 Git 暂存时明确排除 `.build/` 与 `node_modules` junction。

## 输出与 Git

- 中文 Markdown：`docs/yc_cli_generation_path.md`
- 中文 PPT：`docs/yc_cli_generation_path.pptx`
- PPTX SHA256：`B12C039012F3CB2C8C28F9C31DC4251DBC04CA8BD575F02CC784DAE27920AAAD`；7 页均已逐页检查渲染，含 3 张原生表格；finalizer 版面检查 0 findings / 0 warnings，Artifact Tool 可重新导入。渲染字体策略检查通过，但 bundled renderer 未验证 native font rendering；未声称在 PowerPoint 中打开验证。
- 结构、站点、WeatherMan 路径、Gym 接口、PPTX validation receipt、readiness JSON 与本记录：本目录内对应证据文件。
- 只提交本轮交付；不 push。GitHub backup pending explicit user approval。
