$ErrorActionPreference = "Stop"

$repo = Resolve-Path "."
$outDir = Join-Path $repo "figures_hl"
$pptPath = Join-Path $outDir "experiment_record_reward_all_irrigation.pptx"
$rankingPath = Join-Path $repo "output_hl\all_reward_sweep\combined_ppo_ranking.csv"
$diagPath = Join-Path $repo "output_hl\diagnostics\irrigation_policy_summary.csv"

function Add-TextBox {
    param($Slide, [double]$Left, [double]$Top, [double]$Width, [double]$Height, [string]$Text, [int]$Size = 16, [bool]$Bold = $false)
    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Bold = if ($Bold) { -1 } else { 0 }
    $shape.TextFrame.WordWrap = -1
    return $shape
}

function Add-Title {
    param($Slide, [string]$Title, [string]$Subtitle = "")
    Add-TextBox $Slide 38 22 880 44 $Title 25 $true | Out-Null
    if ($Subtitle) {
        Add-TextBox $Slide 40 68 860 26 $Subtitle 11 $false | Out-Null
    }
}

function Add-Bullets {
    param($Slide, [string[]]$Items, [double]$Left = 55, [double]$Top = 112, [double]$Width = 850, [double]$Height = 390, [int]$Size = 15)
    $text = ($Items | ForEach-Object { "• $_" }) -join "`r`n"
    Add-TextBox $Slide $Left $Top $Width $Height $text $Size $false | Out-Null
}

function Add-SimpleTable {
    param($Slide, [object[]]$Rows, [double]$Left, [double]$Top, [double]$Width, [double]$Height, [int]$FontSize = 10)
    if ($Rows.Count -eq 0) { return }
    $headers = $Rows[0].PSObject.Properties.Name
    $tableShape = $Slide.Shapes.AddTable($Rows.Count + 1, $headers.Count, $Left, $Top, $Width, $Height)
    $table = $tableShape.Table
    for ($c = 1; $c -le $headers.Count; $c++) {
        $cell = $table.Cell(1, $c).Shape.TextFrame.TextRange
        $cell.Text = [string]$headers[$c - 1]
        $cell.Font.Name = "Microsoft YaHei"
        $cell.Font.Size = $FontSize
        $cell.Font.Bold = -1
    }
    for ($r = 1; $r -le $Rows.Count; $r++) {
        for ($c = 1; $c -le $headers.Count; $c++) {
            $value = $Rows[$r - 1].PSObject.Properties[$headers[$c - 1]].Value
            $cell = $table.Cell($r + 1, $c).Shape.TextFrame.TextRange
            $cell.Text = [string]$value
            $cell.Font.Name = "Microsoft YaHei"
            $cell.Font.Size = $FontSize
        }
    }
}

function Round-Value {
    param($Value, [int]$Digits = 2)
    if ($null -eq $Value -or $Value -eq "") { return "" }
    try { return [math]::Round([double]$Value, $Digits) } catch { return $Value }
}

$rankingRows = @()
if (Test-Path $rankingPath) {
    $rankingRows = Import-Csv $rankingPath | Select-Object -First 6 | ForEach-Object {
        [pscustomobject]@{
            "组合" = $_.run
            "N惩罚" = $_.penality
            "水成本" = $_.amir_cost
            "TRNU" = Round-Value $_.mean_trnu 3
            "施肥" = Round-Value $_.total_anfer 1
            "灌水" = Round-Value $_.total_amir 1
            "产量" = Round-Value $_.max_grnwt 1
            "Score" = Round-Value $_.score 3
        }
    }
}

$diagRows = @()
if (Test-Path $diagPath) {
    $diagRows = Import-Csv $diagPath | Where-Object { $_.agent -in @("null", "expert", "ppo") } | Select-Object -First 12 | ForEach-Object {
        [pscustomobject]@{
            "站点" = $_.site
            "策略" = $_.agent
            "回报" = Round-Value $_.reward_sum 1
            "产量" = Round-Value $_.max_grnwt 1
            "swfac" = Round-Value $_.mean_swfac 3
            "nstres" = Round-Value $_.mean_nstres 3
            "灌水估计" = Round-Value $_.water_estimate 1
            "数据源" = $_.water_source
        }
    }
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$pres = $ppt.Presentations.Add()
$blank = 12

try {
    $slide = $pres.Slides.Add(1, $blank)
    Add-Title $slide "水氮管理优化试验记录" "DSSAT-PDI + PPO；施肥 reward 参数扫描、all 模式、水分诊断路线"
    Add-Bullets $slide @(
        "目标：选择高 TRNU、高产量、低施肥、低灌水的 reward 参数，并为后续天气预报驱动的水氮联合优化打基础。",
        "当前主线：先在海伦站 2007 年玉米数据上打通流程，再扩展到沈阳、禹城、栾城、封丘等站点。",
        "记录原则：保留原始脚本，新增脚本承载新功能；重要中间结果、图、PPT 和诊断表同步到 GitHub。"
    ) 55 125 820 260 17

    $slide = $pres.Slides.Add(2, $blank)
    Add-Title $slide "已完成尝试总览"
    Add-Bullets $slide @(
        "施肥 reward：原始 maize 参数为 coef=1.0、penality=0.5；后续做 coef/penality 扫描，并补充 TRNU 打印、CSV 保存和 plot_hl.py 统计。",
        "1000 episode 评估：生成 PPO、Null、Expert 三种模式的 evaluation_histories.pkl，并按 plot_hl.py 原样式重绘施肥量与 reward 图。",
        "OOM 处理：改成逐组合、逐进程运行，减少一次性加载历史；必要时分批评估和重启 Python 进程。",
        "all 模式：先备份关键代码，再新增 train_hl_all.py/evaluate_hl_all.py/plot_hl_all.py 相关流程，打通水氮联合动作。",
        "GitHub 备份：已把 reward sweep、all sweep、PPT、诊断脚本等多轮结果推送到 codex-reward-sweep-backup 分支。"
    ) 55 105 850 360 14

    $slide = $pres.Slides.Add(3, $blank)
    Add-Title $slide "施肥 reward 参数扫描结论"
    Add-Bullets $slide @(
        "评价标准：TRNU 尽可能高、施肥量尽可能少、产量尽可能高；不是只看 reward 数值。",
        "原始参数 coef=1.0、penality=0.5 作为基线，不再误记为其他组合。",
        "单独施肥模式能学到优于 Expert/Null 的适配策略，说明该站点在施肥管理上有优化空间。",
        "后续 all 模式中，氮肥惩罚需要明显提高，否则 PPO 倾向通过大量施肥换取产量和即时 reward。"
    ) 55 115 850 300 16

    $slide = $pres.Slides.Add(4, $blank)
    Add-Title $slide "all reward 扫描：当前最佳组合"
    Add-TextBox $slide 55 92 850 34 "综合排序来自 output_hl/all_reward_sweep/combined_ppo_ranking.csv；Score 综合考虑产量、TRNU、总施肥和总灌水。" 12 $false | Out-Null
    Add-SimpleTable $slide $rankingRows 35 135 890 320 8

    $slide = $pres.Slides.Add(5, $blank)
    Add-Title $slide "all 模式阶段性判断"
    Add-Bullets $slide @(
        "较优组合：penality=15、amir_cost=15，在 30 episode 评估中达到 mean_trnu≈1.741、施肥≈830、灌水≈196、产量≈7464。",
        "提高氮肥惩罚能显著压低总施肥量；过高惩罚或硬阈值惩罚可能导致训练陷入坏局部策略。",
        "灌水成本提高后可进一步压低总灌水，但仍需结合水分胁迫诊断判断是否是合理节水，而不是奖励函数压制过强。",
        "下一步：把灌溉优化从是否少灌升级为是否在缺水期才灌。"
    ) 55 112 850 300 16

    $slide = $pres.Slides.Add(6, $blank)
    Add-Title $slide "五站点灌溉异常：Null 为何常常最好"
    Add-Bullets $slide @(
        "现象：沈阳、禹城、栾城、封丘等站点中，Null 不灌水的 reward 和产量经常高于固定 Expert 灌溉。",
        "代码层面：旧脚本中海伦站 swfac/nstres 实际为 NaN，不应解释为 0；部分 action_amir 是归一化动作，不是真实灌水量。",
        "数据层面：若生长季降雨充足或土壤初始含水量高，固定灌溉没有增产边际收益，还会被 reward 水量惩罚扣分。",
        "策略层面：Expert 灌溉日期/水量固定，不随站点、年份、降雨过程调整，因此不一定是真正专家。"
    ) 55 110 850 310 16

    $slide = $pres.Slides.Add(7, $blank)
    Add-Title $slide "已有五站点旧结果诊断摘要"
    Add-TextBox $slide 55 92 850 34 "该表来自现有 output_*/output_*/irrigation 结果；海伦旧 CSV 缺少真实灌水量，因此水量标记为 missing。" 12 $false | Out-Null
    Add-SimpleTable $slide $diagRows 30 135 900 320 8

    $slide = $pres.Slides.Add(8, $blank)
    Add-Title $slide "导师目标下的后续路线"
    Add-Bullets $slide @(
        "湿润/平衡年份：PPO 应学会接近 Null 的少灌水策略，同时保持产量，这本身就是节水优化结果。",
        "缺水年份/站点：PPO 应优于 Null 的产量，并比固定 Expert 更少水或更高综合 reward。",
        "预报接入：第一阶段使用历史天气构造完美预报特征，如未来 3/7/14 天降雨、温度、辐射统计；第二阶段替换为真实天气预报。",
        "正式论文指标：TRNU、产量、总施肥、总灌水、swfac/turfac 胁迫天数、氮胁迫 nstres、runoff/cleach 等环境指标。"
    ) 55 110 850 320 16

    $slide = $pres.Slides.Add(9, $blank)
    Add-Title $slide "下一步：五站点水分胁迫诊断"
    Add-Bullets $slide @(
        "输入文件：my_data 中不带 (1) 的 UFGA8201-FQ/HL/LC/SY/YC.jinja2、对应 WTH、对应 SOL、MZCER048.CUL。",
        "输出表：PRCP、ETCP、PRCP-ETCP、swfac/turfac 胁迫天数、最小 swfac、平均 nstres、产量、Null/Expert 灌水量。",
        "判断标准：若某站点 PRCP<ETCP 且 DSSAT 胁迫天数明显，但 Null 仍最好，则优先检查初始土壤水、灌溉动作、reward 与输出记录。",
        "若五站点均无明显水分胁迫，则新增干旱情景或寻找干旱年份，作为灌溉策略能力验证场景。"
    ) 55 110 850 320 16

    $slide = $pres.Slides.Add(10, $blank)
    Add-Title $slide "文件索引"
    Add-Bullets $slide @(
        "参数排序：output_hl/all_reward_sweep/combined_ppo_ranking.csv",
        "灌溉异常说明：docs/irrigation_null_expert_analysis.md",
        "多站点新增脚本：train_hl_all_multisite.py、evaluate_hl_all_multisite.py、dssat_site_config.py",
        "旧结果诊断：analyze_irrigation_outputs.py、output_hl/diagnostics/irrigation_policy_summary.csv",
        "本 PPT 源脚本：tools/create_experiment_record_ppt.ps1"
    ) 55 112 850 310 15

    if (Test-Path $pptPath) { Remove-Item $pptPath -Force }
    $pres.SaveAs($pptPath)
    Write-Output "Saved: $pptPath"
}
finally {
    if ($pres) { $pres.Close() }
    if ($ppt) { $ppt.Quit() }
}
