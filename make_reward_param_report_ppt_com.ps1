param(
    [string]$MetricsPath = "output_hl\reward_sweep\all_policy_metrics.csv",
    [string]$RankingPath = "output_hl\reward_sweep\ppo_objective_ranking.csv",
    [string]$ImageRoot = "figures_hl\reward_sweep_plot_hl_png",
    [string]$OutputPath = "figures_hl\reward_parameter_selection_report.pptx"
)

$ErrorActionPreference = "Stop"

function Rgb {
    param([int]$R, [int]$G, [int]$B)
    return $R + ($G -shl 8) + ($B -shl 16)
}

function Fmt {
    param($Value, [int]$Digits = 3)
    if ($null -eq $Value -or $Value -eq "") { return "-" }
    return ([double]$Value).ToString("F$Digits")
}

function Add-Text {
    param(
        $Slide,
        [string]$Text,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H,
        [double]$Size = 18,
        [bool]$Bold = $false,
        [int]$Color = $(Rgb 35 39 47),
        [int]$Align = 1
    )
    $shape = $Slide.Shapes.AddTextbox(1, $X, $Y, $W, $H)
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Align
    return $shape
}

function Add-Title {
    param($Slide, [string]$Title, [string]$SubTitle = "")
    Add-Text $Slide $Title 42 24 876 36 23 $true (Rgb 19 28 43) 1 | Out-Null
    $line = $Slide.Shapes.AddShape(1, 42, 66, 876, 2)
    $line.Fill.ForeColor.RGB = Rgb 35 90 150
    $line.Line.Visible = 0
    if ($SubTitle -ne "") {
        Add-Text $Slide $SubTitle 42 76 876 30 11 $false (Rgb 91 101 118) 1 | Out-Null
    }
}

function Add-Table {
    param(
        $Slide,
        [object[]]$Rows,
        [string[]]$Headers,
        [double]$X,
        [double]$Y,
        [double]$W,
        [double]$H
    )
    $rowCount = $Rows.Count + 1
    $colCount = $Headers.Count
    $tableShape = $Slide.Shapes.AddTable($rowCount, $colCount, $X, $Y, $W, $H)
    $table = $tableShape.Table
    for ($c = 1; $c -le $colCount; $c++) {
        $cell = $table.Cell(1, $c).Shape
        $cell.Fill.ForeColor.RGB = Rgb 232 238 247
        $cell.TextFrame.TextRange.Text = $Headers[$c - 1]
        $cell.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
        $cell.TextFrame.TextRange.Font.Size = 9
        $cell.TextFrame.TextRange.Font.Bold = -1
        $cell.TextFrame.TextRange.Font.Color.RGB = Rgb 25 35 50
    }
    for ($r = 0; $r -lt $Rows.Count; $r++) {
        for ($c = 0; $c -lt $colCount; $c++) {
            $cell = $table.Cell($r + 2, $c + 1).Shape
            $cell.Fill.ForeColor.RGB = $(if (($r % 2) -eq 0) { Rgb 255 255 255 } else { Rgb 247 249 252 })
            $cell.TextFrame.TextRange.Text = [string]$Rows[$r][$c]
            $cell.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
            $cell.TextFrame.TextRange.Font.Size = 8.5
            $cell.TextFrame.TextRange.Font.Color.RGB = Rgb 35 39 47
        }
    }
    return $tableShape
}

function Add-NoteBox {
    param($Slide, [string]$Text, [double]$X, [double]$Y, [double]$W, [double]$H, [int]$Fill)
    $box = $Slide.Shapes.AddShape(1, $X, $Y, $W, $H)
    $box.Fill.ForeColor.RGB = $Fill
    $box.Line.ForeColor.RGB = Rgb 205 213 225
    Add-Text $Slide $Text ($X + 14) ($Y + 12) ($W - 28) ($H - 24) 14 $false (Rgb 35 39 47) 1 | Out-Null
}

$root = Resolve-Path "."
$metricsFull = Resolve-Path $MetricsPath
$rankingFull = Resolve-Path $RankingPath
$imageRootFull = Resolve-Path $ImageRoot
$outputFullPath = Join-Path $root $OutputPath
$outputDir = Split-Path $outputFullPath -Parent
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$metrics = Import-Csv $metricsFull
$ranking = Import-Csv $rankingFull

$original = $ranking | Where-Object { $_.run -eq "coef1_pen0.5" } | Select-Object -First 1
$candidate = $ranking | Where-Object { $_.run -eq "coef1.5_pen0.5" } | Select-Object -First 1
$overFert = $ranking | Where-Object { $_.run -eq "coef0.5_pen0.25" } | Select-Object -First 1
$lowN = $ranking | Where-Object { $_.run -eq "coef0.5_pen0.5" } | Select-Object -First 1
$dominated = $ranking | Where-Object { $_.run -eq "coef0.5_pen0.1" } | Select-Object -First 1
$plateau = $ranking | Where-Object {
    [math]::Abs([double]$_.final_trnu_mean - [double]$original.final_trnu_mean) -lt 0.000000001 -and
    [math]::Abs([double]$_.total_anfer_mean - [double]$original.total_anfer_mean) -lt 0.000001
}
$baselineRows = $metrics | Where-Object { $_.run -eq "coef1_pen0.5" } | Sort-Object agent

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$ppt.WindowState = 2

try {
    $presentation = $ppt.Presentations.Add([Microsoft.Office.Core.MsoTriState]::msoTrue)
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    $slide = $presentation.Slides.Add(1, 12)
    Add-Text $slide "施肥 reward 参数选择汇报" 60 110 840 52 32 $true (Rgb 19 28 43) 2 | Out-Null
    Add-Text $slide "目标：在 TRNU 尽可能高、施肥量尽可能少、产量尽可能高之间选择参数" 90 176 780 42 17 $false (Rgb 69 78 94) 2 | Out-Null
    Add-NoteBox $slide "重要纠正：原始 maize 参数是 coef=1.0, penality=0.5；coef=1.5, penality=0.5 不是原始值，只能作为候选参数讨论。" 132 260 696 92 (Rgb 255 248 226)
    Add-Text $slide "数据来源：reward_sweep 20 个组合的 1000 episodes 评估结果" 90 398 780 30 12 $false (Rgb 91 101 118) 2 | Out-Null

    $slide = $presentation.Slides.Add(2, 12)
    Add-Title $slide "评价标准" "导师要求的三个目标需要同时看，不能只看累计 reward"
    Add-NoteBox $slide "1. 氮回收率：final_trnu_mean 越高越好`r`n2. 施肥成本：total_anfer_mean 和 n_applications_mean 越低越好`r`n3. 产量表现：grnwt_mean / topwt_mean 越高越好`r`n4. 基线比较：PPO 至少要优于 expert 和 null，才说明站点在施肥策略上有优化空间" 74 138 812 190 (Rgb 238 245 255)
    Add-NoteBox $slide "本次判断不采用单一 reward 最大值作为唯一标准，因为 reward 数值会随 coef/penality 的尺度改变；最终以农学指标和 Pareto 支配关系为主。" 74 365 812 80 (Rgb 246 248 250)

    $slide = $presentation.Slides.Add(3, 12)
    Add-Title $slide "参数扫描设计" "4 个 coef × 5 个 penality，共 20 个组合"
    $designRows = @(
        @("coef", "0.5, 1.0, 1.5, 2.0"),
        @("penality", "0.1, 0.25, 0.5, 0.75, 1.0"),
        @("模式", "fertilization"),
        @("评估 episodes", "PPO / expert / null 各 1000"),
        @("输出", "daily trace CSV, evaluation_histories.pkl, plot_hl 风格 PDF/PNG")
    )
    Add-Table $slide $designRows @("项目", "设置") 120 130 720 210 | Out-Null
    Add-NoteBox $slide "原始参数 coef=1.0, penality=0.5 被保留在扫描网格内，因此可以直接检验它是否被其它组合支配。" 120 390 720 58 (Rgb 255 248 226)

    $slide = $presentation.Slides.Add(4, 12)
    Add-Title $slide "PPO 与基线对比" "以原始参数 coef=1.0, penality=0.5 的评估结果展示施肥优化空间"
    $policyRows = @()
    foreach ($row in $baselineRows) {
        $policyRows += ,@(
            $row.agent,
            (Fmt $row.final_trnu_mean 3),
            (Fmt $row.total_anfer_mean 1),
            (Fmt $row.n_applications_mean 1),
            (Fmt $row.grnwt_mean 1),
            (Fmt $row.topwt_mean 1)
        )
    }
    Add-Table $slide $policyRows @("策略", "final TRNU", "总施肥量", "施肥次数", "籽粒产量", "生物量") 48 128 864 150 | Out-Null
    Add-NoteBox $slide "结论：PPO 在该站点的施肥任务上明显优于 expert 和 null：TRNU 更高，产量更高。虽然总施肥量高于 expert，但换来了显著产量与 TRNU 提升，说明该站点确实有施肥策略优化空间。" 66 328 828 88 (Rgb 238 245 255)

    $slide = $presentation.Slides.Add(5, 12)
    Add-Title $slide "关键组合对比" "从 20 个组合中抽取代表性结果"
    $comboRows = @(
        @("原始参数", "coef1_pen0.5", (Fmt $original.final_trnu_mean 3), (Fmt $original.total_anfer_mean 1), (Fmt $original.n_applications_mean 1), (Fmt $original.grnwt_mean 1), "Pareto 候选"),
        @("候选参数", "coef1.5_pen0.5", (Fmt $candidate.final_trnu_mean 3), (Fmt $candidate.total_anfer_mean 1), (Fmt $candidate.n_applications_mean 1), (Fmt $candidate.grnwt_mean 1), "与原始结果持平"),
        @("过量施肥", "coef0.5_pen0.25", (Fmt $overFert.final_trnu_mean 3), (Fmt $overFert.total_anfer_mean 1), (Fmt $overFert.n_applications_mean 1), (Fmt $overFert.grnwt_mean 1), "施肥量异常高，剔除"),
        @("低氮牺牲", "coef0.5_pen0.5", (Fmt $lowN.final_trnu_mean 3), (Fmt $lowN.total_anfer_mean 1), (Fmt $lowN.n_applications_mean 1), (Fmt $lowN.grnwt_mean 1), "TRNU/产量下降"),
        @("被支配", "coef0.5_pen0.1", (Fmt $dominated.final_trnu_mean 3), (Fmt $dominated.total_anfer_mean 1), (Fmt $dominated.n_applications_mean 1), (Fmt $dominated.grnwt_mean 1), "不如稳定平台")
    )
    Add-Table $slide $comboRows @("类型", "组合", "TRNU", "总施肥量", "次数", "籽粒产量", "判断") 30 122 900 202 | Out-Null
    Add-NoteBox $slide "观察：有 $($plateau.Count) 个组合落在同一稳定平台，TRNU、施肥量、施肥次数和产量完全一致或数值等同；因此当前数据不能证明 1.5/0.5 比原始 1.0/0.5 更优。" 60 374 840 70 (Rgb 255 248 226)

    $slide = $presentation.Slides.Add(6, 12)
    Add-Title $slide "参数结论" "现阶段建议以保守、可复现的结论向导师汇报"
    Add-NoteBox $slide "首选结论：暂定保留原始参数 coef=1.0, penality=0.5。理由是它位于当前 20 组结果中的 Pareto 稳定平台，未被其它组合支配，并且不会把所谓优化结果误写成原始参数。" 64 124 832 88 (Rgb 238 245 255)
    Add-NoteBox $slide "备选说明：coef=1.5, penality=0.5 可作为敏感性分析候选，因为它与原始参数获得相同策略表现；但它不是原始参数，也不能在当前证据下被说成显著更优。" 64 244 832 88 (Rgb 246 248 250)
    Add-NoteBox $slide "应剔除：coef=0.5, penality=0.25。虽然 TRNU 和产量最高，但总施肥量达到约 6702，明显不满足施肥尽可能少的目标。" 64 364 832 72 (Rgb 255 238 232)

    $slide = $presentation.Slides.Add(7, 12)
    Add-Title $slide "数据质量核查" "这是正式定稿前必须补上的安全检查"
    Add-NoteBox $slide "当前容器内安装包 rewards.py 核查结果：maize 仍是硬编码 coef=1.0, penality=0.5，且没有读取 GYM_DSSAT_REWARD_COEF / GYM_DSSAT_REWARD_PENALITY。" 62 122 836 82 (Rgb 255 248 226)
    Add-NoteBox $slide "含义：如果训练当时也是这个 reward 文件状态，那么本轮输出只能说明不同模型文件的策略表现，不能严格证明 coef/penality 的最优性。" 62 234 836 82 (Rgb 255 238 232)
    Add-NoteBox $slide "处理方案：已新增 tools/patch_dssat_rewards.py。正式重跑前先用该脚本带备份地 patch 安装包 reward，再用 optimize_reward_hl.py 重跑或补跑，之后再把参数结论定稿。" 62 346 836 82 (Rgb 238 245 255)

    $slide = $presentation.Slides.Add(8, 12)
    Add-Title $slide "原始参数图例" "plot_hl 原样式：coef=1.0, penality=0.5"
    $apps = Join-Path $imageRootFull "coef1_pen0.5\fertilizationApplications.png"
    $rewards = Join-Path $imageRootFull "coef1_pen0.5\fertilizationRewards.png"
    if ((Test-Path $apps) -and (Test-Path $rewards)) {
        $slide.Shapes.AddPicture($apps, 0, -1, 42, 112, 420, 360) | Out-Null
        $slide.Shapes.AddPicture($rewards, 0, -1, 500, 112, 420, 360) | Out-Null
    } else {
        Add-NoteBox $slide "未找到 coef1_pen0.5 的 PNG 图，跳过图像嵌入。" 120 190 720 90 (Rgb 255 238 232)
    }

    $slide = $presentation.Slides.Add(9, 12)
    Add-Title $slide "后续 all 模式建议" "施肥与灌溉联合优化，以及未来接入天气预报"
    Add-NoteBox $slide "1. 先不要直接改原始 train_hl/evaluate_hl/plot_hl；新建 all 版本脚本并保留 fertilization 结果作为对照。`r`n2. all_reward 必须返回标量 reward，不能返回 [fertilization_reward, irrigation_reward] 列表。`r`n3. 先用历史天气作为 perfect forecast，把观测扩展和数据流跑通；再替换为真实天气预报接口。`r`n4. 如果灌溉本身优化空间小，应把论文表述放在联合框架可运行、施肥贡献主要收益、灌溉受站点和年份限制。" 62 126 836 210 (Rgb 238 245 255)
    Add-NoteBox $slide "下一步代码任务：备份后新建 train_hl_all.py / evaluate_hl_all.py / plot_hl_all.py，并将 action、baseline、reward logging 和 CSV 输出扩展到 anfer + amir。" 62 382 836 70 (Rgb 246 248 250)

    if (Test-Path $outputFullPath) {
        Remove-Item -LiteralPath $outputFullPath -Force
    }
    $presentation.SaveAs($outputFullPath, 24)
    $presentation.Close()
}
finally {
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}

Write-Host "Saved PPTX: $outputFullPath"
