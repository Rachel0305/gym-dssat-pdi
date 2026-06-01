$ErrorActionPreference = "Stop"

$repo = Resolve-Path "."
$outDir = Join-Path $repo "figures_hl"
$pptPath = Join-Path $outDir "rainfed_weak_irrigation_context_summary.pptx"
$summaryPath = Join-Path $repo "output_hl\all_water_stress_diagnostics\all_water_stress_summary_final.csv"

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
    if ($Subtitle) { Add-TextBox $Slide 40 68 860 26 $Subtitle 11 $false | Out-Null }
}

function Add-Bullets {
    param($Slide, [string[]]$Items, [double]$Top = 112, [int]$Size = 15)
    $text = ($Items | ForEach-Object { "• $_" }) -join "`r`n"
    Add-TextBox $Slide 55 $Top 850 370 $text $Size $false | Out-Null
}

function Add-SimpleTable {
    param($Slide, [object[]]$Rows, [double]$Left, [double]$Top, [double]$Width, [double]$Height, [int]$FontSize = 9)
    if ($Rows.Count -eq 0) { return }
    $headers = $Rows[0].PSObject.Properties.Name
    $shape = $Slide.Shapes.AddTable($Rows.Count + 1, $headers.Count, $Left, $Top, $Width, $Height)
    $table = $shape.Table
    for ($c = 1; $c -le $headers.Count; $c++) {
        $cell = $table.Cell(1, $c).Shape.TextFrame.TextRange
        $cell.Text = [string]$headers[$c - 1]
        $cell.Font.Name = "Microsoft YaHei"
        $cell.Font.Size = $FontSize
        $cell.Font.Bold = -1
    }
    for ($r = 1; $r -le $Rows.Count; $r++) {
        for ($c = 1; $c -le $headers.Count; $c++) {
            $cell = $table.Cell($r + 1, $c).Shape.TextFrame.TextRange
            $cell.Text = [string]$Rows[$r - 1].PSObject.Properties[$headers[$c - 1]].Value
            $cell.Font.Name = "Microsoft YaHei"
            $cell.Font.Size = $FontSize
        }
    }
}

function R2 { param($v) try { [math]::Round([double]$v, 2) } catch { "" } }

$rows = @()
if (Test-Path $summaryPath) {
    $rows = Import-Csv $summaryPath | Where-Object { $_.trace_source -eq "all_water_stress_diagnostics" } | ForEach-Object {
        [pscustomobject]@{
            "站点" = $_.site
            "策略" = $_.agent
            "P-E" = R2 $_.PRCP_minus_ETCP
            "swfac天数" = R2 $_.swfac_stress_days_gt_0.05
            "nstres天数" = R2 $_.nstres_days_gt_0.05
            "产量" = R2 $_.max_grnwt
            "施肥" = R2 $_.total_anfer
            "灌水" = R2 $_.total_amir
        }
    }
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$pres = $ppt.Presentations.Add()
$blank = 12

try {
    $slide = $pres.Slides.Add(1, $blank)
    Add-Title $slide "雨养或弱灌溉需求情景小结" "五站点 all 模式水分/氮素胁迫诊断，可作为论文讨论材料"
    Add-Bullets $slide @(
        "五站点实测年份中，多数站点 PRCP 与 ETCP 接近平衡或降雨充足；海伦虽年尺度亏缺，但 all 模式日尺度 swfac 未显示明显水分胁迫。",
        "在这些情景下，PPO 若学习到减少灌溉或接近不灌溉，并不一定是失败；它可能是在识别灌水边际收益很低的年份。",
        "因此当前五站点更适合作为雨养或弱灌溉需求情景，用于证明模型不会为了灌溉而灌溉，具备节水倾向。"
    ) 120 16

    $slide = $pres.Slides.Add(2, $blank)
    Add-Title $slide "诊断依据"
    Add-SimpleTable $slide $rows 30 105 900 330 8
    Add-TextBox $slide 50 455 850 48 "说明：当前 maize all 模式中 swfac/nstres 已按 1-原值处理，数值越大代表胁迫越强；turfac 暂未安全暴露。" 12 $false | Out-Null

    $slide = $pres.Slides.Add(3, $blank)
    Add-Title $slide "论文可写结论"
    Add-Bullets $slide @(
        "本研究首先在五个典型站点年份上检验模型对水分与氮素状态的响应。结果表明，当前年份下水分胁迫指标 swfac 基本接近 0，灌溉收益有限。",
        "相比之下，不施肥或低施肥情景下 nstres 明显升高，说明氮素管理是当前数据中更主要的限制因子。",
        "因此，PPO 在这些年份中倾向降低灌水量是合理的节水响应，而非模型未能学习灌溉；真正的灌溉能力应在干旱年份或构造干旱情景中进一步验证。"
    ) 115 15

    $slide = $pres.Slides.Add(4, $blank)
    Add-Title $slide "后续实验设计"
    Add-Bullets $slide @(
        "真实数据路线：继续搜集干旱年份 WTH，优先选择 PRCP 显著低于 ETCP 且关键生育期降雨不足的站点年份。",
        "调试路线：在原始 WTH 基础上构造降雨缩放情景，例如 80%、60%、40% 降雨，用于验证 all 模式是否在水分胁迫增强时主动灌溉。",
        "论文表述：历史天气可作为完美预报输入；后续可将未来 3/7/14 天降雨、温度、辐射统计接入 observation。"
    ) 115 15

    if (Test-Path $pptPath) { Remove-Item $pptPath -Force }
    $pres.SaveAs($pptPath)
    Write-Output "Saved: $pptPath"
}
finally {
    if ($pres) { $pres.Close() }
    if ($ppt) { $ppt.Quit() }
}
