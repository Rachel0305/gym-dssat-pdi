param(
    [string]$ImageRoot = "figures_hl\reward_sweep_plot_hl_png",
    [string]$OutputPath = "figures_hl\reward_sweep_appendix.pptx"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path "."
$imageRootPath = Resolve-Path $ImageRoot
$outputFullPath = Join-Path $root $OutputPath
$outputDir = Split-Path $outputFullPath -Parent
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

if (Test-Path $outputFullPath) {
    $backupPath = $outputFullPath -replace "\.pptx$", ".invalid_backup.pptx"
    Move-Item -LiteralPath $outputFullPath -Destination $backupPath -Force
}

function Get-RunSortKey {
    param([string]$Name)
    if ($Name -match '^coef([0-9.]+)_pen([0-9.]+)$') {
        return [pscustomobject]@{
            Coef = [double]$Matches[1]
            Pen  = [double]$Matches[2]
            Name = $Name
        }
    }
    return [pscustomobject]@{ Coef = [double]::PositiveInfinity; Pen = [double]::PositiveInfinity; Name = $Name }
}

$runDirs = Get-ChildItem -LiteralPath $imageRootPath -Directory | ForEach-Object {
    $key = Get-RunSortKey $_.Name
    [pscustomobject]@{ Directory = $_; Coef = $key.Coef; Pen = $key.Pen; Name = $key.Name }
} | Sort-Object Coef, Pen, Name

if ($runDirs.Count -eq 0) {
    throw "No image directories found under $imageRootPath"
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$ppt.WindowState = 2  # ppWindowMinimized

try {
    $presentation = $ppt.Presentations.Add([Microsoft.Office.Core.MsoTriState]::msoTrue)
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    $slideIndex = 0
    foreach ($run in $runDirs) {
        $apps = Join-Path $run.Directory.FullName "fertilizationApplications.png"
        $rewards = Join-Path $run.Directory.FullName "fertilizationRewards.png"
        if (!(Test-Path $apps) -or !(Test-Path $rewards)) {
            throw "Missing PNG pair for $($run.Name)"
        }

        $slideIndex += 1
        $slide = $presentation.Slides.Add($slideIndex, 12)  # ppLayoutBlank

        $title = $slide.Shapes.AddTextbox(1, 30, 12, 900, 28)
        $title.TextFrame.TextRange.Text = "$($run.Name)  (coef=$($run.Coef), penality=$($run.Pen))"
        $title.TextFrame.TextRange.Font.Name = "Arial"
        $title.TextFrame.TextRange.Font.Size = 18
        $title.TextFrame.TextRange.Font.Bold = -1
        $title.TextFrame.TextRange.ParagraphFormat.Alignment = 2

        $leftLabel = $slide.Shapes.AddTextbox(1, 45, 48, 410, 18)
        $leftLabel.TextFrame.TextRange.Text = "Nitrogen fertilizer applications"
        $leftLabel.TextFrame.TextRange.Font.Name = "Arial"
        $leftLabel.TextFrame.TextRange.Font.Size = 10
        $leftLabel.TextFrame.TextRange.ParagraphFormat.Alignment = 2

        $rightLabel = $slide.Shapes.AddTextbox(1, 505, 48, 410, 18)
        $rightLabel.TextFrame.TextRange.Text = "Policy returns"
        $rightLabel.TextFrame.TextRange.Font.Name = "Arial"
        $rightLabel.TextFrame.TextRange.Font.Size = 10
        $rightLabel.TextFrame.TextRange.ParagraphFormat.Alignment = 2

        $slide.Shapes.AddPicture($apps, 0, -1, 28, 72, 430, 430) | Out-Null
        $slide.Shapes.AddPicture($rewards, 0, -1, 502, 72, 430, 430) | Out-Null
    }

    $presentation.SaveAs($outputFullPath, 24) # ppSaveAsOpenXMLPresentation
    $presentation.Close()
}
finally {
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}

Write-Host "Saved PPTX: $outputFullPath"
