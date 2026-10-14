$ErrorActionPreference = 'Stop'

$root = (Get-Location).Path
$resultDir = Join-Path $root 'results\yc_chinaflux_weather_reconciliation'
if (-not (Test-Path -LiteralPath $resultDir)) {
    New-Item -ItemType Directory -Path $resultDir | Out-Null
}

$rainMatches = @(Get-ChildItem -LiteralPath (Join-Path $root 'my_data') -File -Filter 'HLLCYCFQ*.xls')
if ($rainMatches.Count -ne 1) {
    throw "Expected exactly one YC rainfall workbook, found $($rainMatches.Count)"
}

$specs = @(
    @{ File = 'T2.xls'; Variables = @(@{ Name = 'TMAX'; Column = 6 }, @{ Name = 'TMIN'; Column = 8 }) },
    @{ File = 'D32.xls'; Variables = @(@{ Name = 'SRAD'; Column = 5 }) },
    @{ Path = $rainMatches[0].FullName; Variables = @(@{ Name = 'RAIN'; Column = 5 }) }
)

$records = @{}
$sourceFiles = @{}
$sheetColumnMappings = @()
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$excel.ScreenUpdating = $false

function Get-RawCellStatus($cell) {
    if ($null -eq $cell -or [string]::IsNullOrWhiteSpace([string]$cell)) {
        return @{ Value = ''; Status = 'blank' }
    }
    if ([string]$cell -match '^\s*(--+|/|NA|N/A)\s*$') {
        return @{ Value = ''; Status = 'sentinel' }
    }
    $number = 0.0
    $parsed = [double]::TryParse(
        [string]$cell,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$number
    )
    if (-not $parsed -or [double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        return @{ Value = ''; Status = 'unparsed' }
    }
    return @{ Value = $number.ToString('R', [Globalization.CultureInfo]::InvariantCulture); Status = 'numeric' }
}

try {
    foreach ($spec in $specs) {
        $sourcePath = if ($spec.ContainsKey('Path')) { $spec.Path } else { Join-Path (Join-Path $root 'my_data') $spec.File }
        if (-not (Test-Path -LiteralPath $sourcePath)) { throw "Missing source workbook: $sourcePath" }
        $sourceName = [IO.Path]::GetFileName($sourcePath)
        $sourceFiles[$sourceName] = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
        $workbook = $excel.Workbooks.Open($sourcePath, 0, $true)
        try {
            foreach ($sheet in $workbook.Worksheets) {
                $used = $sheet.UsedRange
                $variableColumns = @{}
                foreach ($variable in $spec.Variables) {
                    $column = $variable.Column
                    if ($variable.Name -eq 'RAIN') {
                        $matches = @()
                        for ($candidate = 1; $candidate -le $used.Columns.Count; $candidate++) {
                            $header = [string]$sheet.Cells.Item(2, $candidate).Value2
                            if ($header.Contains('20-20') -and $header.Contains('合计')) {
                                $matches += $candidate
                            }
                        }
                        if ($matches.Count -ne 1) {
                            throw "Expected one 20-20 total precipitation column in $sourceName / $($sheet.Name); found $($matches.Count)"
                        }
                        $column = $matches[0]
                    }
                    $variableColumns[$variable.Name] = $column
                }
                $sheetColumnMappings += [pscustomobject]@{
                    source_file = $sourceName
                    sheet = $sheet.Name
                    variable_columns = $variableColumns
                }
                $filterRange = $sheet.Range($sheet.Cells.Item(2, 1), $sheet.Cells.Item($used.Rows.Count, $used.Columns.Count))
                [void]$filterRange.AutoFilter(1, 'YCA')
                foreach ($year in 2004..2010) {
                    [void]$filterRange.AutoFilter(2, [string]$year)
                    try {
                        $visible = $filterRange.SpecialCells(12)
                    }
                    catch {
                        throw "No visible rows for YCA $year in $sourceName / $($sheet.Name)"
                    }
                    foreach ($area in $visible.Areas) {
                        if ($area.Row -eq 2) { continue }
                        $values = $area.Value2
                        for ($localRow = 1; $localRow -le $area.Rows.Count; $localRow++) {
                            $rowYear = [int]$values[$localRow, 2]
                            if ($rowYear -ne $year) { continue }
                            $month = [int]$values[$localRow, 3]
                            $day = [int]$values[$localRow, 4]
                            $date = '{0:D4}-{1:D2}-{2:D2}' -f $rowYear, $month, $day
                            if (-not $records.ContainsKey($date)) { $records[$date] = @{ date = $date; row_present_in_any_source = $true } }
                            else { $records[$date].row_present_in_any_source = $true }
                            foreach ($variable in $spec.Variables) {
                                $cell = $values[$localRow, $variableColumns[$variable.Name]]
                                $parsed = Get-RawCellStatus $cell
                                $records[$date]["raw_$($variable.Name)"] = $parsed.Value
                                $records[$date]["status_$($variable.Name)"] = $parsed.Status
                                $records[$date]["source_$($variable.Name)"] = $sourceName
                            }
                        }
                    }
                }
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($filterRange)
            }
        }
        finally {
            $workbook.Close($false)
            [void][Runtime.InteropServices.Marshal]::ReleaseComObject($workbook)
        }
    }
}
finally {
    $excel.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$expectedDates = @()
foreach ($year in 2004..2010) {
    $d = [datetime]::new($year, 1, 1)
    while ($d.Year -eq $year) {
        $expectedDates += $d.ToString('yyyy-MM-dd')
        $d = $d.AddDays(1)
    }
}
$missingDates = @($expectedDates | Where-Object { -not $records.ContainsKey($_) })
foreach ($date in $missingDates) {
    $records[$date] = @{ date = $date; row_present_in_any_source = $false }
}

$columns = @(
    'date', 'row_present_in_any_source', 'raw_SRAD', 'status_SRAD', 'source_SRAD',
    'raw_TMAX', 'status_TMAX', 'source_TMAX',
    'raw_TMIN', 'status_TMIN', 'source_TMIN',
    'raw_RAIN', 'status_RAIN', 'source_RAIN'
)
$rows = foreach ($date in $expectedDates) {
    $record = $records[$date]
    foreach ($column in $columns) {
        if (-not $record.ContainsKey($column)) {
            $record[$column] = if ($column.StartsWith('status_') -and -not $record.row_present_in_any_source) { 'no_row' } else { '' }
        }
    }
    [pscustomobject]$record
}
$csvPath = Join-Path $resultDir 'legacy_source_observations_2004_2010.csv'
$rows | Select-Object $columns | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding utf8

$manifest = @{
    station = 'YCA'
    site = 'YC'
    years = @(2004..2010)
    expected_daily_rows = $expectedDates.Count
    extracted_daily_rows = $rows.Count
    dates_without_any_source_row = $missingDates.Count
    dates_without_any_source_row_by_year = @($missingDates | Group-Object { $_.Substring(0, 4) } | ForEach-Object { @{ year = [int]$_.Name; count = $_.Count; dates = @($_.Group) } })
    extraction = 'Excel COM; source workbooks opened ReadOnly=true and closed without saving'
    source_files = $sourceFiles
    sheet_variable_columns = $sheetColumnMappings
    output_csv = 'legacy_source_observations_2004_2010.csv'
    raw_blank_semantics = 'Preserved as blank status; no blank converted to a weather observation.'
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resultDir 'legacy_source_inventory.json') -Encoding utf8
Write-Output "Extracted $($rows.Count) daily YC source rows to $csvPath"
