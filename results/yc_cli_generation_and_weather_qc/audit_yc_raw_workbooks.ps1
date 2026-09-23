$ErrorActionPreference = 'Stop'

$root = (Get-Location).Path
$resultDir = Join-Path $root 'results\yc_cli_generation_and_weather_qc'
$weatherPath = Join-Path $root 'DSSAT_auto_validation\multisite_new_cultivar_inputs_013_lowIC_manual\YC\CNYC0401.WTH'
$specs = @(
    @{ File = 'T2.xls'; Variables = @(@{ Name = 'TMAX'; Column = 6 }, @{ Name = 'TMIN'; Column = 8 }) },
    @{ File = 'D32.xls'; Variables = @(@{ Name = 'SRAD'; Column = 5 }) },
    @{ Pattern = 'HLLCYCFQ*.xls'; Variables = @(@{ Name = 'RAIN'; Column = 5 }) }
)

function New-VariableAudit {
    return @{
        rows = 0
        blank_count = 0
        sentinel_count = 0
        numeric_count = 0
        zero_count = 0
        min = $null
        max = $null
        blank_dates = [System.Collections.Generic.List[string]]::new()
        zero_dates = [System.Collections.Generic.List[string]]::new()
    }
}

$sourceAudits = [System.Collections.Generic.List[object]]::new()
$rainByDate = @{}
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$excel.ScreenUpdating = $false

try {
    foreach ($spec in $specs) {
        if ($spec.ContainsKey('Pattern')) {
            $matches = @(Get-ChildItem -LiteralPath (Join-Path $root 'my_data') -File -Filter $spec.Pattern)
            if ($matches.Count -ne 1) { throw "Expected one source matching $($spec.Pattern), found $($matches.Count)" }
            $sourcePath = $matches[0].FullName
        }
        else {
            $sourcePath = Join-Path $root (Join-Path 'my_data' $spec.File)
        }
        $sourceName = [IO.Path]::GetFileName($sourcePath)
        $workbook = $excel.Workbooks.Open($sourcePath, 0, $true)
        try {
            $sheetAudits = [System.Collections.Generic.List[object]]::new()
            foreach ($sheet in $workbook.Worksheets) {
                $used = $sheet.UsedRange
                $filterRange = $sheet.Range($sheet.Cells.Item(2, 1), $sheet.Cells.Item($used.Rows.Count, $used.Columns.Count))
                [void]$filterRange.AutoFilter(1, 'YCA')
                [void]$filterRange.AutoFilter(2, '2004')
                $visible = $filterRange.SpecialCells(12)
                $variables = @{}
                foreach ($variable in $spec.Variables) {
                    $variables[$variable.Name] = New-VariableAudit
                }
                $dates = [System.Collections.Generic.List[string]]::new()

                foreach ($area in $visible.Areas) {
                    if ($area.Row -eq 2) { continue }
                    $areaValues = $area.Value2
                    for ($localRow = 1; $localRow -le $area.Rows.Count; $localRow++) {
                        $date = '{0:D4}-{1:D2}-{2:D2}' -f [int]$areaValues[$localRow, 2], [int]$areaValues[$localRow, 3], [int]$areaValues[$localRow, 4]
                        $dates.Add($date)
                        foreach ($variable in $spec.Variables) {
                            $audit = $variables[$variable.Name]
                            $audit.rows++
                            $cell = $areaValues[$localRow, $variable.Column]
                            if ($null -eq $cell -or [string]::IsNullOrWhiteSpace([string]$cell)) {
                                $audit.blank_count++
                                $audit.blank_dates.Add($date)
                                if ($variable.Name -eq 'RAIN') {
                                    $rainByDate[$date] = 'blank'
                                }
                                continue
                            }

                            if ([string]$cell -match '^\s*(--+|/|NA|N/A)\s*$') {
                                $audit.sentinel_count++
                                if ($variable.Name -eq 'RAIN') {
                                    $rainByDate[$date] = 'sentinel'
                                }
                                continue
                            }

                            $number = 0.0
                            $parsed = [double]::TryParse(
                                [string]$cell,
                                [Globalization.NumberStyles]::Float,
                                [Globalization.CultureInfo]::InvariantCulture,
                                [ref]$number
                            )
                            if (-not $parsed) {
                                $audit.sentinel_count++
                                if ($variable.Name -eq 'RAIN') {
                                    $rainByDate[$date] = 'unparsed'
                                }
                                continue
                            }

                            $audit.numeric_count++
                            if ($null -eq $audit.min -or $number -lt $audit.min) { $audit.min = $number }
                            if ($null -eq $audit.max -or $number -gt $audit.max) { $audit.max = $number }
                            if ($number -eq 0) {
                                $audit.zero_count++
                                $audit.zero_dates.Add($date)
                            }
                            if ($variable.Name -eq 'RAIN') {
                                $rainByDate[$date] = if ($number -eq 0) { 'zero' } else { 'numeric' }
                            }
                        }
                    }
                }

                $uniqueDates = @($dates | Sort-Object -Unique)
                $duplicateRows = $dates.Count - $uniqueDates.Count
                $sheetAudits.Add(@{
                    sheet = $sheet.Name
                    matching_rows = $dates.Count
                    unique_dates = $uniqueDates.Count
                    first_date = if ($uniqueDates.Count) { $uniqueDates[0] } else { $null }
                    last_date = if ($uniqueDates.Count) { $uniqueDates[-1] } else { $null }
                    duplicate_rows = $duplicateRows
                    variables = $variables
                })
            }

            $sourceAudits.Add(@{
                file = $sourceName
                sha256 = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
                sheets = @($sheetAudits)
            })
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

$wthRows = foreach ($line in Get-Content -LiteralPath $weatherPath) {
    if ($line -match '^\s*(\d{7})\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)') {
        [pscustomobject]@{
            date_code = $matches[1]
            rain = [double]::Parse($matches[5], [Globalization.CultureInfo]::InvariantCulture)
        }
    }
}
$longestStart = $null
$longestEnd = $null
$currentStart = $null
$currentLength = 0
$longestLength = 0
foreach ($row in $wthRows) {
    if ($row.rain -le 0.1) {
        if ($null -eq $currentStart) { $currentStart = $row.date_code }
        $currentLength++
        if ($currentLength -gt $longestLength) {
            $longestLength = $currentLength
            $longestStart = $currentStart
            $longestEnd = $row.date_code
        }
    }
    else {
        $currentStart = $null
        $currentLength = 0
    }
}

$drySpellRawCounts = @{ blank = 0; zero = 0; numeric = 0; sentinel = 0; unparsed = 0; absent = 0 }
if ($longestLength -gt 0) {
    $dryStartYear = [int]$longestStart.Substring(0, 4)
    $dryStartDoy = [int]$longestStart.Substring(4, 3)
    $dryEndDoy = [int]$longestEnd.Substring(4, 3)
    for ($doy = $dryStartDoy; $doy -le $dryEndDoy; $doy++) {
        $date = (Get-Date -Year $dryStartYear -Month 1 -Day 1).AddDays($doy - 1).ToString('yyyy-MM-dd')
        $category = if ($rainByDate.ContainsKey($date)) { $rainByDate[$date] } else { 'absent' }
        $drySpellRawCounts[$category]++
    }
}

$result = @{
    site = 'YC'
    station_id = 'YCA'
    target_year = 2004
    audit_method = 'Microsoft Excel COM; workbook opened with ReadOnly=true; no workbook saved or modified'
    source_files = @($sourceAudits)
    cleaning_rule_evidence = @{
        script = 'src/weather_preprocess.py'
        behavior = 'RAIN values are coerced to numeric, then NaN values are replaced with 0 before station/date filtering.'
        lines = '148-150'
    }
    wth_longest_dry_spell_le_0p1mm = @{
        length_days = $longestLength
        start_date_code = $longestStart
        end_date_code = $longestEnd
        raw_rain_source_categories = $drySpellRawCounts
    }
    interpretation = 'Raw blanks and numeric zero are reported separately. Because the preprocessing pipeline converts rainfall blanks to zero, blank counts within the WTH dry spell remain provenance-ambiguous until observation semantics are independently confirmed.'
}
$outPath = Join-Path $resultDir 'weather_2004_raw_excel_audit.json'
$result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $outPath -Encoding utf8
Write-Output ("Saved {0}; dry_spell={1} days ({2}-{3}); raw categories={4}" -f $outPath, $longestLength, $longestStart, $longestEnd, ($drySpellRawCounts | ConvertTo-Json -Compress))
