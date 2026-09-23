$ErrorActionPreference = 'Stop'

$root = (Get-Location).Path
$outDir = Join-Path $root 'results\yc_cli_generation_and_weather_qc'
$inputDir = Join-Path $root 'DSSAT_auto_validation\multisite_new_cultivar_inputs_013_lowIC_manual\YC'
$cleanCsvPath = Join-Path $root 'weather_clean\YCA_weather_cleaned.csv'
$beforeFillPath = Join-Path $root 'weather_clean\data_check_by_year_before_fill.csv'
$fillLogPath = Join-Path $root 'weather_clean\missing_value_fill_log.csv'
$rawAuditPath = Join-Path $outDir 'weather_2004_raw_excel_audit.json'
$yearRows = [System.Collections.Generic.List[object]]::new()
$monthRows = [System.Collections.Generic.List[object]]::new()
$cleanIndex = @{}
$sourceMissing = @{}

function Convert-InvariantDouble([string]$value) {
    return [double]::Parse($value, [Globalization.CultureInfo]::InvariantCulture)
}

foreach ($row in (Import-Csv -LiteralPath $cleanCsvPath)) {
    $cleanIndex[[string]$row.year_doy] = $row
}
foreach ($row in (Import-Csv -LiteralPath $beforeFillPath | Where-Object {
    $_.station -eq 'YCA' -and [int]$_.year -ge 2004 -and [int]$_.year -le 2013
})) {
    $sourceMissing[[int]$row.year] = $row
}

foreach ($year in 2004..2013) {
    $stem = 'CNYC{0:D2}01' -f ($year % 100)
    $weatherPath = Join-Path $inputDir ($stem + '.WTH')
    if (-not (Test-Path -LiteralPath $weatherPath)) { throw "Missing train weather file: $weatherPath" }
    $rows = [System.Collections.Generic.List[object]]::new()
    $parseErrors = 0
    foreach ($line in (Get-Content -LiteralPath $weatherPath)) {
        if ($line -match '^\s*(\d{7})\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)') {
            $dateCode = $matches[1]
            try {
                $rows.Add([pscustomobject]@{
                    date_code = $dateCode
                    year = [int]$dateCode.Substring(0, 4)
                    doy = [int]$dateCode.Substring(4, 3)
                    month = ([DateTime]::new([int]$dateCode.Substring(0, 4), 1, 1).AddDays([int]$dateCode.Substring(4, 3) - 1)).Month
                    SRAD = Convert-InvariantDouble $matches[2]
                    TMAX = Convert-InvariantDouble $matches[3]
                    TMIN = Convert-InvariantDouble $matches[4]
                    RAIN = Convert-InvariantDouble $matches[5]
                })
            }
            catch { $parseErrors++ }
        }
    }

    $expectedDays = if ([DateTime]::IsLeapYear($year)) { 366 } else { 365 }
    $duplicates = @($rows | Group-Object date_code | Where-Object Count -gt 1)
    $seenDoys = @{}
    foreach ($row in $rows) { $seenDoys[$row.doy] = $true }
    $missingDoys = @((1..$expectedDays | Where-Object { -not $seenDoys.ContainsKey($_) }))
    $badDateCount = @($rows | Where-Object { $_.year -ne $year -or $_.doy -lt 1 -or $_.doy -gt $expectedDays }).Count
    $sentinelCount = 0
    $knownSentinels = @(-99, -999, -9999, 99, 999, 9999)
    foreach ($row in $rows) {
        if (@(@($row.SRAD, $row.TMAX, $row.TMIN, $row.RAIN) | Where-Object { $knownSentinels -contains $_ }).Count -gt 0) { $sentinelCount++ }
    }
    $negativeRain = @($rows | Where-Object RAIN -lt 0).Count
    $negativeSrad = @($rows | Where-Object SRAD -lt 0).Count
    $tmaxBelowTmin = @($rows | Where-Object { $_.TMAX -lt $_.TMIN }).Count
    $rainTotal = ($rows | Measure-Object -Property RAIN -Sum).Sum
    $wetDays = @($rows | Where-Object RAIN -gt 0.1).Count

    $maxDry = 0; $currentDry = 0; $maxWet = 0; $currentWet = 0
    foreach ($row in $rows) {
        if ($row.RAIN -le 0.1) { $currentDry++; $currentWet = 0; $maxDry = [Math]::Max($maxDry, $currentDry) }
        else { $currentWet++; $currentDry = 0; $maxWet = [Math]::Max($maxWet, $currentWet) }
    }

    $deltas = @{ SRAD = [System.Collections.Generic.List[double]]::new(); TMAX = [System.Collections.Generic.List[double]]::new(); TMIN = [System.Collections.Generic.List[double]]::new(); RAIN = [System.Collections.Generic.List[double]]::new() }
    $unmatched = 0
    foreach ($row in $rows) {
        if (-not $cleanIndex.ContainsKey($row.date_code)) { $unmatched++; continue }
        $clean = $cleanIndex[$row.date_code]
        foreach ($variable in @('SRAD', 'TMAX', 'TMIN', 'RAIN')) {
            $deltas[$variable].Add([Math]::Abs($row.$variable - (Convert-InvariantDouble ([string]$clean.$variable))))
        }
    }

    $monthly = $rows | Group-Object month | Sort-Object { [int]$_.Name }
    foreach ($group in $monthly) {
        $data = @($group.Group)
        $monthlyRain = ($data | Measure-Object -Property RAIN -Sum).Sum
        $wet = @($data | Where-Object RAIN -gt 0.1)
        $dry = @($data | Where-Object RAIN -le 0.1)
        $monthRows.Add([pscustomobject]@{
            site = 'YC'; year = $year; month = [int]$group.Name; days = $data.Count
            rain_total_mm = [Math]::Round($monthlyRain, 3)
            wet_days_gt_0p1mm = $wet.Count
            mean_srad_mj_m2 = [Math]::Round(($data | Measure-Object -Property SRAD -Average).Average, 4)
            mean_tmax_c = [Math]::Round(($data | Measure-Object -Property TMAX -Average).Average, 4)
            mean_tmin_c = [Math]::Round(($data | Measure-Object -Property TMIN -Average).Average, 4)
            mean_dtr_c = [Math]::Round((($data | Measure-Object -Property TMAX -Average).Average) - (($data | Measure-Object -Property TMIN -Average).Average), 4)
            mean_rain_wet_days_mm = if ($wet.Count) { [Math]::Round(($wet | Measure-Object -Property RAIN -Average).Average, 4) } else { $null }
            mean_rain_dry_days_mm = if ($dry.Count) { [Math]::Round(($dry | Measure-Object -Property RAIN -Average).Average, 4) } else { $null }
        })
    }

    $source = $sourceMissing[$year]
    $hash = (Get-FileHash -LiteralPath $weatherPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $record = [ordered]@{
        site = 'YC'; station = 'YCA'; year = $year; weather_file = $weatherPath.Substring($root.Length + 1).Replace('\', '/')
        sha256 = $hash; file_bytes = (Get-Item -LiteralPath $weatherPath).Length
        record_count = $rows.Count; expected_record_count = $expectedDays
        first_date_code = if ($rows.Count) { $rows[0].date_code } else { $null }
        last_date_code = if ($rows.Count) { $rows[-1].date_code } else { $null }
        duplicate_date_count = (($duplicates | Measure-Object -Property Count -Sum).Sum -as [int])
        missing_date_count = $missingDoys.Count; bad_date_count = $badDateCount; parse_error_count = $parseErrors
        sentinel_or_nonfinite_count = $sentinelCount; negative_rain_count = $negativeRain
        negative_srad_count = $negativeSrad; tmax_below_tmin_count = $tmaxBelowTmin
        annual_rain_mm = [Math]::Round($rainTotal, 3); wet_day_count_gt_0p1mm = $wetDays
        max_daily_rain_mm = [Math]::Round(($rows | Measure-Object -Property RAIN -Maximum).Maximum, 3)
        days_rain_over_100mm = @($rows | Where-Object RAIN -gt 100).Count
        max_dry_spell_days_rain_le_0p1mm = $maxDry; max_wet_spell_days_rain_gt_0p1mm = $maxWet
        source_missing_srad_days = [int]$source.SRAD; source_missing_tmax_days = [int]$source.TMAX
        source_missing_tmin_days = [int]$source.TMIN; source_missing_rain_days = [int]$source.RAIN
        source_missing_variable_values = [int]$source.SRAD + [int]$source.TMAX + [int]$source.TMIN + [int]$source.RAIN
        cleaned_csv_unmatched_dates = $unmatched
        source_integrity_status = if (([int]$source.SRAD + [int]$source.TMAX + [int]$source.TMIN + [int]$source.RAIN) -gt 0) { 'incomplete_source_values_imputed' } else { 'no_source_gaps_recorded' }
        format_qc_status = if ($rows.Count -eq $expectedDays -and $duplicates.Count -eq 0 -and $missingDoys.Count -eq 0 -and $badDateCount -eq 0 -and $parseErrors -eq 0 -and $sentinelCount -eq 0 -and $negativeRain -eq 0 -and $negativeSrad -eq 0 -and $tmaxBelowTmin -eq 0) { 'pass' } else { 'fail' }
    }
    foreach ($variable in @('SRAD', 'TMAX', 'TMIN', 'RAIN')) {
        $maxDelta = if ($deltas[$variable].Count) { ($deltas[$variable] | Measure-Object -Maximum).Maximum } else { $null }
        $within = @($deltas[$variable] | Where-Object { $_ -le 0.051 }).Count
        $record["${variable}_csv_match_count_within_0p051"] = $within
        $record["${variable}_max_abs_delta_vs_cleaned_csv"] = if ($null -ne $maxDelta) { [Math]::Round($maxDelta, 6) } else { $null }
    }
    $yearRows.Add([pscustomobject]$record)
}

$trainCsv = Join-Path $outDir 'train_weather_integrity.csv'
$monthCsv = Join-Path $outDir 'train_weather_monthly_summary.csv'
$yearRows | Export-Csv -LiteralPath $trainCsv -NoTypeInformation -Encoding utf8
$monthRows | Export-Csv -LiteralPath $monthCsv -NoTypeInformation -Encoding utf8

$rawAudit = Get-Content -LiteralPath $rawAuditPath -Raw | ConvertFrom-Json
$rawSourceSummary = foreach ($source in $rawAudit.source_files) {
    $matchedSheet = $source.sheets | Where-Object matching_rows -gt 0 | Select-Object -First 1
    if ($matchedSheet) {
        $vars = [ordered]@{}
        foreach ($variable in $matchedSheet.variables.PSObject.Properties.Name) {
            $v = $matchedSheet.variables.$variable
            $vars[$variable] = @{
                rows = $v.rows; blank_count = $v.blank_count; numeric_count = $v.numeric_count
                zero_count = $v.zero_count; sentinel_count = $v.sentinel_count
                min = $v.min; max = $v.max
            }
        }
        @{
            file = $source.file; sha256 = $source.sha256; sheet = $matchedSheet.sheet
            rows = $matchedSheet.matching_rows; unique_dates = $matchedSheet.unique_dates
            first_date = $matchedSheet.first_date; last_date = $matchedSheet.last_date
            duplicate_rows = $matchedSheet.duplicate_rows; variables = $vars
        }
    }
}
$year2004 = $yearRows | Where-Object year -eq 2004
$provenance = @{
    status = 'BLOCKED_WEATHER_INTEGRITY'
    site = 'YC'; station_id = 'YCA'; year = 2004
    weather_file = $year2004.weather_file; weather_sha256 = $year2004.sha256
    wth_annual_rain_mm = $year2004.annual_rain_mm
    wth_max_dry_spell_days = $year2004.max_dry_spell_days_rain_le_0p1mm
    raw_source_files = @($rawSourceSummary)
    raw_dry_spell_crosswalk = $rawAudit.wth_longest_dry_spell_le_0p1mm
    preprocessing = @{
        script = 'src/weather_preprocess.py'; rain_blank_rule = 'coerce missing to zero before source-date merge';
        temperature_and_srad_fill = 'station-month mean, then station annual mean fallback';
        exact_rain_rule_lines = '148-150'; exact_fill_rule_lines = '260-289'
    }
    cleaned_weather_daily_match = @{
        cleaned_csv = 'weather_clean/YCA_weather_cleaned.csv'
        tolerance = 0.051
        srad_matches = $year2004.SRAD_csv_match_count_within_0p051
        tmax_matches = $year2004.TMAX_csv_match_count_within_0p051
        tmin_matches = $year2004.TMIN_csv_match_count_within_0p051
        rain_matches = $year2004.RAIN_csv_match_count_within_0p051
        source_missing_days = @{
            srad = $year2004.source_missing_srad_days; tmax = $year2004.source_missing_tmax_days
            tmin = $year2004.source_missing_tmin_days; rain = $year2004.source_missing_rain_days
        }
    }
    conclusion = 'The WTH is full-year and physically parseable, but not a complete raw-observation record: SRAD/TMAX/TMIN have source blanks filled by monthly means, and the 257-day WTH dry spell maps to raw rainfall blanks that preprocessing converted to zero. The earlier blanket blank-means-no-rain confirmation does not independently establish completeness for this contiguous interval; do not use 2004 to fit WGEN until the station source coverage/semantics are resolved.'
}
$provenance | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $outDir 'weather_2004_provenance.json') -Encoding utf8

$splitAuditPath = Join-Path $root 'results\yc_wgen_cli_pilot\split_audit.json'
$split = Get-Content -LiteralPath $splitAuditPath -Raw | ConvertFrom-Json
$testStatus = @{
    training_years = $split.training_years
    validation_years = $split.validation_years
    configured_independent_test_years = $split.configured_independent_test_years
    candidate_years_2000_2003 = $split.extra_year_candidates_outside_train_and_validation
    independent_test_status = 'not_available_or_not_verified'
    reason = $split.independent_test_reason
    prior_selection_evidence = @($split.split_evidence) + @($split.broad_null_run_evidence)
    test_split_does_not_block_weather_technical_audit = $true
}
$testStatus | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outDir 'test_split_status.json') -Encoding utf8

$seedStatus = @{
    status = 'not_attempted_blocked_before_cli'
    weather_generation_seeds_planned = @(101, 102, 103, 104, 105)
    ppo_seed = $null
    weather_generation_seed = $null
    same_seed_reproducibility = 'not_tested'
    different_seed_weather_difference = 'not_tested'
    generated_weather_hashes = @()
    cli_sha256 = $null
    reason = 'Gate B failed before CLI generation. No WGEN seed execution or weather capture was attempted.'
    wrapper_contract_evidence = @{
        source = 'references/dssat_pdi.py'; random_weather_false_in_active_args = $true
        when_enabled_random_weather_selects_wgen_mode = $true
        weather_seed_uses_same_environment_numpy_rng = $true
        independent_weather_generation_seed_api = 'not present in checked-in wrapper; a pilot-only adapter would be needed after gates pass'
    }
}
$seedStatus | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outDir 'seed_reproducibility.json') -Encoding utf8

$mapping = @{
    status = 'verified_static_station_prefix_mapping_runtime_not_executed'
    site = 'YC'; station_code = 'YCA'; station_id = 'CNYC'
    filex_path = 'DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYC0801.MZX'
    experiment_number = 1
    filex_sha256 = (Get-FileHash -LiteralPath (Join-Path $inputDir 'CNYC0801.MZX') -Algorithm SHA256).Hash.ToLowerInvariant()
    active_config = 'configs/055_00_yca_lowIC_expanded_action_maskableppo.json'
    active_config_sha256 = (Get-FileHash -LiteralPath (Join-Path $root 'configs\055_00_yca_lowIC_expanded_action_maskableppo.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    source_filex_wsta = @('CNYC0801', 'CNYC1401')
    wsta_in_filex = 'CNYC0801 (template row); CNYC1401 (second treatment row)'
    active_training_wsta_example = 'CNYC0401'
    rendered_wsta_pattern = 'CNYCyy01'
    historical_weather_prefix = 'CNYC'
    expected_cli_name = 'CNYC.CLI'
    mapping_verified = $true
    current_run_weather_mode = 'measured'; current_run_random_weather = $false
    current_run_cli_lookup_exercised = $false
    external_runtime_version_verified = $false
    evidence = @(
        'DSSAT_auto_validation/multisite_new_cultivar_inputs_013_lowIC_manual/YC/CNYC0801.MZX:21-23',
        'src/ppo_safe_rendering.py:42-48',
        'src/ppo_safe_rendering.py:77-91',
        'src/ppo_safe_rendering.py:258-289',
        'src/ppo_safe_rendering.py:300-304',
        'src/ppo_safe_rendering.py:317-341',
        'references/dssat_pdi.py:88-92',
        'official DSSAT SECLI.for: https://github.com/DSSAT/dssat-csm-os/blob/develop/InputModule/SECLI.for'
    )
    mapping_note = 'Gym-DSSAT passes a copied year-specific CNYCyy01.WTH in measured mode. DSSAT SECLI uses the 4-character CNYC site prefix for its climate file name (CNYC.CLI). Static identifier/name mapping is established; runtime WGEN lookup was not executed and the installed runtime version was not inspected outside the project boundary.'
}
$mapping | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outDir 'wsta_mapping.json') -Encoding utf8

$summary = @{
    final_status = 'BLOCKED_WEATHER_INTEGRITY'
    first_failed_gate = 'B'
    train_years = @(2004..2013)
    all_train_years_have_source_gaps = (@($yearRows | Where-Object source_integrity_status -eq 'incomplete_source_values_imputed').Count -eq 10)
    format_qc_passed_years = @($yearRows | Where-Object format_qc_status -eq 'pass' | ForEach-Object year)
    daily_cleaned_csv_match_passed_years = @($yearRows | Where-Object {
        $_.cleaned_csv_unmatched_dates -eq 0 -and $_.SRAD_csv_match_count_within_0p051 -eq $_.expected_record_count -and
        $_.TMAX_csv_match_count_within_0p051 -eq $_.expected_record_count -and $_.TMIN_csv_match_count_within_0p051 -eq $_.expected_record_count -and
        $_.RAIN_csv_match_count_within_0p051 -eq $_.expected_record_count
    } | ForEach-Object year)
    all_train_years_source_gap_values = @($yearRows | ForEach-Object {
        @{ year = $_.year; srad = $_.source_missing_srad_days; tmax = $_.source_missing_tmax_days; tmin = $_.source_missing_tmin_days; rain = $_.source_missing_rain_days }
    })
    cli_generated = $false; wgen_realizations = 0; dssat_smoke_runs = 0; ppo_runs = 0
    input_weather_hashes = @($yearRows | ForEach-Object { @{ year = $_.year; sha256 = $_.sha256 } })
    tools_found_in_project = @{ weather_man_or_cli_parameter_estimator = $false; official_dssat_wgen_runtime_execution = $false }
    result_files = @('train_weather_integrity.csv', 'train_weather_monthly_summary.csv', 'weather_2004_provenance.json', 'weather_2004_raw_excel_audit.json', 'test_split_status.json', 'seed_reproducibility.json', 'wsta_mapping.json')
}
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outDir 'audit_summary.json') -Encoding utf8
Write-Output ("Years={0}; full-format-pass={1}; source-gap-years={2}; 2004 daily CSV matches SRAD/TMAX/TMIN/RAIN={3}/{4}/{5}/{6}; final={7}" -f $yearRows.Count, @($yearRows | Where-Object format_qc_status -eq 'pass').Count, @($yearRows | Where-Object source_integrity_status -eq 'incomplete_source_values_imputed').Count, $year2004.SRAD_csv_match_count_within_0p051, $year2004.TMAX_csv_match_count_within_0p051, $year2004.TMIN_csv_match_count_within_0p051, $year2004.RAIN_csv_match_count_within_0p051, $summary.final_status)
