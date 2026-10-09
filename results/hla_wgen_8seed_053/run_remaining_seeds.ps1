param()

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$eventLog = Join-Path $PSScriptRoot 'orchestrator_events.jsonl'

function Write-Event([string]$stage, [int]$seed, [string]$status, [string]$detail) {
    $record = [ordered]@{
        utc = (Get-Date).ToUniversalTime().ToString('o')
        stage = $stage
        seed = $seed
        status = $status
        detail = $detail
    }
    Add-Content -LiteralPath $eventLog -Value ($record | ConvertTo-Json -Compress) -Encoding utf8
}

function Get-RunStatus([int]$seed) {
    $resultPath = Join-Path $PSScriptRoot ('seed_{0:D2}\attempt_01\run_result.json' -f $seed)
    if (-not (Test-Path -LiteralPath $resultPath)) { return $null }
    try { return (Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json).status }
    catch { return $null }
}

function Audit-Seed([int]$seed) {
    $auditLog = Join-Path $PSScriptRoot ('seed_{0:D2}_audit.log' -f $seed)
    if (Test-Path -LiteralPath $auditLog) { throw "Audit log already exists: $auditLog" }
    & docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python results/hla_wgen_8seed_053/audit_seed.py --seed $seed --run formal 2>&1 | Out-File -LiteralPath $auditLog -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw "Audit failed seed $seed; see $auditLog" }
    $gatePath = Join-Path $PSScriptRoot ('seed_{0:D2}\attempt_01\audit_gate.json' -f $seed)
    $gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
    if ($gate.status -ne 'PASS_100K_ARCHIVE_ONLY') { throw "Audit gate failed seed ${seed}: $($gate.status)" }
    Write-Event 'audit' $seed 'PASS' $gatePath
}

try {
    Write-Event 'orchestrator' -1 'STARTED' 'Wait for seed 0, then run seeds 1 to 7 sequentially.'
    while ($true) {
        $status = Get-RunStatus 0
        if ($null -ne $status) { break }
        Start-Sleep -Seconds 30
    }
    if ($status -ne 'PASS_100K_ARCHIVE_ONLY') { throw "Seed 0 training failed: $status" }
    Audit-Seed 0

    foreach ($seed in 1..7) {
        $trainLog = Join-Path $PSScriptRoot ('seed_{0:D2}_train.log' -f $seed)
        if (Test-Path -LiteralPath $trainLog) { throw "Training log already exists: $trainLog" }
        Write-Event 'training' $seed 'STARTED' $trainLog
        & docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python results/hla_wgen_8seed_053/run_seed_100k.py --seed $seed 2>&1 | Out-File -LiteralPath $trainLog -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "Training command failed seed $seed; see $trainLog" }
        if ((Get-RunStatus $seed) -ne 'PASS_100K_ARCHIVE_ONLY') { throw "Training result gate failed seed $seed" }
        Write-Event 'training' $seed 'PASS' $trainLog
        Audit-Seed $seed
    }
    Write-Event 'orchestrator' -1 'ALL_8_TRAINED_AND_AUDITED' 'Validation and figures are a separate next stage.'
} catch {
    Write-Event 'orchestrator' -1 'STOPPED_ON_FAILURE' $_.Exception.Message
    exit 2
}
