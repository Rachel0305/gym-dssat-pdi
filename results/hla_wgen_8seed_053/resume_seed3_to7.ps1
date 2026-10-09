param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$events = Join-Path $root 'resume_events.jsonl'

function Event([string]$stage, [int]$seed, [string]$status, [string]$detail) {
    $entry = [ordered]@{ utc = (Get-Date).ToUniversalTime().ToString('o'); stage = $stage; seed = $seed; status = $status; detail = $detail }
    Add-Content -LiteralPath $events -Value ($entry | ConvertTo-Json -Compress) -Encoding utf8
}

function Gate([int]$seed, [string]$name) {
    $path = Join-Path $root ('seed_{0:D2}\attempt_01\{1}' -f $seed, $name)
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing $path" }
    $value = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    if ($value.status -ne 'PASS_100K_ARCHIVE_ONLY') { throw "Failed gate $path : $($value.status)" }
}

try {
    foreach ($seed in 0..2) { Gate $seed 'run_result.json'; Gate $seed 'audit_gate.json' }
    foreach ($seed in 3..7) {
        $dir = Join-Path $root ('seed_{0:D2}' -f $seed)
        $trainLog = Join-Path $root ('seed_{0:D2}_train.log' -f $seed)
        $auditLog = Join-Path $root ('seed_{0:D2}_audit.log' -f $seed)
        if ((Test-Path -LiteralPath $dir) -or (Test-Path -LiteralPath $trainLog) -or (Test-Path -LiteralPath $auditLog)) {
            throw "Existing seed $seed output; refusing overwrite"
        }
    }
    Event 'resume' 2 'AUDITED' 'Seed 2 completed and passed audit; original orchestrator exit cause not recorded.'
    foreach ($seed in 3..7) {
        $trainLog = Join-Path $root ('seed_{0:D2}_train.log' -f $seed)
        $auditLog = Join-Path $root ('seed_{0:D2}_audit.log' -f $seed)
        Event 'training' $seed 'STARTED' $trainLog
        & docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python results/hla_wgen_8seed_053/run_seed_100k.py --seed $seed 2>&1 | Out-File -LiteralPath $trainLog -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "Training command failed seed ${seed}; see $trainLog" }
        Gate $seed 'run_result.json'
        Event 'training' $seed 'PASS' $trainLog
        Event 'audit' $seed 'STARTED' $auditLog
        & docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python results/hla_wgen_8seed_053/audit_seed.py --seed $seed --run formal 2>&1 | Out-File -LiteralPath $auditLog -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "Audit command failed seed ${seed}; see $auditLog" }
        Gate $seed 'audit_gate.json'
        Event 'audit' $seed 'PASS' $auditLog
    }
    Event 'resume' -1 'ALL_8_TRAINED_AND_AUDITED' 'Validation and figures are separate.'
} catch {
    Event 'resume' -1 'STOPPED_ON_FAILURE' $_.Exception.Message
    exit 2
}
