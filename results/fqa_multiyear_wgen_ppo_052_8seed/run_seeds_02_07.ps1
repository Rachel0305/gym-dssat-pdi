$ErrorActionPreference = 'Stop'
$Root = (Get-Location).Path
$RunScript = Join-Path $Root 'results/fqa_multiyear_wgen_ppo_052_8seed/run_seed_100k.py'
$OutRoot = Join-Path $Root 'results/fqa_multiyear_wgen_ppo_052_8seed'

foreach ($Seed in 2..7) {
    $Attempt = Join-Path $OutRoot ("seed_{0:D2}/attempt_01" -f $Seed)
    if (Test-Path -LiteralPath $Attempt) {
        throw "Refusing to overwrite existing seed attempt: $Attempt"
    }
    $Log = Join-Path $OutRoot ("seed_{0:D2}_formal_console.log" -f $Seed)
    $AuditLog = Join-Path $OutRoot ("seed_{0:D2}_formal_audit.log" -f $Seed)
    Write-Host ("START seed {0} at {1}" -f $Seed, (Get-Date -Format o))
    docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python /workspace/results/fqa_multiyear_wgen_ppo_052_8seed/run_seed_100k.py --seed $Seed 2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE -ne 0) {
        throw "Seed $Seed training process exited with code $LASTEXITCODE; preserving evidence and stopping queue."
    }
    docker exec -w /workspace nifty_taussig /opt/gym_dssat_pdi/bin/python /workspace/results/fqa_multiyear_wgen_ppo_052_8seed/audit_seed.py --seed $Seed --run formal 2>&1 | Tee-Object -FilePath $AuditLog
    if ($LASTEXITCODE -ne 0) {
        throw "Seed $Seed formal archive audit failed with code $LASTEXITCODE; preserving evidence and stopping queue."
    }
    $GatePath = Join-Path $Attempt 'audit_gate.json'
    $Gate = Get-Content -LiteralPath $GatePath -Raw | ConvertFrom-Json
    if ($Gate.status -ne 'PASS_100K_ARCHIVE_ONLY') {
        throw "Seed $Seed audit gate status is $($Gate.status); stopping queue."
    }
    Write-Host ("PASS seed {0} at {1}" -f $Seed, (Get-Date -Format o))
}
