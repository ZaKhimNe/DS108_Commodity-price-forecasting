$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$PY = (Get-Command py).Source

function Run-Step {
    param([string]$Script, [string]$LogFile, [string]$ErrFile)
    Write-Host "$(Get-Date -Format 'HH:mm:ss') [START] $Script"
    $p = Start-Process -FilePath $PY `
        -ArgumentList "-3.14", "-u", "-X", "utf8", $Script `
        -RedirectStandardOutput $LogFile `
        -RedirectStandardError  $ErrFile `
        -NoNewWindow -PassThru -Wait
    Write-Host "$(Get-Date -Format 'HH:mm:ss') [DONE]  $Script (exit=$($p.ExitCode))"
    return $p.ExitCode
}

function Wait-AllLogs {
    param([string[]]$LogFiles, [string[]]$DonePatterns)
    Write-Host "$(Get-Date -Format 'HH:mm:ss') Waiting for: $($LogFiles -join ', ')"
    while ($true) {
        $allDone = $true
        for ($i = 0; $i -lt $LogFiles.Count; $i++) {
            $log = $LogFiles[$i]
            $pat = $DonePatterns[$i]
            if (-not (Test-Path $log)) { $allDone = $false; continue }
            $content = Get-Content $log -Raw -ErrorAction SilentlyContinue
            if ($null -eq $content -or -not ($content -match $pat)) {
                $allDone = $false
            }
        }
        if ($allDone) { break }
        Start-Sleep -Seconds 60
        Write-Host "$(Get-Date -Format 'HH:mm:ss') Still waiting..."
    }
    Write-Host "$(Get-Date -Format 'HH:mm:ss') All done!"
}

# === WAIT FOR 3 TRAINING MODELS ===
$trainLogs = @(
    "logs/16_lstm_restart.log",
    "logs/16c_lstm_restart.log",
    "logs/17c_tcn_restart.log"
)
$trainPatterns = @(
    "Models saved to",
    "Models saved to",
    "Models saved to"
)
Wait-AllLogs -LogFiles $trainLogs -DonePatterns $trainPatterns

Write-Host ""
Write-Host "=========================================="
Write-Host "ALL 3 MODELS COMPLETE. STARTING STACKING."
Write-Host "=========================================="

# === BINARY PIPELINE ===
Run-Step "src/modeling/18_stacking_ensemble.py" "logs/18_stack.log" "logs/18_stack_err.log"
Run-Step "src/modeling/19_backtesting_engine.py" "logs/19_backtest.log" "logs/19_backtest_err.log"

# === REGRESSION PIPELINE ===
Run-Step "src/modeling/18c_stacking_regression.py" "logs/18c_stack.log" "logs/18c_stack_err.log"
Run-Step "src/modeling/19c_backtesting_reg.py" "logs/19c_backtest.log" "logs/19c_backtest_err.log"

# === REPORT ===
Run-Step "src/20_pipeline_report.py" "logs/20_report.log" "logs/20_report_err.log"

Write-Host ""
Write-Host "=========================================="
Write-Host "FULL PIPELINE COMPLETE!"
Write-Host "=========================================="
