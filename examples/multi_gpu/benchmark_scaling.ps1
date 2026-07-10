# ===================================================================
#  Dream multi-core scaling benchmark (Windows / PowerShell).
#  Put this next to data_parallel_train.dream and run it.
#  If PowerShell blocks scripts, first run:
#    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# ===================================================================
param([string]$Dream = "dream")

$script = "data_parallel_train.dream"
if (-not (Test-Path $script)) {
    Write-Host "ERROR: $script not found in $((Get-Location).Path)"; exit 1
}
$cores = [Environment]::ProcessorCount

Write-Host "=== Dream multi-core scaling benchmark ==="
Write-Host "compiler: $Dream    CPU cores: $cores"
Write-Host ""
Write-Host "(warming up / precompiling runtime, one-time)..."
& $Dream run $script *> $null
Write-Host ""
Write-Host ("{0,-8} {1,-16} {2,-8}" -f "threads", "wall_time(s)", "speedup")
Write-Host "------------------------------------"

$base = $null
foreach ($t in 1,2,4,8) {
    if ($t -gt $cores) { continue }
    $env:OMP_NUM_THREADS = "$t"
    $out = & $Dream run $script 2>$null
    # find the WALLTIME= line and take the number after '='
    $line = $out | Where-Object { $_ -match '^WALLTIME=' } | Select-Object -Last 1
    $wt = if ($line) { [double]($line -replace '^WALLTIME=', '') } else { 0 }
    if ($null -eq $base) { $base = $wt }
    $sp = if ($wt -gt 0) { [math]::Round($base / $wt, 2) } else { 0 }
    Write-Host ("{0,-8} {1,-16} {2,-8}" -f $t, $wt, "${sp}x")
}
Write-Host ""
Write-Host "speedup > 1 at higher thread counts = multi-core is faster."
Write-Host "(2 threads ideal is ~2x; the gap is parallelization overhead on small models)"
