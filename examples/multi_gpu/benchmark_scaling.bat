@echo off
REM ===================================================================
REM  Dream multi-core scaling benchmark (Windows / cmd).
REM  Put this .bat in the SAME folder as data_parallel_train.dream, then run.
REM  Assumes `dream` is on your PATH (pass a path as arg 1 to override).
REM ===================================================================
setlocal enabledelayedexpansion

if not "%~1"=="" ( set "DREAM=%~1" ) else ( set "DREAM=dream" )
set "SCRIPT=data_parallel_train.dream"

if not exist "%SCRIPT%" (
    echo ERROR: %SCRIPT% not found in this folder: %CD%
    pause & exit /b 1
)

echo === Dream multi-core scaling benchmark ===
echo compiler: %DREAM%    CPU cores: %NUMBER_OF_PROCESSORS%
echo.
echo (warming up / precompiling runtime, one-time)...
"%DREAM%" run "%SCRIPT%" >nul 2>nul
echo.
echo threads    wall_time^(s^)    speedup
echo ------------------------------------
set "BASE="

for %%T in (1 2 4 8) do (
    if %%T LEQ %NUMBER_OF_PROCESSORS% (
        set "OMP_NUM_THREADS=%%T"
        set "WT="
        REM extract the WALLTIME=... line, keep only the number after '='
        for /f "usebackq tokens=2 delims==" %%V in (`"%DREAM%" run "%SCRIPT%" 2^>nul ^| findstr /b "WALLTIME="`) do set "WT=%%V"
        if not defined BASE set "BASE=!WT!"
        REM speedup = BASE / WT  (cmd has no float math; use PowerShell for the divide)
        for /f %%S in ('powershell -NoProfile -Command "[math]::Round(!BASE! / !WT!, 2)"') do set "SP=%%S"
        echo   %%T        !WT!        !SP!x
    )
)

echo.
echo speedup ^> 1 at higher thread counts = multi-core is faster.
echo (2 threads ideal is ~2x; the gap is parallelization overhead on small models)
endlocal
pause
