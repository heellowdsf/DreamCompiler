@echo off
REM Diagnostic: run the training ONCE with full output visible, so we can see
REM whether Dream runs at all and what it prints. No output hiding.
setlocal

if not "%~1"=="" ( set "DREAM=%~1" ) else ( set "DREAM=dream" )

echo === diagnostic run ===
echo compiler command: %DREAM%
echo current folder:   %CD%
echo.
echo checking the .dream script exists:
if exist data_parallel_train.dream (echo   FOUND data_parallel_train.dream) else (echo   MISSING data_parallel_train.dream)
echo.
echo running once with FULL output (this may take a bit on first run):
echo -------------------------------------------------------------
"%DREAM%" run data_parallel_train.dream
echo -------------------------------------------------------------
echo exit code was: %ERRORLEVEL%
echo.
echo If you see "final accuracy: 1" and a number above, Dream works.
echo The last number is the wall time the benchmark reads.
endlocal
pause
