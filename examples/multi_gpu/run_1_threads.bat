@echo off
REM Run Dream training with 1 CPU thread(s). Just double-click.
REM The number after WALLTIME= is the wall time in seconds -- send me all 4.
set OMP_NUM_THREADS=1
echo ============================================
echo  Running with 1 thread(s)  (OMP_NUM_THREADS=1)
echo ============================================
dream run data_parallel_train.dream
echo.
echo ^^^ The WALLTIME= number above is for 1 thread(s).
pause
