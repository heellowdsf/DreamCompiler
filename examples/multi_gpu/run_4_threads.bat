@echo off
REM Run Dream training with 4 CPU thread(s). Just double-click.
REM The number after WALLTIME= is the wall time in seconds -- send me all 4.
set OMP_NUM_THREADS=4
echo ============================================
echo  Running with 4 thread(s)  (OMP_NUM_THREADS=4)
echo ============================================
dream run data_parallel_train.dream
echo.
echo ^^^ The WALLTIME= number above is for 4 thread(s).
pause
