@echo off
setlocal enableextensions

REM ===================================================================
REM  Pearl AMD miner - tensor-core (RDNA3 / coopmat) mining launcher
REM  Edit ADDRESS below to your own prl1... wallet and run.
REM ===================================================================

REM ----- EDIT HERE: your Pearl wallet (prl1...) -----
set "ADDRESS=prl1p5vtjsxajasd805qtc2xp5zp3tl99egklxzfr0th7m0v8ue858uvs7hrhhs"

REM ----- Worker name (anything; shown in pool stats) -----
set "WORKER=rx7900xt"

REM ----- Pool (defaults match the script) -----
set "POOL_HOST=eu1.alphapool.tech"
set "POOL_PORT=5566"

REM ----- coopmat: distinct shares to find+submit per round -----
set "COOPMAT_BATCH=64"

REM ----- Run effectively forever (demo caps; raise to keep mining) -----
REM   MAX_HITS        : stop after this many submitted shares
REM   OBSERVE_SECONDS : stop after this many wall-clock seconds
REM   (must stay under ~49 days; Windows thread-wait caps the timeout)
set "MAX_HITS=1000000000"
set "OBSERVE_SECONDS=2592000"

REM ===================================================================
REM  No need to edit below this line
REM ===================================================================

REM Work from the folder this .bat lives in (= repo root)
cd /d "%~dp0"

REM Python from the project venv; fall back to system python if missing
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo.
echo   Wallet : %ADDRESS%
echo   Worker : %WORKER%
echo   Pool   : %POOL_HOST%:%POOL_PORT%
echo   Path   : coopmat (tensor cores) + submit
echo.

"%PY%" src\scripts\35_run_miner_live.py --host "%POOL_HOST%" --port %POOL_PORT% --address "%ADDRESS%" --worker "%WORKER%" --coopmat --coopmat-batch %COOPMAT_BATCH% --max-hits %MAX_HITS% --observe-seconds %OBSERVE_SECONDS% --submit --coopmat-submit-threads 8 --password "x;d=3000000"

echo.
echo   Miner exited (code %ERRORLEVEL%).
endlocal
pause
