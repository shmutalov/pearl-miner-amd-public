@echo off
setlocal enableextensions

REM ===================================================================
REM  Pearl AMD miner - HeroMiners (TLS, pearl/v1 challenge-first) launcher
REM  Standard-stratum pool: the pdiff share target (0xFFFF<<208)/D applies
REM  cleanly here (unlike AlphaPool, whose per-job target diverges from the
REM  standard path -- ARC's docs/POOLS.md notes a direct AlphaPool connect is
REM  rejected as "below target"). Endpoint from ARC-miner docs/POOLS.md.
REM  Edit ADDRESS below to your own prl1... wallet and run.
REM ===================================================================

REM ----- EDIT HERE: your Pearl wallet (prl1...) -----
set "ADDRESS=prl1p5vtjsxajasd805qtc2xp5zp3tl99egklxzfr0th7m0v8ue858uvs7hrhhs"

REM ----- Worker name (anything; shown in pool stats) -----
set "WORKER=rx7900xt"

REM ----- Pool (HeroMiners, TLS). Pick the region nearest you; others:
REM        ca.pearl.herominers.com / de.pearl.herominers.com /
REM        sg.pearl.herominers.com / br.pearl.herominers.com (port 1200, TLS).
REM        Always confirm the current host/port on the pool's own site. -----
set "POOL_HOST=ca.pearl.herominers.com"
set "POOL_PORT=1200"

REM ----- coopmat: distinct shares to find+submit per round -----
set "COOPMAT_BATCH=64"

REM ----- Run effectively forever (demo caps; raise to keep mining) -----
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
echo   Pool   : %POOL_HOST%:%POOL_PORT% (TLS)
echo   Path   : coopmat (tensor cores) + submit
echo.

REM  --tls: HeroMiners serves stratum over TLS on port 1200.
REM  --password "x;d=1": request the lowest difficulty the pool allows so the
REM  (correct, pdiff) share target is reachable; HeroMiners vardiff may raise it.
REM  Drop the --password arg entirely to let the pool's vardiff manage it.
"%PY%" src\scripts\35_run_miner_live.py --host "%POOL_HOST%" --port %POOL_PORT% --tls --address "%ADDRESS%" --worker "%WORKER%" --coopmat --coopmat-batch %COOPMAT_BATCH% --max-hits %MAX_HITS% --observe-seconds %OBSERVE_SECONDS% --submit --password "x;d=1"

echo.
echo   Miner exited (code %ERRORLEVEL%).
endlocal
pause
