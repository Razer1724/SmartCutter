@echo off
setlocal
cd /d "%~dp0"
if not exist "env\python.exe" (
    echo Run install.bat first.
    pause
    exit /b 1
)
if not exist "runs\v9\best.pt" (
    echo Train a v9 checkpoint first with run_train_v9.bat.
    pause
    exit /b 1
)
env\python.exe inference_v9.py infer_input --checkpoint runs\v9\best.pt --output infer_output
pause
