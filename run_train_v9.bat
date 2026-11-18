@echo off
setlocal
cd /d "%~dp0"
if not exist "env\python.exe" (
    echo Run install.bat first.
    pause
    exit /b 1
)
if not exist "data\manifest.jsonl" (
    echo Prepare data\manifest.jsonl as described in docs\UPGRADE.md.
    pause
    exit /b 1
)
env\python.exe train_v9.py data\manifest.jsonl --config configs\v9.json --output runs\v9
pause
