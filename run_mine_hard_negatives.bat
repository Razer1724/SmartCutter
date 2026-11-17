@echo off

if /i "%cd%"=="C:\Windows\System32" (
    color 0C
    echo SmartCutter shouldn't be run with admin perms. Don't do that.
    echo.
    pause
    exit /b 1
)

setlocal
title SmartCutter Hard-Negative Mining

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

env\python.exe mine_hard_negatives.py
echo.
pause
