@echo off
setlocal
cd /d %~dp0
cls
title FYP-Implementation Launcher
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u launch_docker.py
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found on PATH and .venv\Scripts\python.exe does not exist.
        pause
        exit /b 1
    )
    python -u launch_docker.py
)
pause
