@echo off
REM Sprint 1 - Quick Docker Commands for Windows

echo ============================================
echo FYP Implementation - Docker Helper
echo ============================================
echo.

if "%1"=="build" (
    echo Building Docker image...
    docker build -t fyp-implementation -f docker/Dockerfile .
    goto :end
)

if "%1"=="test" (
    echo Running quick test ^(2 epochs^)...
    docker run -it --rm -v %cd%:/workspace fyp-implementation python test_supervised.py
    goto :end
)

if "%1"=="train" (
    echo Running full supervised training...
    docker run -it --rm -v %cd%:/workspace fyp-implementation python train_supervised.py
    goto :end
)

if "%1"=="train-gpu" (
    echo Running full supervised training with GPU...
    docker run -it --rm --gpus all -v %cd%:/workspace fyp-implementation python train_supervised.py
    goto :end
)

if "%1"=="shell" (
    echo Opening Docker shell...
    docker run -it --rm -v %cd%:/workspace fyp-implementation /bin/bash
    goto :end
)

echo Usage: docker_helper.bat [command]
echo.
echo Commands:
echo   build       - Build the Docker image
echo   test        - Run quick validation test ^(2 epochs^)
echo   train       - Run full supervised training ^(CPU^)
echo   train-gpu   - Run full supervised training ^(GPU^)
echo   shell       - Open interactive shell in container
echo.
echo Examples:
echo   docker_helper.bat build
echo   docker_helper.bat test
echo   docker_helper.bat train-gpu
echo.

:end
