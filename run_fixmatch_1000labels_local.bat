@echo off
setlocal

set PY=C:\Users\momot\Documents\GitHub\FYP-Implementation\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Python executable not found: %PY%
    exit /b 1
)

echo ============================================
echo FixMatch 1000-label benchmark run
echo ============================================

"%PY%" train_ssl.py --config configs/benchmarks/ssl_fixmatch_cifar10_1000labels.yaml
if errorlevel 1 exit /b 1

"%PY%" compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_1000labels.yaml --ssl-config configs/benchmarks/ssl_fixmatch_cifar10_1000labels.yaml
if errorlevel 1 exit /b 1

echo.
echo FixMatch 1000-label run completed successfully.

endlocal
