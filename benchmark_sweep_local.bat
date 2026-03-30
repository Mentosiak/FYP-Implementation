@echo off
setlocal

set PY=C:\Users\momot\Documents\GitHub\FYP-Implementation\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo Python executable not found: %PY%
    exit /b 1
)

echo ============================================
echo CIFAR-10 Benchmark Sweep ^(Local Python^)
echo ============================================
echo Standard SSL protocol splits:
echo   250 total labels  = 25 labels per class
echo   1000 total labels = 100 labels per class
echo   4000 total labels = 400 labels per class
echo.

call :run_pair 250 configs/benchmarks/supervised_limited_cifar10_250labels.yaml configs/benchmarks/ssl_pseudolabel_cifar10_250labels.yaml
if errorlevel 1 goto :fail

call :run_pair 1000 configs/benchmarks/supervised_limited_cifar10_1000labels.yaml configs/benchmarks/ssl_pseudolabel_cifar10_1000labels.yaml
if errorlevel 1 goto :fail

call :run_pair 4000 configs/benchmarks/supervised_limited_cifar10_4000labels.yaml configs/benchmarks/ssl_pseudolabel_cifar10_4000labels.yaml
if errorlevel 1 goto :fail

call :run_ssl_only mixmatch_250 configs/benchmarks/ssl_mixmatch_cifar10_250labels.yaml
if errorlevel 1 goto :fail

call :run_ssl_only mixmatch_1000 configs/benchmarks/ssl_mixmatch_cifar10_1000labels.yaml
if errorlevel 1 goto :fail

call :run_ssl_only fixmatch_4000 configs/benchmarks/ssl_fixmatch_cifar10_4000labels.yaml
if errorlevel 1 goto :fail

call :run_ssl_only flexmatch_250 configs/benchmarks/ssl_flexmatch_cifar10_250labels.yaml
if errorlevel 1 goto :fail

call :run_ssl_only flexmatch_1000 configs/benchmarks/ssl_flexmatch_cifar10_1000labels.yaml
if errorlevel 1 goto :fail

call :run_ssl_only flexmatch_4000 configs/benchmarks/ssl_flexmatch_cifar10_4000labels.yaml
if errorlevel 1 goto :fail

echo.
echo Local benchmark sweep completed successfully.
goto :end

:run_pair
set SPLIT=%1
set SUP_CFG=%2
set SSL_CFG=%3

echo --------------------------------------------
echo Split: %SPLIT% labeled images
echo Supervised config: %SUP_CFG%
echo SSL config:        %SSL_CFG%
echo --------------------------------------------

"%PY%" train_supervised_limited.py --config %SUP_CFG%
if errorlevel 1 exit /b 1

"%PY%" train_ssl.py --config %SSL_CFG%
if errorlevel 1 exit /b 1

"%PY%" compare_methods.py --supervised-config %SUP_CFG% --ssl-config %SSL_CFG%
if errorlevel 1 exit /b 1

exit /b 0

:run_ssl_only
set NAME=%1
set SSL_CFG=%2

echo --------------------------------------------
echo SSL-only run: %NAME%
echo SSL config: %SSL_CFG%
echo --------------------------------------------

"%PY%" train_ssl.py --config %SSL_CFG%
if errorlevel 1 exit /b 1

exit /b 0

:fail
echo.
echo Local benchmark sweep failed.
exit /b 1

:end
endlocal
