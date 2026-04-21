@echo off
setlocal

set IMAGE=fyp-ssl:latest

echo ============================================
echo Missing Comparison Queue ^(Docker CUDA^)
echo ============================================

echo [1/9] FixMatch 250 vs Supervised 250...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_250labels.yaml --ssl-config configs/benchmarks/ssl_fixmatch_cifar10_250labels.yaml
if errorlevel 1 goto :fail

echo [2/9] FixMatch 1000 vs Supervised 1000...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_1000labels.yaml --ssl-config configs/benchmarks/ssl_fixmatch_cifar10_1000labels.yaml
if errorlevel 1 goto :fail

echo [3/9] FixMatch 4000 vs Supervised 4000...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_4000labels.yaml --ssl-config configs/benchmarks/ssl_fixmatch_cifar10_4000labels.yaml
if errorlevel 1 goto :fail

echo [4/9] MixMatch 250 vs Supervised 250...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_250labels.yaml --ssl-config configs/benchmarks/ssl_mixmatch_cifar10_250labels.yaml
if errorlevel 1 goto :fail

echo [5/9] MixMatch 1000 vs Supervised 1000...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_1000labels.yaml --ssl-config configs/benchmarks/ssl_mixmatch_cifar10_1000labels.yaml
if errorlevel 1 goto :fail

echo [6/9] MixMatch 4000 vs Supervised 4000...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_4000labels.yaml --ssl-config configs/benchmarks/ssl_mixmatch_cifar10_4000labels.yaml
if errorlevel 1 goto :fail

echo [7/9] FlexMatch 250 vs Supervised 250...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_250labels.yaml --ssl-config configs/benchmarks/ssl_flexmatch_cifar10_250labels.yaml
if errorlevel 1 goto :fail

echo [8/9] FlexMatch 1000 vs Supervised 1000...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_1000labels.yaml --ssl-config configs/benchmarks/ssl_flexmatch_cifar10_1000labels.yaml
if errorlevel 1 goto :fail

echo [9/9] FlexMatch 4000 vs Supervised 4000...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_4000labels.yaml --ssl-config configs/benchmarks/ssl_flexmatch_cifar10_4000labels.yaml
if errorlevel 1 goto :fail

echo.
echo Missing comparison queue completed successfully.
goto :end

:fail
echo.
echo Missing comparison queue failed.
exit /b 1

:end
endlocal
