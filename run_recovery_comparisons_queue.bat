@echo off
setlocal

set IMAGE=fyp-ssl:latest

echo ============================================
echo Recovery Comparison Queue ^(Docker CUDA^)
echo ============================================

echo [1/5] Supervised-250 vs FixMatch-250 (literature-aligned)...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_250labels.yaml --ssl-config configs/benchmarks/ssl_fixmatch_cifar10_250labels_lit.yaml
if errorlevel 1 goto :fail

echo [2/5] Supervised-250 vs MixMatch-250 (literature-aligned)...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_250labels.yaml --ssl-config configs/benchmarks/ssl_mixmatch_cifar10_250labels_lit.yaml
if errorlevel 1 goto :fail

echo [3/5] Supervised-250 vs FlexMatch-250 (literature-aligned)...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_250labels.yaml --ssl-config configs/benchmarks/ssl_flexmatch_cifar10_250labels_lit.yaml
if errorlevel 1 goto :fail

echo [4/5] Supervised-250 vs Pseudo-Label-250 conf90...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_250labels.yaml --ssl-config configs/benchmarks/ssl_pseudolabel_cifar10_250labels_conf90.yaml
if errorlevel 1 goto :fail

echo [5/5] Supervised-250 vs Pseudo-Label-250 conf80...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python compare_methods.py --supervised-config configs/benchmarks/supervised_limited_cifar10_250labels.yaml --ssl-config configs/benchmarks/ssl_pseudolabel_cifar10_250labels_conf80.yaml
if errorlevel 1 goto :fail

echo.
echo Recovery comparison queue completed successfully.
goto :end

:fail
echo.
echo Recovery comparison queue failed.
exit /b 1

:end
endlocal
