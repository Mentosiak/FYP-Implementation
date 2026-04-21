@echo off
setlocal

set IMAGE=fyp-ssl:latest

echo ============================================
echo Recovery and Sweep Queue ^(Docker CUDA^)
echo ============================================

echo [1/5] Rerun FixMatch 250 ^(literature-aligned^)...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_fixmatch_cifar10_250labels_lit.yaml
if errorlevel 1 goto :fail

echo [2/5] Rerun MixMatch 250 ^(literature-aligned^)...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_mixmatch_cifar10_250labels_lit.yaml
if errorlevel 1 goto :fail

echo [3/5] Rerun FlexMatch 250 ^(literature-aligned^)...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_flexmatch_cifar10_250labels_lit.yaml
if errorlevel 1 goto :fail

echo [4/5] Pseudo-Label 250 threshold sweep ^(0.90 confidence^)...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_pseudolabel_cifar10_250labels_conf90.yaml
if errorlevel 1 goto :fail

echo [5/5] Pseudo-Label 250 threshold sweep ^(0.80 confidence^)...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_pseudolabel_cifar10_250labels_conf80.yaml
if errorlevel 1 goto :fail

echo.
echo Recovery and sweep queue completed successfully.
echo Next: run comparison queue for reruns and regenerate benchmark summary.
goto :end

:fail
echo.
echo Recovery and sweep queue failed.
exit /b 1

:end
endlocal
