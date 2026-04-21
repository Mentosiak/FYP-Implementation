@echo off
setlocal

set IMAGE=fyp-ssl:latest

echo ============================================
echo Missing Training Queue ^(Docker CUDA^)
echo ============================================

echo [1/4] Run FixMatch 250 labels...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_fixmatch_cifar10_250labels.yaml
if errorlevel 1 goto :fail

echo [2/4] Resume or complete MixMatch 1000 labels...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_mixmatch_cifar10_1000labels.yaml --resume checkpoints/ssl_mixmatch_cifar10_1000labels/ssl_mixmatch_cifar10_1000labels_best.pt
if errorlevel 1 goto :fail

echo [3/4] Run FlexMatch 250 labels...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_flexmatch_cifar10_250labels.yaml
if errorlevel 1 goto :fail

echo [4/4] Run FlexMatch 4000 labels...
docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_flexmatch_cifar10_4000labels.yaml
if errorlevel 1 goto :fail

echo.
echo Missing training queue completed successfully.
echo Next: run summarize_benchmarks.py and then run comparison queue.
goto :end

:fail
echo.
echo Missing training queue failed.
exit /b 1

:end
endlocal
