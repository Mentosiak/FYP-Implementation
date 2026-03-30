@echo off
setlocal

set IMAGE=fyp-ssl:latest

echo ============================================
echo Remaining Training Queue ^(Docker CUDA^)
echo ============================================

docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_fixmatch_cifar10_4000labels.yaml --resume checkpoints/ssl_fixmatch_cifar10_4000labels/ssl_fixmatch_cifar10_4000labels_best.pt
if errorlevel 1 goto :fail

docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_mixmatch_cifar10_250labels_tuned.yaml
if errorlevel 1 goto :fail

docker run --rm --gpus all -v %cd%:/workspace %IMAGE% python train_ssl.py --config configs/benchmarks/ssl_flexmatch_cifar10_1000labels.yaml
if errorlevel 1 goto :fail

echo.
echo Remaining training queue completed successfully.
goto :end

:fail
echo.
echo Remaining training queue failed.
exit /b 1

:end
endlocal
