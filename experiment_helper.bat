@echo off
REM Helper script for common SSL vs Supervised tasks (Windows)

echo ============================================
echo FYP Implementation - Experiment Helper
echo ============================================
echo.

if "%1"=="verify" (
    echo Verifying installation...
    python verify_installation.py
    goto :end
)

if "%1"=="test" (
    echo Running quick tests...
    echo.
    echo [1/2] Testing supervised pipeline...
    python test_supervised.py
    echo.
    echo [2/2] Testing SSL pipeline...
    python test_ssl.py
    goto :end
)

if "%1"=="train-supervised" (
    echo Training supervised model with limited labels...
    python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml
    goto :end
)

if "%1"=="train-ssl" (
    echo Training SSL Pseudo-Label model...
    python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
    goto :end
)

if "%1"=="compare" (
    echo Comparing SSL vs Supervised...
    python compare_methods.py --supervised-config configs/supervised_limited_cifar10.yaml --ssl-config configs/ssl_pseudolabel_cifar10.yaml
    goto :end
)

if "%1"=="compare-force" (
    echo Comparing SSL vs Supervised (force retrain)...
    python compare_methods.py --supervised-config configs/supervised_limited_cifar10.yaml --ssl-config configs/ssl_pseudolabel_cifar10.yaml --force-train
    goto :end
)

if "%1"=="full-experiment" (
    echo Running full experiment workflow...
    echo.
    echo [1/3] Training supervised with limited labels...
    python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml
    echo.
    echo [2/3] Training SSL Pseudo-Label...
    python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
    echo.
    echo [3/3] Comparing results...
    python compare_methods.py --supervised-config configs/supervised_limited_cifar10.yaml --ssl-config configs/ssl_pseudolabel_cifar10.yaml
    goto :end
)

echo Usage: experiment_helper.bat [command]
echo.
echo Commands:
echo   verify              - Verify installation and imports
echo   test                - Run quick validation tests (2 epochs each)
echo   train-supervised    - Train supervised with limited labels
echo   train-ssl           - Train SSL Pseudo-Label
echo   compare             - Compare methods (uses checkpoints if available)
echo   compare-force       - Compare methods (force retrain)
echo   full-experiment     - Run complete experiment workflow
echo.
echo Examples:
echo   experiment_helper.bat verify
echo   experiment_helper.bat test
echo   experiment_helper.bat full-experiment
echo.

:end
