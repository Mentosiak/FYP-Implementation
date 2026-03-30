@echo off
REM Enhanced Docker Helper for SSL Experiments

echo ============================================
echo FYP SSL Implementation - Docker Runner
echo ============================================
echo.

if "%1"=="build" (
    echo Building Docker image...
    docker build -t fyp-ssl:latest -f docker/Dockerfile .
    goto :end
)

if "%1"=="verify" (
    echo Running verification in Docker...
    docker run --rm -v %cd%:/workspace fyp-ssl:latest python verify_installation.py
    goto :end
)

if "%1"=="test-ssl" (
    echo Running SSL quick test in Docker ^(2 epochs^)...
    docker run --rm --gpus all -v %cd%:/workspace fyp-ssl:latest python test_ssl.py
    goto :end
)

if "%1"=="test-ssl-cpu" (
    echo Running SSL quick test in Docker ^(CPU, 2 epochs^)...
    docker run --rm -v %cd%:/workspace fyp-ssl:latest python test_ssl.py
    goto :end
)

if "%1"=="train-supervised" (
    echo Training supervised with limited labels...
    docker run --rm --gpus all -v %cd%:/workspace fyp-ssl:latest python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml
    goto :end
)

if "%1"=="train-ssl" (
    echo Training SSL Pseudo-Label...
    docker run --rm --gpus all -v %cd%:/workspace fyp-ssl:latest python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
    goto :end
)

if "%1"=="compare" (
    echo Comparing SSL vs Supervised...
    docker run --rm --gpus all -v %cd%:/workspace fyp-ssl:latest python compare_methods.py --supervised-config configs/supervised_limited_cifar10.yaml --ssl-config configs/ssl_pseudolabel_cifar10.yaml
    goto :end
)

if "%1"=="full-experiment" (
    echo Running full experiment workflow...
    echo This will train both models and compare results...
    echo.
    docker run --rm --gpus all -v %cd%:/workspace fyp-ssl:latest bash -c "python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml && python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml && python compare_methods.py --supervised-config configs/supervised_limited_cifar10.yaml --ssl-config configs/ssl_pseudolabel_cifar10.yaml"
    goto :end
)

if "%1"=="benchmark-sweep" (
    call benchmark_sweep.bat
    goto :end
)

if "%1"=="benchmark-sweep-cpu" (
    call benchmark_sweep.bat cpu
    goto :end
)

if "%1"=="cli" (
    python train_cli_docker.py
    goto :end
)

if "%1"=="shell" (
    echo Opening interactive Docker shell...
    docker run -it --rm --gpus all -v %cd%:/workspace fyp-ssl:latest /bin/bash
    goto :end
)

if "%1"=="shell-cpu" (
    echo Opening interactive Docker shell ^(CPU only^)...
    docker run -it --rm -v %cd%:/workspace fyp-ssl:latest /bin/bash
    goto :end
)

if "%1"=="compose-build" (
    echo Building with Docker Compose...
    docker-compose build
    goto :end
)

if "%1"=="compose-verify" (
    echo Running verification with Docker Compose...
    docker-compose run --rm verify
    goto :end
)

if "%1"=="compose-test" (
    echo Running SSL test with Docker Compose...
    docker-compose run --rm test-ssl
    goto :end
)

if "%1"=="compose-full" (
    echo Running full experiment with Docker Compose...
    docker-compose run --rm full-experiment
    goto :end
)

echo Usage: run_docker.bat [command]
echo.
echo Docker Commands:
echo   build                  - Build the Docker image
echo   verify                 - Verify installation in Docker
echo   test-ssl              - Quick SSL test with GPU ^(2 epochs^)
echo   test-ssl-cpu          - Quick SSL test with CPU ^(2 epochs^)
echo   train-supervised      - Train supervised model with limited labels ^(GPU^)
echo   train-ssl             - Train SSL Pseudo-Label ^(GPU^)
echo   compare               - Compare SSL vs Supervised ^(GPU^)
echo   full-experiment       - Run complete experiment workflow ^(GPU^)
echo   benchmark-sweep       - Run 250/1000/4000 label benchmark sweep ^(GPU^)
echo   benchmark-sweep-cpu   - Run benchmark sweep without GPU
echo   cli                   - Interactive launcher for Docker CUDA training
echo   shell                 - Open interactive shell with GPU
echo   shell-cpu             - Open interactive shell without GPU
echo.
echo Docker Compose Commands:
echo   compose-build         - Build with docker-compose
echo   compose-verify        - Verify with docker-compose
echo   compose-test          - Quick test with docker-compose
echo   compose-full          - Full experiment with docker-compose
echo.
echo Examples:
echo   run_docker.bat build
echo   run_docker.bat test-ssl
echo   run_docker.bat full-experiment
echo   run_docker.bat shell
echo.
echo Note: GPU commands require NVIDIA Docker runtime
echo      Use -cpu variants for CPU-only execution
echo.

:end
