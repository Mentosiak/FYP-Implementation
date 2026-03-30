"""Interactive CLI launcher for Docker/CUDA training runs."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


ALGORITHMS = ["supervised", "pseudolabel", "fixmatch", "mixmatch", "flexmatch"]
DATASETS = ["cifar10", "cifar100"]
MODELS = ["resnet18", "resnet34", "wideresnet"]


def _prompt_choice(title: str, choices: list[str], default: str) -> str:
    print(f"\n{title}")
    for idx, item in enumerate(choices, start=1):
        marker = " (default)" if item == default else ""
        print(f"  {idx}. {item}{marker}")
    raw = input("Select option: ").strip()
    if not raw:
        return default
    if raw.isdigit():
        choice_idx = int(raw) - 1
        if 0 <= choice_idx < len(choices):
            return choices[choice_idx]
    if raw in choices:
        return raw
    print(f"Invalid selection, using default: {default}")
    return default


def _prompt_int(title: str, default: int) -> int:
    raw = input(f"{title} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid integer, using default: {default}")
        return default


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


def _cuda_available_in_docker() -> bool:
    try:
        probe = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "nvidia/cuda:12.1.0-base-ubuntu22.04",
                "nvidia-smi",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return "NVIDIA-SMI" in probe.stdout
    except Exception:
        return False


def _num_classes(dataset: str) -> int:
    if dataset == "cifar100":
        return 100
    return 10


def _build_generated_config(
    algorithm: str,
    dataset: str,
    model: str,
    labels_per_class: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> dict:
    num_classes = _num_classes(dataset)
    run_name = f"{algorithm}_{dataset}_{labels_per_class}lpc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cfg = {
        "run_name": run_name,
        "dataset": {
            "name": dataset,
            "data_dir": "./data",
            "num_classes": num_classes,
        },
        "model": {
            "architecture": model,
            "depth": 28,
            "widen_factor": 2,
            "dropout_rate": 0.0,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 0,
            "validation_split": 0.1,
            "stop_loss_threshold": 3.0,
            "stop_loss_warmup_epochs": 12,
            "supervised_algorithm": "mixup",
            "mixup_alpha": 0.2,
            "save_best": True,
            "save_last": True,
        },
        "ssl": {
            "enabled": algorithm != "supervised",
            "algorithm": algorithm,
            "labels_per_class": labels_per_class,
            "label_fraction": None,
            "unlabeled_batch_size": max(128, batch_size * 4),
            "confidence_threshold": 0.95,
            "unlabeled_loss_weight": 1.0,
            "strong_augment": True,
            "temperature": 1.0,
            "mixmatch_alpha": 0.75,
            "mixmatch_temperature": 0.5,
            "seed": 42,
        },
        "system": {
            "seed": 42,
            "num_workers": 2,
            "device": "cuda",
            "checkpoint_dir": "checkpoints",
            "log_dir": "logs",
            "experiment_dir": "experiments",
        },
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Docker CUDA training launcher")
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--labels-per-class", type=int)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image", default="fyp-ssl:latest")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--cpu-fallback", action="store_true")
    args = parser.parse_args()

    if not _docker_available():
        raise SystemExit("Docker is not available. Start Docker Desktop and retry.")

    algorithm = args.algorithm or _prompt_choice("Algorithm", ALGORITHMS, "pseudolabel")
    dataset = args.dataset or _prompt_choice("Dataset", DATASETS, "cifar10")
    model = args.model or _prompt_choice("Model", MODELS, "wideresnet")

    if args.labels_per_class is not None:
        labels_per_class = args.labels_per_class
    else:
        labels_per_class = _prompt_int("Labels per class", 100)

    epochs = args.epochs if args.epochs is not None else _prompt_int("Epochs", 300)
    lr = args.learning_rate
    batch_size = args.batch_size

    if args.build_image:
        print("Building docker image...")
        subprocess.run(["docker", "build", "-t", args.image, "-f", "docker/Dockerfile", "."], check=True)

    use_gpu = _cuda_available_in_docker()
    if not use_gpu and not args.cpu_fallback:
        raise SystemExit("CUDA not available inside Docker. Use --cpu-fallback to continue without GPU.")

    generated_cfg = _build_generated_config(
        algorithm=algorithm,
        dataset=dataset,
        model=model,
        labels_per_class=labels_per_class,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    generated_dir = Path("configs") / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    config_path = generated_dir / f"{generated_cfg['run_name']}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(generated_cfg, f, sort_keys=False)

    project_root = Path.cwd().resolve()
    mount = f"{project_root}:/workspace"
    train_script = "train_supervised_limited.py" if algorithm == "supervised" else "train_ssl.py"

    cmd = ["docker", "run", "--rm"]
    if use_gpu:
        cmd.extend(["--gpus", "all"])
    cmd.extend(["-v", mount, args.image, "python", train_script, "--config", config_path.as_posix()])

    print("\nLaunching training container:")
    print(" ".join(cmd))
    print(f"Generated config: {config_path.as_posix()}")

    if args.background:
        subprocess.Popen(cmd)
        print("Training started in background container process.")
    else:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
