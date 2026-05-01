from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "configs" / "benchmarks"
IMAGE_NAME = "fyp-ssl:latest"
CONTAINER_WORKDIR = "/workspace"

DISPLAY_NAMES = {
    "cifar10": "Cifar-10",
    "cifar100": "Cifar-100",
    "supervised": "Supervised",
    "pseudolabel": "Pseudo-Label",
    "fixmatch": "FixMatch",
    "mixmatch": "MixMatch",
    "flexmatch": "FlexMatch",
    "40labels": "40",
    "250labels": "250",
    "1000labels": "1000",
    "4000labels": "4000",
    "1percent_val": "1 Percent Val",
}

TRAINERS = {
    "supervised": "train_supervised_limited.py",
    "pseudolabel": "train_ssl.py",
    "fixmatch": "train_ssl.py",
    "mixmatch": "train_ssl.py",
    "flexmatch": "train_ssl.py",
}

FILENAME_PATTERN = re.compile(r"^(?P<prefix>supervised_limited|ssl_[a-z]+)_(?P<dataset>[a-z0-9]+)_(?P<split>.+)\.yaml$")


def discover_runs() -> dict[tuple[str, str, str], Path]:
    runs: dict[tuple[str, str, str], Path] = {}
    for path in sorted(BENCHMARK_DIR.glob("*.yaml")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue
        prefix = match.group("prefix")
        dataset = match.group("dataset")
        split = match.group("split")
        algorithm = "supervised" if prefix == "supervised_limited" else prefix.removeprefix("ssl_")
        runs[(dataset, algorithm, split)] = path
    return runs


def display_name(value: str) -> str:
    return DISPLAY_NAMES.get(value, value.replace("_", " ").title())


def ask_choice(prompt: str, options: list[str]) -> str:
    print()
    print(prompt)
    for index, option in enumerate(options, start=1):
        print(f"[{index}] {display_name(option)}")

    while True:
        answer = input("Enter a number: ").strip()
        if answer.isdigit():
            choice = int(answer)
            if 1 <= choice <= len(options):
                return options[choice - 1]
        print("Please enter one of the listed numbers.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{prompt} {suffix} ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def ask_extra_args() -> list[str]:
    raw = input("Extra training args (optional, press Enter to skip): ").strip()
    if not raw:
        return []
    return shlex.split(raw)


def build_parser(runs: dict[tuple[str, str, str], Path]) -> argparse.ArgumentParser:
    datasets = sorted({dataset for dataset, _, _ in runs})
    algorithms = sorted({algorithm for _, algorithm, _ in runs})

    parser = argparse.ArgumentParser(description="Launch a benchmark run in Docker")
    parser.add_argument("--dataset", choices=datasets, help="Dataset to run")
    parser.add_argument("--algorithm", choices=algorithms, help="Algorithm to run")
    parser.add_argument("--split", help="Benchmark split to use")
    parser.add_argument("--image", default=IMAGE_NAME, help="Docker image to run")
    parser.add_argument("--dry-run", action="store_true", help="Print the Docker command without running it")
    parser.add_argument("--list", action="store_true", help="Show available runs and exit")
    return parser


def choose_run(runs: dict[tuple[str, str, str], Path], args: argparse.Namespace) -> tuple[str, str, str]:
    datasets = sorted({dataset for dataset, _, _ in runs})
    if not args.dataset:
        args.dataset = ask_choice("What Dataset Would you like to train on?", datasets)

    available_algorithms = sorted({algorithm for dataset, algorithm, _ in runs if dataset == args.dataset})
    if not args.algorithm:
        args.algorithm = ask_choice("What Algorithm would you like to train?", available_algorithms)

    available_splits = sorted(
        split for dataset, algorithm, split in runs if dataset == args.dataset and algorithm == args.algorithm
    )
    if not available_splits:
        raise SystemExit(f"No benchmark splits found for dataset={args.dataset!r} and algorithm={args.algorithm!r}")
    if not args.split:
        args.split = ask_choice("What Data Split?", available_splits)

    return args.dataset, args.algorithm, args.split


def main() -> int:
    runs = discover_runs()
    parser = build_parser(runs)
    args, extra_args = parser.parse_known_args()

    if args.list:
        print()
        print("Available runs:")
        for dataset, algorithm, split in sorted(runs):
            print(f"  {display_name(dataset):10}  {display_name(algorithm):11}  {display_name(split)}")
        return 0

    dataset, algorithm, split = choose_run(runs, args)
    config_path = runs.get((dataset, algorithm, split))
    if config_path is None:
        raise SystemExit(f"No benchmark config found for dataset={dataset!r}, algorithm={algorithm!r}, split={split!r}")

    use_gpu = ask_yes_no("Use GPU?", default=True)
    if not extra_args:
        extra_args = ask_extra_args()

    if shutil.which("docker") is None:
        print("Docker is not installed or not on PATH.", file=sys.stderr)
        return 1

    trainer_script = TRAINERS[algorithm]
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"{algorithm}_{split}",
        "-v",
        f"{ROOT}:{CONTAINER_WORKDIR}",
        "-w",
        CONTAINER_WORKDIR,
    ]
    if use_gpu:
        command.extend(["--gpus", "all"])
    command.extend([
        args.image,
        "python",
        trainer_script,
        "--config",
        config_path.relative_to(ROOT).as_posix(),
        *extra_args,
    ])

    print()
    print("Running:")
    print(" ".join(command))

    if args.dry_run:
        return 0

    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
