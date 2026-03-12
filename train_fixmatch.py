"""Convenience entrypoint for FixMatch training."""

from __future__ import annotations

import argparse
import sys

from train_ssl import main as train_ssl_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default='configs/ssl_fixmatch_cifar10.yaml')
    known_args, remaining = parser.parse_known_args()

    # Rebuild argv so train_ssl.py receives this default config unless overridden.
    sys.argv = ['train_ssl.py', '--config', known_args.config, *remaining]
    train_ssl_main()
