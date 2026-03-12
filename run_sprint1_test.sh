#!/bin/bash

# Sprint 1 - Test Supervised Training in Docker Container

echo "=============================================="
echo "Sprint 1: Testing Supervised Training Pipeline"
echo "=============================================="

echo ""
echo "Running quick validation test (2 epochs)..."
python test_supervised.py

echo ""
echo "=============================================="
echo "If the test passed, you can run full training:"
echo "  python train_supervised.py --epochs 200"
echo "=============================================="
