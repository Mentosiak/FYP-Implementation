"""
Verification script to check all imports and basic functionality.
Run this to ensure everything is properly installed and configured.
"""

import sys
import importlib


def check_import(module_name, item_name=None):
    """Check if a module/item can be imported."""
    try:
        module = importlib.import_module(module_name)
        if item_name:
            getattr(module, item_name)
            print(f"✅ {module_name}.{item_name}")
        else:
            print(f"✅ {module_name}")
        return True
    except Exception as e:
        print(f"❌ {module_name}{('.' + item_name) if item_name else ''}: {e}")
        return False


def main():
    print("="*60)
    print("FYP Implementation - Verification Script")
    print("="*60)
    
    all_ok = True
    
    # Check core dependencies
    print("\n📦 Checking core dependencies...")
    deps = ['torch', 'torchvision', 'numpy', 'yaml', 'matplotlib']
    for dep in deps:
        all_ok &= check_import(dep)
    
    # Check project modules
    print("\n🔧 Checking project modules...")
    
    # Data module
    print("\nData module:")
    all_ok &= check_import('src.data', 'get_cifar10_loaders')
    all_ok &= check_import('src.data', 'get_cifar_supervised_loaders')
    all_ok &= check_import('src.data', 'get_cifar_ssl_loaders')
    all_ok &= check_import('src.data', 'SplitConfig')
    
    # Models module
    print("\nModels module:")
    all_ok &= check_import('src.models', 'ResNet18')
    all_ok &= check_import('src.models', 'ResNet34')
    all_ok &= check_import('src.models', 'WideResNet')
    all_ok &= check_import('src.models', 'build_model')
    
    # Training module
    print("\nTraining module:")
    all_ok &= check_import('src.training', 'SupervisedTrainer')
    all_ok &= check_import('src.training', 'PseudoLabelTrainer')
    
    # Utils module
    print("\nUtils module:")
    all_ok &= check_import('src.utils', 'set_seed')
    all_ok &= check_import('src.utils', 'get_device')
    all_ok &= check_import('src.utils', 'get_logger')
    all_ok &= check_import('src.utils', 'load_config')
    all_ok &= check_import('src.utils', 'ExperimentConfig')
    
    # Check config files
    print("\n📝 Checking configuration files...")
    import os
    configs = [
        'configs/supervised_cifar10.yaml',
        'configs/supervised_limited_cifar10.yaml',
        'configs/ssl_pseudolabel_cifar10.yaml',
    ]
    for config_path in configs:
        if os.path.exists(config_path):
            print(f"✅ {config_path}")
        else:
            print(f"❌ {config_path} not found")
            all_ok = False
    
    # Test config loading
    print("\n⚙️  Testing config loading...")
    try:
        from src.utils import load_config
        config = load_config('configs/supervised_cifar10.yaml')
        print(f"✅ Loaded config: {config.run_name}")
        print(f"   Dataset: {config.dataset.name}")
        print(f"   Model: {config.model.architecture}")
        print(f"   Epochs: {config.training.epochs}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        all_ok = False
    
    # Test model building
    print("\n🏗️  Testing model building...")
    try:
        from src.models import build_model
        from src.utils import load_config
        config = load_config('configs/supervised_cifar10.yaml')
        model = build_model(config.model, num_classes=10)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Built {config.model.architecture} with {param_count:,} parameters")
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        all_ok = False
    
    # Check CUDA
    print("\n🖥️  Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA not available (will use CPU)")
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✅ All checks passed! System is ready.")
        print("\nNext steps:")
        print("1. Run: python test_supervised.py")
        print("2. Run: python test_ssl.py")
        print("3. Start training: python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("="*60)


if __name__ == '__main__':
    main()
