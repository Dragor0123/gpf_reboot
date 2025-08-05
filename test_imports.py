#!/usr/bin/env python3
"""Simple test script to validate import structure."""

import sys
import traceback

def test_import(module_name):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except Exception as e:
        print(f"❌ {module_name} failed to import: {e}")
        traceback.print_exc()
        return False

def main():
    """Test all critical imports."""
    modules_to_test = [
        'core',
        'core.config',
        'core.device', 
        'core.logging',
        'core.reproducibility',
        'models',
        'datasets',
        'prompts',
        'training',
        'training.losses',
        'training.regularizers',
    ]
    
    print("Testing import structure...")
    print("=" * 50)
    
    failed_imports = []
    
    for module in modules_to_test:
        if not test_import(module):
            failed_imports.append(module)
    
    print("=" * 50)
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} modules failed to import:")
        for module in failed_imports:
            print(f"  - {module}")
        return False
    else:
        print("✅ All modules imported successfully!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)