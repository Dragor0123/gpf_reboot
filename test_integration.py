"""Integration test for dynamic anchor selection with existing training pipeline."""

import torch
import yaml
from core.config import ConfigManager
from training.losses import TargetCentricLoss
from core.logging import setup_logging


def test_config_loading():
    """Test if config loading works with dynamic anchor settings."""
    print("🧪 Testing config loading with dynamic anchor settings...")
    
    # Test loading config.yaml
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config("config.yaml")
        
        print(f"  ✅ Config loaded successfully")
        
        # Check dynamic anchor settings
        dynamic_config = config.get('dynamic_anchor', {})
        print(f"  📊 Dynamic anchor enabled: {dynamic_config.get('enable', False)}")
        print(f"  📊 Selection ratio: {dynamic_config.get('selection_ratio', 'N/A')}")
        print(f"  📊 Update frequency: {dynamic_config.get('update_frequency', 'N/A')}")
        
        # Check target_centric settings
        tc_config = config.get('target_centric', {})
        print(f"  📊 Target-centric enabled: {tc_config.get('enable', False)}")
        
        return config
        
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        return None


def test_loss_function_initialization():
    """Test TargetCentricLoss initialization with dynamic anchor config."""
    print("🧪 Testing TargetCentricLoss initialization...")
    
    # Load config
    config = test_config_loading()
    if config is None:
        return False
    
    try:
        # Initialize loss function
        loss_fn = TargetCentricLoss(config)
        print(f"  ✅ TargetCentricLoss initialized")
        print(f"  📊 Regularizer type: {loss_fn.regularizer_type}")
        
        # Get status
        status = loss_fn.get_regularizer_status()
        print(f"  📋 Status: {status}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ TargetCentricLoss initialization failed: {e}")
        return False


def test_synthetic_training_loop():
    """Test a few epochs of synthetic training with dynamic anchors."""
    print("🧪 Testing synthetic training loop with dynamic anchors...")
    
    # Load config
    config = test_config_loading()
    if config is None:
        return False
    
    try:
        # Initialize loss function
        loss_fn = TargetCentricLoss(config)
        
        # Create synthetic data
        num_nodes, num_classes = 100, 3
        hidden_dim = 64
        
        torch.manual_seed(42)
        embeddings = torch.randn(num_nodes, hidden_dim)
        logits = torch.randn(num_nodes, num_classes)
        labels = torch.randint(0, num_classes, (num_nodes,))
        
        # Create proper training mask
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_indices = torch.randperm(num_nodes)[:70]
        mask[train_indices] = True
        
        # Set fallback anchors if using dynamic anchors
        if loss_fn.regularizer_type == 'dynamic':
            fallback_anchors = torch.randn(50, hidden_dim)
            loss_fn.initialize_regularizer_with_fixed_anchors(fallback_anchors)
            print(f"  ✅ Fallback anchors set for dynamic regularizer")
        
        print(f"  🔄 Running synthetic training epochs...")
        
        # Run a few training epochs
        for epoch in range(15):  # Test across anchor update boundary
            # Forward pass
            loss_dict = loss_fn(
                logits=logits,
                labels=labels,
                embeddings=embeddings,
                mask=mask,
                epoch=epoch
            )
            
            # Log key information
            if epoch % 5 == 0:
                total_loss = loss_dict['total_loss'].item()
                task_loss = loss_dict['task_loss'].item()
                reg_loss = loss_dict['reg_loss'].item()
                
                log_msg = f"    Epoch {epoch:02d}: Total={total_loss:.4f}, Task={task_loss:.4f}, Reg={reg_loss:.4f}"
                
                if 'anchor_update_info' in loss_dict:
                    update_info = loss_dict['anchor_update_info']
                    if update_info['anchors_updated']:
                        log_msg += " | Anchors Updated"
                    if update_info['using_fallback']:
                        log_msg += " | Using Fallback"
                
                print(log_msg)
        
        print(f"  ✅ Synthetic training completed successfully")
        
        # Check final regularizer status
        if hasattr(loss_fn, 'get_regularizer_status'):
            final_status = loss_fn.get_regularizer_status()
            print(f"  📋 Final status: initialized={final_status.get('initialized', 'N/A')}")
            print(f"  📋 Using fallback: {final_status.get('using_fallback', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Synthetic training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_compatibility():
    """Test backward compatibility with existing config format."""
    print("🧪 Testing backward compatibility...")
    
    # Test with dynamic anchors disabled
    legacy_config = {
        'target_centric': {
            'enable': True,
            'regularization': {
                'weight': 0.1,
                'anchor': {'type': 'mog'},
                'divergence': {'type': 'mmd', 'params': {'sigma': 1.0}}
            }
        },
        'dynamic_anchor': {
            'enable': False  # Explicitly disabled
        }
    }
    
    try:
        loss_fn = TargetCentricLoss(legacy_config)
        print(f"  ✅ Legacy config works: type={loss_fn.regularizer_type}")
        
        # Test with no dynamic_anchor section at all
        minimal_config = {
            'target_centric': {
                'enable': False
            }
        }
        
        loss_fn_minimal = TargetCentricLoss(minimal_config)
        print(f"  ✅ Minimal config works: type={loss_fn_minimal.regularizer_type}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run integration tests."""
    print("🚀 Starting Integration Tests for Dynamic Anchor Selection\n")
    
    # Setup basic logging
    setup_logging("INFO")
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Loss Function Initialization", test_loss_function_initialization),
        ("Synthetic Training Loop", test_synthetic_training_loop),
        ("Backward Compatibility", test_config_compatibility)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{'='*60}")
        print(f"Testing: {test_name}")
        print('='*60)
        
        try:
            if test_name == "Config Loading":
                # Config loading returns config or None
                result = test_func()
                success = result is not None
            else:
                success = test_func()
            
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} PASSED\n")
            else:
                print(f"❌ {test_name} FAILED\n")
                
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print(f"{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} integration tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! Ready for end-to-end testing.")
    else:
        print("⚠️ Some integration tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)