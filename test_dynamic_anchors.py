"""Simple unit tests for dynamic anchor selection components."""

import torch
import torch.nn.functional as F
import numpy as np
from core.performance_evaluator import NodePerformanceEvaluator
from core.dynamic_anchor_selector import DynamicPerformanceAnchorSelector
from training.dynamic_regularizers import DynamicAnchorRegularizer


def test_node_performance_evaluator():
    """Test NodePerformanceEvaluator basic functionality."""
    print("🧪 Testing NodePerformanceEvaluator...")
    
    # Create synthetic data
    num_nodes, num_classes = 100, 3
    hidden_dim = 64
    
    # Create logits with some variation in quality
    torch.manual_seed(42)
    logits = torch.randn(num_nodes, num_classes)
    labels = torch.randint(0, num_classes, (num_nodes,))
    # Create proper boolean mask for training nodes
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_indices = torch.randperm(num_nodes)[:70]  # Select 70 random nodes
    mask[train_indices] = True
    
    # Make some nodes perform better
    good_nodes = torch.randperm(num_nodes)[:20]
    logits[good_nodes, labels[good_nodes]] += 2.0  # Boost correct class scores
    
    # Initialize evaluator
    evaluator = NodePerformanceEvaluator(loss_weight=0.7, confidence_weight=0.3)
    
    # Test basic scoring
    scores = evaluator.evaluate_node_performance(logits, labels, mask)
    print(f"  ✅ Performance scores computed: shape {scores.shape}")
    print(f"  📊 Score stats: mean={scores.mean():.3f}, std={scores.std():.3f}")
    
    # Test top-k selection
    top_indices = evaluator.get_top_performing_indices(logits, labels, mask, selection_ratio=0.2)
    print(f"  ✅ Top indices selected: {len(top_indices)} nodes")
    
    # Test statistics
    stats = evaluator.compute_performance_statistics(logits, labels, mask)
    print(f"  📈 Overall accuracy: {stats['overall_accuracy']:.3f}")
    
    return True


def test_dynamic_anchor_selector():
    """Test DynamicPerformanceAnchorSelector basic functionality."""
    print("🧪 Testing DynamicPerformanceAnchorSelector...")
    
    # Create synthetic data
    num_nodes, num_classes = 100, 3
    hidden_dim = 64
    
    torch.manual_seed(42)
    embeddings = torch.randn(num_nodes, hidden_dim)
    logits = torch.randn(num_nodes, num_classes)
    labels = torch.randint(0, num_classes, (num_nodes,))
    # Create proper boolean mask for training nodes
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_indices = torch.randperm(num_nodes)[:70]  # Select 70 random nodes
    mask[train_indices] = True
    
    # Make some nodes perform better
    good_nodes = torch.randperm(num_nodes)[:20]
    logits[good_nodes, labels[good_nodes]] += 2.0
    
    # Initialize selector
    selector = DynamicPerformanceAnchorSelector(
        selection_ratio=0.2,
        update_frequency=5,
        loss_weight=0.7,
        confidence_weight=0.3
    )
    
    # Test anchor selection
    epoch = 0
    anchors, selection_info = selector.select_anchors_by_performance(
        embeddings, logits, labels, mask, epoch
    )
    print(f"  ✅ Anchors selected: shape {anchors.shape}")
    print(f"  📊 Selection info: {selection_info['num_anchors']} anchors at epoch {selection_info['epoch']}")
    
    # Test update frequency
    should_update_5 = selector.should_update_anchors(5)
    should_update_3 = selector.should_update_anchors(3)
    print(f"  ⏰ Update at epoch 5: {should_update_5}, epoch 3: {should_update_3}")
    
    # Test quality tracking
    if selector.quality_tracking:
        quality_checks = selector.check_anchor_quality(embeddings, mask)
        print(f"  🔍 Quality checks: {quality_checks}")
    
    return True


def test_dynamic_anchor_regularizer():
    """Test DynamicAnchorRegularizer basic functionality."""
    print("🧪 Testing DynamicAnchorRegularizer...")
    
    # Create test config
    config = {
        'dynamic_anchor': {
            'selection_ratio': 0.2,
            'update_frequency': 5,
            'criteria': {
                'loss_weight': 0.7,
                'confidence_weight': 0.3
            },
            'quality_tracking': True,
            'soft_update_momentum': 0.9,
            'fallback': {
                'enable': True,
                'method': 'mog',
                'conditions': {
                    'performance_drop': 0.05,
                    'anchor_diversity': 0.3,
                    'selection_instability': 0.5
                }
            }
        },
        'target_centric': {
            'regularization': {
                'weight': 0.1
            }
        },
        'divergence': {
            'type': 'mmd',
            'params': {'sigma': 1.0}
        },
        'beta': 0.1
    }
    
    # Initialize regularizer
    regularizer = DynamicAnchorRegularizer(config)
    print(f"  ✅ Regularizer initialized")
    
    # Create synthetic data
    num_nodes, num_classes = 100, 3
    hidden_dim = 64
    
    torch.manual_seed(42)
    embeddings = torch.randn(num_nodes, hidden_dim)
    logits = torch.randn(num_nodes, num_classes)
    labels = torch.randint(0, num_classes, (num_nodes,))
    # Create proper boolean mask for training nodes
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_indices = torch.randperm(num_nodes)[:70]  # Select 70 random nodes
    mask[train_indices] = True
    
    # Make some nodes perform better
    good_nodes = torch.randperm(num_nodes)[:20]
    logits[good_nodes, labels[good_nodes]] += 2.0
    
    # Set fallback anchors
    fallback_anchors = torch.randn(50, hidden_dim)
    regularizer.set_fallback_anchors(fallback_anchors)
    print(f"  ✅ Fallback anchors set: shape {fallback_anchors.shape}")
    
    # Test anchor update
    epoch = 0
    update_info = regularizer.update_anchors(embeddings, logits, labels, mask, epoch)
    print(f"  🔄 Anchor update completed: {update_info}")
    
    # Test regularization loss computation
    try:
        reg_loss = regularizer.compute_regularization_loss(embeddings)
        print(f"  📊 Regularization loss: {reg_loss.item():.4f}")
        print(f"  ✅ Loss computation successful")
    except Exception as e:
        print(f"  ❌ Loss computation failed: {e}")
        return False
    
    # Test status
    status = regularizer.get_regularizer_status()
    print(f"  📋 Regularizer status: initialized={status['initialized']}, using_fallback={status['using_fallback']}")
    
    return True


def test_target_centric_loss_integration():
    """Test TargetCentricLoss with dynamic anchor support."""
    print("🧪 Testing TargetCentricLoss with dynamic anchors...")
    
    from training.losses import TargetCentricLoss
    
    # Create test config for dynamic anchors
    config = {
        'dynamic_anchor': {
            'enable': True,
            'selection_ratio': 0.2,
            'update_frequency': 5,
            'criteria': {
                'loss_weight': 0.7,
                'confidence_weight': 0.3
            },
            'quality_tracking': True,
            'fallback': {
                'enable': True,
                'method': 'mog',
                'conditions': {
                    'performance_drop': 0.05,
                    'anchor_diversity': 0.3,
                    'selection_instability': 0.5
                }
            }
        },
        'target_centric': {
            'enable': True,
            'regularization': {
                'weight': 0.1,
                'divergence': {
                    'type': 'mmd',
                    'params': {'sigma': 1.0}
                }
            }
        }
    }
    
    # Initialize loss function
    loss_fn = TargetCentricLoss(config)
    print(f"  ✅ TargetCentricLoss initialized with type: {loss_fn.regularizer_type}")
    
    # Create synthetic data
    num_nodes, num_classes = 100, 3
    hidden_dim = 64
    
    torch.manual_seed(42)
    embeddings = torch.randn(num_nodes, hidden_dim)
    logits = torch.randn(num_nodes, num_classes)
    labels = torch.randint(0, num_classes, (num_nodes,))
    # Create proper boolean mask for training nodes
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_indices = torch.randperm(num_nodes)[:70]  # Select 70 random nodes
    mask[train_indices] = True
    
    # Set fallback anchors
    fallback_anchors = torch.randn(50, hidden_dim)
    loss_fn.initialize_regularizer_with_fixed_anchors(fallback_anchors)
    print(f"  ✅ Fallback anchors set")
    
    # Test forward pass
    epoch = 0
    loss_dict = loss_fn(logits, labels, embeddings, mask, epoch=epoch)
    
    print(f"  📊 Loss components:")
    print(f"    Task loss: {loss_dict['task_loss'].item():.4f}")
    print(f"    Reg loss: {loss_dict['reg_loss'].item():.4f}")
    print(f"    Total loss: {loss_dict['total_loss'].item():.4f}")
    
    if 'anchor_update_info' in loss_dict:
        update_info = loss_dict['anchor_update_info']
        print(f"    Anchors updated: {update_info['anchors_updated']}")
        print(f"    Using fallback: {update_info['using_fallback']}")
    
    # Test status
    status = loss_fn.get_regularizer_status()
    print(f"  📋 Status: type={status['type']}, enabled={status['enabled']}")
    
    return True


def main():
    """Run all tests."""
    print("🚀 Starting Dynamic Anchor Selection Tests\n")
    
    tests = [
        ("NodePerformanceEvaluator", test_node_performance_evaluator),
        ("DynamicPerformanceAnchorSelector", test_dynamic_anchor_selector),
        ("DynamicAnchorRegularizer", test_dynamic_anchor_regularizer),
        ("TargetCentricLoss Integration", test_target_centric_loss_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dynamic anchor selection is ready.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)