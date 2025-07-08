#!/usr/bin/env python3
"""
긴급 데이터 검증 스크립트
CUDA Index Out of Bounds 원인 파악
"""

import torch
from datasets.load_dataset import _load_single_dataset, _create_splits
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_dataset_integrity(dataset_name):
    """데이터셋의 edge_index 무결성 검사"""
    print(f"\n{'='*60}")
    print(f"🔍 Checking dataset: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # 원본 데이터 로드
        dataset = _load_single_dataset(dataset_name)
        data = dataset[0]
        
        # 기본 정보
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        num_features = data.num_node_features
        
        print(f"📊 Basic Info:")
        print(f"   Nodes: {num_nodes}")
        print(f"   Edges: {num_edges}")
        print(f"   Features: {num_features}")
        
        # Edge index 검사
        edge_index = data.edge_index
        print(f"\n🔍 Edge Index Analysis:")
        print(f"   Shape: {edge_index.shape}")
        print(f"   Min value: {edge_index.min().item()}")
        print(f"   Max value: {edge_index.max().item()}")
        print(f"   Valid range: [0, {num_nodes - 1}]")
        
        # 🚨 핵심 체크: Index Out of Bounds 검사
        max_node_id = edge_index.max().item()
        min_node_id = edge_index.min().item()
        
        if max_node_id >= num_nodes:
            print(f"❌ INDEX OUT OF BOUNDS DETECTED!")
            print(f"   Max node ID in edges: {max_node_id}")
            print(f"   But only {num_nodes} nodes exist (0~{num_nodes-1})")
            
            # 문제가 있는 edge 개수 계산
            invalid_edges = ((edge_index >= num_nodes) | (edge_index < 0)).any(dim=0).sum()
            print(f"   Invalid edges: {invalid_edges.item()}/{num_edges}")
            
            return False
        
        if min_node_id < 0:
            print(f"❌ NEGATIVE INDEX DETECTED!")
            print(f"   Min node ID: {min_node_id}")
            return False
        
        print(f"✅ Edge index is valid!")
        
        # Split 후에도 검사
        print(f"\n🔍 After Split:")
        data_split = _create_splits(data, val_ratio=0.1, test_ratio=0.2, shuffle=True)
        
        # Split 후에도 edge_index는 동일해야 함
        if not torch.equal(data.edge_index, data_split.edge_index):
            print(f"⚠️  Edge index changed after split!")
        else:
            print(f"✅ Edge index unchanged after split")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return False


def main():
    """모든 데이터셋 검사"""
    datasets = ['cora', 'citeseer', 'pubmed', 'computers', 'photo']
    
    print("🚨 EMERGENCY DATA INTEGRITY CHECK")
    print("Looking for CUDA Index Out of Bounds causes...")
    
    results = {}
    
    for dataset_name in datasets:
        try:
            is_valid = check_dataset_integrity(dataset_name)
            results[dataset_name] = is_valid
        except Exception as e:
            print(f"❌ Failed to check {dataset_name}: {e}")
            results[dataset_name] = False
    
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    for dataset, is_valid in results.items():
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"{dataset:>10}: {status}")
    
    # 문제가 있는 데이터셋 식별
    invalid_datasets = [name for name, valid in results.items() if not valid]
    
    if invalid_datasets:
        print(f"\n🚨 PROBLEMATIC DATASETS: {invalid_datasets}")
        print("These datasets likely cause CUDA Index Out of Bounds!")
        
        print(f"\n🔧 RECOMMENDED FIXES:")
        print("1. Add edge index cleaning in data loading")
        print("2. Remove invalid edges before training")
        print("3. Check original dataset files")
    else:
        print(f"\n🤔 All datasets seem valid...")
        print("The Index Out of Bounds might be caused by:")
        print("1. Data augmentation (edge dropping)")
        print("2. Graph transformation during training")
        print("3. Device transfer issues")


if __name__ == "__main__":
    main()
    