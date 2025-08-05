#!/usr/bin/env python3
"""
단계별 학습 과정 디버깅
CUDA Index Out of Bounds 정확한 발생 지점 추적
"""

import torch
import logging
from pathlib import Path
import os

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils import (
    set_seed, get_device, load_config,
    setup_logging, log_experiment_info
)
from models import create_model, Classifier
from datasets import load_dataset
from prompts.prompt_function import GPFPrompt
from losses import TargetCentricLoss


def safe_tensor_info(tensor, name):
    """안전한 텐서 정보 출력"""
    if tensor is None:
        print(f"❌ {name}: None")
        return False
    
    try:
        print(f"🔍 {name}:")
        print(f"   Shape: {tensor.shape}")
        print(f"   Device: {tensor.device}")
        print(f"   Dtype: {tensor.dtype}")
        
        if tensor.numel() > 0:
            print(f"   Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
            print(f"   Has NaN: {torch.isnan(tensor).any().item()}")
            print(f"   Has Inf: {torch.isinf(tensor).any().item()}")
        
        return True
    except Exception as e:
        print(f"❌ Error inspecting {name}: {e}")
        return False


def debug_pretraining_step():
    """Pre-training 단계 디버깅"""
    print("\n" + "="*80)
    print("🔍 DEBUGGING PRE-TRAINING STEP")
    print("="*80)
    
    # Config 로드
    config = load_config("config.yaml")
    set_seed(config['experiment']['seed'])
    device = get_device(config['experiment']['device'])
    
    print(f"Device: {device}")
    print(f"Source dataset: {config['experiment']['source_dataset']}")
    print(f"Target dataset: {config['experiment']['target_dataset']}")
    
    try:
        # Source 데이터 로드
        print("\n🔍 Loading source dataset...")
        (source_info, target_info, source_loader, _, _, _, 
         source_svd_reducer) = load_dataset(config)
        
        source_data = next(iter(source_loader))
        source_data = source_data.to(device)
        
        print(f"✅ Source data loaded successfully")
        safe_tensor_info(source_data.x, "Source features")
        safe_tensor_info(source_data.edge_index, "Source edge_index")
        
        # Edge index 검증
        num_nodes = source_data.x.size(0)
        edge_max = source_data.edge_index.max().item() if source_data.edge_index.numel() > 0 else -1
        edge_min = source_data.edge_index.min().item() if source_data.edge_index.numel() > 0 else -1
        
        print(f"🔍 Source edge validation:")
        print(f"   Num nodes: {num_nodes}")
        print(f"   Edge range: [{edge_min}, {edge_max}]")
        print(f"   Valid: {edge_max < num_nodes and edge_min >= 0}")
        
        if edge_max >= num_nodes:
            print(f"❌ FOUND ISSUE: Edge index {edge_max} >= num_nodes {num_nodes}")
            return False
        
        # 간단한 모델 생성 및 테스트
        print(f"\n🔍 Testing model creation...")
        encoder = create_model(
            model_type=config['model']['type'],
            input_dim=source_info['num_features'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        ).to(device)
        
        print(f"✅ Model created successfully")
        
        # Forward pass 테스트
        print(f"\n🔍 Testing forward pass...")
        encoder.eval()
        
        with torch.no_grad():
            try:
                h = encoder(source_data.x, source_data.edge_index)
                safe_tensor_info(h, "Encoder output")
                print(f"✅ Forward pass successful")
                return True
            except RuntimeError as e:
                print(f"❌ Forward pass failed: {e}")
                return False
        
    except Exception as e:
        print(f"❌ Pre-training debug failed: {e}")
        return False


def debug_prompt_tuning_step():
    """Prompt tuning 단계 디버깅"""
    print("\n" + "="*80)
    print("🔍 DEBUGGING PROMPT TUNING STEP")
    print("="*80)
    
    # Config 로드
    config = load_config("config.yaml")
    set_seed(config['experiment']['seed'])
    device = get_device(config['experiment']['device'])
    
    try:
        # Target 데이터 로드
        print("\n🔍 Loading target dataset...")
        (source_info, target_info, source_loader, target_train_loader, 
         target_val_loader, target_test_loader, source_svd_reducer) = load_dataset(config)
        
        target_data = next(iter(target_train_loader))
        target_data = target_data.to(device)
        
        print(f"✅ Target data loaded successfully")
        safe_tensor_info(target_data.x, "Target features")
        safe_tensor_info(target_data.edge_index, "Target edge_index")
        
        # Edge index 검증
        num_nodes = target_data.x.size(0)
        edge_max = target_data.edge_index.max().item() if target_data.edge_index.numel() > 0 else -1
        edge_min = target_data.edge_index.min().item() if target_data.edge_index.numel() > 0 else -1
        
        print(f"🔍 Target edge validation:")
        print(f"   Num nodes: {num_nodes}")
        print(f"   Edge range: [{edge_min}, {edge_max}]")
        print(f"   Valid: {edge_max < num_nodes and edge_min >= 0}")
        
        if edge_max >= num_nodes:
            print(f"❌ FOUND ISSUE: Edge index {edge_max} >= num_nodes {num_nodes}")
            return False
        
        # Encoder 로드
        source_dataset = config['experiment']['source_dataset']
        checkpoint_path = f"checkpoints/{source_dataset}_encoder_final.pt"
        
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("   Run pre-training first!")
            return False
        
        print(f"🔍 Loading encoder from: {checkpoint_path}")
        
        encoder = create_model(
            model_type=config['model']['type'],
            input_dim=target_info['num_features'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        ).to(device)
        
        # Checkpoint 로드
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['model_state_dict'], strict=True)
        encoder.eval()
        
        print(f"✅ Encoder loaded successfully")
        
        # Prompt 생성
        print(f"\n🔍 Creating prompt...")
        prompt = GPFPrompt(
            input_dim=target_info['num_features'],
            p_num=config['prompt']['num_prompts']
        ).to(device)
        
        print(f"✅ Prompt created successfully")
        
        # Classifier 생성
        classifier = Classifier(
            input_dim=config['model']['hidden_dim'],
            num_classes=target_info['num_classes']
        ).to(device)
        
        print(f"✅ Classifier created successfully")
        
        # Step-by-step forward pass
        print(f"\n🔍 Step-by-step forward pass...")
        
        x = target_data.x
        edge_index = target_data.edge_index
        
        # Step 1: Prompt
        print(f"Step 1: Applying prompt...")
        safe_tensor_info(x, "Input features")
        
        try:
            prompted_x = prompt.add(x)
            safe_tensor_info(prompted_x, "Prompted features")
            print(f"✅ Prompt applied successfully")
        except Exception as e:
            print(f"❌ Prompt failed: {e}")
            return False
        
        # Step 2: Encoder
        print(f"Step 2: Encoder forward...")
        safe_tensor_info(prompted_x, "Encoder input")
        safe_tensor_info(edge_index, "Edge index")
        
        try:
            with torch.no_grad():
                h = encoder(prompted_x, edge_index)
            safe_tensor_info(h, "Encoder output")
            print(f"✅ Encoder forward successful")
        except RuntimeError as e:
            print(f"❌ Encoder forward failed: {e}")
            print(f"   This is likely where CUDA Index Out of Bounds occurs!")
            
            # 더 자세한 정보
            print(f"🔍 Detailed debug info:")
            print(f"   Prompted_x shape: {prompted_x.shape}")
            print(f"   Edge_index shape: {edge_index.shape}")
            print(f"   Edge_index device: {edge_index.device}")
            print(f"   Prompted_x device: {prompted_x.device}")
            
            if edge_index.numel() > 0:
                print(f"   Edge_index min: {edge_index.min()}")
                print(f"   Edge_index max: {edge_index.max()}")
                print(f"   Num nodes (from features): {prompted_x.size(0)}")
            
            return False
        
        # Step 3: Classifier
        print(f"Step 3: Classifier forward...")
        
        try:
            logits = classifier(h)
            safe_tensor_info(logits, "Classifier output")
            print(f"✅ Classifier forward successful")
        except Exception as e:
            print(f"❌ Classifier failed: {e}")
            return False
        
        print(f"\n✅ All forward pass steps completed successfully!")
        
        # Target-Centric 테스트 (문제가 여기서 발생할 수도 있음)
        if config['target_centric']['enable']:
            print(f"\n🔍 Testing Target-Centric initialization...")
            
            try:
                loss_fn = TargetCentricLoss(config).to(device)
                loss_fn.initialize_regularizer_with_target_features(
                    target_data.x, encoder, target_data.edge_index
                )
                print(f"✅ Target-Centric initialized successfully")
            except Exception as e:
                print(f"❌ Target-Centric failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt tuning debug failed: {e}")
        return False


def main():
    """전체 디버깅 실행"""
    print("🚨 COMPREHENSIVE DEBUGGING SESSION")
    print("Tracking CUDA Index Out of Bounds...")
    
    # 1. Pre-training 단계 체크
    pretrain_ok = debug_pretraining_step()
    
    # 2. Prompt tuning 단계 체크
    prompt_ok = debug_prompt_tuning_step()
    
    print("\n" + "="*80)
    print("📋 DEBUGGING SUMMARY")
    print("="*80)
    print(f"Pre-training step: {'✅ OK' if pretrain_ok else '❌ FAILED'}")
    print(f"Prompt tuning step: {'✅ OK' if prompt_ok else '❌ FAILED'}")
    
    if not pretrain_ok:
        print(f"\n🚨 ISSUE FOUND IN PRE-TRAINING!")
        print("Check source dataset or model creation")
    elif not prompt_ok:
        print(f"\n🚨 ISSUE FOUND IN PROMPT TUNING!")
        print("This is likely where CUDA Index Out of Bounds occurs")
        print("Check target dataset or encoder loading")
    else:
        print(f"\n🤔 Both steps seem OK individually...")
        print("The issue might occur during actual training loop")
        print("Try running with CUDA_LAUNCH_BLOCKING=1 for more details")


if __name__ == "__main__":
    main()