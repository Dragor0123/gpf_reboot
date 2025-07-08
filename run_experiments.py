#!/usr/bin/env python3
"""
Target-Centric Prior Modeling 전체 조합 실험 스크립트

실험 설계:
1. GPF Baseline (target_centric.enable = false)
2. GPF + Target-Centric Prior Modeling (target_centric.enable = true)

모든 가능한 source-target 조합에서 성능 비교 (5*4 = 20가지)
"""

import subprocess
import yaml
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import itertools
from datetime import datetime
import json


# 실험 설정
DATASETS = ['cora', 'citeseer', 'pubmed', 'computer', 'photo']
REGULARIZATION_TYPES = ['mmd', 'entropy', 'manifold']
BETA_VALUES = [0.01, 0.1, 0.5, 1.0]

# 모든 가능한 크로스 도메인 조합 생성 (source != target)
def generate_all_cross_domain_pairs(datasets: List[str]) -> List[Tuple[str, str]]:
    """
    모든 가능한 source-target 조합 생성
    
    Args:
        datasets: 데이터셋 리스트
    
    Returns:
        (source, target) 튜플 리스트
    """
    pairs = []
    for source in datasets:
        for target in datasets:
            if source != target:  # source와 target이 다른 경우만
                pairs.append((source, target))
    return pairs


CROSS_DOMAIN_PAIRS = generate_all_cross_domain_pairs(DATASETS)


def setup_experiment_logging():
    """실험 로깅 설정"""
    log_filename = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Experiment log file: {log_filename}")


def create_config(source_dataset: str, target_dataset: str, 
                 target_centric_enabled: bool = False, 
                 regularization_type: str = 'mmd',
                 beta: float = 0.1) -> Dict[str, Any]:
    """
    실험용 설정 생성
    
    Args:
        source_dataset: 소스 데이터셋
        target_dataset: 타겟 데이터셋
        target_centric_enabled: Target-centric 활성화 여부
        regularization_type: 정규화 타입
        beta: 정규화 가중치
    
    Returns:
        실험 설정 딕셔너리
    """
    config = {
        'experiment': {
            'type': 'cross_domain',
            'source_dataset': source_dataset,
            'target_dataset': target_dataset,
            'seed': 42,
            'device': 'auto',
            'log_level': 'INFO'
        },
        'dataset': {
            'split': {
                'val_ratio': 0.1,
                'test_ratio': 0.2,
                'shuffle': True
            }
        },
        'model': {
            'type': 'gin',
            'hidden_dim': 128,
            'num_layers': 5,
            'dropout': 0.5
        },
        'pretrain': {
            'lr': 0.001,
            'weight_decay': 0.0005,
            'epochs': 1000,
            'save_path': f'checkpoints/{source_dataset}_encoder.pt',
            'augmentation': {
                'view1': 'dropN',
                'view2': 'permE',
                'aug_ratio': 0.2,
                'temperature': 0.5
            }
        },
        'prompt_tuning': {
            'lr': 0.01,
            'weight_decay': 0.0005,
            'epochs': 200,
            'early_stopping': {
                'enable': True,
                'patience': 30,
                'min_delta': 0.001
            }
        },
        'prompt': {
            'type': 'gpf_plus',
            'num_prompts': 10
        },
        'target_centric': {
            'enable': target_centric_enabled,
            'regularization': {
                'type': regularization_type,
                'beta': beta,
                'mmd': {
                    'sigma': 1.0,
                    'anchor_type': 'random',
                    'num_anchors': 100
                },
                'entropy': {
                    'temperature': 1.0
                },
                'manifold': {
                    'neighbor_k': 5
                }
            }
        },
        'evaluation': {
            'metrics': ['accuracy', 'f1_macro', 'f1_micro'],
            'save_results': True,
            'results_dir': 'results'
        }
    }
    
    return config


def save_config(config: Dict[str, Any], filepath: str):
    """설정을 YAML 파일로 저장"""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_single_experiment(config: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    """
    단일 실험 실행
    
    Args:
        config: 실험 설정
        experiment_name: 실험 이름
    
    Returns:
        실험 결과
    """
    logging.info(f"🚀 Starting experiment: {experiment_name}")
    
    # 설정 파일 저장
    config_path = f"config_temp_{experiment_name}.yaml"
    save_config(config, config_path)
    
    try:
        # 현재 config.yaml을 백업
        if Path("config.yaml").exists():
            subprocess.run(["cp", "config.yaml", "config_backup.yaml"], check=True)
        
        # 임시 설정을 메인 설정으로 복사
        subprocess.run(["cp", config_path, "config.yaml"], check=True)
        
        # 사전훈련 실행 (source 데이터셋에서)
        logging.info(f"   Running pretraining on {config['experiment']['source_dataset']}...")
        
        start_time = time.time()
        result = subprocess.run(
            ["python", "train_pretrain.py"], 
            capture_output=True, 
            text=True,
            timeout=3600  # 1시간 타임아웃
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Pretraining failed: {result.stderr}")
        
        pretrain_time = time.time() - start_time
        logging.info(f"   Pretraining completed in {pretrain_time:.1f}s")
        
        # 프롬프트 튜닝 실행 (target 데이터셋에서)
        logging.info(f"   Running prompt tuning on {config['experiment']['target_dataset']}...")
        
        start_time = time.time()
        result = subprocess.run(
            ["python", "train_prompt_tuning.py"], 
            capture_output=True, 
            text=True,
            timeout=1800  # 30분 타임아웃
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Prompt tuning failed: {result.stderr}")
        
        tuning_time = time.time() - start_time
        logging.info(f"   Prompt tuning completed in {tuning_time:.1f}s")
        
        # 결과 파싱 시도
        test_metrics = parse_results_from_output(result.stdout)
        
        logging.info(f"✅ Completed experiment: {experiment_name}")
        if test_metrics:
            logging.info(f"   Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
        
        return {
            'status': 'success',
            'experiment_name': experiment_name,
            'pretrain_time': pretrain_time,
            'tuning_time': tuning_time,
            'total_time': pretrain_time + tuning_time,
            'test_metrics': test_metrics
        }
        
    except subprocess.TimeoutExpired:
        logging.error(f"❌ Experiment {experiment_name} timed out")
        return {
            'status': 'timeout',
            'experiment_name': experiment_name,
            'error': 'Experiment timed out'
        }
        
    except Exception as e:
        logging.error(f"❌ Failed experiment {experiment_name}: {str(e)}")
        return {
            'status': 'failed',
            'experiment_name': experiment_name,
            'error': str(e)
        }
    
    finally:
        # 설정 파일 정리
        Path(config_path).unlink(missing_ok=True)
        
        # 백업된 설정 복원
        if Path("config_backup.yaml").exists():
            subprocess.run(["mv", "config_backup.yaml", "config.yaml"], check=False)


def parse_results_from_output(output: str) -> Dict[str, float]:
    """
    실행 결과에서 메트릭 파싱
    """
    metrics = {}
    lines = output.split('\n')
    
    # "FINAL TEST RESULTS" 섹션 찾기
    in_results_section = False
    
    for line in lines:
        if "FINAL TEST RESULTS" in line:
            in_results_section = True
            continue
            
        if in_results_section and "=" in line and "=" * 10 in line:
            in_results_section = False
            continue
            
        if in_results_section and ":" in line:
            try:
                parts = line.split(":")
                if len(parts) == 2:
                    metric_name = parts[0].strip().lower()
                    metric_value = float(parts[1].strip())
                    metrics[metric_name] = metric_value
            except (ValueError, IndexError):
                continue
    
    return metrics


def run_baseline_vs_target_centric():
    """
    모든 조합에서 Baseline vs Target-Centric Prior Modeling 비교 실험
    """
    logging.info("🔬 Starting comprehensive Baseline vs Target-Centric comparison")
    logging.info(f"📊 Total combinations: {len(CROSS_DOMAIN_PAIRS)} (5×4 = 20)")
    
    results = []
    
    for i, (source, target) in enumerate(CROSS_DOMAIN_PAIRS, 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"📊 Combination {i}/{len(CROSS_DOMAIN_PAIRS)}: {source} → {target}")
        logging.info(f"{'='*60}")
        
        # 1. Baseline 실험 (Target-Centric 비활성화)
        logging.info(f"🔵 Running Baseline GPF...")
        baseline_config = create_config(
            source_dataset=source,
            target_dataset=target,
            target_centric_enabled=False
        )
        
        baseline_name = f"baseline_{source}_to_{target}"
        baseline_result = run_single_experiment(baseline_config, baseline_name)
        baseline_result.update({
            'experiment_type': 'baseline',
            'source': source,
            'target': target,
            'target_centric': False
        })
        results.append(baseline_result)
        
        # 2. Target-Centric 실험 (MMD 정규화)
        logging.info(f"🎯 Running Target-Centric GPF...")
        target_centric_config = create_config(
            source_dataset=source,
            target_dataset=target,
            target_centric_enabled=True,
            regularization_type='mmd',
            beta=0.1
        )
        
        target_centric_name = f"target_centric_{source}_to_{target}"
        target_centric_result = run_single_experiment(target_centric_config, target_centric_name)
        target_centric_result.update({
            'experiment_type': 'target_centric',
            'source': source,
            'target': target,
            'target_centric': True,
            'regularization_type': 'mmd',
            'beta': 0.1
        })
        results.append(target_centric_result)
        
        # 중간 결과 저장 (실험 중단 시 복구용)
        save_intermediate_results(results, f"intermediate_results_{i}.json")
        
        # 잠시 대기 (시스템 안정성)
        time.sleep(3)
    
    return results


def run_regularization_ablation():
    """
    정규화 타입별 ablation study (선택된 조합에서만)
    """
    logging.info("🔬 Starting regularization ablation study")
    
    # 대표적인 도메인 조합 선택 (너무 많으면 시간이 오래 걸림)
    selected_pairs = [
        ('cora', 'computer'),    # Citation → Amazon
        ('computer', 'cora'),    # Amazon → Citation
        ('citeseer', 'photo'),   # Citation → Amazon
        ('photo', 'citeseer'),   # Amazon → Citation
        ('pubmed', 'computer')   # Citation → Amazon
    ]
    
    logging.info(f"📊 Ablation on {len(selected_pairs)} representative pairs")
    
    results = []
    
    for source, target in selected_pairs:
        logging.info(f"\n📊 Ablation study: {source} → {target}")
        
        for reg_type in REGULARIZATION_TYPES:
            # 각 정규화 타입마다 중간 beta 값 하나만 테스트
            beta = 0.1
            
            config = create_config(
                source_dataset=source,
                target_dataset=target,
                target_centric_enabled=True,
                regularization_type=reg_type,
                beta=beta
            )
            
            experiment_name = f"ablation_{source}_to_{target}_{reg_type}_beta{beta}"
            result = run_single_experiment(config, experiment_name)
            
            result.update({
                'experiment_type': 'ablation',
                'source': source,
                'target': target,
                'target_centric': True,
                'regularization_type': reg_type,
                'beta': beta
            })
            results.append(result)
            
            time.sleep(2)
    
    return results


def save_intermediate_results(results: List[Dict[str, Any]], filename: str):
    """중간 결과 저장"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def analyze_results(results: List[Dict[str, Any]]):
    """
    실험 결과 분석 및 요약
    """
    logging.info("📈 Analyzing experimental results...")
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        logging.warning("No results to analyze")
        return
    
    # 성공한 실험만 필터링
    successful_df = df[df['status'] == 'success'].copy()
    
    logging.info(f"Successful experiments: {len(successful_df)}/{len(df)}")
    
    # 기본 통계
    logging.info("\n" + "="*60)
    logging.info("📊 EXPERIMENT SUMMARY")
    logging.info("="*60)
    logging.info(f"Total experiments: {len(df)}")
    logging.info(f"Successful: {len(successful_df)}")
    logging.info(f"Failed: {len(df[df['status'] == 'failed'])}")
    logging.info(f"Timeout: {len(df[df['status'] == 'timeout'])}")
    
    if len(successful_df) > 0:
        # Baseline vs Target-Centric 비교
        baseline_df = successful_df[successful_df['experiment_type'] == 'baseline']
        tc_df = successful_df[successful_df['experiment_type'] == 'target_centric']
        
        if len(baseline_df) > 0 and len(tc_df) > 0:
            logging.info("\n📊 BASELINE vs TARGET-CENTRIC COMPARISON:")
            
            # 평균 성능
            baseline_acc = [r.get('accuracy', 0) for r in baseline_df['test_metrics'] if r]
            tc_acc = [r.get('accuracy', 0) for r in tc_df['test_metrics'] if r]
            
            if baseline_acc and tc_acc:
                baseline_mean = sum(baseline_acc) / len(baseline_acc)
                tc_mean = sum(tc_acc) / len(tc_acc)
                
                logging.info(f"Baseline Average Accuracy: {baseline_mean:.4f}")
                logging.info(f"Target-Centric Average Accuracy: {tc_mean:.4f}")
                
                improvement = (tc_mean - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0
                logging.info(f"Average Improvement: {improvement:+.2f}%")
                
                # 개별 조합별 결과
                logging.info("\n📋 DETAILED RESULTS BY TRANSFER PAIR:")
                
                for source in DATASETS:
                    for target in DATASETS:
                        if source == target:
                            continue
                            
                        baseline_result = baseline_df[
                            (baseline_df['source'] == source) & 
                            (baseline_df['target'] == target)
                        ]
                        tc_result = tc_df[
                            (tc_df['source'] == source) & 
                            (tc_df['target'] == target)
                        ]
                        
                        if len(baseline_result) > 0 and len(tc_result) > 0:
                            baseline_metrics = baseline_result.iloc[0]['test_metrics']
                            tc_metrics = tc_result.iloc[0]['test_metrics']
                            
                            if baseline_metrics and tc_metrics:
                                baseline_acc = baseline_metrics.get('accuracy', 0)
                                tc_acc = tc_metrics.get('accuracy', 0)
                                
                                if baseline_acc > 0:
                                    improvement = (tc_acc - baseline_acc) / baseline_acc * 100
                                    status = "🟢" if improvement > 0 else "🔴"
                                    
                                    logging.info(f"{status} {source:>8} → {target:<8}: "
                                               f"Baseline={baseline_acc:.4f}, "
                                               f"TC={tc_acc:.4f}, "
                                               f"Δ={improvement:+.2f}%")
    
    # 실패한 실험 로깅
    failed_df = df[df['status'] != 'success']
    if len(failed_df) > 0:
        logging.info("\n⚠️  FAILED EXPERIMENTS:")
        for _, row in failed_df.iterrows():
            logging.info(f"   {row['experiment_name']}: {row.get('error', 'Unknown error')}")
    
    logging.info("="*60)


def main():
    """
    메인 실험 실행 함수
    """
    setup_experiment_logging()
    
    logging.info("🧪 Starting Comprehensive Target-Centric Prior Modeling Experiments")
    logging.info(f"📅 Experiment started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"🔢 Total cross-domain combinations: {len(CROSS_DOMAIN_PAIRS)}")
    
    # 결과 디렉토리 생성
    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    all_results = []
    
    try:
        # 1. 모든 조합에서 Baseline vs Target-Centric 비교
        logging.info("\n" + "="*80)
        logging.info("PHASE 1: Comprehensive Baseline vs Target-Centric Comparison")
        logging.info("="*80)
        
        baseline_results = run_baseline_vs_target_centric()
        all_results.extend(baseline_results)
        
        # 2. 선택된 조합에서 정규화 타입 ablation study
        logging.info("\n" + "="*80)
        logging.info("PHASE 2: Regularization Ablation Study")
        logging.info("="*80)
        
        ablation_results = run_regularization_ablation()
        all_results.extend(ablation_results)
        
        # 3. 결과 분석
        logging.info("\n" + "="*80)
        logging.info("PHASE 3: Results Analysis")
        logging.info("="*80)
        
        analyze_results(all_results)
        
        # 최종 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON 형식으로 상세 결과 저장
        results_json = f"results/comprehensive_results_{timestamp}.json"
        with open(results_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # CSV 형식으로 요약 저장
        results_df = pd.DataFrame(all_results)
        results_csv = f"results/comprehensive_summary_{timestamp}.csv"
        results_df.to_csv(results_csv, index=False)
        
        logging.info("🎉 All experiments completed successfully!")
        logging.info(f"📁 Detailed results: {results_json}")
        logging.info(f"📊 Summary CSV: {results_csv}")
        
    except KeyboardInterrupt:
        logging.info("🛑 Experiment interrupted by user")
        logging.info("💾 Saving partial results...")
        
        if all_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            partial_results = f"results/partial_results_{timestamp}.json"
            with open(partial_results, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logging.info(f"📁 Partial results saved: {partial_results}")
        
    except Exception as e:
        logging.error(f"❌ Experiment suite failed: {str(e)}")
        raise
    
    finally:
        logging.info(f"🏁 Experiment ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()