"""
utils.py
기본적인 유틸리티 함수들과 설정 관련 함수들
Source-Free 환경 지원 추가
SVD Feature Reducer 추가
"""

import os
import random
import logging
import torch
import numpy as np
import yaml
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
from datetime import datetime
from sklearn.decomposition import TruncatedSVD


class SVDFeatureReducer:
    """
    SVD를 이용한 feature 차원 축소 클래스
    모든 데이터셋을 동일한 차원(target_dim)으로 표준화
    """
    
    def __init__(self, target_dim: int = 100):
        self.target_dim = target_dim
        self.mean_ = None
        self.svd_model = None
        self.is_fitted = False
        self.original_dim = None
        self.explained_variance_ratio = None
        
    def fit(self, X: torch.Tensor) -> 'SVDFeatureReducer':
        """
        SVD 모델 학습
        
        Args:
            X: [N, D] 형태의 feature matrix
        """
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        self.original_dim = X_np.shape[1]
        
        # Center the data
        self.mean_ = np.mean(X_np, axis=0)
        X_centered = X_np - self.mean_
        
        # Use TruncatedSVD for efficiency
        # target_dim을 original_dim보다 작게 제한
        actual_target_dim = min(self.target_dim, self.original_dim - 1)
        
        self.svd_model = TruncatedSVD(
            n_components=actual_target_dim,
            random_state=42,
            algorithm='randomized'
        )
        
        # Fit SVD
        self.svd_model.fit(X_centered)
        
        # Store information
        self.explained_variance_ratio = self.svd_model.explained_variance_ratio_
        self.is_fitted = True
        
        logging.info(f"✅ SVD fitted: {self.original_dim}D → {actual_target_dim}D")
        logging.info(f"   Explained variance ratio: {self.explained_variance_ratio.sum():.4f}")
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        학습된 SVD로 차원 축소 수행
        
        Args:
            X: [N, D] 형태의 feature matrix
            
        Returns:
            [N, target_dim] 형태의 축소된 feature matrix
        """
        if not self.is_fitted:
            raise ValueError("SVD reducer must be fitted before transform")
            
        device = X.device if isinstance(X, torch.Tensor) else None
        
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        # Handle dimension mismatch
        if X_np.shape[1] != self.original_dim:
            logging.warning(f"⚠️  Dimension mismatch: expected {self.original_dim}, got {X_np.shape[1]}")
            if X_np.shape[1] < self.original_dim:
                # Zero-pad
                padded = np.zeros((X_np.shape[0], self.original_dim))
                padded[:, :X_np.shape[1]] = X_np
                X_np = padded
            else:
                # Truncate
                X_np = X_np[:, :self.original_dim]
        
        # Center and transform
        X_centered = X_np - self.mean_
        X_reduced = self.svd_model.transform(X_centered)
        
        # Convert back to tensor
        if device is not None:
            return torch.tensor(X_reduced, dtype=torch.float32, device=device)
        else:
            return torch.tensor(X_reduced, dtype=torch.float32)
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        SVD 학습 및 변환을 한번에 수행
        """
        return self.fit(X).transform(X)
    
    def save(self, filepath: str):
        """
        SVD reducer 저장
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted SVD reducer")
            
        save_data = {
            'target_dim': self.target_dim,
            'original_dim': self.original_dim,
            'mean_': self.mean_,
            'svd_model': self.svd_model,
            'explained_variance_ratio': self.explained_variance_ratio,
            'is_fitted': self.is_fitted
        }
        
        # Create directory if not exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
            
        logging.info(f"💾 SVD reducer saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SVDFeatureReducer':
        """
        저장된 SVD reducer 로드
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"SVD reducer file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create instance and restore state
        reducer = cls(target_dim=save_data['target_dim'])
        reducer.original_dim = save_data['original_dim']
        reducer.mean_ = save_data['mean_']
        reducer.svd_model = save_data['svd_model']
        reducer.explained_variance_ratio = save_data['explained_variance_ratio']
        reducer.is_fitted = save_data['is_fitted']
        
        logging.info(f"📂 SVD reducer loaded: {filepath}")
        logging.info(f"   {reducer.original_dim}D → {reducer.target_dim}D")
        
        return reducer
    
    def get_info(self) -> Dict[str, Any]:
        """
        SVD reducer 정보 반환
        """
        if not self.is_fitted:
            return {"fitted": False}
            
        return {
            "fitted": True,
            "original_dim": self.original_dim,
            "target_dim": self.target_dim,
            "explained_variance_ratio": float(self.explained_variance_ratio.sum()),
            "n_components": len(self.explained_variance_ratio)
        }


def set_seed(seed: int = 42):
    """
    모든 랜덤 시드 고정 (재현 가능한 실험을 위해)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 추가적인 CUDA 재현성 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """
    사용할 디바이스 결정
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    elif device_str == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device: {device_str}")
    
    print(f"🖥️  Using device: {device}")
    return device


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    YAML 설정 파일 로드 및 검증
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # 설정 검증
    config = validate_config(config)
    
    print(f"📝 Loaded config from: {config_path}")
    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    설정 파일 검증 및 기본값 설정
    """
    # 필수 섹션 체크
    required_sections = ['experiment', 'dataset', 'model']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # 실험 타입별 설정 검증
    experiment_type = config['experiment'].get('type', 'single_domain')
    
    if experiment_type == 'cross_domain':
        if 'source_dataset' not in config['experiment']:
            raise ValueError("source_dataset required for cross_domain experiments")
        if 'target_dataset' not in config['experiment']:
            raise ValueError("target_dataset required for cross_domain experiments")
    elif experiment_type == 'single_domain':
        if 'name' not in config['dataset']:
            raise ValueError("dataset.name required for single_domain experiments")
    
    # 기본값 설정
    config = set_default_config(config)
    
    return config


def set_default_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    설정 파일에 기본값 설정
    """
    # Experiment defaults
    config['experiment'].setdefault('seed', 42)
    config['experiment'].setdefault('device', 'auto')
    config['experiment'].setdefault('log_level', 'INFO')
    
    # Dataset defaults
    config['dataset'].setdefault('split', {})
    config['dataset']['split'].setdefault('val_ratio', 0.1)
    config['dataset']['split'].setdefault('test_ratio', 0.2)
    config['dataset']['split'].setdefault('shuffle', True)
    
    config['dataset'].setdefault('data_loading', {})
    config['dataset']['data_loading'].setdefault('cache_datasets', True)
    
    # SVD Feature Reduction defaults
    config.setdefault('feature_reduction', {})
    config['feature_reduction'].setdefault('enable', True)
    config['feature_reduction'].setdefault('method', 'svd')
    config['feature_reduction'].setdefault('target_dim', 100)
    config['feature_reduction'].setdefault('save_reducer', True)
    
    # Target-centric defaults
    config.setdefault('target_centric', {})
    config['target_centric'].setdefault('enable', False)
    config['target_centric'].setdefault('regularization', {})
    config['target_centric']['regularization'].setdefault('type', 'mmd')
    config['target_centric']['regularization'].setdefault('beta', 0.1)
    
    # Evaluation defaults
    config.setdefault('evaluation', {})
    config['evaluation'].setdefault('metrics', ['accuracy', 'f1_macro'])
    config['evaluation'].setdefault('save_results', True)
    config['evaluation'].setdefault('results_dir', 'results')
    
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    로깅 설정
    """
    # 로그 레벨 설정
    level = getattr(logging, log_level.upper())
    
    # 로그 포맷
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 핸들러 설정
    handlers = []
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_file:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # 루트 로거 설정
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )


def create_run_name(config: Dict[str, Any]) -> str:
    """
    실험 실행 이름 생성
    """
    experiment_type = config['experiment']['type']
    
    if experiment_type == 'cross_domain':
        source = config['experiment']['source_dataset']
        target = config['experiment']['target_dataset']
        run_name = f"{source}_to_{target}"
        
        # SVD 정보 추가
        if config['feature_reduction']['enable']:
            svd_dim = config['feature_reduction']['target_dim']
            run_name += f"_svd{svd_dim}"
        
        # Target-centric 정보 추가
        if config['target_centric']['enable']:
            reg_type = config['target_centric']['regularization']['type']
            beta = config['target_centric']['regularization']['beta']
            run_name += f"_tc_{reg_type}_{beta}"
        else:
            run_name += "_baseline"
            
    else:
        dataset = config['dataset']['name']
        model = config['model']['type']
        run_name = f"{dataset}_{model}"
        
        if config['feature_reduction']['enable']:
            svd_dim = config['feature_reduction']['target_dim']
            run_name += f"_svd{svd_dim}"
    
    # 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name += f"_{timestamp}"
    
    return run_name


def save_ckpt(model: torch.nn.Module, 
              optimizer: torch.optim.Optimizer,
              epoch: int,
              loss: float,
              filepath: str,
              **kwargs):
    """
    모델 체크포인트 저장 (SVD 정보 포함)
    """
    checkpoint_dir = Path(filepath).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_ckpt(filepath: str, 
              model: torch.nn.Module,
              optimizer: Optional[torch.optim.Optimizer] = None,
              device: Optional[torch.device] = None,
              strict: bool = True) -> Dict[str, Any]:
    """
    모델 체크포인트 로드 (Source-Free 환경 지원)
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # 디바이스 설정
    map_location = device if device else torch.device('cpu')
    
    # FIX: Add weights_only=True to suppress warning for safe loading
    try:
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(filepath, map_location=map_location)
    
    # 모델 상태 로드
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    except RuntimeError as e:
        if not strict:
            logging.warning(f"Model loading with strict=False: {e}")
        else:
            raise e
    
    # 옵티마이저 상태 로드 (선택적)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            logging.warning("Failed to load optimizer state, skipping...")
    
    print(f"📂 Checkpoint loaded: {filepath}")
    print(f"   - Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   - Loss: {checkpoint.get('loss', 'Unknown')}")    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """
    모델의 학습 가능한 파라미터 수 계산
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """
    모델 정보 출력
    """
    total_params = count_parameters(model)
    
    print(f"\n📊 {model_name} Information:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 각 모듈별 파라미터 수
    print(f"   - Module breakdown:")
    for name, module in model.named_children():
        module_params = count_parameters(module)
        print(f"     - {name}: {module_params:,}")


def log_experiment_info(config: Dict[str, Any]):
    """
    실험 정보 로깅
    """
    logger = logging.getLogger(__name__)
    
    experiment_type = config['experiment']['type']
    
    logger.info("🧪 Experiment Information:")
    logger.info(f"   - Type: {experiment_type}")
    
    if experiment_type == 'cross_domain':
        logger.info(f"   - Source: {config['experiment']['source_dataset']}")
        logger.info(f"   - Target: {config['experiment']['target_dataset']}")
    else:
        logger.info(f"   - Dataset: {config['dataset']['name']}")
    
    logger.info(f"   - Model: {config['model']['type']}")
    
    # SVD 정보
    if config['feature_reduction']['enable']:
        logger.info(f"   - SVD Reduction: {config['feature_reduction']['target_dim']}D")
    
    logger.info(f"   - Target-centric: {config['target_centric']['enable']}")
    
    if config['target_centric']['enable']:
        reg_config = config['target_centric']['regularization']
        logger.info(f"   - Regularization: {reg_config['type']} (β={reg_config['beta']})")
    
    logger.info(f"   - Device: {config['experiment']['device']}")
    logger.info(f"   - Seed: {config['experiment']['seed']}")


class EarlyStopping:
    """
    Early Stopping 클래스
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def ensure_dir(directory: str):
    """
    디렉토리가 없으면 생성
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_results(results: Dict[str, Any], config: Dict[str, Any]):
    """
    실험 결과 저장
    """
    results_dir = Path(config['evaluation']['results_dir'])
    ensure_dir(results_dir)
    
    run_name = config['experiment'].get('run_name') or create_run_name(config)
    results_file = results_dir / f"{run_name}_results.yaml"
    
    # 결과와 설정을 함께 저장
    output = {
        'results': results,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_file, 'w') as f:
        yaml.dump(output, f, default_flow_style=False)
    
    logging.info(f"Results saved: {results_file}")


def get_git_hash() -> Optional[str]:
    """
    현재 Git commit hash 가져오기
    """
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return None