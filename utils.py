"""
utils.py
기본적인 유틸리티 함수들과 설정 관련 함수들
"""

import os
import random
import logging
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def set_seed(seed: int = 42):
    """
    모든 랜덤 시드 고정 (재현 가능한 실험을 위해)
    원본 GPF의 시드 설정 방식 참고
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
    
    Args:
        device_str: "auto", "cuda", "cpu" 중 하나
        
    Returns:
        torch.device 객체
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
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        # yaml.safe_load 대신 yaml.load with SafeLoader 사용
        # 이렇게 하면 과학적 표기법이 올바르게 파싱됩니다
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    print(f"📝 Loaded config from: {config_path}")
    return config


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    로깅 설정
    
    Args:
        log_level: 로그 레벨 ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: 로그 파일 경로 (None이면 콘솔에만 출력)
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
        force=True  # 기존 설정 덮어쓰기
    )


def count_parameters(model: torch.nn.Module) -> int:
    """
    모델의 학습 가능한 파라미터 수 계산
    
    Args:
        model: PyTorch 모델
        
    Returns:
        학습 가능한 파라미터 수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_ckpt(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str,
                   **kwargs):
    """
    모델 체크포인트 저장
    
    Args:
        model: 저장할 모델
        optimizer: 옵티마이저
        epoch: 현재 에포크
        loss: 현재 손실값
        filepath: 저장할 파일 경로
        **kwargs: 추가 저장할 정보
    """
    checkpoint_dir = Path(filepath).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_ckpt(filepath: str, 
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    모델 체크포인트 로드
    
    Args:
        filepath: 체크포인트 파일 경로
        model: 로드할 모델
        optimizer: 옵티마이저 (선택적)
        device: 디바이스 (선택적)
        
    Returns:
        체크포인트 정보 딕셔너리
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # 디바이스 설정
    map_location = device if device else torch.device('cpu')
    
    checkpoint = torch.load(filepath, map_location=map_location)
    
    # 모델 상태 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저 상태 로드 (선택적)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📂 Checkpoint loaded: {filepath}")
    print(f"   - Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   - Loss: {checkpoint.get('loss', 'Unknown')}")
    
    return checkpoint


def create_run_name(config: Dict[str, Any]) -> str:
    """
    실험 실행 이름 생성 (로깅 및 체크포인트용)
    
    Args:
        config: 설정 딕셔너리
        
    Returns:
        실행 이름 문자열
    """
    dataset = config.get('dataset', {}).get('name', 'unknown')
    model = config.get('model', {}).get('type', 'unknown')
    prompt = config.get('prompt', {}).get('type', 'none')
    
    run_name = f"{dataset}_{model}_{prompt}"
    
    # 추가 파라미터들
    if prompt != 'none':
        num_prompts = config.get('prompt', {}).get('num_prompts')
        if num_prompts:
            run_name += f"_p{num_prompts}"
    
    return run_name


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """
    모델 정보 출력
    
    Args:
        model: PyTorch 모델
        model_name: 모델 이름
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


def ensure_dir(directory: str):
    """
    디렉토리가 없으면 생성
    
    Args:
        directory: 생성할 디렉토리 경로
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_git_hash() -> Optional[str]:
    """
    현재 Git commit hash 가져오기 (재현성을 위해)
    
    Returns:
        Git commit hash 또는 None
    """
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # 짧은 해시
    except:
        pass
    return None


def log_experiment_info(config: Dict[str, Any]):
    """
    실험 정보 로깅
    
    Args:
        config: 설정 딕셔너리
    """
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Experiment Information:")
    logger.info(f"   - Dataset: {config.get('dataset', {}).get('name', 'Unknown')}")
    logger.info(f"   - Model: {config.get('model', {}).get('type', 'Unknown')}")
    logger.info(f"   - Prompt: {config.get('prompt', {}).get('type', 'None')}")
    logger.info(f"   - Device: {config.get('experiment', {}).get('device', 'auto')}")
    logger.info(f"   - Seed: {config.get('experiment', {}).get('seed', 42)}")
    
    # Git 정보
    git_hash = get_git_hash()
    if git_hash:
        logger.info(f"   - Git commit: {git_hash}")


class EarlyStopping:
    """
    Early Stopping 클래스
    원본 GPF의 조기 종료 방식 참고
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Args:
            patience: 개선이 없을 때 기다릴 에포크 수
            min_delta: 개선으로 간주할 최소 변화량
            mode: 'max' (높을수록 좋음) 또는 'min' (낮을수록 좋음)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.wait = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Args:
            score: 현재 성능 점수
            
        Returns:
            조기 종료 여부
        """
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
        """개선 여부 판단"""
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
        
        