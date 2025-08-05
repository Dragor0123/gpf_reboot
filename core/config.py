"""Configuration management for GPF experiments."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Structured configuration for GPF experiments."""
    
    # Experiment settings
    type: str = "cross_domain"
    source_dataset: Optional[str] = None
    target_dataset: Optional[str] = None
    seed: int = 42
    device: str = "auto"
    log_level: str = "INFO"
    run_name: Optional[str] = None
    
    # Dataset settings
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    shuffle: bool = True
    cache_datasets: bool = True
    
    # Feature reduction settings
    feature_reduction_enable: bool = True
    feature_reduction_method: str = "svd"
    feature_reduction_target_dim: int = 100
    feature_reduction_save_reducer: bool = True
    feature_reduction_explained_variance_threshold: float = 0.95
    
    # Model settings
    model_type: str = "gcnii"
    model_hidden_dim: int = 128
    model_num_layers: int = 5
    model_dropout: float = 0.5
    model_alpha: float = 0.2  # GCNII specific
    model_theta: float = 1.0  # GCNII specific
    
    # Pretraining settings
    pretrain_lr: float = 0.001
    pretrain_weight_decay: float = 0.0005
    pretrain_epochs: int = 1000
    pretrain_aug_view1: str = "dropN"
    pretrain_aug_view2: str = "permE"
    pretrain_aug_ratio: float = 0.2
    pretrain_temperature: float = 0.5
    
    # Fine-tuning settings
    finetune_lr: float = 0.01
    finetune_weight_decay: float = 0.0005
    finetune_epochs: int = 200
    finetune_early_stopping_enable: bool = True
    finetune_early_stopping_patience: int = 30
    finetune_early_stopping_min_delta: float = 0.001
    
    # Prompt settings
    prompt_type: str = "gpf_plus"
    prompt_num_prompts: int = 10
    
    # Target-centric settings
    target_centric_enable: bool = False
    target_centric_beta: float = 0.1
    target_centric_anchor_type: str = "random"
    target_centric_anchor_num_anchors: int = 500
    target_centric_anchor_num_components: int = 8  # For MoG
    target_centric_anchor_use_sklearn_gmm: bool = True
    target_centric_mapper_type: str = "identity"
    target_centric_divergence_type: str = "mmd"
    target_centric_divergence_sigma: float = 0.25
    
    # Evaluation settings
    evaluation_metrics: list = field(default_factory=lambda: ["accuracy", "f1_macro", "f1_micro"])
    evaluation_save_results: bool = True
    evaluation_results_dir: str = "results"
    
    # Reproducibility settings
    reproducibility_deterministic: bool = True
    reproducibility_benchmark: bool = False


class ConfigManager:
    """Manages configuration loading, validation, and conversion."""
    
    def __init__(self):
        self.config_cache = {}
    
    def load_config(self, config_path: str = "config.yaml") -> ExperimentConfig:
        """Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated experiment configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Check cache
        mtime = config_file.stat().st_mtime
        if config_path in self.config_cache:
            cached_config, cached_mtime = self.config_cache[config_path]
            if cached_mtime == mtime:
                return cached_config
        
        # Load and parse YAML
        with open(config_file, 'r', encoding='utf-8') as f:
            raw_config = yaml.load(f, Loader=yaml.SafeLoader)
        
        # Convert to structured config
        structured_config = self._convert_to_structured_config(raw_config)
        
        # Validate
        self._validate_config(structured_config)
        
        # Cache and return
        self.config_cache[config_path] = (structured_config, mtime)
        return structured_config
    
    def _convert_to_structured_config(self, raw_config: Dict[str, Any]) -> ExperimentConfig:
        """Convert raw YAML config to structured ExperimentConfig."""
        config = ExperimentConfig()
        
        # Experiment section
        if 'experiment' in raw_config:
            exp = raw_config['experiment']
            config.type = exp.get('type', config.type)
            config.source_dataset = exp.get('source_dataset', config.source_dataset)
            config.target_dataset = exp.get('target_dataset', config.target_dataset)
            config.seed = exp.get('seed', config.seed)
            config.device = exp.get('device', config.device)
            config.log_level = exp.get('log_level', config.log_level)
            config.run_name = exp.get('run_name', config.run_name)
        
        # Dataset section
        if 'dataset' in raw_config:
            dataset = raw_config['dataset']
            if 'split' in dataset:
                split = dataset['split']
                config.val_ratio = split.get('val_ratio', config.val_ratio)
                config.test_ratio = split.get('test_ratio', config.test_ratio)
                config.shuffle = split.get('shuffle', config.shuffle)
            
            if 'data_loading' in dataset:
                data_loading = dataset['data_loading']
                config.cache_datasets = data_loading.get('cache_datasets', config.cache_datasets)
            
            # Single domain dataset name
            if config.type == 'single_domain' and 'name' in dataset:
                config.source_dataset = dataset['name']
        
        # Feature reduction section
        if 'feature_reduction' in raw_config:
            fr = raw_config['feature_reduction']
            config.feature_reduction_enable = fr.get('enable', config.feature_reduction_enable)
            config.feature_reduction_method = fr.get('method', config.feature_reduction_method)
            config.feature_reduction_target_dim = fr.get('target_dim', config.feature_reduction_target_dim)
            config.feature_reduction_save_reducer = fr.get('save_reducer', config.feature_reduction_save_reducer)
            config.feature_reduction_explained_variance_threshold = fr.get(
                'explained_variance_threshold', config.feature_reduction_explained_variance_threshold
            )
        
        # Model section
        if 'model' in raw_config:
            model = raw_config['model']
            config.model_type = model.get('type', config.model_type)
            config.model_hidden_dim = model.get('hidden_dim', config.model_hidden_dim)
            config.model_num_layers = model.get('num_layers', config.model_num_layers)
            config.model_dropout = model.get('dropout', config.model_dropout)
            config.model_alpha = model.get('alpha', config.model_alpha)
            config.model_theta = model.get('theta', config.model_theta)
        
        # Pretraining section
        if 'pretrain' in raw_config:
            pretrain = raw_config['pretrain']
            config.pretrain_lr = pretrain.get('lr', config.pretrain_lr)
            config.pretrain_weight_decay = pretrain.get('weight_decay', config.pretrain_weight_decay)
            config.pretrain_epochs = pretrain.get('epochs', config.pretrain_epochs)
            
            if 'augmentation' in pretrain:
                aug = pretrain['augmentation']
                config.pretrain_aug_view1 = aug.get('view1', config.pretrain_aug_view1)
                config.pretrain_aug_view2 = aug.get('view2', config.pretrain_aug_view2)
                config.pretrain_aug_ratio = aug.get('aug_ratio', config.pretrain_aug_ratio)
                config.pretrain_temperature = aug.get('temperature', config.pretrain_temperature)
        
        # Prompt tuning section (mapped to fine-tuning)
        if 'prompt_tuning' in raw_config:
            pt = raw_config['prompt_tuning']
            config.finetune_lr = pt.get('lr', config.finetune_lr)
            config.finetune_weight_decay = pt.get('weight_decay', config.finetune_weight_decay)
            config.finetune_epochs = pt.get('epochs', config.finetune_epochs)
            
            if 'early_stopping' in pt:
                es = pt['early_stopping']
                config.finetune_early_stopping_enable = es.get('enable', config.finetune_early_stopping_enable)
                config.finetune_early_stopping_patience = es.get('patience', config.finetune_early_stopping_patience)
                config.finetune_early_stopping_min_delta = es.get('min_delta', config.finetune_early_stopping_min_delta)
        
        # Prompt section
        if 'prompt' in raw_config:
            prompt = raw_config['prompt']
            config.prompt_type = prompt.get('type', config.prompt_type)
            config.prompt_num_prompts = prompt.get('num_prompts', config.prompt_num_prompts)
        
        # Target-centric section
        if 'target_centric' in raw_config:
            tc = raw_config['target_centric']
            config.target_centric_enable = tc.get('enable', config.target_centric_enable)
            
            if 'regularization' in tc:
                reg = tc['regularization']
                config.target_centric_beta = reg.get('beta', config.target_centric_beta)
                
                if 'anchor' in reg:
                    anchor = reg['anchor']
                    config.target_centric_anchor_type = anchor.get('type', config.target_centric_anchor_type)
                    config.target_centric_anchor_num_anchors = anchor.get('num_anchors', config.target_centric_anchor_num_anchors)
                    config.target_centric_anchor_num_components = anchor.get('num_components', config.target_centric_anchor_num_components)
                    config.target_centric_anchor_use_sklearn_gmm = anchor.get('use_sklearn_gmm', config.target_centric_anchor_use_sklearn_gmm)
                
                if 'mapper' in reg:
                    mapper = reg['mapper']
                    config.target_centric_mapper_type = mapper.get('type', config.target_centric_mapper_type)
                
                if 'divergence' in reg:
                    div = reg['divergence']
                    config.target_centric_divergence_type = div.get('type', config.target_centric_divergence_type)
                    if 'params' in div:
                        params = div['params']
                        config.target_centric_divergence_sigma = params.get('sigma', config.target_centric_divergence_sigma)
        
        # Evaluation section
        if 'evaluation' in raw_config:
            eval_section = raw_config['evaluation']
            config.evaluation_metrics = eval_section.get('metrics', config.evaluation_metrics)
            config.evaluation_save_results = eval_section.get('save_results', config.evaluation_save_results)
            config.evaluation_results_dir = eval_section.get('results_dir', config.evaluation_results_dir)
        
        # Reproducibility section
        if 'reproducibility' in raw_config:
            repro = raw_config['reproducibility']
            config.reproducibility_deterministic = repro.get('deterministic', config.reproducibility_deterministic)
            config.reproducibility_benchmark = repro.get('benchmark', config.reproducibility_benchmark)
        
        return config
    
    def _validate_config(self, config: ExperimentConfig) -> None:
        """Validate configuration for consistency and completeness.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate experiment type specific requirements
        if config.type == 'cross_domain':
            if not config.source_dataset:
                raise ValueError("source_dataset required for cross_domain experiments")
            if not config.target_dataset:
                raise ValueError("target_dataset required for cross_domain experiments")
        elif config.type == 'single_domain':
            if not config.source_dataset:
                raise ValueError("dataset name required for single_domain experiments")
        else:
            raise ValueError(f"Unknown experiment type: {config.type}")
        
        # Validate numeric ranges
        if not 0 < config.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if not 0 < config.test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        if config.val_ratio + config.test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be less than 1")
        
        # Validate model parameters
        if config.model_hidden_dim <= 0:
            raise ValueError("model_hidden_dim must be positive")
        if config.model_num_layers <= 0:
            raise ValueError("model_num_layers must be positive")
        if not 0 <= config.model_dropout <= 1:
            raise ValueError("model_dropout must be between 0 and 1")
        
        # Validate training parameters
        if config.pretrain_lr <= 0:
            raise ValueError("pretrain_lr must be positive")
        if config.finetune_lr <= 0:
            raise ValueError("finetune_lr must be positive")
        if config.pretrain_epochs <= 0:
            raise ValueError("pretrain_epochs must be positive")
        if config.finetune_epochs <= 0:
            raise ValueError("finetune_epochs must be positive")
        
        # Validate target-centric parameters
        if config.target_centric_enable:
            if config.target_centric_beta < 0:
                raise ValueError("target_centric_beta must be non-negative")
            if config.target_centric_anchor_num_anchors <= 0:
                raise ValueError("target_centric_anchor_num_anchors must be positive")
    
    def save_config(self, config: ExperimentConfig, filepath: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            filepath: Output file path
        """
        # Convert structured config back to nested dict format
        config_dict = self._convert_to_yaml_format(config)
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to YAML
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _convert_to_yaml_format(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Convert structured config back to YAML dictionary format."""
        return {
            'experiment': {
                'type': config.type,
                'source_dataset': config.source_dataset,
                'target_dataset': config.target_dataset,
                'seed': config.seed,
                'device': config.device,
                'log_level': config.log_level,
                'run_name': config.run_name,
            },
            'dataset': {
                'split': {
                    'val_ratio': config.val_ratio,
                    'test_ratio': config.test_ratio,
                    'shuffle': config.shuffle,
                },
                'data_loading': {
                    'cache_datasets': config.cache_datasets,
                }
            },
            'feature_reduction': {
                'enable': config.feature_reduction_enable,
                'method': config.feature_reduction_method,
                'target_dim': config.feature_reduction_target_dim,
                'save_reducer': config.feature_reduction_save_reducer,
                'explained_variance_threshold': config.feature_reduction_explained_variance_threshold,
            },
            'model': {
                'type': config.model_type,
                'hidden_dim': config.model_hidden_dim,
                'num_layers': config.model_num_layers,
                'dropout': config.model_dropout,
                'alpha': config.model_alpha,
                'theta': config.model_theta,
            },
            'pretrain': {
                'lr': config.pretrain_lr,
                'weight_decay': config.pretrain_weight_decay,
                'epochs': config.pretrain_epochs,
                'augmentation': {
                    'view1': config.pretrain_aug_view1,
                    'view2': config.pretrain_aug_view2,
                    'aug_ratio': config.pretrain_aug_ratio,
                    'temperature': config.pretrain_temperature,
                }
            },
            'prompt_tuning': {
                'lr': config.finetune_lr,
                'weight_decay': config.finetune_weight_decay,
                'epochs': config.finetune_epochs,
                'early_stopping': {
                    'enable': config.finetune_early_stopping_enable,
                    'patience': config.finetune_early_stopping_patience,
                    'min_delta': config.finetune_early_stopping_min_delta,
                }
            },
            'prompt': {
                'type': config.prompt_type,
                'num_prompts': config.prompt_num_prompts,
            },
            'target_centric': {
                'enable': config.target_centric_enable,
                'regularization': {
                    'beta': config.target_centric_beta,
                    'anchor': {
                        'type': config.target_centric_anchor_type,
                        'num_anchors': config.target_centric_anchor_num_anchors,
                        'num_components': config.target_centric_anchor_num_components,
                        'use_sklearn_gmm': config.target_centric_anchor_use_sklearn_gmm,
                    },
                    'mapper': {
                        'type': config.target_centric_mapper_type,
                    },
                    'divergence': {
                        'type': config.target_centric_divergence_type,
                        'params': {
                            'sigma': config.target_centric_divergence_sigma,
                        }
                    }
                }
            },
            'evaluation': {
                'metrics': config.evaluation_metrics,
                'save_results': config.evaluation_save_results,
                'results_dir': config.evaluation_results_dir,
            },
            'reproducibility': {
                'deterministic': config.reproducibility_deterministic,
                'benchmark': config.reproducibility_benchmark,
            }
        }