"""Logging utilities for GPF experiments."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        
        return super().format(record)


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    enable_colors: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration for GPF experiments.
    
    Args:
        level: Logging level (string or logging constant)
        log_file: Optional file to write logs to
        enable_colors: Whether to use colored output for console
        format_string: Custom format string
        
    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if enable_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        
        # No colors for file output
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_path}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """Enhanced logger for experiment tracking."""
    
    def __init__(self, name: str, experiment_dir: Optional[Path] = None):
        self.logger = get_logger(name)
        self.experiment_dir = experiment_dir
        self.start_time = datetime.now()
        
        if experiment_dir:
            experiment_dir.mkdir(parents=True, exist_ok=True)
            log_file = experiment_dir / f"{name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
            
            # Add file handler for this specific experiment
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Experiment log: {log_file}")
    
    def log_experiment_start(self, config_dict: dict) -> None:
        """Log experiment start with configuration."""
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT START")
        self.logger.info("="*80)
        self.logger.info(f"Start time: {self.start_time}")
        
        # Log key configuration items
        self.logger.info("Configuration:")
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    self.logger.info(f"    {subkey}: {subvalue}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_experiment_end(self, results: Optional[dict] = None) -> None:
        """Log experiment end with results."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT END")
        self.logger.info("="*80)
        self.logger.info(f"End time: {end_time}")
        self.logger.info(f"Duration: {duration}")
        
        if results:
            self.logger.info("Results:")
            for key, value in results.items():
                self.logger.info(f"  {key}: {value}")
    
    def log_phase_start(self, phase_name: str) -> None:
        """Log the start of an experiment phase."""
        self.logger.info("-" * 60)
        self.logger.info(f"PHASE START: {phase_name}")
        self.logger.info("-" * 60)
    
    def log_phase_end(self, phase_name: str, metrics: Optional[dict] = None) -> None:
        """Log the end of an experiment phase."""
        self.logger.info(f"PHASE END: {phase_name}")
        if metrics:
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("-" * 60)
    
    def log_model_info(self, model, model_name: str = "Model") -> None:
        """Log detailed model information."""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"{model_name} Information:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Log module breakdown
        self.logger.info("  Module breakdown:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            self.logger.info(f"    {name}: {module_params:,} parameters")
    
    def log_training_progress(self, epoch: int, total_epochs: int, 
                            losses: dict, metrics: Optional[dict] = None) -> None:
        """Log training progress."""
        progress = f"Epoch {epoch:04d}/{total_epochs:04d}"
        
        loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        log_msg = f"{progress} | {loss_str}"
        
        if metrics:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_msg += f" | {metric_str}"
        
        self.logger.info(log_msg)