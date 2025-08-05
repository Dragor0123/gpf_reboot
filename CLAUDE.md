# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modern, refactored implementation of Graph Prompt Feature (GPF) for cross-domain graph neural network transfer learning. The project focuses on learning transferable representations across different graph datasets using contrastive pretraining, SVD feature alignment, and target-centric prior modeling.

## Key Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running Experiments
```bash
# Single cross-domain experiment (cora → computers)
python train_pretrain.py    # Pretrain encoder on source dataset
python train_fine_tuning.py # Fine-tune on target dataset

# Comprehensive evaluation across all dataset pairs
python run_experiments.py   # Runs 20 cross-domain combinations with baseline vs target-centric comparison

# Debug and data verification
python debug_data_check.py     # Verify dataset loading and SVD alignment
python debug_step_by_step.py   # Step-by-step debugging of the pipeline
```

### Configuration
All experiments are controlled via `config.yaml` with structured validation:
- `experiment.type`: "single_domain" or "cross_domain" 
- `experiment.source_dataset` / `target_dataset`: Dataset names
- `feature_reduction.enable`: SVD dimensionality reduction (recommended: true)
- `target_centric.enable`: Enable target-centric prior modeling

## Refactored Architecture Overview

### Modular Design
The codebase has been completely refactored into a clean, modular architecture:

```
core/              # Core utilities and configuration
├── config.py      # Structured configuration management
├── device.py      # Device management and optimization
├── logging.py     # Enhanced logging with colors and experiment tracking
├── reproducibility.py  # Comprehensive reproducibility utilities
└── svd_reducer.py # Robust SVD feature reduction

models/            # Graph neural network architectures
├── base.py        # Abstract base classes and interfaces
└── architectures.py  # Concrete GNN implementations

datasets/          # Dataset loading and management
├── base.py        # Abstract dataset interfaces
├── loaders.py     # Concrete dataset loaders
└── manager.py     # High-level dataset management with SVD integration

training/          # Training components
├── losses.py      # Loss functions and target-centric losses
├── regularizers.py # Target-centric regularization strategies
├── augmentation.py # Graph data augmentation
└── trainer.py     # Training loop abstractions
```

### Core Components

**Structured Configuration** (`core/config.py`):
- Type-safe configuration with dataclasses (`ExperimentConfig`)
- Automatic validation and default value handling
- Support for both YAML and programmatic configuration
- Backward compatibility with existing config files

**Enhanced Dataset Management** (`datasets/`):
- Abstract base classes for extensible dataset loading
- Automatic SVD alignment for cross-domain compatibility
- Comprehensive dataset statistics and validation
- Support for custom datasets and transforms

**Robust SVD Feature Alignment** (`core/svd_reducer.py`):
- Improved error handling and dimension mismatch resolution
- Comprehensive serialization and caching
- Reconstruction error analysis and inverse transforms
- Thread-safe operation with proper state management

**Modern GNN Architectures** (`models/`):
- Abstract `GraphEncoder` base class for consistent interfaces
- Enhanced implementations of GIN, GCN, GAT, SAGE, GCNII
- Flexible projection heads and classifiers
- Comprehensive model introspection and logging

**Advanced Training Components** (`training/`):
- Modular loss functions with proper abstractions
- Sophisticated target-centric regularization strategies
- Graph augmentation pipeline with multiple strategies
- Abstract trainer classes for different training phases

### Key Design Patterns

**Factory Pattern**: Consistent factory functions for models, datasets, and training components enable easy experimentation and extension.

**Strategy Pattern**: Pluggable strategies for anchoring, regularization, and augmentation allow flexible algorithm composition.

**Template Method**: Abstract base classes define training workflows while allowing customization of specific steps.

**Configuration-Driven Design**: Structured configuration management eliminates magic numbers and enables reproducible experiments.

## Important Implementation Details

### Enhanced Error Handling
- Comprehensive validation at all levels (configuration, data, models)
- Graceful degradation with informative error messages
- Automatic fallback strategies for common failure modes
- Detailed logging for debugging and monitoring

### Improved Reproducibility
- Comprehensive seed management across all random sources
- Deterministic CUDA operations with performance trade-offs
- Random state capture and restoration utilities
- Verification tools to ensure reproducible results

### Cross-Domain Robustness
- Automatic dimension alignment with multiple fallback strategies
- Comprehensive compatibility checking between datasets
- Robust SVD transformation with error recovery
- Extensive validation and logging throughout the pipeline

### Advanced Logging and Monitoring
- Colored console output with structured formatting
- Experiment-specific log files with detailed tracking
- Progress monitoring with phase-based organization
- Performance metrics and resource utilization tracking

## Development Guidelines

### Adding New Models
1. Inherit from `GraphEncoder` base class in `models/base.py`
2. Implement required abstract methods with proper type hints
3. Add model to factory function in `models/architectures.py`
4. Update model requirements dictionary for configuration validation

### Adding New Datasets
1. Create loader class inheriting from `BaseDatasetLoader`
2. Implement `load_raw_dataset()` and `get_supported_datasets()`
3. Register with `DatasetManager` for integrated pipeline support
4. Add comprehensive metadata and statistics computation

### Adding New Training Components
1. Use appropriate abstract base classes from `training/`
2. Follow factory pattern conventions for extensibility
3. Implement proper configuration integration
4. Add comprehensive logging and error handling

### Configuration Management
- Use structured `ExperimentConfig` for type safety
- Add validation logic to `ConfigManager._validate_config()`
- Maintain backward compatibility with existing YAML files
- Document all new configuration parameters

The refactored architecture provides a solid foundation for advanced graph neural network research with emphasis on maintainability, extensibility, and reproducibility.