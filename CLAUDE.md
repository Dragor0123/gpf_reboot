```markdown
# CLAUDE.md - Strategy 1 Implementation Guide

## Project Overview
Implement **Strategy 1: Gradient-Based Optimization** for source-free reference distribution generation in graph neural networks. The goal is to create an ideal reference distribution by optimizing input features to maximize encoder output quality, without accessing source domain data.

## Implementation Strategy

### Phase 1: Core Infrastructure Setup

#### 1.1 Create Gradient-Based Optimizer Module
```python
# File: core/gradient_optimizer.py
class GradientBasedReferenceGenerator:
    def __init__(self, encoder, target_data, device):
        self.encoder = encoder
        self.target_data = target_data
        self.device = device
    
    def generate_reference_distribution(self, num_anchors=1000):
        # Main entry point for reference distribution generation
        pass
    
    def optimize_input_for_encoder_output(self, objectives, num_iterations=100):
        # Core optimization loop using gradient ascent
        pass
```

#### 1.2 Define Objective Functions
```python
# File: core/objectives.py
class EncoderObjectives:
    @staticmethod
    def high_norm(embeddings):
        # Encourage strong activations
        pass
    
    @staticmethod
    def feature_diversity(embeddings):
        # Maximize embedding diversity
        pass
    
    @staticmethod
    def task_alignment(embeddings, labels=None):
        # Align with downstream task if labels available
        pass
    
    @staticmethod
    def graph_homophily(embeddings, edge_index):
        # Preserve graph structural properties
        pass
```

### Phase 2: Core Implementation

#### 2.1 Implement Optimization Engine
- Create learnable input features that are optimized via gradient ascent
- Implement multi-objective optimization with weighted combination
- Add regularization to prevent divergence from original target distribution
- Include convergence criteria and early stopping

#### 2.2 Graph-Specific Objectives
- Implement homophily preservation objective
- Add structural diversity metrics
- Create message passing quality assessment
- Design node centrality preservation objective

#### 2.3 Regularization Mechanisms
```python
def regularized_optimization(optimized_features, original_features, lambda_reg=0.1):
    # Prevent optimized features from deviating too far from originals
    distance_penalty = torch.norm(optimized_features - original_features, p=2)
    return base_loss - lambda_reg * distance_penalty
```

### Phase 3: Integration with Existing Codebase

#### 3.1 Replace MoG Anchor Generation
- Locate current MoG anchor generation in `anchor_factory.py`
- Replace `generate_mog_anchors()` function with gradient-based approach
- Maintain same interface: `generate_gradient_optimized_anchors(embeddings, num_anchors)`

#### 3.2 Update Training Pipeline
- Modify `train_prompt_tuning.py` to use new anchor generation method
- Update configuration system to support gradient optimization parameters
- Add new hyperparameters: learning rate, optimization steps, objective weights

#### 3.3 Configuration Updates
```yaml
# Add to config.yaml
target_centric:
  regularization:
    anchor:
      type: "gradient_optimized"  # New anchor type
      num_anchors: 1000
      optimization:
        learning_rate: 0.01
        num_iterations: 100
        objectives:
          high_norm: 0.3
          diversity: 0.4
          task_alignment: 0.3
        regularization_lambda: 0.1
```

### Phase 4: Objective Function Design

#### 4.1 Primary Objectives Implementation
1. **Activation Strength**: `torch.norm(embeddings, dim=1).mean()`
2. **Feature Diversity**: Pairwise distance maximization
3. **Task Relevance**: Classification loss if labels available
4. **Structural Preservation**: Graph homophily maintenance

#### 4.2 Multi-Objective Optimization
- Implement weighted combination of objectives
- Add hyperparameter tuning for objective weights
- Create ablation study framework for objective contribution analysis

### Phase 5: Experimental Validation

#### 5.1 Comparison Framework
- Compare against current MoG approach
- Implement baseline: random anchor generation
- Add comparison with Gaussian prior anchors

#### 5.2 Ablation Studies
- Individual objective contribution analysis
- Regularization strength impact study
- Optimization iteration count sensitivity analysis

#### 5.3 Performance Metrics
- Downstream task accuracy comparison
- MMD loss convergence analysis
- Training stability metrics
- Computational cost measurement

## Implementation Priority

### High Priority (Implement First)
1. `GradientBasedReferenceGenerator` class
2. Basic objective functions (high_norm, diversity)
3. Simple optimization loop with gradient ascent
4. Integration point in `anchor_factory.py`

### Medium Priority
1. Graph-specific objectives (homophily, structural)
2. Multi-objective optimization framework
3. Regularization mechanisms
4. Configuration system updates

### Low Priority (Polish Phase)
1. Advanced optimization techniques (Adam, momentum)
2. Adaptive objective weighting
3. Extensive ablation study framework
4. Performance profiling and optimization

## Key Files to Modify

### Core Implementation
- `core/gradient_optimizer.py` (NEW)
- `core/objectives.py` (NEW)
- `anchor_factory.py` (MODIFY)

### Training Pipeline
- `train_prompt_tuning.py` (MODIFY)
- `config.yaml` (MODIFY)

### Loss Functions
- `training/losses.py` (MINOR MODIFY)
- `training/regularizers.py` (MINOR MODIFY)

## Success Criteria

### Functional Success
- [ ] Gradient optimization generates stable anchor distributions
- [ ] Integration with existing training pipeline works seamlessly
- [ ] No performance degradation compared to current MoG approach

### Performance Success
- [ ] At least 2% improvement in downstream task accuracy
- [ ] Faster convergence in MMD loss
- [ ] Computational overhead < 20% of current approach

### Research Success
- [ ] Clear ablation study showing objective contribution
- [ ] Demonstrated source-free capability
- [ ] Reproducible results across different graph domains

## Implementation Notes

### Critical Considerations
1. **Gradient Flow**: Ensure gradients flow properly through frozen encoder
2. **Numerical Stability**: Add gradient clipping and learning rate scheduling
3. **Memory Efficiency**: Implement batch processing for large graphs
4. **Reproducibility**: Fix random seeds and document all hyperparameters

### Potential Pitfalls
1. **Optimization Instability**: May need careful learning rate tuning
2. **Local Minima**: Consider multiple random initializations
3. **Overfitting**: Balance between optimization and regularization
4. **Computational Cost**: Monitor and optimize if too expensive

## Testing Strategy

### Unit Tests
- Individual objective function correctness
- Optimization convergence on toy examples
- Gradient computation accuracy

### Integration Tests
- End-to-end pipeline with gradient-optimized anchors
- Comparison with existing MoG approach
- Cross-domain transfer performance

### Performance Tests
- Large-scale graph handling
- Memory usage profiling
- Training time comparison

---

**Implementation Goal**: Replace current MoG-based anchor generation with gradient-optimized approach that leverages pretrained encoder knowledge to create better reference distributions for source-free domain adaptation.
```