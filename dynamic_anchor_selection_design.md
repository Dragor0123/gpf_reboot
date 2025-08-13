# Dynamic Performance-Based Anchor Selection for Source-Free Graph Domain Adaptation

## Executive Summary

This document outlines the design and implementation strategy for a novel **Dynamic Performance-Based Anchor Selection** approach in graph neural network domain adaptation. Unlike existing static methods (e.g., MoG-based anchors), our approach dynamically selects high-performing nodes as reference anchors during training, enabling more effective source-free domain adaptation.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Current Approach vs. Proposed Solution](#current-approach-vs-proposed-solution)
3. [Core Design Philosophy](#core-design-philosophy)
4. [Technical Architecture](#technical-architecture)
5. [Implementation Details](#implementation-details)
6. [Performance Metrics](#performance-metrics)
7. [Integration Strategy](#integration-strategy)
8. [Experimental Validation](#experimental-validation)
9. [Risk Assessment](#risk-assessment)
10. [Future Extensions](#future-extensions)

## Problem Statement

### Current Limitations
- **Static Anchor Generation**: Existing MoG-based approaches generate anchors once at the beginning of training and keep them fixed
- **Target-Only Distribution**: Using target domain embeddings to model "ideal" distribution is logically inconsistent
- **Source Dependency**: Many approaches require access to source domain data, limiting real-world applicability

### Research Objective
Develop a **source-free, dynamic anchor selection mechanism** that:
1. Identifies high-performing nodes during training
2. Uses these nodes as reference anchors for regularization
3. Adapts anchor selection based on evolving model performance
4. Maintains task-relevance without source domain access

## Current Approach vs. Proposed Solution

### Current MoG Approach
```python
# Static anchor generation (once at training start)
with torch.no_grad():
    latent_z = encoder(target_data.x, target_data.edge_index)
    anchors = generate_mog_anchors(latent_z, num_components=8, num_anchors=1000)
    # ❌ Fixed for entire training duration
```

### Proposed Dynamic Approach
```python
# Dynamic anchor selection (updated during training)
for epoch in range(total_epochs):
    prompted_embeddings = encoder(prompted_x, edge_index)
    logits = classifier(prompted_embeddings)
    
    # ✅ Select anchors based on current performance
    high_perf_anchors = select_high_performance_nodes(
        prompted_embeddings, logits, labels, train_mask
    )
    
    # ✅ Use selected embeddings as reference distribution
    mmd_loss = compute_mmd(prompted_embeddings, high_perf_anchors)
```

## Core Design Philosophy

### 1. Performance-Driven Selection
**Principle**: Nodes that achieve high task performance represent "ideal" representations that other nodes should align with.

**Rationale**: 
- High-performing nodes indicate successful knowledge transfer from pretrained encoder
- These nodes serve as natural reference points for the target domain
- Task-specific relevance is automatically ensured

### 2. Dynamic Adaptation
**Principle**: Anchor selection should evolve as model performance improves during training.

**Benefits**:
- Early training: Use initially well-performing nodes
- Late training: Use nodes that have learned optimal representations
- Continuous improvement of reference quality

### 3. Self-Supervised Discovery
**Principle**: The target domain data itself contains signals about optimal representations.

**Implementation**:
- No external reference needed
- Leverages pretrained encoder knowledge implicitly
- Maintains source-free constraint

## Technical Architecture

### Core Components

#### 1. Performance Evaluator
```python
class NodePerformanceEvaluator:
    """Evaluates individual node performance for anchor selection"""
    
    def compute_loss_scores(self, logits, labels, mask):
        """Lower loss = higher performance score"""
        
    def compute_confidence_scores(self, logits, mask):
        """Higher prediction confidence = higher score"""
        
    def compute_combined_scores(self, loss_scores, confidence_scores):
        """Weighted combination of multiple criteria"""
```

#### 2. Dynamic Anchor Selector
```python
class DynamicPerformanceAnchorSelector:
    """Main component for dynamic anchor selection"""
    
    def __init__(self, selection_ratio=0.2, update_frequency=10):
        self.selection_ratio = selection_ratio
        self.update_frequency = update_frequency
        self.performance_history = []
    
    def select_anchors_by_performance(self, embeddings, logits, labels, mask, epoch):
        """Core anchor selection logic"""
        
    def update_anchor_quality_metrics(self, selected_anchors, all_embeddings):
        """Track anchor quality over time"""
```

#### 3. Regularizer Integration
```python
class DynamicTargetCentricRegularizer(nn.Module):
    """Integrates dynamic anchors with existing regularization framework"""
    
    def update_anchors(self, new_anchors):
        """Updates anchor references for MMD computation"""
        
    def compute_regularization_loss(self, prompted_embeddings):
        """Computes MMD/Wasserstein distance with current anchors"""
```

### Selection Criteria

#### Primary Criterion: Individual Node Loss
```python
def compute_loss_scores(logits, labels, mask):
    """
    Rationale: Nodes with lower individual loss demonstrate better task alignment
    """
    node_losses = F.cross_entropy(logits, labels, reduction='none')  # [N]
    train_losses = node_losses[mask]
    
    # Invert and normalize: lower loss → higher score
    min_loss, max_loss = train_losses.min(), train_losses.max()
    normalized = (train_losses - min_loss) / (max_loss - min_loss + 1e-8)
    scores = 1.0 - normalized
    
    return scores
```

#### Secondary Criterion: Prediction Confidence
```python
def compute_confidence_scores(logits, mask):
    """
    Rationale: High confidence predictions indicate stable, reliable representations
    """
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    confidence = -entropy  # Lower entropy = higher confidence
    
    train_confidence = confidence[mask]
    
    # Normalize to [0, 1] range
    min_conf, max_conf = train_confidence.min(), train_confidence.max()
    scores = (train_confidence - min_conf) / (max_conf - min_conf + 1e-8)
    
    return scores
```

#### Combined Scoring
```python
def compute_combined_scores(loss_scores, confidence_scores, alpha=0.7):
    """
    Weighted combination prioritizing loss performance
    """
    return alpha * loss_scores + (1 - alpha) * confidence_scores
```

## Implementation Details

### 1. Integration with Existing Codebase

#### File Modifications
```bash
# New files
core/dynamic_anchor_selector.py          # Core implementation
core/performance_evaluator.py            # Performance metrics

# Modified files  
training/regularizers.py                 # Add dynamic anchor support
train_prompt_tuning.py                   # Integrate with training loop
config.yaml                              # Add configuration options
```

#### Configuration Updates
```yaml
target_centric:
  enable: true
  regularization:
    anchor:
      type: "dynamic_performance"  # New anchor type
      selection_ratio: 0.2          # Top 20% performing nodes
      update_frequency: 10          # Update every 10 epochs
      criteria:
        loss_weight: 0.7           # Loss criterion weight
        confidence_weight: 0.3     # Confidence criterion weight
      quality_tracking: true       # Track anchor quality metrics
```

### 2. Training Loop Integration

```python
def train_with_dynamic_anchors(config, data, dataset_info, ...):
    # Initialize dynamic anchor selector
    anchor_selector = DynamicPerformanceAnchorSelector(
        selection_ratio=config['target_centric']['regularization']['anchor']['selection_ratio'],
        update_frequency=config['target_centric']['regularization']['anchor']['update_frequency']
    )
    
    # Initialize regularizer with dynamic capability
    loss_fn = TargetCentricLoss(config)
    loss_fn.regularizer.enable_dynamic_anchors(anchor_selector)
    
    for epoch in range(cfg['epochs']):
        # Standard forward pass
        prompted_x = prompt.add(x)
        prompted_embeddings = encoder(prompted_x, edge_index)
        logits = classifier(prompted_embeddings)
        
        # Dynamic anchor selection and update
        if epoch % anchor_selector.update_frequency == 0:
            selected_anchors = anchor_selector.select_anchors_by_performance(
                prompted_embeddings, logits, y, train_mask, epoch
            )
            loss_fn.regularizer.update_anchors(selected_anchors)
        
        # Compute loss with updated anchors
        loss_dict = loss_fn(
            logits=logits,
            labels=y,
            embeddings=prompted_embeddings,
            mask=train_mask
        )
        
        # Continue with standard training loop...
```

### 3. Anchor Quality Metrics

```python
def compute_anchor_quality_metrics(anchors, all_embeddings, labels, mask):
    """
    Comprehensive anchor quality assessment
    """
    metrics = {}
    
    # 1. Diversity: How diverse are the selected anchors?
    anchor_distances = torch.cdist(anchors, anchors, p=2)
    upper_triangle = torch.triu(anchor_distances, diagonal=1)
    metrics['diversity'] = upper_triangle[upper_triangle > 0].mean().item()
    
    # 2. Representativeness: How well do anchors represent the full distribution?
    all_train_embeddings = all_embeddings[mask]
    distances_to_anchors = torch.cdist(all_train_embeddings, anchors, p=2)
    min_distances = distances_to_anchors.min(dim=1)[0]
    metrics['representativeness'] = min_distances.mean().item()
    
    # 3. Stability: How consistent are anchor selections across epochs?
    # (Requires maintaining anchor history)
    
    # 4. Task Alignment: Performance of anchor-based regularization
    # (Measured through downstream task accuracy)
    
    return metrics
```

## Performance Metrics

### Success Criteria

#### Functional Requirements
- [ ] Dynamic anchor selection operates without training instability
- [ ] Integration with existing codebase maintains backward compatibility
- [ ] Computational overhead remains under 15% of baseline training time

#### Performance Requirements
- [ ] **Primary**: 2-5% improvement in downstream task accuracy over static MoG approach
- [ ] **Secondary**: Faster convergence (10-20% fewer epochs to reach target performance)
- [ ] **Tertiary**: Better cross-domain generalization across different graph types

#### Quality Requirements
- [ ] Anchor diversity score > 0.8 (normalized)
- [ ] Anchor representativeness score < 0.5 (lower is better)
- [ ] Selection stability > 70% overlap between consecutive updates

### Evaluation Framework

#### Baseline Comparisons
1. **Current MoG Approach**: Static anchor generation
2. **Random Selection**: Random node selection as anchors
3. **Gaussian Prior**: Standard Gaussian anchor generation
4. **No Regularization**: Training without target-centric regularization

#### Test Scenarios
1. **Cross-Domain**: Photo→PubMed, Computer→CiteSeer, etc.
2. **Limited Labels**: 5-shot, 10-shot learning scenarios  
3. **Large Graphs**: Scalability testing on Reddit, ogbn-products
4. **Heterogeneous Graphs**: Non-homophilic graph types

## Integration Strategy

### Phase 1: Core Implementation (Weeks 1-2)
```python
# Priority 1: Basic dynamic anchor selector
class DynamicPerformanceAnchorSelector:
    def select_anchors_by_performance(self, ...):
        # Implement loss-based selection only
        
# Priority 2: Integration with regularizer
class DynamicTargetCentricRegularizer:
    def update_anchors(self, new_anchors):
        # Replace fixed anchors with dynamic ones
```

### Phase 2: Enhanced Metrics (Weeks 3-4)
```python
# Add confidence-based scoring
def compute_confidence_scores(self, logits, mask):
    # Implement entropy-based confidence

# Add quality tracking
def compute_anchor_quality_metrics(self, ...):
    # Implement diversity and representativeness metrics
```

### Phase 3: Optimization (Weeks 5-6)
```python
# Add curriculum learning aspects
class CurriculumAnchorScheduler:
    def adapt_selection_criteria(self, epoch, performance):
        # Implement adaptive selection criteria

# Performance optimization
def optimize_anchor_selection_speed(self, ...):
    # Implement efficient selection algorithms
```

## Experimental Validation

### Ablation Studies

#### 1. Selection Criteria Impact
```python
experiments = [
    {'loss_weight': 1.0, 'confidence_weight': 0.0},  # Loss only
    {'loss_weight': 0.0, 'confidence_weight': 1.0},  # Confidence only
    {'loss_weight': 0.7, 'confidence_weight': 0.3},  # Combined (proposed)
    {'loss_weight': 0.5, 'confidence_weight': 0.5},  # Equal weights
]
```

#### 2. Update Frequency Impact
```python
update_frequencies = [1, 5, 10, 20, 50]  # epochs
# Measure: performance vs. computational cost trade-off
```

#### 3. Selection Ratio Impact
```python
selection_ratios = [0.1, 0.15, 0.2, 0.25, 0.3]  # percentage of nodes
# Measure: anchor quality vs. diversity trade-off
```

### Comparative Analysis

#### Performance Comparison Table
| Method | Accuracy | F1-Macro | Convergence Speed | Computational Cost |
|--------|----------|----------|-------------------|-------------------|
| Static MoG | Baseline | Baseline | Baseline | Baseline |
| Random Selection | -X% | -Y% | +Z epochs | -W% |
| Dynamic Performance | **+A%** | **+B%** | **-C epochs** | +D% |
| Gaussian Prior | -E% | -F% | +G epochs | -H% |

### Statistical Validation
- **Significance Testing**: t-tests with p < 0.05 across 5 random seeds
- **Effect Size**: Cohen's d > 0.5 for meaningful improvements
- **Confidence Intervals**: 95% CI for all reported metrics

## Risk Assessment

### Technical Risks

#### High Risk: Training Instability
**Risk**: Dynamic anchor changes might destabilize training
**Mitigation**: 
- Implement smooth anchor transitions with momentum
- Add stability checks before anchor updates
- Fallback to static anchors if instability detected

#### Medium Risk: Computational Overhead
**Risk**: Node performance evaluation adds significant compute cost
**Mitigation**:
- Cache performance scores between updates
- Use efficient batch operations
- Implement early stopping for anchor selection

#### Low Risk: Overfitting to High-Performing Nodes
**Risk**: Selection might become too narrow, reducing diversity
**Mitigation**:
- Include diversity term in selection criteria
- Implement minimum anchor diversity constraints
- Monitor anchor distribution spread

### Implementation Risks

#### Integration Complexity
**Risk**: Complex integration with existing codebase
**Mitigation**:
- Maintain backward compatibility
- Implement feature flags for gradual rollout
- Comprehensive testing at each integration step

#### Hyperparameter Sensitivity
**Risk**: Method might be sensitive to hyperparameter choices
**Mitigation**:
- Extensive hyperparameter sensitivity analysis
- Provide robust default configurations
- Implement adaptive hyperparameter adjustment

## Future Extensions

### 1. Multi-Criteria Expansion
```python
# Additional criteria for anchor selection
criteria = {
    'gradient_stability': 0.1,    # Consistent gradient directions
    'structural_centrality': 0.1, # Graph centrality measures
    'representation_novelty': 0.1 # Novel representation discovery
}
```

### 2. Hierarchical Anchor Selection
```python
# Different anchor types for different purposes
class HierarchicalAnchorSelector:
    def select_task_specific_anchors(self, ...):  # Task performance focus
    def select_structural_anchors(self, ...):     # Graph structure focus
    def select_diversity_anchors(self, ...):      # Representation diversity focus
```

### 3. Cross-Domain Anchor Transfer
```python
# Transfer high-performing anchors across related domains
class CrossDomainAnchorTransfer:
    def transfer_anchors_between_domains(self, source_anchors, target_domain):
        # Intelligent anchor adaptation across domains
```

### 4. Theoretical Analysis
- **Convergence Guarantees**: Theoretical analysis of convergence properties
- **Generalization Bounds**: PAC-Bayes bounds for dynamic anchor selection
- **Optimization Landscape**: Analysis of loss surface properties

## Conclusion

The Dynamic Performance-Based Anchor Selection approach represents a significant advancement over static anchor generation methods. By leveraging real-time node performance metrics, this method creates more task-relevant reference distributions while maintaining the source-free constraint essential for practical domain adaptation scenarios.

### Key Innovations
1. **Performance-Driven**: Anchors are selected based on actual task performance
2. **Dynamic Adaptation**: Anchor quality improves as training progresses
3. **Source-Free**: No dependency on source domain data access
4. **Self-Supervised**: Target domain provides its own reference signals

### Expected Impact
- **Immediate**: 2-5% accuracy improvement over baseline methods
- **Medium-term**: Broader applicability to diverse graph domains
- **Long-term**: Foundation for adaptive regularization in graph neural networks

This design provides a robust framework for implementation while maintaining flexibility for future enhancements and research directions.

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-15  
**Authors**: Research Team  
**Status**: Design Phase - Ready for Implementation