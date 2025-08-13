import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Callable


class GradientBasedReferenceGenerator:
    """
    Strategy 1: Gradient-Based Optimization for source-free reference distribution generation.
    
    This class implements gradient-based optimization to create ideal reference distributions
    by optimizing input features to maximize encoder output quality, without accessing
    source domain data.
    """
    
    def __init__(self, encoder: nn.Module, target_data: torch.Tensor, device: str = 'cpu'):
        """
        Initialize the gradient-based reference generator.
        
        Args:
            encoder: Pretrained frozen encoder
            target_data: Target domain data for optimization context
            device: Device for computation
        """
        self.encoder = encoder
        self.target_data = target_data
        self.device = device
        self.encoder.eval()
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        logging.info("🎯 GradientBasedReferenceGenerator initialized")
        logging.info(f"   Target data shape: {target_data.shape}")
        logging.info(f"   Device: {device}")
    
    def generate_reference_distribution(self, 
                                      num_anchors: int = 1000,
                                      num_iterations: int = 100,
                                      learning_rate: float = 0.01,
                                      objective_weights: Optional[Dict[str, float]] = None,
                                      regularization_lambda: float = 0.1,
                                      edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Main entry point for reference distribution generation.
        
        Args:
            num_anchors: Number of anchor points to generate
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient ascent
            objective_weights: Weights for different objectives
            regularization_lambda: Regularization strength
            edge_index: Graph edge indices for structural objectives
            
        Returns:
            Generated anchor distribution tensor [num_anchors, feature_dim]
        """
        if objective_weights is None:
            objective_weights = {
                'high_norm': 0.3,
                'diversity': 0.4,
                'task_alignment': 0.3
            }
        
        logging.info(f"🚀 Starting gradient-based anchor generation:")
        logging.info(f"   Num anchors: {num_anchors}")
        logging.info(f"   Iterations: {num_iterations}")
        logging.info(f"   Learning rate: {learning_rate}")
        logging.info(f"   Objective weights: {objective_weights}")
        logging.info(f"   Regularization λ: {regularization_lambda}")
        
        # Initialize learnable features from target distribution
        optimized_features = self._initialize_features(num_anchors)
        original_features = optimized_features.clone().detach()
        
        # Define objectives
        objectives = self._get_objective_functions(objective_weights, edge_index)
        
        # Optimize input features
        final_features = self.optimize_input_for_encoder_output(
            optimized_features=optimized_features,
            original_features=original_features,
            objectives=objectives,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            regularization_lambda=regularization_lambda,
            edge_index=edge_index
        )
        
        logging.info(f"✅ Gradient optimization completed")
        logging.info(f"   Generated anchors shape: {final_features.shape}")
        
        return final_features.detach()
    
    def optimize_input_for_encoder_output(self,
                                        optimized_features: torch.Tensor,
                                        original_features: torch.Tensor,
                                        objectives: List[Callable],
                                        num_iterations: int = 100,
                                        learning_rate: float = 0.01,
                                        regularization_lambda: float = 0.1,
                                        edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Core optimization loop using gradient ascent.
        
        Args:
            optimized_features: Learnable input features
            original_features: Original features for regularization
            objectives: List of objective functions
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate
            regularization_lambda: Regularization strength
            edge_index: Graph edge indices
            
        Returns:
            Optimized features
        """
        # Make features learnable
        optimized_features.requires_grad_(True)
        optimizer = torch.optim.Adam([optimized_features], lr=learning_rate)
        
        best_loss = float('-inf')
        best_features = optimized_features.clone()
        convergence_threshold = 1e-6
        patience = 20
        no_improve_count = 0
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Get encoder embeddings for anchor features (use empty edge_index)
            # Create empty edge_index since we're optimizing isolated anchor features
            device = optimized_features.device
            empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            embeddings = self.encoder(optimized_features, empty_edge_index)
            
            # Compute multi-objective loss
            total_objective = 0.0
            objective_details = {}
            
            for obj_name, obj_func, weight in objectives:
                # Skip objectives with zero weight
                if weight == 0.0:
                    continue
                    
                # Skip graph objectives when using empty edge_index for anchor optimization
                if obj_name == 'graph_homophily':
                    continue  # Skip graph objectives during anchor optimization
                    
                obj_value = obj_func(embeddings, optimized_features, empty_edge_index)
                total_objective += weight * obj_value
                objective_details[obj_name] = obj_value.item()
            
            # Add regularization to prevent divergence
            regularization_penalty = regularization_lambda * torch.norm(
                optimized_features - original_features, p=2
            )
            
            # Final loss (negative because we want to maximize objectives)
            loss = total_objective - regularization_penalty
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([optimized_features], max_norm=1.0)
            
            optimizer.step()
            
            # Convergence check and logging
            if iteration % 20 == 0 or iteration == num_iterations - 1:
                logging.info(f"   Iter {iteration:03d} | Loss: {loss.item():.4f} | "
                           f"Reg: {regularization_penalty.item():.4f}")
                
                # Track best solution
                if loss.item() > best_loss:
                    best_loss = loss.item()
                    best_features = optimized_features.clone().detach()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Early stopping
            if no_improve_count >= patience:
                logging.info(f"   Early stopping at iteration {iteration}")
                break
                
            # Check for convergence
            if iteration > 0 and abs(loss.item() - best_loss) < convergence_threshold:
                logging.info(f"   Converged at iteration {iteration}")
                break
        
        return best_features
    
    def _initialize_features(self, num_anchors: int) -> torch.Tensor:
        """
        Initialize learnable features from target data distribution.
        
        Args:
            num_anchors: Number of anchor points to initialize
            
        Returns:
            Initialized features tensor
        """
        feature_dim = self.target_data.shape[1]
        
        # Strategy: Sample from target data distribution with noise
        if len(self.target_data) >= num_anchors:
            # Sample from existing target data
            indices = torch.randperm(len(self.target_data))[:num_anchors]
            base_features = self.target_data[indices].clone()
        else:
            # Repeat and sample with replacement
            repeat_factor = (num_anchors // len(self.target_data)) + 1
            repeated_data = self.target_data.repeat(repeat_factor, 1)
            indices = torch.randperm(len(repeated_data))[:num_anchors]
            base_features = repeated_data[indices].clone()
        
        # Add small random noise for diversity
        noise = torch.randn_like(base_features) * 0.1
        initialized_features = base_features + noise
        
        return initialized_features.to(self.device)
    
    def _get_objective_functions(self, 
                               objective_weights: Dict[str, float],
                               edge_index: Optional[torch.Tensor] = None) -> List[tuple]:
        """
        Get list of objective functions with their weights.
        
        Args:
            objective_weights: Dictionary of objective names and weights
            edge_index: Graph edge indices
            
        Returns:
            List of (name, function, weight) tuples
        """
        from .objectives import EncoderObjectives
        
        objectives = []
        
        if 'high_norm' in objective_weights:
            objectives.append((
                'high_norm', 
                EncoderObjectives.high_norm, 
                objective_weights['high_norm']
            ))
        
        if 'diversity' in objective_weights:
            objectives.append((
                'diversity', 
                EncoderObjectives.feature_diversity, 
                objective_weights['diversity']
            ))
        
        if 'task_alignment' in objective_weights:
            objectives.append((
                'task_alignment', 
                EncoderObjectives.task_alignment, 
                objective_weights['task_alignment']
            ))
        
        if 'graph_homophily' in objective_weights and edge_index is not None:
            objectives.append((
                'graph_homophily', 
                EncoderObjectives.graph_homophily, 
                objective_weights['graph_homophily']
            ))
        
        return objectives
    
    def _create_dummy_edge_index(self, num_nodes: int) -> torch.Tensor:
        """
        Create dummy edge index for non-graph data.
        Creates a simple chain graph: 0-1-2-...-n
        
        Args:
            num_nodes: Number of nodes
            
        Returns:
            Edge index tensor
        """
        if num_nodes <= 1:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])  # Undirected
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        return edge_index