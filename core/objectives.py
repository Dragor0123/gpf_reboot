import torch
import torch.nn.functional as F
from typing import Optional


class EncoderObjectives:
    """
    Objective functions for gradient-based reference distribution generation.
    
    These objectives guide the optimization process to create better anchor
    distributions by maximizing encoder output quality without accessing
    source domain data.
    """
    
    @staticmethod
    def high_norm(embeddings: torch.Tensor, 
                  features: Optional[torch.Tensor] = None,
                  edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encourage strong activations in embeddings.
        
        High-norm embeddings indicate that the encoder is producing meaningful
        representations rather than collapsing to near-zero values.
        
        Args:
            embeddings: Encoder output embeddings [N, D]
            features: Input features (unused)
            edge_index: Graph edges (unused)
            
        Returns:
            Objective value (higher is better)
        """
        # L2 norm of embeddings
        norms = torch.norm(embeddings, dim=1)
        return norms.mean()
    
    @staticmethod
    def feature_diversity(embeddings: torch.Tensor,
                         features: Optional[torch.Tensor] = None,
                         edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Maximize embedding diversity to avoid mode collapse.
        
        Encourages the generated embeddings to be diverse and cover
        different regions of the embedding space.
        
        Args:
            embeddings: Encoder output embeddings [N, D]
            features: Input features (unused)
            edge_index: Graph edges (unused)
            
        Returns:
            Objective value (higher is better)
        """
        # Pairwise distance maximization
        # Compute pairwise L2 distances
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        distances = torch.cdist(embeddings_normalized, embeddings_normalized, p=2)
        
        # Avoid self-distances (diagonal)
        mask = ~torch.eye(len(embeddings), device=embeddings.device, dtype=bool)
        valid_distances = distances[mask]
        
        # Maximize minimum distances to encourage diversity
        min_distances = valid_distances.view(len(embeddings), -1).min(dim=1)[0]
        return min_distances.mean()
    
    @staticmethod
    def task_alignment(embeddings: torch.Tensor,
                      features: Optional[torch.Tensor] = None,
                      edge_index: Optional[torch.Tensor] = None,
                      labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Align embeddings with downstream task requirements.
        
        When labels are available, encourages embeddings that would be
        useful for the classification task. When not available, uses
        structural properties as proxy.
        
        Args:
            embeddings: Encoder output embeddings [N, D]
            features: Input features (unused)
            edge_index: Graph edges (unused)
            labels: Ground truth labels (optional)
            
        Returns:
            Objective value (higher is better)
        """
        if labels is not None:
            # Use label information if available
            # Maximize inter-class distances, minimize intra-class distances
            unique_labels = torch.unique(labels)
            
            if len(unique_labels) <= 1:
                # Fallback to diversity if only one class
                return EncoderObjectives.feature_diversity(embeddings)
            
            inter_class_dist = 0.0
            intra_class_dist = 0.0
            
            for label in unique_labels:
                mask = labels == label
                if mask.sum() <= 1:
                    continue
                    
                class_embeddings = embeddings[mask]
                class_center = class_embeddings.mean(dim=0)
                
                # Intra-class distance (minimize)
                intra_dist = torch.norm(class_embeddings - class_center, dim=1).mean()
                intra_class_dist += intra_dist
                
                # Inter-class distance (maximize)
                other_labels = unique_labels[unique_labels != label]
                for other_label in other_labels:
                    other_mask = labels == other_label
                    if other_mask.sum() > 0:
                        other_center = embeddings[other_mask].mean(dim=0)
                        inter_dist = torch.norm(class_center - other_center)
                        inter_class_dist += inter_dist
            
            # Maximize inter-class, minimize intra-class
            return inter_class_dist - intra_class_dist
        
        else:
            # Fallback: encourage embeddings with good separability properties
            # Use variance as a proxy for task-relevant structure
            embedding_variance = torch.var(embeddings, dim=0).mean()
            return embedding_variance
    
    @staticmethod
    def graph_homophily(embeddings: torch.Tensor,
                       features: Optional[torch.Tensor] = None,
                       edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Preserve graph structural properties (homophily).
        
        Encourages connected nodes to have similar embeddings,
        preserving the graph's structural information.
        
        Args:
            embeddings: Encoder output embeddings [N, D]
            features: Input features (unused)
            edge_index: Graph edge indices [2, E]
            
        Returns:
            Objective value (higher is better)
        """
        if edge_index is None or edge_index.size(1) == 0:
            # No edges available, return neutral value
            return torch.tensor(0.0, device=embeddings.device)
        
        # Get edge endpoints
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Get embeddings for connected nodes
        src_embeddings = embeddings[src_nodes]
        dst_embeddings = embeddings[dst_nodes]
        
        # Compute similarity between connected nodes
        similarities = F.cosine_similarity(src_embeddings, dst_embeddings, dim=1)
        
        # Maximize average similarity (homophily)
        return similarities.mean()
    
    @staticmethod
    def structural_preservation(embeddings: torch.Tensor,
                              features: Optional[torch.Tensor] = None,
                              edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Preserve structural properties through embedding geometry.
        
        Encourages embeddings to preserve distances between nodes
        based on graph structure.
        
        Args:
            embeddings: Encoder output embeddings [N, D]
            features: Input features (unused)
            edge_index: Graph edge indices [2, E]
            
        Returns:
            Objective value (higher is better)
        """
        if edge_index is None or edge_index.size(1) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute node degrees as structural importance
        num_nodes = embeddings.size(0)
        degrees = torch.zeros(num_nodes, device=embeddings.device)
        degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
        
        # Encourage high-degree nodes to have distinctive embeddings
        degree_weights = degrees / (degrees.max() + 1e-8)
        weighted_norms = torch.norm(embeddings, dim=1) * degree_weights
        
        return weighted_norms.mean()
    
    @staticmethod
    def embedding_smoothness(embeddings: torch.Tensor,
                           features: Optional[torch.Tensor] = None,
                           edge_index: Optional[torch.Tensor] = None,
                           alpha: float = 0.5) -> torch.Tensor:
        """
        Encourage smooth embeddings while preserving discriminative power.
        
        Balances between smooth embeddings (similar to graph_homophily)
        and discriminative embeddings (similar to feature_diversity).
        
        Args:
            embeddings: Encoder output embeddings [N, D]
            features: Input features (unused)
            edge_index: Graph edge indices [2, E]
            alpha: Balance between smoothness and diversity
            
        Returns:
            Objective value (higher is better)
        """
        smoothness = EncoderObjectives.graph_homophily(embeddings, features, edge_index)
        diversity = EncoderObjectives.feature_diversity(embeddings, features, edge_index)
        
        return alpha * smoothness + (1 - alpha) * diversity