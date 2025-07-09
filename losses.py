import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging
from abc import ABC, abstractmethod


# =============================================================================
# Legacy Functions (기존 코드 호환성 유지)
# =============================================================================

def compute_ce_loss(logits, labels):
    """
    Standard cross-entropy loss for classification.
    """
    return F.cross_entropy(logits, labels)


def compute_kl_divergence(p_logits, q_logits, reduction='batchmean'):
    """
    KL divergence between two probability distributions from logits.
    """
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction=reduction)


def gaussian_kernel(x, y, sigma=1.0):
    """
    RBF (Gaussian) kernel between two batches of vectors.
    """
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.exp(-dist / (2 * sigma ** 2 + 1e-8))


def compute_mmd_loss(x_samples, y_samples, sigma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between x_samples and y_samples.
    """
    K_xx = gaussian_kernel(x_samples, x_samples, sigma)
    K_yy = gaussian_kernel(y_samples, y_samples, sigma)
    K_xy = gaussian_kernel(x_samples, y_samples, sigma)

    m = x_samples.size(0)
    n = y_samples.size(0)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


def total_loss(ce=None, kl=None, mmd=None, weights={'ce':1.0, 'kl':0.1, 'mmd':0.1}):
    """
    Legacy aggregate loss function with optional CE, KL, and MMD terms.
    """
    loss = 0.0
    if ce is not None:
        loss += weights['ce'] * ce
    if kl is not None:
        loss += weights['kl'] * kl
    if mmd is not None:
        loss += weights['mmd'] * mmd
    return loss


# =============================================================================
# Enhanced Functions for Target-Centric
# =============================================================================

def compute_ce_loss_with_mask(logits, labels, mask=None):
    """Enhanced CE loss with mask support."""
    if mask is not None:
        logits = logits[mask]
        labels = labels[mask]
    return F.cross_entropy(logits, labels)


def compute_unbiased_mmd_loss(x_samples, y_samples, sigma=1.0):
    """Unbiased MMD estimate."""
    K_xx = gaussian_kernel(x_samples, x_samples, sigma)
    K_yy = gaussian_kernel(y_samples, y_samples, sigma)
    K_xy = gaussian_kernel(x_samples, y_samples, sigma)

    m = x_samples.size(0)
    n = y_samples.size(0)

    # Unbiased MMD estimate
    mmd = (K_xx.sum() - K_xx.diag().sum()) / (m * (m - 1)) if m > 1 else 0.0
    mmd += (K_yy.sum() - K_yy.diag().sum()) / (n * (n - 1)) if n > 1 else 0.0
    mmd -= 2 * K_xy.mean()
    
    return mmd


def compute_wasserstein_distance(x_samples, y_samples):
    """Simplified Wasserstein distance."""
    return torch.norm(x_samples.mean(dim=0) - y_samples.mean(dim=0), p=2)


def compute_cosine_similarity_loss(x_samples, y_samples):
    """Negative cosine similarity as loss."""
    x_mean = F.normalize(x_samples.mean(dim=0), dim=0)
    y_mean = F.normalize(y_samples.mean(dim=0), dim=0)
    return -F.cosine_similarity(x_mean, y_mean, dim=0)


# =============================================================================
# Target-Centric Abstract Classes
# =============================================================================

class AnchorSelector(ABC):
    """Abstract base class for anchor selection strategies."""
    
    @abstractmethod
    def select_anchors(self, target_features: torch.Tensor, 
                      edge_index: Optional[torch.Tensor] = None,
                      num_anchors: int = 100) -> torch.Tensor:
        pass


class AnchorMapper(ABC):
    """Abstract base class for mapping anchors to hidden space."""
    
    @abstractmethod
    def initialize(self, anchor_features: torch.Tensor, 
                  encoder: nn.Module, edge_index: Optional[torch.Tensor] = None):
        pass
    
    @abstractmethod
    def get_anchor_representations(self) -> torch.Tensor:
        pass


class DivergenceMetric(ABC):
    """Abstract base class for divergence metrics."""
    
    @abstractmethod
    def compute_divergence(self, prompted_embeddings: torch.Tensor,
                          anchor_representations: torch.Tensor) -> torch.Tensor:
        pass


# =============================================================================
# Concrete Implementations
# =============================================================================

class RandomAnchorSelector(AnchorSelector):
    """Random sampling of anchor nodes."""
    
    def select_anchors(self, target_features: torch.Tensor, 
                      edge_index: Optional[torch.Tensor] = None,
                      num_anchors: int = 100) -> torch.Tensor:
        num_nodes = target_features.size(0)
        num_anchors = min(num_anchors, num_nodes)
        indices = torch.randperm(num_nodes)[:num_anchors]
        return target_features[indices].clone()


class HighDegreeAnchorSelector(AnchorSelector):
    """Select high-degree nodes as anchors."""
    
    def select_anchors(self, target_features: torch.Tensor, 
                      edge_index: Optional[torch.Tensor] = None,
                      num_anchors: int = 100) -> torch.Tensor:
        if edge_index is None:
            logging.warning("No edge_index provided, falling back to random selection")
            return RandomAnchorSelector().select_anchors(target_features, None, num_anchors)
        
        # Compute node degrees
        num_nodes = target_features.size(0)
        degrees = torch.zeros(num_nodes, device=target_features.device)
        degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float))
        
        # Select top-degree nodes
        num_anchors = min(num_anchors, num_nodes)
        _, top_indices = torch.topk(degrees, num_anchors)
        return target_features[top_indices].clone()


class DiverseAnchorSelector(AnchorSelector):
    """
    개선된 Diverse Anchor Selector
    
    k-means로 클러스터링 후, 각 클러스터에서 중심점과 가장 가까운
    실제 노드들을 앵커로 선택
    """
    
    def __init__(self, selection_strategy: str = "closest_to_centroid", 
                 nodes_per_cluster: int = 1, confidence_threshold: float = None):
        """
        Args:
            selection_strategy: 노드 선택 전략
                - "closest_to_centroid": 중심점과 가장 가까운 노드
                - "most_confident": 클러스터 내에서 가장 confident한 노드들
                - "diverse_within_cluster": 클러스터 내에서도 다양성 고려
            nodes_per_cluster: 클러스터당 선택할 노드 수
            confidence_threshold: confidence 임계값 (선택적)
        """
        self.selection_strategy = selection_strategy
        self.nodes_per_cluster = nodes_per_cluster
        self.confidence_threshold = confidence_threshold
    
    def select_anchors(self, target_features: torch.Tensor, 
                      edge_index: Optional[torch.Tensor] = None,
                      num_anchors: int = 100) -> torch.Tensor:
        """
        다양성을 고려하여 실제 노드들을 앵커로 선택
        """
        # 클러스터 수 계산 (nodes_per_cluster 고려)
        num_clusters = max(1, num_anchors // self.nodes_per_cluster)
        
        # k-means 클러스터링 수행
        centroids, assignments = self._perform_kmeans(target_features, num_clusters)
        
        # 각 클러스터에서 노드 선택
        if self.selection_strategy == "closest_to_centroid":
            selected_nodes = self._select_closest_to_centroids(
                target_features, centroids, assignments, num_anchors
            )
        elif self.selection_strategy == "most_confident":
            selected_nodes = self._select_most_confident(
                target_features, centroids, assignments, num_anchors
            )
        elif self.selection_strategy == "diverse_within_cluster":
            selected_nodes = self._select_diverse_within_clusters(
                target_features, centroids, assignments, num_anchors
            )
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        logging.info(f"Selected {selected_nodes.size(0)} diverse anchor nodes using "
                    f"{self.selection_strategy} strategy")
        
        return selected_nodes
    
    def _perform_kmeans(self, features: torch.Tensor, k: int, 
                       max_iters: int = 20, tolerance: float = 1e-4):
        """
        개선된 k-means 클러스터링
        
        Returns:
            centroids: [k, d] 클러스터 중심점들
            assignments: [n] 각 노드의 클러스터 할당
        """
        n, d = features.shape
        k = min(k, n)
        device = features.device
        
        # k-means++ 초기화 (더 나은 초기 중심점 선택)
        centroids = self._kmeans_plus_plus_init(features, k)
        
        prev_loss = float('inf')
        
        for iteration in range(max_iters):
            # E-step: 클러스터 할당
            distances = torch.cdist(features, centroids)  # [n, k]
            assignments = torch.argmin(distances, dim=1)  # [n]
            
            # M-step: 중심점 업데이트
            new_centroids = torch.zeros_like(centroids)
            for i in range(k):
                cluster_mask = assignments == i
                if cluster_mask.sum() > 0:
                    new_centroids[i] = features[cluster_mask].mean(dim=0)
                else:
                    # 빈 클러스터 처리: 가장 먼 노드를 새 중심점으로
                    distances_to_all = torch.cdist(features, centroids).min(dim=1)[0]
                    farthest_idx = torch.argmax(distances_to_all)
                    new_centroids[i] = features[farthest_idx]
            
            # 수렴 체크
            centroid_shift = torch.norm(new_centroids - centroids)
            current_loss = self._compute_kmeans_loss(features, new_centroids, assignments)
            
            centroids = new_centroids
            
            # 수렴 조건
            if centroid_shift < tolerance or abs(prev_loss - current_loss) < tolerance:
                logging.debug(f"K-means converged at iteration {iteration}")
                break
            
            prev_loss = current_loss
        
        return centroids, assignments
    
    def _kmeans_plus_plus_init(self, features: torch.Tensor, k: int) -> torch.Tensor:
        """k-means++ 초기화로 더 나은 시작점 선택"""
        n, d = features.shape
        centroids = torch.zeros(k, d, device=features.device)
        
        # 첫 번째 중심점은 랜덤하게
        centroids[0] = features[torch.randint(0, n, (1,))]
        
        for i in range(1, k):
            # 기존 중심점들과의 최소 거리 계산
            distances = torch.cdist(features, centroids[:i])  # [n, i]
            min_distances = torch.min(distances, dim=1)[0]  # [n]
            
            # 거리에 비례한 확률로 다음 중심점 선택
            probabilities = min_distances / min_distances.sum()
            next_idx = torch.multinomial(probabilities, 1)
            centroids[i] = features[next_idx]
        
        return centroids
    
    def _compute_kmeans_loss(self, features: torch.Tensor, centroids: torch.Tensor, 
                            assignments: torch.Tensor) -> float:
        """k-means 손실 계산 (Within-Cluster Sum of Squares)"""
        total_loss = 0.0
        for i in range(centroids.size(0)):
            cluster_mask = assignments == i
            if cluster_mask.sum() > 0:
                cluster_points = features[cluster_mask]
                distances = torch.norm(cluster_points - centroids[i], dim=1)
                total_loss += (distances ** 2).sum().item()
        return total_loss
    
    def _select_closest_to_centroids(self, features: torch.Tensor, 
                                   centroids: torch.Tensor, assignments: torch.Tensor,
                                   num_anchors: int) -> torch.Tensor:
        """각 클러스터에서 중심점과 가장 가까운 노드들 선택"""
        selected_nodes = []
        k = centroids.size(0)
        
        for i in range(k):
            cluster_mask = assignments == i
            if cluster_mask.sum() == 0:
                continue
                
            cluster_features = features[cluster_mask]
            cluster_indices = torch.where(cluster_mask)[0]
            
            # 중심점과의 거리 계산
            distances = torch.norm(cluster_features - centroids[i], dim=1)
            
            # 가장 가까운 노드들 선택
            num_to_select = min(self.nodes_per_cluster, len(cluster_features))
            _, closest_indices = torch.topk(distances, num_to_select, largest=False)
            
            for idx in closest_indices:
                selected_nodes.append(cluster_features[idx])
                
                # 충분한 앵커를 선택했으면 종료
                if len(selected_nodes) >= num_anchors:
                    break
            
            if len(selected_nodes) >= num_anchors:
                break
        
        # 정확히 num_anchors개만 반환
        selected_nodes = selected_nodes[:num_anchors]
        return torch.stack(selected_nodes) if selected_nodes else features[:num_anchors]
    
    def _select_most_confident(self, features: torch.Tensor, 
                             centroids: torch.Tensor, assignments: torch.Tensor,
                             num_anchors: int) -> torch.Tensor:
        """
        가장 confident한 노드들 선택
        
        Confidence = 1 / (1 + distance_to_centroid)
        즉, 중심점에 가까울수록 높은 confidence
        """
        confidences = []
        node_features = []
        
        k = centroids.size(0)
        
        for i in range(k):
            cluster_mask = assignments == i
            if cluster_mask.sum() == 0:
                continue
                
            cluster_features = features[cluster_mask]
            distances = torch.norm(cluster_features - centroids[i], dim=1)
            
            # Confidence 계산 (거리가 가까울수록 높음)
            cluster_confidences = 1.0 / (1.0 + distances)
            
            confidences.extend(cluster_confidences.tolist())
            node_features.extend(cluster_features.tolist())
        
        # 모든 노드를 confidence 순으로 정렬
        if confidences:
            confidence_tensor = torch.tensor(confidences)
            features_tensor = torch.tensor(node_features)
            
            # 상위 confidence 노드들 선택
            _, top_indices = torch.topk(confidence_tensor, 
                                      min(num_anchors, len(confidences)), 
                                      largest=True)
            
            selected_features = features_tensor[top_indices]
            
            # Confidence threshold 적용 (선택적)
            if self.confidence_threshold is not None:
                high_conf_mask = confidence_tensor[top_indices] >= self.confidence_threshold
                selected_features = selected_features[high_conf_mask]
                
                if selected_features.size(0) == 0:
                    logging.warning("No nodes meet confidence threshold, using top nodes")
                    selected_features = features_tensor[top_indices[:num_anchors]]
            
            return selected_features
        else:
            return features[:num_anchors]
    
    def _select_diverse_within_clusters(self, features: torch.Tensor, 
                                      centroids: torch.Tensor, assignments: torch.Tensor,
                                      num_anchors: int) -> torch.Tensor:
        """클러스터 내에서도 다양성을 고려한 선택"""
        selected_nodes = []
        k = centroids.size(0)
        
        for i in range(k):
            cluster_mask = assignments == i
            if cluster_mask.sum() == 0:
                continue
                
            cluster_features = features[cluster_mask]
            
            if cluster_features.size(0) <= self.nodes_per_cluster:
                # 클러스터가 작으면 모든 노드 선택
                selected_nodes.extend(cluster_features.tolist())
            else:
                # 클러스터 내에서 다시 k-means 적용하여 다양성 확보
                mini_centroids, mini_assignments = self._perform_kmeans(
                    cluster_features, self.nodes_per_cluster
                )
                
                # 각 미니 클러스터에서 중심점과 가장 가까운 노드 선택
                for j in range(self.nodes_per_cluster):
                    mini_cluster_mask = mini_assignments == j
                    if mini_cluster_mask.sum() > 0:
                        mini_cluster_features = cluster_features[mini_cluster_mask]
                        distances = torch.norm(mini_cluster_features - mini_centroids[j], dim=1)
                        closest_idx = torch.argmin(distances)
                        selected_nodes.append(mini_cluster_features[closest_idx])
            
            if len(selected_nodes) >= num_anchors:
                break
        
        selected_nodes = selected_nodes[:num_anchors]
        return torch.stack(selected_nodes) if selected_nodes else features[:num_anchors]


class EncoderAnchorMapper(AnchorMapper):
    """Map anchors using the frozen encoder (Method 3)."""
    
    def __init__(self):
        self.anchor_representations = None
    
    def initialize(self, anchor_features: torch.Tensor, 
                  encoder: nn.Module, edge_index: Optional[torch.Tensor] = None):
        """Initialize anchor representations using encoder."""
        with torch.no_grad():
            encoder.eval()
            
            # Create a temporary graph with only anchor nodes
            num_anchors = anchor_features.size(0)
            device = anchor_features.device
            
            if edge_index is not None:
                # Use full graph context for encoding anchors
                # Create a subgraph or use full graph - for simplicity, use isolated nodes
                empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                anchor_embeddings = encoder(anchor_features, empty_edge_index)
            else:
                # Isolated nodes
                empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                anchor_embeddings = encoder(anchor_features, empty_edge_index)
            
            self.anchor_representations = anchor_embeddings.clone()
    
    def get_anchor_representations(self) -> torch.Tensor:
        if self.anchor_representations is None:
            raise RuntimeError("Anchor mapper not initialized")
        return self.anchor_representations


class ProjectionAnchorMapper(AnchorMapper):
    """Map anchors using learnable projection (Method 1) - 차원 자동 감지."""
    
    def __init__(self, feature_dim: Optional[int] = None, hidden_dim: int = 128):
        """
        Args:
            feature_dim: 입력 특성 차원 (None이면 초기화 시 자동 감지)
            hidden_dim: 출력 히든 차원
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.projector = None  # 초기화 시에 생성
        self.anchor_features = None
    
    def initialize(self, anchor_features: torch.Tensor, 
                  encoder: nn.Module, edge_index: Optional[torch.Tensor] = None):
        """Store anchor features for projection."""
        self.anchor_features = anchor_features.clone()
        
        # 실제 특성 차원 자동 감지
        actual_feature_dim = anchor_features.size(1)
        
        if self.projector is None:
            # 처음 초기화할 때 실제 차원으로 projector 생성
            self.projector = nn.Linear(actual_feature_dim, self.hidden_dim)
            self.projector = self.projector.to(anchor_features.device)
            
            logging.info(f"🔧 ProjectionAnchorMapper: Auto-detected feature_dim={actual_feature_dim}, "
                        f"hidden_dim={self.hidden_dim}")
        
        elif self.projector.in_features != actual_feature_dim:
            # 차원이 맞지 않으면 새로 생성
            logging.warning(f"⚠️  Feature dimension mismatch! Expected {self.projector.in_features}, "
                           f"got {actual_feature_dim}. Re-creating projector.")
            
            self.projector = nn.Linear(actual_feature_dim, self.hidden_dim)
            self.projector = self.projector.to(anchor_features.device)
    
    def get_anchor_representations(self) -> torch.Tensor:
        if self.anchor_features is None:
            raise RuntimeError("Anchor mapper not initialized")
        if self.projector is None:
            raise RuntimeError("Projector not initialized")
        
        return self.projector(self.anchor_features)
    

class MMDDivergence(DivergenceMetric):
    """MMD divergence metric."""
    
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
    
    def compute_divergence(self, prompted_embeddings: torch.Tensor,
                          anchor_representations: torch.Tensor) -> torch.Tensor:
        return compute_unbiased_mmd_loss(prompted_embeddings, anchor_representations, self.sigma)


class WassersteinDivergence(DivergenceMetric):
    """Wasserstein divergence metric."""
    
    def compute_divergence(self, prompted_embeddings: torch.Tensor,
                          anchor_representations: torch.Tensor) -> torch.Tensor:
        return compute_wasserstein_distance(prompted_embeddings, anchor_representations)


class CosineDivergence(DivergenceMetric):
    """Cosine similarity based divergence."""
    
    def compute_divergence(self, prompted_embeddings: torch.Tensor,
                          anchor_representations: torch.Tensor) -> torch.Tensor:
        return compute_cosine_similarity_loss(prompted_embeddings, anchor_representations)


# =============================================================================
# Factory Functions
# =============================================================================

def create_anchor_selector(selector_type: str, **kwargs) -> AnchorSelector:
    """Factory function for anchor selectors."""
    selectors = {
        'random': RandomAnchorSelector,
        'high_degree': HighDegreeAnchorSelector,
        'diverse': lambda: DiverseAnchorSelector(
            selection_strategy=kwargs.get('strategy', 'closest_to_centroid'),
            nodes_per_cluster=kwargs.get('nodes_per_cluster', 1),
            confidence_threshold=kwargs.get('confidence_threshold', None)
        ),
    }
    
    if selector_type not in selectors:
        raise ValueError(f"Unknown anchor selector: {selector_type}")
    
    return selectors[selector_type]()



# Factory 함수도 업데이트
def create_anchor_mapper(mapper_type: str, **kwargs) -> AnchorMapper:
    """Factory function for anchor mappers."""
    if mapper_type == 'encoder':
        return EncoderAnchorMapper()
    elif mapper_type == 'projection':
        # feature_dim을 None으로 설정하여 자동 감지 활성화
        return ProjectionAnchorMapper(
            feature_dim=kwargs.get('feature_dim', None),  # None이면 자동 감지
            hidden_dim=kwargs.get('hidden_dim', 128)
        )
    else:
        raise ValueError(f"Unknown anchor mapper: {mapper_type}")


def create_divergence_metric(metric_type: str, **kwargs) -> DivergenceMetric:
    """Factory function for divergence metrics."""
    metrics = {
        'mmd': lambda: MMDDivergence(kwargs.get('sigma', 1.0)),
        'wasserstein': WassersteinDivergence,
        'cosine': CosineDivergence,
    }
    
    if metric_type not in metrics:
        raise ValueError(f"Unknown divergence metric: {metric_type}")
    
    return metrics[metric_type]()


# =============================================================================
# Target-Centric Regularizer
# =============================================================================
# TargetCentricRegularizer도 업데이트하여 projector 등록 처리
class TargetCentricRegularizer(nn.Module):
    """Target-Centric Prior Modeling Regularizer with flexible design."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Parse configuration with defaults
        anchor_config = config.get('anchor', {})
        mapper_config = config.get('mapper', {})
        divergence_config = config.get('divergence', {})
        
        # Handle Gaussian anchor separately (direct anchor injection mode)
        anchor_type = anchor_config.get('type', 'random')
        self.is_fixed_anchor_mode = (anchor_type == 'gaussian')  # 선택지 A용 조건
        
        if self.is_fixed_anchor_mode:
            self.anchor_selector = None  # selector 사용하지 않음
            self.anchor_mapper = None    # mapper도 비활성
        else:
            self.anchor_selector = create_anchor_selector(
                anchor_type,
                **anchor_config.get('params', {})
            )
            
            self.anchor_mapper = create_anchor_mapper(
                mapper_config.get('type', 'encoder'),
                **mapper_config.get('params', {})
            )

        self.divergence_metric = create_divergence_metric(
            divergence_config.get('type', 'mmd'),
            **divergence_config.get('params', {})
        )
        
        self.beta = config.get('beta', 0.1)
        self.num_anchors = anchor_config.get('num_anchors', 100)
        
        # projector 등록은 초기화 후에 수행
        self.projector_registered = False
        self.fixed_anchors = None  # 선택지 A용 수동 anchor 보관 공간

    def initialize_fixed_anchors(self, anchor_vectors: torch.Tensor):
        """
        선택지 A (gaussian anchor prior)에서 직접 anchor 벡터를 주입받는 경우.
        """
        self.fixed_anchors = anchor_vectors.detach()
        logging.info(f"✅ [Gaussian] Fixed anchors registered with shape: {self.fixed_anchors.shape}")

    def initialize_anchors(self, target_features: torch.Tensor, 
                          encoder: nn.Module, edge_index: Optional[torch.Tensor] = None):
        """Initialize anchor points and their representations."""
        if self.is_fixed_anchor_mode:
            # 선택지 A에서는 이 함수가 불리면 안 됨
            raise RuntimeError("❌ Cannot initialize anchors when using fixed Gaussian prior.")
        
        # Step 1: Select anchor nodes from original target features
        anchor_features = self.anchor_selector.select_anchors(
            target_features, edge_index, self.num_anchors
        )
        
        logging.info(f"Selected {anchor_features.size(0)} anchors using {type(self.anchor_selector).__name__}")
        
        # Step 2: Map anchors to hidden space
        self.anchor_mapper.initialize(anchor_features, encoder, edge_index)
        
        # Step 3: Register projector parameters if needed (for Method 1)
        if hasattr(self.anchor_mapper, 'projector') and not self.projector_registered:
            if self.anchor_mapper.projector is not None:
                self.projector = self.anchor_mapper.projector
                self.projector_registered = True
                logging.info("🔧 Registered projector parameters for training")
        
        logging.info(f"Mapped anchors using {type(self.anchor_mapper).__name__}")
    
    def forward(self, prompted_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss."""
        if self.is_fixed_anchor_mode:
            anchor_representations = self.fixed_anchors
        else:
            anchor_representations = self.anchor_mapper.get_anchor_representations()
        
        divergence = self.divergence_metric.compute_divergence(
            prompted_embeddings, anchor_representations
        )
        
        return self.beta * divergence


# =============================================================================
# Enhanced Target-Centric Loss (호환성 유지)
# =============================================================================

class TargetCentricLoss(nn.Module):
    """
    Enhanced Target-Centric Prior Modeling Loss Function.
    
    Backward compatible with existing TargetCentricLoss interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.target_centric_enabled = config['target_centric']['enable']
        
        if self.target_centric_enabled:
            reg_config = config['target_centric']['regularization']
            
            # Handle both old and new config formats
            if 'anchor' in reg_config:
                # New flexible format
                self.regularizer = TargetCentricRegularizer(reg_config)
            else:
                # Old format - convert to new format
                old_config = {
                    'beta': reg_config.get('beta', 0.1),
                    'anchor': {
                        'type': 'random',
                        'num_anchors': reg_config.get('mmd', {}).get('num_anchors', 100)
                    },
                    'mapper': {
                        'type': 'encoder'
                    },
                    'divergence': {
                        'type': reg_config.get('type', 'mmd'),
                        'params': {
                            'sigma': reg_config.get('mmd', {}).get('sigma', 1.0)
                        }
                    }
                }
                self.regularizer = TargetCentricRegularizer(old_config)
        else:
            self.regularizer = None
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, 
                embeddings: torch.Tensor, mask: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None):
        """
        Compute total loss.
        
        Args:
            logits: Model predictions [N, C]
            labels: True labels [N]
            embeddings: Node embeddings [N, D] (prompted embeddings)
            mask: Training mask [N]
            edge_index: Graph edges [2, E]
        """
        # Task alignment loss
        task_loss = compute_ce_loss_with_mask(logits, labels, mask)
        
        losses = {
            'task_loss': task_loss,
            'total_loss': task_loss
        }
        
        # Add regularization if enabled
        if self.target_centric_enabled and self.regularizer is not None:
            reg_loss = self.regularizer(embeddings)
            losses['reg_loss'] = reg_loss
            losses['total_loss'] = task_loss + reg_loss
        else:
            losses['reg_loss'] = torch.tensor(0.0, device=task_loss.device)
        
        return losses
    
    def initialize_regularizer(self, embeddings: torch.Tensor):
        """
        Legacy interface - initialize with embeddings.
        NOTE: This is the OLD way, new way uses target_features + encoder.
        """
        if self.regularizer is not None:
            # For backward compatibility, create dummy anchor features
            logging.warning("Using legacy initialize_regularizer interface")
            # This won't work properly with the new Target-Centric design
            # Use initialize_regularizer_with_target_features instead
    
    def initialize_regularizer_with_target_features(self, target_features: torch.Tensor, 
                                                   encoder: nn.Module, 
                                                   edge_index: Optional[torch.Tensor] = None):
        """
        New interface - initialize with original target features and encoder.
        """
        if self.regularizer is not None:
            logging.info("🔧 Initializing Target-Centric regularizer with target features")
            self.regularizer.initialize_anchors(target_features, encoder, edge_index)
            
    def initialize_regularizer_with_fixed_anchors(self, anchors: torch.Tensor):
        self.anchor_vectors = anchors.detach()
        self.mapper_type = "identity"
