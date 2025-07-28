import torch
from sklearn.cluster import KMeans
import logging

def generate_gaussian_anchors(num_anchors: int, latent_dim: int, device='cpu'):
    """
    Generate anchors from isotropic Gaussian prior N(0, I)
    """
    return torch.randn(num_anchors, latent_dim).to(device)

# ========== [Optional Placeholder for 선택지 B] ==========
def generate_latent_space_anchors(embeddings: torch.Tensor, top_k: int, strategy="random"):
    """
    Select anchors from existing latent embeddings (for 선택지 B).
    
    Args:
        embeddings: Tensor of shape [N, D] from encoder output
        top_k: Number of anchor vectors to select
        strategy: "random" or "high_entropy" or "kmeans"

    Returns:
        Tensor of shape [top_k, D]
    """
    N = embeddings.size(0)

    if strategy == "random":
        idx = torch.randperm(N)[:top_k]
        return embeddings[idx].detach()

    # TODO: 확장 가능한 전략 - 예: high entropy, k-means 중심 등
    raise NotImplementedError(f"Strategy '{strategy}' not implemented yet.")


def generate_mog_anchors(embeddings: torch.Tensor, 
                        num_components: int = 5, 
                        num_anchors: int = 500):
    """
    Generate anchors from Mixture of Gaussians fitted to embeddings
    
    Args:
        embeddings: Target embeddings from pretrained encoder [N, D]
        num_components: Number of Gaussian components
        num_anchors: Total number of anchor vectors to generate
    
    Returns:
        Generated anchor vectors [num_anchors, D]
    """
    from sklearn.mixture import GaussianMixture
    
    # Fit GMM to embeddings
    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type='full',  # Full covariance matrix
        random_state=42,
        n_init=3
    )
    
    embeddings_np = embeddings.cpu().numpy()
    gmm.fit(embeddings_np)
    
    # Sample from fitted GMM
    anchors_np, _ = gmm.sample(num_anchors)
    
    # Convert back to tensor
    anchors = torch.tensor(anchors_np, dtype=torch.float32, device=embeddings.device)
    
    logging.info(f"🎯 Generated {num_anchors} MoG anchors from {num_components} components")
    logging.info(f"   GMM converged: {gmm.converged_}")
    logging.info(f"   Log-likelihood: {gmm.score(embeddings_np):.4f}")
    
    return anchors


def generate_mog_anchors_simple(embeddings: torch.Tensor, 
                               num_components: int = 5,
                               num_anchors: int = 500):
    """
    Simplified MoG using KMeans + Gaussian sampling around centers
    (현재 구현의 개선된 버전)
    """
    from sklearn.cluster import KMeans
    import numpy as np
    
    # K-means clustering
    kmeans = KMeans(n_clusters=num_components, n_init=10, random_state=42)
    embeddings_np = embeddings.cpu().numpy()
    labels = kmeans.fit_predict(embeddings_np)
    centers = kmeans.cluster_centers_
    
    # Calculate variance for each cluster
    anchors_list = []
    samples_per_component = num_anchors // num_components
    
    for i in range(num_components):
        cluster_mask = labels == i
        cluster_points = embeddings_np[cluster_mask]
        
        if len(cluster_points) > 1:
            # Calculate cluster covariance
            cluster_cov = np.cov(cluster_points.T)
            
            # Add small regularization to diagonal
            cluster_cov += np.eye(cluster_cov.shape[0]) * 0.01
            
            # Sample from multivariate normal
            cluster_anchors = np.random.multivariate_normal(
                mean=centers[i],
                cov=cluster_cov,
                size=samples_per_component
            )
        else:
            # Fallback: isotropic Gaussian
            std = 0.5
            cluster_anchors = np.random.normal(
                loc=centers[i],
                scale=std,
                size=(samples_per_component, centers.shape[1])
            )
        
        anchors_list.append(cluster_anchors)
    
    # Combine all anchors
    all_anchors = np.vstack(anchors_list)
    
    # Handle remainder
    remainder = num_anchors - len(all_anchors)
    if remainder > 0:
        # Sample remainder from overall distribution
        overall_mean = embeddings_np.mean(axis=0)
        overall_cov = np.cov(embeddings_np.T) + np.eye(embeddings_np.shape[1]) * 0.01
        extra_anchors = np.random.multivariate_normal(
            mean=overall_mean,
            cov=overall_cov,
            size=remainder
        )
        all_anchors = np.vstack([all_anchors, extra_anchors])
    
    return torch.tensor(all_anchors[:num_anchors], 
                       dtype=torch.float32, device=embeddings.device)