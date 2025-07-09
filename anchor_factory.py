import torch
from sklearn.cluster import KMeans


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


def generate_mog_anchors(embeddings: torch.Tensor, num_components: int = 5):
    """
    Fit Mixture of Gaussians (KMeans centers only) as anchor prior
    Args:
        embeddings: torch.Tensor of shape [N, D]
        num_components: number of Gaussian modes

    Returns:
        torch.Tensor of shape [num_components, D]
    """
    kmeans = KMeans(n_clusters=num_components, n_init=10)
    centers = kmeans.fit(embeddings.cpu().numpy()).cluster_centers_
    return torch.tensor(centers, dtype=torch.float32, device=embeddings.device)