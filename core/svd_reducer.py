"""SVD-based feature dimensionality reduction for cross-domain alignment."""

import torch
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class SVDFeatureReducer:
    """SVD-based feature dimensionality reducer.
    
    Standardizes all datasets to the same target dimension for cross-domain
    compatibility. Uses TruncatedSVD for efficiency and handles dimension
    mismatches gracefully.
    
    Args:
        target_dim: Target feature dimension (default: 100)
        algorithm: SVD algorithm ('randomized' or 'arpack')
        random_state: Random state for reproducible results
    """
    
    def __init__(self, target_dim: int = 100, algorithm: str = 'randomized', 
                 random_state: int = 42):
        self.target_dim = target_dim
        self.algorithm = algorithm
        self.random_state = random_state
        
        # Internal state
        self.mean_ = None
        self.svd_model = None
        self.is_fitted = False
        self.original_dim = None
        self.explained_variance_ratio = None
        self.actual_components = None
    
    def fit(self, X: torch.Tensor) -> 'SVDFeatureReducer':
        """Fit SVD model to feature matrix.
        
        Args:
            X: Feature matrix of shape [N, D]
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If input tensor has wrong dimensions
        """
        if X.dim() != 2:
            raise ValueError(f"Input tensor must be 2D, got {X.dim()}D")
        
        # Convert to numpy for sklearn compatibility
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        self.original_dim = X_np.shape[1]
        
        # Validate target dimension
        max_components = min(X_np.shape) - 1
        self.actual_components = min(self.target_dim, max_components)
        
        if self.actual_components <= 0:
            raise ValueError(f"Cannot reduce to {self.target_dim} components from "
                           f"input shape {X_np.shape}")
        
        logger.info(f"Fitting SVD: {self.original_dim}D → {self.actual_components}D")
        
        # Center the data
        self.mean_ = np.mean(X_np, axis=0)
        X_centered = X_np - self.mean_
        
        # Fit SVD model
        self.svd_model = TruncatedSVD(
            n_components=self.actual_components,
            algorithm=self.algorithm,
            random_state=self.random_state
        )
        
        try:
            self.svd_model.fit(X_centered)
            self.explained_variance_ratio = self.svd_model.explained_variance_ratio_
            self.is_fitted = True
            
            total_variance = self.explained_variance_ratio.sum()
            logger.info(f"✅ SVD fitted successfully")
            logger.info(f"   Components: {self.actual_components}")
            logger.info(f"   Explained variance: {total_variance:.4f}")
            
        except Exception as e:
            logger.error(f"SVD fitting failed: {e}")
            raise
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform features using fitted SVD.
        
        Args:
            X: Feature matrix of shape [N, D]
            
        Returns:
            Reduced feature matrix of shape [N, target_dim]
            
        Raises:
            ValueError: If SVD reducer is not fitted
            RuntimeError: If transformation fails
        """
        if not self.is_fitted:
            raise ValueError("SVD reducer must be fitted before transform")
        
        device = X.device if isinstance(X, torch.Tensor) else None
        
        # Convert to numpy
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
        
        # Handle dimension mismatch
        if X_np.shape[1] != self.original_dim:
            X_np = self._handle_dimension_mismatch(X_np)
        
        try:
            # Center and transform
            X_centered = X_np - self.mean_
            X_reduced = self.svd_model.transform(X_centered)
            
            # Pad or truncate to exact target dimension if needed
            if X_reduced.shape[1] != self.target_dim:
                X_reduced = self._adjust_to_target_dim(X_reduced)
            
            # Convert back to tensor
            if device is not None:
                return torch.tensor(X_reduced, dtype=torch.float32, device=device)
            else:
                return torch.tensor(X_reduced, dtype=torch.float32)
                
        except Exception as e:
            logger.error(f"SVD transformation failed: {e}")
            raise RuntimeError(f"SVD transformation failed: {e}")
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit SVD and transform features in one step.
        
        Args:
            X: Feature matrix of shape [N, D]
            
        Returns:
            Reduced feature matrix of shape [N, target_dim]
        """
        return self.fit(X).transform(X)
    
    def _handle_dimension_mismatch(self, X_np: np.ndarray) -> np.ndarray:
        """Handle dimension mismatches between training and inference data."""
        current_dim = X_np.shape[1]
        
        if current_dim < self.original_dim:
            # Zero-pad to match original dimension
            logger.warning(f"Input dimension {current_dim} < original {self.original_dim}, "
                         f"zero-padding")
            padded = np.zeros((X_np.shape[0], self.original_dim))
            padded[:, :current_dim] = X_np
            return padded
        
        elif current_dim > self.original_dim:
            # Truncate to match original dimension
            logger.warning(f"Input dimension {current_dim} > original {self.original_dim}, "
                         f"truncating")
            return X_np[:, :self.original_dim]
        
        return X_np
    
    def _adjust_to_target_dim(self, X_reduced: np.ndarray) -> np.ndarray:
        """Adjust reduced features to exact target dimension."""
        current_dim = X_reduced.shape[1]
        
        if current_dim < self.target_dim:
            # Pad with zeros
            padded = np.zeros((X_reduced.shape[0], self.target_dim))
            padded[:, :current_dim] = X_reduced
            logger.debug(f"Padded from {current_dim}D to {self.target_dim}D")
            return padded
        
        elif current_dim > self.target_dim:
            # Truncate (should not happen in normal operation)
            logger.warning(f"Truncating from {current_dim}D to {self.target_dim}D")
            return X_reduced[:, :self.target_dim]
        
        return X_reduced
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save SVD reducer to file.
        
        Args:
            filepath: Path to save the reducer
            
        Raises:
            ValueError: If reducer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted SVD reducer")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'target_dim': self.target_dim,
            'algorithm': self.algorithm,
            'random_state': self.random_state,
            'original_dim': self.original_dim,
            'actual_components': self.actual_components,
            'mean_': self.mean_,
            'svd_model': self.svd_model,
            'explained_variance_ratio': self.explained_variance_ratio,
            'is_fitted': self.is_fitted,
            'version': '1.0'  # For future compatibility
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"💾 SVD reducer saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save SVD reducer: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SVDFeatureReducer':
        """Load SVD reducer from file.
        
        Args:
            filepath: Path to the saved reducer
            
        Returns:
            Loaded SVD reducer instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is corrupted or incompatible
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"SVD reducer file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Validate saved data
            required_keys = ['target_dim', 'original_dim', 'mean_', 'svd_model', 
                           'explained_variance_ratio', 'is_fitted']
            
            for key in required_keys:
                if key not in save_data:
                    raise ValueError(f"Corrupted save file: missing key '{key}'")
            
            # Create instance and restore state
            reducer = cls(
                target_dim=save_data['target_dim'],
                algorithm=save_data.get('algorithm', 'randomized'),
                random_state=save_data.get('random_state', 42)
            )
            
            reducer.original_dim = save_data['original_dim']
            reducer.actual_components = save_data.get('actual_components', 
                                                    save_data['target_dim'])
            reducer.mean_ = save_data['mean_']
            reducer.svd_model = save_data['svd_model']
            reducer.explained_variance_ratio = save_data['explained_variance_ratio']
            reducer.is_fitted = save_data['is_fitted']
            
            logger.info(f"📂 SVD reducer loaded: {filepath}")
            logger.info(f"   {reducer.original_dim}D → {reducer.actual_components}D")
            
            return reducer
            
        except Exception as e:
            logger.error(f"Failed to load SVD reducer: {e}")
            raise ValueError(f"Failed to load SVD reducer from {filepath}: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the SVD reducer.
        
        Returns:
            Dictionary containing reducer information
        """
        if not self.is_fitted:
            return {
                "fitted": False,
                "target_dim": self.target_dim,
                "algorithm": self.algorithm,
                "random_state": self.random_state
            }
        
        return {
            "fitted": True,
            "original_dim": self.original_dim,
            "target_dim": self.target_dim,
            "actual_components": self.actual_components,
            "explained_variance_ratio": float(self.explained_variance_ratio.sum()),
            "individual_explained_variance": self.explained_variance_ratio.tolist(),
            "algorithm": self.algorithm,
            "random_state": self.random_state
        }
    
    def get_reconstruction_error(self, X: torch.Tensor) -> float:
        """Calculate reconstruction error for the given data.
        
        Args:
            X: Original feature matrix
            
        Returns:
            Mean squared reconstruction error
            
        Raises:
            ValueError: If reducer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("SVD reducer must be fitted to calculate reconstruction error")
        
        # Transform and inverse transform
        X_reduced = self.transform(X)
        X_reconstructed = self.inverse_transform(X_reduced)
        
        # Calculate MSE
        if isinstance(X, torch.Tensor):
            mse = torch.mean((X - X_reconstructed) ** 2).item()
        else:
            mse = np.mean((X - X_reconstructed) ** 2)
        
        return mse
    
    def inverse_transform(self, X_reduced: torch.Tensor) -> torch.Tensor:
        """Inverse transform reduced features back to original space.
        
        Args:
            X_reduced: Reduced feature matrix of shape [N, target_dim]
            
        Returns:
            Reconstructed feature matrix of shape [N, original_dim]
            
        Raises:
            ValueError: If reducer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("SVD reducer must be fitted for inverse transform")
        
        device = X_reduced.device if isinstance(X_reduced, torch.Tensor) else None
        
        # Convert to numpy
        if isinstance(X_reduced, torch.Tensor):
            X_np = X_reduced.cpu().numpy()
        else:
            X_np = X_reduced
        
        # Handle dimension mismatch (take only actual components)
        if X_np.shape[1] > self.actual_components:
            X_np = X_np[:, :self.actual_components]
        elif X_np.shape[1] < self.actual_components:
            # Pad with zeros
            padded = np.zeros((X_np.shape[0], self.actual_components))
            padded[:, :X_np.shape[1]] = X_np
            X_np = padded
        
        # Inverse transform
        X_reconstructed = self.svd_model.inverse_transform(X_np)
        
        # Add back the mean
        X_reconstructed = X_reconstructed + self.mean_
        
        # Convert back to tensor
        if device is not None:
            return torch.tensor(X_reconstructed, dtype=torch.float32, device=device)
        else:
            return torch.tensor(X_reconstructed, dtype=torch.float32)
    
    def __repr__(self) -> str:
        """String representation of the SVD reducer."""
        if self.is_fitted:
            return (f"SVDFeatureReducer(fitted=True, "
                   f"original_dim={self.original_dim}, "
                   f"target_dim={self.target_dim}, "
                   f"explained_variance={self.explained_variance_ratio.sum():.4f})")
        else:
            return (f"SVDFeatureReducer(fitted=False, "
                   f"target_dim={self.target_dim})")
    
    def __getstate__(self):
        """Support for pickling."""
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        """Support for unpickling."""
        self.__dict__.update(state)