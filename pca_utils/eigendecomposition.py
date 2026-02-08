"""
Eigendecomposition Module

Functions for performing eigendecomposition on the covariance matrix.
"""

import numpy as np


def perform_eigendecomposition(cov_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of covariance matrix.
    
    Eigendecomposition decomposes the covariance matrix into eigenvalues and 
    eigenvectors. The eigenvectors represent the principal components (directions 
    of maximum variance), and the eigenvalues represent the amount of variance 
    explained by each principal component.
    
    Args:
        cov_matrix: Covariance matrix of shape (n_features, n_features)
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
        - eigenvalues: 1D array of shape (n_features,) containing eigenvalues
        - eigenvectors: 2D array of shape (n_features, n_features) where each 
          column is an eigenvector corresponding to the eigenvalue at the same index
    
    Raises:
        ValueError: If the covariance matrix is not square
        ValueError: If eigendecomposition fails (singular matrix or other numerical issues)
    
    Note:
        - Small negative eigenvalues (due to numerical errors) are set to zero
        - The function uses np.linalg.eig which returns complex eigenvalues/eigenvectors
          for non-symmetric matrices, but covariance matrices should be symmetric and 
          produce real eigenvalues
    """
    # Validate input
    if cov_matrix.ndim != 2:
        raise ValueError(f"Covariance matrix must be 2D, got {cov_matrix.ndim}D")
    
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError(
            f"Covariance matrix must be square, got shape {cov_matrix.shape}"
        )
    
    try:
        # Perform eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Convert to real values (covariance matrices should produce real eigenvalues)
        # Any imaginary parts are due to numerical errors
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Handle small negative eigenvalues (numerical errors)
        # These can occur due to floating-point precision issues
        if np.any(eigenvalues < -1e-10):
            raise ValueError(
                f"Covariance matrix has significantly negative eigenvalues: "
                f"min eigenvalue = {np.min(eigenvalues)}"
            )
        
        # Set small negative values to zero
        eigenvalues = np.maximum(eigenvalues, 0)
        
        return eigenvalues, eigenvectors
        
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to compute eigendecomposition - matrix may be singular: {str(e)}"
        )
